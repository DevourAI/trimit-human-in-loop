import io
from tqdm.asyncio import tqdm as tqdm_async
from schema import Schema
from trimit.utils.cache import CacheMixin
from trimit.app import VOLUME_DIR
from trimit.utils.frame_extraction import extract_frames, encode_image
from trimit.utils.prompt_engineering import parse_prompt_template
from trimit.models import Video, Scene
from trimit.utils.scene_extraction import extract_scenes_to_disk
from trimit.utils.model_utils import get_scene_folder, get_frame_folder
from trimit.backend.utils import get_agent_output_modal_or_local
from trimit.models.backend_models import FinalLLMOutput
from beanie import BulkWriter
from pathlib import Path
import random
from collections import defaultdict
import os


class SpeakerInFrameDetection(CacheMixin):
    def __init__(self, cache=None, volume_dir=VOLUME_DIR):
        super().__init__(cache=cache, cache_prefix="speaker_in_frame/")
        self.volume_dir = volume_dir

    async def detect_speaker_in_frame_from_videos(
        self,
        videos: list[Video],
        nframes=10,
        use_existing_output=True,
        nsamples_per_speaker=5,
        min_word_count_per_scene=10,
    ):
        async with BulkWriter() as bulk_writer:
            for video in videos:
                if not video.transcription:
                    print(
                        f"Skipping video {video.md5_hash} because it has no transcription"
                    )
                    continue
                if not video.frame_rate:
                    print(
                        f"Skipping video {video.md5_hash} because it has no frame rate"
                    )
                    continue
                print(f"Detecting speakers in frame for video {video.md5_hash}")
                if use_existing_output and video.speakers_in_frame:
                    print(
                        f"Skipping video {video.md5_hash} because speakers in frame already detected"
                    )
                    continue

                video_path = video.path(self.volume_dir)
                output_dir = get_scene_folder(
                    self.volume_dir, video.user_email, video.upload_datetime
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                # TODO scene creation should go elsewhere, just pass in scenes here
                speakers_to_segments = defaultdict(list)
                all_speakers = set()
                for segment in video.transcription["segments"]:
                    speaker = segment.get("speaker")
                    all_speakers.add(speaker)
                    speakers_in_segment = set(
                        [
                            word["speaker"]
                            for word in segment["words"]
                            if "speaker" in word
                        ]
                    )
                    # Assume all actual interviewees (speakers) will have an entire segment that's just them speaking
                    if len(speakers_in_segment) == 0:
                        print(f"Skipping segment with no speakers: {segment}")
                        continue
                    elif len(speakers_in_segment) == 1:
                        speaker = list(speakers_in_segment)[0]
                        speakers_to_segments[speaker].append(segment)
                speaker_segment_samples = {
                    speaker: random.sample(
                        segments, min(nsamples_per_speaker, len(segments))
                    )
                    for speaker, segments in speakers_to_segments.items()
                    if len(segments)
                }
                speaker_to_scenes = defaultdict(list)

                scenes_to_write_to_disk = []
                scenes_to_detect_speaker = defaultdict(list)

                # Assumes only the same speakers are in or out of frame for whole video
                for speaker, segments in speaker_segment_samples.items():
                    for segment in segments:
                        start_frame = int(round(segment["start"] * video.frame_rate))
                        end_frame = int(round(segment["end"] * video.frame_rate))
                        scene = await Scene.from_video(
                            video, start_frame, end_frame, save=True
                        )
                        speaker_to_scenes[speaker].append(scene)
                        output_file = str(Path(output_dir) / scene.filename)
                        scenes_to_detect_speaker[speaker].append(scene)
                        if not os.path.exists(output_file):
                            scenes_to_write_to_disk.append(scene)

                if len(scenes_to_write_to_disk) > 0:
                    print(
                        f"Writing {len(scenes_to_write_to_disk)} scenes for video {video.md5_hash} to disk"
                    )
                    await extract_scenes_to_disk(
                        video_path,
                        scenes_to_write_to_disk,
                        str(output_dir),
                        frame_rate=video.frame_rate,
                        codec=video.codec,
                    )

                speaker_to_speaker_in_frame = (
                    await self.detect_all_speakers_from_speaker_scene_dict(
                        scenes_to_detect_speaker,
                        nframes=nframes,
                        use_existing_output=use_existing_output,
                    )
                )
                video.speakers_in_frame = []
                for speaker, in_frame in speaker_to_speaker_in_frame.items():
                    if in_frame:
                        existing_speakers_in_frame = video.speakers_in_frame or []
                        video.speakers_in_frame = sorted(
                            list(set(existing_speakers_in_frame + [speaker]))
                        )
                await video.save_with_retry()

        await bulk_writer.commit()

    async def detect_all_speakers_from_speaker_scene_dict(
        self,
        speaker_to_scenes: dict[str, list[Scene]],
        nframes=10,
        use_existing_output=True,
    ):
        tasks = []
        for speaker, scenes in speaker_to_scenes.items():
            tasks.append(
                self.detect_speaker_in_frame_from_scenes(
                    scenes, nframes=nframes, use_existing_output=use_existing_output
                )
            )
        task_results = await tqdm_async.gather(
            *tasks, desc="Detecting speakers in frame"
        )
        speaker_to_speaker_in_frame = {}
        for speaker, speaker_in_frame in zip(speaker_to_scenes.keys(), task_results):
            speaker_to_speaker_in_frame[speaker] = speaker_in_frame
        return speaker_to_speaker_in_frame

    async def detect_speaker_in_frame_from_scenes(
        self, scenes: list[Scene], nframes=10, use_existing_output=True
    ):
        frame_bytes = [None] * len(scenes)
        for i, scene in enumerate(scenes):
            if use_existing_output:
                _frame_bytes = self.get_frame_bytes_from_cache(scene)
                if _frame_bytes:
                    frame_bytes[i] = _frame_bytes
                    continue
            print(f"getting frame buffer for scene {scene.name}")
            output_folder = (
                Path(
                    get_frame_folder(
                        self.volume_dir, scene.user_email, scene.video.upload_datetime
                    )
                )
                / scene.name
            )
            frame_buffer = await extract_frames(
                scene.path(self.volume_dir),
                output_folder=output_folder,
                output_to_buffer=True,
                # TODO add back when this is accurate
                # total_duration=scene.video.duration,
                use_existing_output=use_existing_output,
                # TODO calculate this from duration, and nframes
                # max_frame_rate=
            )
            if frame_buffer is None:
                print(f"Could not extract frames for scene {scene.name}")
                return False
            assert isinstance(frame_buffer, io.BytesIO)
            _frame_bytes = frame_buffer.read()
            self.save_frame_bytes_to_cache(scene, _frame_bytes)
            frame_bytes[i] = _frame_bytes
        prompt = parse_prompt_template("speaker_in_frame")
        schema = Schema({"is_speaking": bool}).json_schema("SpeakerInFrameDetection")
        output = None
        async for output, is_last in get_agent_output_modal_or_local(
            prompt, json_mode=True, schema=schema, images=frame_bytes
        ):
            if is_last:
                break
        if (
            not isinstance(output, FinalLLMOutput)
            or not isinstance(output.json_value, dict)
            or "is_speaking" not in output.json_value
        ):
            print(f"Unexpected response from GPT, returning False: {output}")
            return False
        is_speaking = output.json_value["is_speaking"]
        return is_speaking

    def save_frame_bytes_to_cache(self, scene, frame_bytes):
        self.cache_set(f"frame_bytes/{scene.name}", frame_bytes)

    def get_frame_bytes_from_cache(self, scene):
        return self.cache_get(f"frame_bytes/{scene.name}")
