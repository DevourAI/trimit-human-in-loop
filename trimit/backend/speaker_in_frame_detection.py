from trimit.utils.openai import GPTMixin
from trimit.app import VOLUME_DIR
from trimit.utils.frame_extraction import extract_frames, encode_image
from trimit.utils.prompt_engineering import parse_prompt_template
from trimit.models import Video, Scene
from trimit.utils.scene_extraction import extract_scenes_to_disk
from trimit.utils.model_utils import get_scene_folder, get_frame_folder
from beanie import BulkWriter
from pathlib import Path
from tqdm import tqdm
import os


class SpeakerInFrameDetection(GPTMixin):
    def __init__(self, cache=None, volume_dir=VOLUME_DIR, min_scene_length_words=10):
        super().__init__(cache=cache, cache_prefix="speaker_in_frame/")
        self.volume_dir = volume_dir
        self.min_scene_length_words = min_scene_length_words

    # TODO should the video writing go here or outside?
    async def detect_speaker_in_frame_from_videos(
        self, videos: list[Video], nframes=10, use_existing_output=True
    ):
        all_scenes = []
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
                # TODO scene creation should go elsewhere, just pass in scenes here
                video_scenes = []
                cur_start_frame = 0
                cur_end_frame = 0
                cur_num_words = 0
                next_start_is_cur_start = True
                for segment in video.transcription["segments"]:
                    cur_num_words += len(segment["words"])
                    if next_start_is_cur_start:
                        cur_start_frame = int(
                            round(segment["start"] * video.frame_rate)
                        )
                    end_frame = int(round(segment["end"] * video.frame_rate))
                    next_start_is_cur_start = False
                    if cur_num_words > self.min_scene_length_words:
                        cur_end_frame = end_frame
                        scene = await Scene.from_video(
                            video, cur_start_frame, cur_end_frame, save=True
                        )
                        video_scenes.append(scene)
                        cur_start_frame = cur_end_frame + 1
                        cur_end_frame = cur_start_frame
                        cur_num_words = 0
                        next_start_is_cur_start = True

                if end_frame > cur_end_frame:
                    scene = await Scene.from_video(
                        video, cur_start_frame, end_frame, save=True
                    )
                    video_scenes.append(scene)

                video_path = video.path(self.volume_dir)
                output_dir = get_scene_folder(
                    self.volume_dir, video.user_email, video.upload_datetime
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                scenes_to_write_to_disk = []
                for scene in video_scenes:
                    output_file = str(Path(output_dir) / scene.filename)
                    if not use_existing_output or not os.path.exists(output_file):
                        scenes_to_write_to_disk.append(scene)
                if len(scenes_to_write_to_disk) > 0:
                    print(
                        f"Writing {len(scenes_to_write_to_disk)} scenes for video {video.md5_hash} to disk"
                    )
                    await extract_scenes_to_disk(
                        video_path,
                        scenes_to_write_to_disk,
                        output_dir,
                        frame_rate=video.frame_rate,
                        codec=video.codec,
                    )

                scenes_to_detect_speaker = []
                for scene in video_scenes:
                    if not use_existing_output or scene.speaker_in_frame is None:
                        scenes_to_detect_speaker.append(scene)
                scene_to_speaker_in_frame = self.detect_speaker_in_frame_from_scenes(
                    scenes_to_detect_speaker,
                    nframes=nframes,
                    use_existing_output=use_existing_output,
                )
                for scene in scenes_to_detect_speaker:
                    scene.speaker_in_frame = scene_to_speaker_in_frame[scene.name]
                    await scene.save_with_retry()
                all_scenes.extend(video_scenes)

        await bulk_writer.commit()
        return all_scenes

    def detect_speaker_in_frame_from_scenes(
        self, scenes: list[Scene], nframes=10, use_existing_output=True
    ):
        # TODO parallelize once we have bumped our openai subscription to allow that
        scene_to_speaker_in_frame = {}
        for scene in tqdm(scenes, desc="Detecting speakers in frame"):
            speaker_in_frame = self.detect_speaker_in_frame_from_scene(
                scene, nframes=nframes, use_existing_output=use_existing_output
            )
            scene_to_speaker_in_frame[scene.name] = speaker_in_frame
        return scene_to_speaker_in_frame

    def detect_speaker_in_frame_from_scene(
        self, scene: Scene, nframes=10, use_existing_output=True
    ):
        print(f"detecting speaker in frame for scene {scene.name}")
        if use_existing_output:
            is_speaking = self.get_is_speaking_from_cache(scene)
            if is_speaking is not None:
                print(f"found is_speaking in cache for scene {scene.name}")
                return is_speaking
        frame_buffer = None
        if use_existing_output:
            frame_buffer = self.get_frame_buffer_from_cache(scene)
        if frame_buffer is None:
            print(f"getting frame buffer for scene {scene.name}")
            output_folder = (
                Path(
                    get_frame_folder(
                        self.volume_dir, scene.user_email, scene.video.upload_datetime
                    )
                )
                / scene.name
            )
            frame_buffer = extract_frames(
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
            self.save_frame_buffer_to_cache(scene, frame_buffer)
        base64_frame = encode_image(frame_buffer)
        prompt = parse_prompt_template("speaker_in_frame")
        print(f"calling gpt for scene {scene.name}")
        response = self.call_gpt(
            prompt,
            use_existing_output=use_existing_output,
            model="gpt-4o",
            base64_images=[base64_frame],
        )
        # TODO perhaps more complex parsing
        is_speaking = '{"speaking": true}' in response
        print(f"is_speaking: {is_speaking}")
        if not is_speaking:
            if '{"speaking": false}' not in response:
                print(f"Unexpected response from GPT, returning False: {response}")
        self.save_is_speaking_to_cache(scene, is_speaking)
        print(f"saved is_spaking to cache for scene {scene.name}")
        return is_speaking

    def save_frame_buffer_to_cache(self, scene, frame_buffer):
        self.cache.set(f"frame_buffer/{scene.name}", frame_buffer)

    def get_frame_buffer_from_cache(self, scene):
        return self.cache.get(f"frame_buffer/{scene.name}")

    def save_is_speaking_to_cache(self, scene, is_speaking):
        self.cache.set(f"is_speaking/{scene.name}", is_speaking)

    def get_is_speaking_from_cache(self, scene):
        return self.cache.get(f"is_speaking/{scene.name}")
