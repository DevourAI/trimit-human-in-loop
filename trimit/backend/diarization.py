import os
from beanie.operators import In
import asyncio
from trimit.app import VOLUME_DIR
from trimit.models import Video, DiarizationSegment, SpeechSegment
from trimit.utils.audio_utils import load_multiple_audio_files_as_single_waveform
from trimit.utils.cache import CacheMixin


class Diarization(CacheMixin):
    def __init__(self, cache=None, volume_dir=VOLUME_DIR, min_duration_off=0.1):
        from pyannote.audio import Pipeline
        import torch

        super().__init__(cache=cache, cache_prefix="diarization/")
        self.volume_dir = volume_dir
        self.speaker_to_centroids = self.cache_get("speaker_to_centroids", None)
        self.pipeline = self.cache_get(
            "pipeline",
            Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=os.environ["HF_API_KEY"],
            ),
        )
        self.pipeline.segmentation.min_duration_off = min_duration_off

        if torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))
        else:
            print("Diarization: CUDA not available. Using CPU.")

    def diarize_audio_files(
        self,
        audio_file_paths,
        use_existing_output=True,
        sample_rate=None,
        buffer_s=5,
        min_speakers=None,
    ):
        waveform, segments, sample_rate = load_multiple_audio_files_as_single_waveform(
            [
                fp
                for fp in audio_file_paths
                if not use_existing_output or not self.in_cache(fp)
            ],
            sample_rate=sample_rate,
            flatten=False,
            buffer_s=buffer_s,
        )
        cache_output = {
            audio_file_path: self.cache_get(audio_file_path)
            for audio_file_path in audio_file_paths
            if use_existing_output and self.in_cache(audio_file_path)
        }
        if len(segments) == 0:
            return cache_output

        diarization_segments = self.diarize_waveform_with_segments(
            segments, waveform, sample_rate, min_speakers=min_speakers
        )
        audio_file_path_to_diarization_segments = {}
        for audio_file_path, diarization_segment in zip(
            audio_file_paths, diarization_segments
        ):
            audio_file_path_to_diarization_segments[audio_file_path] = (
                diarization_segment
            )
            self.cache_set(audio_file_path, diarization_segment)
        return audio_file_path_to_diarization_segments

    def diarize_waveform_with_segments(
        self, segments, waveform, sample_rate, min_speakers=None
    ):
        from pyannote.audio.pipelines.utils.hook import ProgressHook

        with ProgressHook() as hook:
            diarization, centroids = self.pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                hook=hook,
                return_embeddings=True,
                min_speakers=min_speakers,
            )
        new_speaker_to_centroids = get_speaker_to_centroids(
            diarization.labels(), centroids
        )
        if self.speaker_to_centroids is None:
            self.speaker_to_centroids = new_speaker_to_centroids
        else:
            self.speaker_to_centroids.update(new_speaker_to_centroids)
        self.cache_set("speaker_to_centroids", self.speaker_to_centroids)
        # TODO this might not work if on gpu
        self.cache_set("pipeline", self.pipeline)
        return self.realign_segments(diarization, segments)

    def realign_segments(self, diarization, segments, turn_boundary_size_cutoff=1):
        from pyannote.core import Annotation, Segment

        current_segment_index = 0
        seconds_before_current = 0
        separated_diarizations = [Annotation()]
        # TODO use methods from Segment (and, or, overlaps, etc)
        for turn, track, speaker in diarization.itertracks(yield_label=True):
            current_segment = segments[current_segment_index]
            while turn.start >= seconds_before_current + current_segment.end:
                seconds_before_current += current_segment.end - current_segment.start
                current_segment_index += 1
                separated_diarizations.append(Annotation())
                if current_segment_index >= len(segments):
                    break
                current_segment = segments[current_segment_index]
            if current_segment_index >= len(segments):
                break
            if turn.end > seconds_before_current + current_segment.end:
                split_turn = Segment(
                    start=turn.start - seconds_before_current,
                    end=seconds_before_current + current_segment.end,
                )

                turn = Segment(
                    start=seconds_before_current + current_segment.end, end=turn.end
                )

                # This is a heuristic for fuzziness between segment (audio file) boundaries
                if (
                    split_turn.duration > turn_boundary_size_cutoff
                    or turn.duration < turn_boundary_size_cutoff
                ):
                    separated_diarizations[-1][(split_turn, track)] = speaker
                elif turn.duration < 1:
                    turn = None
                current_segment_index += 1
                seconds_before_current += current_segment.end - current_segment.start
                if current_segment_index >= len(segments):
                    break
                separated_diarizations.append(Annotation())
            if turn is not None:
                turn = Segment(
                    start=turn.start - seconds_before_current,
                    end=turn.end - seconds_before_current,
                )
                separated_diarizations[-1][(turn, track)] = speaker
        return separated_diarizations

    def diarize_audio_file(
        self, audio_file_path, use_existing_output=True, buffer_s=5, min_speakers=None
    ):
        return self.diarize_audio_files(
            audio_file_path,
            use_existing_output=use_existing_output,
            buffer_s=buffer_s,
            min_speakers=min_speakers,
        )

    def diarize_video(self, video: Video, use_existing_output=True):
        return self.diarize_audio_file(
            video.audio_path(self.volume_dir),
            use_existing_output=use_existing_output,
            buffer_s=5,
        )

    def diarize_videos(
        self,
        videos: list[Video],
        sample_rate=None,
        use_existing_output=True,
        buffer_s=5,
        min_speakers=None,
    ):
        # multiple threads not advantageous unless we actually split to multiple GPUs
        # TODO: figure out multiple gpus

        audio_file_paths = [video.audio_path(self.volume_dir) for video in videos]
        audio_file_path_to_diarization_segments = self.diarize_audio_files(
            audio_file_paths,
            use_existing_output=use_existing_output,
            sample_rate=sample_rate,
            buffer_s=buffer_s,
            min_speakers=min_speakers,
        )
        self.cache_set("existing_video_hashes", [video.md5_hash for video in videos])
        return {
            video.md5_hash: audio_file_path_to_diarization_segments[audio_file_path]
            for video, audio_file_path in zip(videos, audio_file_paths)
        }

    async def match_new_speakers(
        self,
        new_videos: list[Video],
        sample_rate=None,
        use_existing_output=True,
        buffer_s=5,
        min_speakers=None,
    ):
        existing_video_hashes = self.cache_get("existing_video_hashes", [])
        existing_videos = await Video.find(
            In(Video.md5_hash, existing_video_hashes)
        ).to_list()
        # TODO when pyannote guy gets back to me
        return self.diarize_videos(
            existing_videos + new_videos,
            sample_rate=sample_rate,
            use_existing_output=use_existing_output,
            buffer_s=buffer_s,
            min_speakers=min_speakers,
        )
        if self.speaker_to_centroids is None:
            raise ValueError(
                "No speaker centroids to match against. First run diarize_videos."
            )
        #  from scipy.spatial.distance import cdist
        #  from pyannote.audio import Audio
        import torch

        if sample_rate is None:
            sample_rate = 16000
        video_save_tasks = []
        for video in new_videos:
            # TODO figure out how to convert my segments to pyannote binary_segmentations or just use theirs in the beginning
            #  speech_segments = video.speech_segments
            #  if speech_segments is None:
            #  speech_segments = [SpeechSegment(start=0, end=-1)]
            diarization_segments = []
            waveform = load_audio_with_segments(
                video.audio_path(self.volume_dir),
                [SpeechSegment(start=0, end=-1)],
                flatten=False,
                sample_rate=sample_rate,
            )[0]
            # audio = Audio()({'waveform': waveform, 'sample_rate': sample_rate})
            #  segmentations = self.pipeline.get_segmentations(audio)
            #  segmentations = self.pipeline.get_segmentations({'waveform': waveform, 'sample_rate': sample_rate})
            segmentations = self.pipeline.get_segmentations(
                {"waveform": torch.from_numpy(waveform), "sample_rate": sample_rate}
            )
            #  splits = [(segment, data) for segment, data in segmentations]
            #  embeddings = self.pipeline.get_embeddings(audio,segmentations)
            #  inactive_speakers = np.sum(segmentations.data, axis=1) == 0

            discrete_diarization = self.pipeline.reconstruct(
                segmentations, self.pipeline.hard_clusters, self.pipeline.count
            )
            diarization = self.pipeline.to_annotation(
                discrete_diarization,
                min_duration_on=0.0,
                min_duration_off=self.pipeline.segmentation.min_duration_off,
            )
            mapping = {
                label: expected_label
                for label, expected_label in zip(
                    diarization.labels(), self.pipeline.classes()
                )
            }
            diarization = diarization.rename_labels(mapping=mapping)
            video.diarization = [
                DiarizationSegment(start=turn.start, end=turn.end, speaker=speaker)
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]
            breakpoint()
            #  for (segment, _), segment_embedding in zip(splits, embeddings):
            #  min_distance_idx = np.argmin(
            #  [
            #  np.min(
            #  cdist(
            #  segment_embedding,
            #  center[np.newaxis, :],
            #  metric="cosine",
            #  )
            #  )
            #  for center in self.speaker_to_centroids.values()
            #  ]
            #  )
            #  speaker = list(self.speaker_to_centroids.keys())[min_distance_idx]
            #  breakpoint()
            #  diarization_segments.append(
            #  DiarizationSegment(
            #  start=segment.start, end=segment.end, speaker=speaker
            #  )
            #  )
            # video.diarization = diarization_segments
            video_save_tasks.append(video.save_with_retry())

        await asyncio.gather(*video_save_tasks)


def get_speaker_to_centroids(labels, centroids):
    return {speaker: centroid for speaker, centroid in zip(labels, centroids)}
