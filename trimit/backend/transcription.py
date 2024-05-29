from trimit.models import Video
from trimit.app import VOLUME_DIR
from trimit.utils.audio_utils import load_audio
from trimit.utils.cache import CacheMixin
from typing import Optional
import tempfile
import os
import subprocess


def estimate_time(words_from_segments, i, transcript_start, transcript_end):
    """adapted from https://github.com/m-bain/whisperX/issues/349"""
    word = words_from_segments[i]
    prev_word_end = transcript_start
    pointer = 1  # let the pointer to the next word start at 1
    if i > 0:
        prev_word_end = words_from_segments[i - 1][
            "end"
        ]  # grab the end from the last one (should always work)

    if i + pointer >= len(words_from_segments):
        next_word_start = transcript_end
    else:
        next_word_start = None
        try:
            next_word_start = words_from_segments[i + pointer][
                "start"
            ]  # try to grab the start from the next one
        except KeyError:  # if trying to grab results in the error
            pointer += 1  # we'll increment pointer to next

        while next_word_start is None and i + pointer < len(words_from_segments):
            try:
                next_word_start = words_from_segments[i + pointer][
                    "start"
                ]  # grab the start time from the next word
                # if successful: find difference, and then divide to find increment. Add increment to prev_word_end and assign.
                next_word_start = (
                    (next_word_start - prev_word_end) / pointer
                ) + prev_word_end
            except KeyError:
                pointer += 1  # if another error, increment the pointer
        if i + pointer >= len(words_from_segments):  # if we reach the end of the list
            next_word_start = transcript_end

    if next_word_start is None:
        next_word_start = transcript_end
    word["start"] = word.get("start", prev_word_end + 0.01)
    word["end"] = word.get("end", next_word_start - 0.01)
    word["score"] = word.get("score", 0.5)


def add_missing_times(align_result):
    for segment in align_result["segments"]:
        for i, word in enumerate(segment["words"]):
            if "start" not in word or "end" not in word:
                estimate_time(segment["words"], i, segment["start"], segment["end"])
            word["score"] = word.get("score", 0.5)

    if "start" not in align_result:
        align_result["start"] = align_result["segments"][0]["start"]
    if "end" not in align_result:
        align_result["end"] = align_result["segments"][-1]["end"]

    for i, word_segment in enumerate(align_result["word_segments"]):
        if "start" not in word_segment or "end" not in word_segment:
            estimate_time(
                align_result["word_segments"],
                i,
                align_result["start"],
                align_result["end"],
            )


class Transcription(CacheMixin):
    def __init__(self, model_dir, cache=None, volume_dir=VOLUME_DIR):
        self.volume_dir = volume_dir
        self.model_dir = model_dir
        self.initialized = False
        super().__init__(cache=cache, cache_prefix="transcription/")

    def lazy_init(self):
        if self.initialized:
            return
        import whisperx
        import torch

        self.load_align_model = whisperx.load_align_model
        self.align = whisperx.align
        self.device = "cpu"
        compute_type = "float32"
        if torch.cuda.is_available():
            self.device = "cuda"
            compute_type = "float16"
        self.whisper_model = whisperx.load_model(
            "large-v2",
            self.device,
            compute_type=compute_type,
            download_root=self.model_dir,
        )
        self.whisper_align_models = {
            "en": whisperx.load_align_model(
                language_code="en", device=self.device, model_dir=self.model_dir
            )
        }
        self.diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=os.environ["HF_API_KEY"], device=self.device
        )
        self.assign_word_speakers = whisperx.assign_word_speakers
        self.subprocess = subprocess
        self.tempfile = tempfile
        self.initialized = True

    def transcribe_audio(self, audio_file_path, sample_rate: int | None = None):
        if not self.initialized:
            self.lazy_init()
        print(f"Transcribing audio file: {audio_file_path}")
        waveform, _ = load_audio(audio_file_path, flatten=True, sr=sample_rate)
        result = self.whisper_model.transcribe(waveform)
        if not result["segments"]:
            print("No dialogue detected in any of passed audio.")
            # TODO is the right return
            return {}
        if "language" not in result or not result["language"]:
            print("No language detected for alignment.")
            # TODO is the right return
            return {}

        lang = result["language"]

        if lang not in self.whisper_align_models:
            try:
                self.whisper_align_models[lang] = self.load_align_model(
                    language_code=lang, device=self.device, model_dir=self.model_dir
                )
            except ValueError as e:
                print(
                    f"could not load alignment model for language {lang}, defaulting to english: {e}"
                )
                lang = "en"

        align_model, align_metadata = self.whisper_align_models[lang]
        align_result = self.align(
            result["segments"],
            align_model,
            align_metadata,
            waveform,
            self.device,
            return_char_alignments=False,
        )
        diarize_segments = self.diarize_model(waveform)
        align_result = self.assign_word_speakers(diarize_segments, align_result)
        add_missing_times(align_result)

        return align_result

    def transcribe_video(self, video: Video, with_cache: bool = True):
        return self.transcribe_videos([video], with_cache=with_cache)[video.md5_hash]

    def transcribe_videos(self, videos: list[Video], with_cache: bool = True):
        # TODO async/concurrent
        transcriptions = {}
        for video in videos:
            cache_result = None
            if with_cache:
                cache_result = self.cache_get(video.md5_hash)
            if cache_result is None:
                fp = video.audio_path(self.volume_dir)
                transcription = self.transcribe_audio(audio_file_path=fp)
                self.cache_set(video.md5_hash, transcription)
                transcriptions[video.md5_hash] = transcription
            else:
                transcriptions[video.md5_hash] = cache_result
        return transcriptions
