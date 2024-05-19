import pytest
from trimit.backend.speaker_in_frame_detection import SpeakerInFrameDetection
from trimit.backend.diarization import Diarization
from trimit.backend.transcription import Transcription
from trimit.utils.fs_utils import ensure_audio_path_on_volume
from .conftest import TEST_VOLUME_DIR, TEST_CACHE_DIR, TEST_MODEL_DIR
from pyannote.core import Segment
import diskcache as dc
import numpy as np

pytestmark = pytest.mark.asyncio(scope="session")


@pytest.mark.long
async def test_speaker_in_frame(test_video_1, test_video_2):
    await ensure_audio_path_on_volume(test_video_1, volume_dir=TEST_VOLUME_DIR)
    await ensure_audio_path_on_volume(test_video_2, volume_dir=TEST_VOLUME_DIR)
    diarization = Diarization(
        volume_dir=TEST_VOLUME_DIR, cache=dc.Cache(TEST_CACHE_DIR)
    )

    transcription = Transcription(
        TEST_MODEL_DIR, volume_dir=TEST_VOLUME_DIR, cache=dc.Cache(TEST_CACHE_DIR)
    )
    md5_hash_to_diarization = diarization.diarize_videos(
        [test_video_1, test_video_2], use_existing_output=True
    )
    test_video_1.diarization = md5_hash_to_diarization[test_video_1.md5_hash]
    test_video_2.diarization = md5_hash_to_diarization[test_video_2.md5_hash]
    md5_hash_to_transcription = transcription.transcribe_videos(
        [test_video_1, test_video_2]
    )
    test_video_1.transcription = md5_hash_to_transcription[test_video_1.md5_hash]
    test_video_2.transcription = md5_hash_to_transcription[test_video_2.md5_hash]

    speaker_in_frame_detection = SpeakerInFrameDetection(
        volume_dir=TEST_VOLUME_DIR, cache=dc.Cache(TEST_CACHE_DIR)
    )
    scenes = await speaker_in_frame_detection.detect_speaker_in_frame_from_videos(
        [test_video_1, test_video_2], nframes=10, use_existing_output=False
    )
    assert all(speaker_in_frame is not None for speaker_in_frame in scenes)
    assert len(scenes) == 11
    expected_speaker_in_frames = [True] * 3 + [False] * 8
    assert [scene.speaker_in_frame for scene in scenes] == expected_speaker_in_frames
