import pytest
from trimit.backend.speaker_in_frame_detection import SpeakerInFrameDetection
from trimit.backend.transcription import Transcription
from trimit.utils.fs_utils import ensure_audio_path_on_volume
from ..conftest import TEST_VOLUME_DIR, TEST_CACHE_DIR, TEST_MODEL_DIR
import diskcache as dc

pytestmark = pytest.mark.asyncio(scope="session")


@pytest.mark.long
async def test_speaker_in_frame(conversation_video, kitchen_conversation_video):
    test_video_1 = conversation_video
    test_video_2 = kitchen_conversation_video
    await ensure_audio_path_on_volume(test_video_1, volume_dir=TEST_VOLUME_DIR)
    await ensure_audio_path_on_volume(test_video_2, volume_dir=TEST_VOLUME_DIR)

    transcription = Transcription(
        TEST_MODEL_DIR, volume_dir=TEST_VOLUME_DIR, cache=dc.Cache(TEST_CACHE_DIR)
    )

    md5_hash_to_transcription = transcription.transcribe_videos(
        [test_video_1, test_video_2], with_cache=True
    )
    test_video_1.transcription = md5_hash_to_transcription[test_video_1.md5_hash]
    test_video_2.transcription = md5_hash_to_transcription[test_video_2.md5_hash]

    speaker_in_frame_detection = SpeakerInFrameDetection(
        volume_dir=TEST_VOLUME_DIR, cache=dc.Cache(TEST_CACHE_DIR)
    )
    await speaker_in_frame_detection.detect_speaker_in_frame_from_videos(
        [test_video_1, test_video_2], use_existing_output=False
    )
    assert test_video_1.speakers_in_frame == []
    assert test_video_2.speakers_in_frame == ["SPEAKER_00"]
