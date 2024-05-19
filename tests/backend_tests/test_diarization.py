import pytest
from trimit.backend.diarization import Diarization
from trimit.utils.fs_utils import ensure_audio_path_on_volume
from .conftest import TEST_VOLUME_DIR, TEST_CACHE_DIR
from pyannote.core import Segment
import diskcache as dc
import numpy as np

pytestmark = pytest.mark.asyncio(scope="session")


def check_expected_segments(expected_segments, actual_segments):
    for i, (segment, track, speaker) in enumerate(
        actual_segments.itertracks(yield_label=True)
    ):
        expected_segment, expected_track, expected_speaker = expected_segments[i]
        assert np.isclose(segment.start, expected_segment.start)
        assert np.isclose(segment.end, expected_segment.end)
        assert speaker == expected_speaker
        assert track == expected_track


@pytest.mark.long
async def test_diarize_videos(test_video_1, test_video_2):
    await ensure_audio_path_on_volume(test_video_1, volume_dir=TEST_VOLUME_DIR)
    await ensure_audio_path_on_volume(test_video_2, volume_dir=TEST_VOLUME_DIR)
    diarization = Diarization(
        volume_dir=TEST_VOLUME_DIR, cache=dc.Cache(TEST_CACHE_DIR)
    )
    md5_hash_to_diarization_segments = diarization.diarize_videos(
        [test_video_1, test_video_2], use_existing_output=True
    )
    video_1_expected_segments = [
        (Segment(5.00909, 6.59534), "A", "SPEAKER_01"),
        (Segment(17.6147, 18.1547), "B", "SPEAKER_02"),
        (Segment(18.9647, 19.3528), "C", "SPEAKER_03"),
        (Segment(20.7535, 21.3272), "D", "SPEAKER_03"),
        (Segment(22.5085, 23.0485), "E", "SPEAKER_03"),
        (Segment(26.1535, 26.7272), "F", "SPEAKER_03"),
    ]
    video_2_expected_segments = [
        (Segment(30.3553, 31.6885), "G", "SPEAKER_04"),
        (Segment(33.4097, 34.6753), "H", "SPEAKER_00"),
        (Segment(34.9622, 35.6203), "I", "SPEAKER_04"),
        (Segment(37.4428, 39.4678), "J", "SPEAKER_00"),
        (Segment(40.5647, 43.6697), "K", "SPEAKER_04"),
    ]

    expected_video_hashes = [
        "bbb6141b26fb79cca20c7ecee899e9bb",
        "c4064248015b0ca75d95ee34f7cea8cb",
    ]
    for video_hash, expected_segments in zip(
        expected_video_hashes, [video_1_expected_segments, video_2_expected_segments]
    ):
        check_expected_segments(
            expected_segments, md5_hash_to_diarization_segments[video_hash]
        )


@pytest.mark.long
async def test_match_new_speaker(test_video_1, test_video_2, test_video_3):
    await ensure_audio_path_on_volume(test_video_1, volume_dir=TEST_VOLUME_DIR)
    await ensure_audio_path_on_volume(test_video_2, volume_dir=TEST_VOLUME_DIR)
    await ensure_audio_path_on_volume(test_video_3, volume_dir=TEST_VOLUME_DIR)
    test_video_2.diarization = []
    await test_video_2.save_with_retry()
    diarization = Diarization(
        volume_dir=TEST_VOLUME_DIR, cache=dc.Cache(TEST_CACHE_DIR)
    )
    md5_hash_to_diarization_segments = diarization.diarize_videos(
        [test_video_1, test_video_3], use_existing_output=False
    )
    test_video_1.diarization = md5_hash_to_diarization_segments[test_video_1.md5_hash]
    await test_video_1.save_with_retry()
    test_video_3.diarization = md5_hash_to_diarization_segments[test_video_3.md5_hash]
    await test_video_3.save_with_retry()

    assert len(test_video_1.diarization) > 0
    assert len(test_video_3.diarization) > 0
    assert len(test_video_2.diarization) == 0

    md5_hash_to_diarization_segments = await diarization.match_new_speakers(
        [test_video_2], use_existing_output=False
    )
    actual_segments = md5_hash_to_diarization_segments[test_video_2.md5_hash]
    expected_segments = [
        (Segment(30.3553, 31.6885), "G", "SPEAKER_04"),
        (Segment(33.4097, 34.6753), "H", "SPEAKER_00"),
        (Segment(34.9622, 35.6203), "I", "SPEAKER_04"),
        (Segment(37.4428, 39.4678), "J", "SPEAKER_00"),
        (Segment(40.5647, 43.6697), "K", "SPEAKER_04"),
    ]
    # TODO this doesnt produce similar segments and diarizations
    check_expected_segments(expected_segments, actual_segments)
