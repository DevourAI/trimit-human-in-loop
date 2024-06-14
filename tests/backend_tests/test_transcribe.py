import pytest
from trimit.utils.fs_utils import ensure_audio_path_on_volume
from trimit.backend.transcription import add_missing_speakers, add_missing_times
from ..conftest import TEST_VOLUME_DIR
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
async def test_add_missing_speakers():
    align_result = {
        "segments": [
            {"words": [{"word": "word1", "start": 1, "end": 2}]},
            {
                "words": [
                    {"word": "word2", "speaker": "SPEAKER_01", "start": 1, "end": 2},
                    {"word": "word3", "speaker": "SPEAKER_01", "start": 2, "end": 3},
                ],
                "speaker": "SPEAKER_01",
            },
            {"words": [{"word": "word4", "start": 3, "end": 4}]},
            {"words": [{"word": "word5", "start": 6, "end": 7}]},
            {
                "words": [
                    {"word": "word6", "start": 7, "end": 8},
                    {"word": "word7", "start": 8, "end": 9, "speaker": "SPEAKER_02"},
                ],
                "speaker": "SPEAKER_02",
            },
            {
                "words": [
                    {"word": "word8", "speaker": "SPEAKER_02", "start": 7, "end": 8},
                    {"word": "word9", "speaker": "SPEAKER_02", "start": 8, "end": 9},
                ],
                "speaker": "SPEAKER_02",
            },
            {"words": [{"word": "word10", "start": 9, "end": 10}]},
        ],
        "word_segments": [],
    }
    for seg in align_result["segments"]:
        for word in seg["words"]:
            align_result["word_segments"].append(word)
    add_missing_times(align_result)
    add_missing_speakers(align_result)
    desired_seg_speakers = [
        "SPEAKER_01",
        "SPEAKER_01",
        "SPEAKER_01",
        "SPEAKER_02",
        "SPEAKER_02",
        "SPEAKER_02",
        "SPEAKER_02",
    ]
    desired_word_speakers = [
        "SPEAKER_01",
        "SPEAKER_01",
        "SPEAKER_01",
        "SPEAKER_01",
        "SPEAKER_02",
        "SPEAKER_02",
        "SPEAKER_02",
        "SPEAKER_02",
        "SPEAKER_02",
        "SPEAKER_02",
    ]
    segment_speakers = [s.get("speaker") for s in align_result["segments"]]
    word_speakers = [
        w.get("speaker") for s in align_result["segments"] for w in s["words"]
    ]
    word_seg_speakers = [w.get("speaker") for w in align_result["word_segments"]]
    assert segment_speakers == desired_seg_speakers
    assert word_speakers == desired_word_speakers
    assert word_seg_speakers == desired_word_speakers


@pytest.mark.long
async def test_transcribe_videos(test_video_1, test_video_2, transcription):
    await ensure_audio_path_on_volume(test_video_1, volume_dir=TEST_VOLUME_DIR)
    await ensure_audio_path_on_volume(test_video_2, volume_dir=TEST_VOLUME_DIR)
    await test_video_1.save_with_retry()
    await test_video_2.save_with_retry()

    md5_hash_to_transcription = transcription.transcribe_videos(
        [test_video_1, test_video_2], with_cache=False
    )
    no_speaker = [
        (md5_hash, t)
        for (md5_hash, t) in md5_hash_to_transcription.items()
        if len([seg for seg in t["segments"] if seg.get("speaker") is not None]) == 0
    ]
    assert len(no_speaker) == 0
    assert md5_hash_to_transcription == {
        "bbb6141b26fb79cca20c7ecee899e9bb": {
            "segments": [
                {
                    "start": 5.189,
                    "end": 18.077,
                    "text": " or my favorite, queso.",
                    "words": [
                        {
                            "word": "or",
                            "start": 5.189,
                            "end": 5.309,
                            "score": 0.752,
                            "speaker": "SPEAKER_01",
                        },
                        {
                            "word": "my",
                            "start": 5.689,
                            "end": 5.91,
                            "score": 0.884,
                            "speaker": "SPEAKER_01",
                        },
                        {
                            "word": "favorite,",
                            "start": 6.01,
                            "end": 6.51,
                            "score": 0.844,
                            "speaker": "SPEAKER_01",
                        },
                        {
                            "word": "queso.",
                            "start": 17.657,
                            "end": 18.077,
                            "score": 0.661,
                            "speaker": "SPEAKER_02",
                        },
                    ],
                    "speaker": "SPEAKER_01",
                },
                {
                    "start": 20.819,
                    "end": 21.259,
                    "text": "Queso.",
                    "words": [
                        {
                            "word": "Queso.",
                            "start": 20.819,
                            "end": 21.259,
                            "score": 0.599,
                            "speaker": "SPEAKER_03",
                        }
                    ],
                    "speaker": "SPEAKER_02",
                },
                {
                    "start": 26.223,
                    "end": 26.623,
                    "text": "Queso.",
                    "words": [
                        {
                            "word": "Queso.",
                            "start": 26.223,
                            "end": 26.623,
                            "score": 0.794,
                            "speaker": "SPEAKER_03",
                        }
                    ],
                    "speaker": "SPEAKER_03",
                },
            ],
            "word_segments": [
                {
                    "word": "or",
                    "start": 5.189,
                    "end": 5.309,
                    "score": 0.752,
                    "speaker": "SPEAKER_01",
                },
                {
                    "word": "my",
                    "start": 5.689,
                    "end": 5.91,
                    "score": 0.884,
                    "speaker": "SPEAKER_01",
                },
                {
                    "word": "favorite,",
                    "start": 6.01,
                    "end": 6.51,
                    "score": 0.844,
                    "speaker": "SPEAKER_01",
                },
                {
                    "word": "queso.",
                    "start": 17.657,
                    "end": 18.077,
                    "score": 0.661,
                    "speaker": "SPEAKER_02",
                },
                {
                    "word": "Queso.",
                    "start": 20.819,
                    "end": 21.259,
                    "score": 0.599,
                    "speaker": "SPEAKER_03",
                },
                {
                    "word": "Queso.",
                    "start": 26.223,
                    "end": 26.623,
                    "score": 0.794,
                    "speaker": "SPEAKER_03",
                },
            ],
        },
        "c4064248015b0ca75d95ee34f7cea8cb": {
            "segments": [
                {
                    "start": 1.238,
                    "end": 2.299,
                    "text": " Yeah, sorry, what is his number?",
                    "words": [
                        {
                            "word": "Yeah,",
                            "start": 1.238,
                            "end": 1.478,
                            "score": 0.719,
                            "speaker": "SPEAKER_04",
                        },
                        {
                            "word": "sorry,",
                            "start": 1.498,
                            "end": 1.718,
                            "score": 0.806,
                            "speaker": "SPEAKER_04",
                        },
                        {
                            "word": "what",
                            "start": 1.738,
                            "end": 1.858,
                            "score": 0.976,
                            "speaker": "SPEAKER_04",
                        },
                        {
                            "word": "is",
                            "start": 1.898,
                            "end": 1.938,
                            "score": 0.994,
                            "speaker": "SPEAKER_04",
                        },
                        {
                            "word": "his",
                            "start": 1.998,
                            "end": 2.058,
                            "score": 0.979,
                            "speaker": "SPEAKER_04",
                        },
                        {
                            "word": "number?",
                            "start": 2.098,
                            "end": 2.299,
                            "score": 0.918,
                            "speaker": "SPEAKER_04",
                        },
                    ],
                    "speaker": "SPEAKER_04",
                },
                {
                    "start": 4.26,
                    "end": 5.26,
                    "text": "My case number.",
                    "words": [
                        {
                            "word": "My",
                            "start": 4.26,
                            "end": 4.4,
                            "score": 0.648,
                            "speaker": "SPEAKER_04",
                        },
                        {
                            "word": "case",
                            "start": 4.44,
                            "end": 4.8,
                            "score": 0.484,
                            "speaker": "SPEAKER_04",
                        },
                        {
                            "word": "number.",
                            "start": 4.84,
                            "end": 5.26,
                            "score": 0.494,
                            "speaker": "SPEAKER_04",
                        },
                    ],
                    "speaker": "SPEAKER_04",
                },
                {
                    "start": 5.82,
                    "end": 6.341,
                    "text": "OK, yeah.",
                    "words": [
                        {
                            "word": "OK,",
                            "start": 5.82,
                            "end": 6.141,
                            "score": 0.418,
                            "speaker": "SPEAKER_04",
                        },
                        {
                            "word": "yeah.",
                            "start": 6.181,
                            "end": 6.341,
                            "score": 0.648,
                            "speaker": "SPEAKER_04",
                        },
                    ],
                    "speaker": "SPEAKER_04",
                },
                {
                    "start": 8.322,
                    "end": 9.122,
                    "text": "Oh, fuck.",
                    "words": [
                        {
                            "word": "Oh,",
                            "start": 8.322,
                            "end": 8.482,
                            "score": 0.81,
                            "speaker": "SPEAKER_04",
                        },
                        {
                            "word": "fuck.",
                            "start": 8.762,
                            "end": 9.122,
                            "score": 0.57,
                            "speaker": "SPEAKER_04",
                        },
                    ],
                    "speaker": "SPEAKER_04",
                },
                {
                    "start": 9.482,
                    "end": 10.203,
                    "text": "Your case number.",
                    "words": [
                        {
                            "word": "Your",
                            "start": 9.482,
                            "end": 9.662,
                            "score": 0.849,
                            "speaker": "SPEAKER_04",
                        },
                        {
                            "word": "case",
                            "start": 9.802,
                            "end": 10.022,
                            "score": 0.467,
                            "speaker": "SPEAKER_04",
                        },
                        {
                            "word": "number.",
                            "start": 10.043,
                            "end": 10.203,
                            "score": 0.14,
                            "speaker": "SPEAKER_04",
                        },
                    ],
                    "speaker": "SPEAKER_04",
                },
                {
                    "start": 11.483,
                    "end": 11.763,
                    "text": "Sorry.",
                    "words": [
                        {
                            "word": "Sorry.",
                            "start": 11.483,
                            "end": 11.763,
                            "score": 0.948,
                            "speaker": "SPEAKER_04",
                        }
                    ],
                    "speaker": "SPEAKER_04",
                },
                {
                    "start": 11.783,
                    "end": 12.844,
                    "text": "Sorry, I'm not something.",
                    "words": [
                        {
                            "word": "Sorry,",
                            "start": 11.783,
                            "end": 12.084,
                            "score": 0.309,
                            "speaker": "SPEAKER_04",
                        },
                        {
                            "word": "I'm",
                            "start": 12.104,
                            "end": 12.264,
                            "score": 0.468,
                            "speaker": "SPEAKER_04",
                        },
                        {
                            "word": "not",
                            "start": 12.304,
                            "end": 12.544,
                            "score": 0.561,
                            "speaker": "SPEAKER_04",
                        },
                        {
                            "word": "something.",
                            "start": 12.564,
                            "end": 12.844,
                            "score": 0.415,
                            "speaker": "SPEAKER_04",
                        },
                    ],
                    "speaker": "SPEAKER_04",
                },
                {
                    "start": 12.864,
                    "end": 13.644,
                    "text": "Sorry?",
                    "words": [
                        {
                            "word": "Sorry?",
                            "start": 12.864,
                            "end": 13.644,
                            "score": 0.213,
                            "speaker": "SPEAKER_04",
                        }
                    ],
                    "speaker": "SPEAKER_04",
                },
                {
                    "start": 13.664,
                    "end": 14.345,
                    "text": "Yeah, I'm ready, yeah.",
                    "words": [
                        {
                            "word": "Yeah,",
                            "start": 13.664,
                            "end": 13.764,
                            "score": 0.109,
                            "speaker": "SPEAKER_04",
                        },
                        {
                            "word": "I'm",
                            "start": 13.804,
                            "end": 13.904,
                            "score": 0.734,
                            "speaker": "SPEAKER_04",
                        },
                        {
                            "word": "ready,",
                            "start": 13.944,
                            "end": 14.145,
                            "score": 0.942,
                            "speaker": "SPEAKER_04",
                        },
                        {
                            "word": "yeah.",
                            "start": 14.165,
                            "end": 14.345,
                            "score": 0.493,
                            "speaker": "SPEAKER_04",
                        },
                    ],
                    "speaker": "SPEAKER_04",
                },
            ],
            "word_segments": [
                {
                    "word": "Yeah,",
                    "start": 1.238,
                    "end": 1.478,
                    "score": 0.719,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "sorry,",
                    "start": 1.498,
                    "end": 1.718,
                    "score": 0.806,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "what",
                    "start": 1.738,
                    "end": 1.858,
                    "score": 0.976,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "is",
                    "start": 1.898,
                    "end": 1.938,
                    "score": 0.994,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "his",
                    "start": 1.998,
                    "end": 2.058,
                    "score": 0.979,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "number?",
                    "start": 2.098,
                    "end": 2.299,
                    "score": 0.918,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "My",
                    "start": 4.26,
                    "end": 4.4,
                    "score": 0.648,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "case",
                    "start": 4.44,
                    "end": 4.8,
                    "score": 0.484,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "number.",
                    "start": 4.84,
                    "end": 5.26,
                    "score": 0.494,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "OK,",
                    "start": 5.82,
                    "end": 6.141,
                    "score": 0.418,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "yeah.",
                    "start": 6.181,
                    "end": 6.341,
                    "score": 0.648,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "Oh,",
                    "start": 8.322,
                    "end": 8.482,
                    "score": 0.81,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "fuck.",
                    "start": 8.762,
                    "end": 9.122,
                    "score": 0.57,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "Your",
                    "start": 9.482,
                    "end": 9.662,
                    "score": 0.849,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "case",
                    "start": 9.802,
                    "end": 10.022,
                    "score": 0.467,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "number.",
                    "start": 10.043,
                    "end": 10.203,
                    "score": 0.14,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "Sorry.",
                    "start": 11.483,
                    "end": 11.763,
                    "score": 0.948,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "Sorry,",
                    "start": 11.783,
                    "end": 12.084,
                    "score": 0.309,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "I'm",
                    "start": 12.104,
                    "end": 12.264,
                    "score": 0.468,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "not",
                    "start": 12.304,
                    "end": 12.544,
                    "score": 0.561,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "something.",
                    "start": 12.564,
                    "end": 12.844,
                    "score": 0.415,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "Sorry?",
                    "start": 12.864,
                    "end": 13.644,
                    "score": 0.213,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "Yeah,",
                    "start": 13.664,
                    "end": 13.764,
                    "score": 0.109,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "I'm",
                    "start": 13.804,
                    "end": 13.904,
                    "score": 0.734,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "ready,",
                    "start": 13.944,
                    "end": 14.145,
                    "score": 0.942,
                    "speaker": "SPEAKER_04",
                },
                {
                    "word": "yeah.",
                    "start": 14.165,
                    "end": 14.345,
                    "score": 0.493,
                    "speaker": "SPEAKER_04",
                },
            ],
            "start": 1.238,
            "end": 14.345,
        },
    }
