import pytest
from pathlib import Path
from trimit.backend.conf import CONF

CONF["chunk_delay"] = 0
from trimit.models.backend_models import Transcript, TranscriptChunk, SoundbitesChunk


@pytest.fixture(scope="session")
def raw_transcript():
    return {
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
    }


def load_transcript(video_hash):
    return Transcript.load_from_file(
        f"tests/fixtures/objects/transcript_{video_hash}.p"
    )


def load_transcript_chunk(video_hash, chunk):
    return TranscriptChunk.load_from_file(
        f"tests/fixtures/objects/transcript_chunk_{chunk}_{video_hash}.p"
    )


def load_story(video_hash):
    with open(f"tests/fixtures/objects/story_{video_hash}.txt", "r") as f:
        story = f.read()
    return story


def load_soundbites_chunk(video_hash, chunk):
    return SoundbitesChunk.load_from_file(
        f"tests/fixtures/objects/soundbites_chunk_{chunk}_{video_hash}.p"
    )


@pytest.fixture(scope="function")
def transcript_3909774043_small():
    return load_transcript("3909774043_small")


@pytest.fixture(scope="function")
def transcript_chunk_0_3909774043():
    return load_transcript_chunk("3909774043", 0)


@pytest.fixture(scope="function")
def story_3909774043():
    return load_story("3909774043")


@pytest.fixture(scope="function")
def soundbites_chunk_0_3909774043():
    return load_soundbites_chunk("3909774043", 0)


@pytest.fixture(scope="function")
def soundbites_chunk_1_3909774043():
    return load_soundbites_chunk("3909774043", 1)


@pytest.fixture(scope="function")
def soundbites_chunk_2_3909774043():
    return load_soundbites_chunk("3909774043", 2)


@pytest.fixture(scope="function")
def soundbites_chunk_3_3909774043():
    return load_soundbites_chunk("3909774043", 3)


@pytest.fixture(scope="function")
def user_prompt_3909774043():
    return 'Create a customer testimonial video with the following title: "Proving Marketing Impact: How Circana and LiveRamp Empower Advertisers to Deliver High-Value Campaign Results". Make sure to include specific numeric details- people love hearing specific numbers and data.'


@pytest.fixture(scope="module")
def transcript_cache_file_dir():
    path = Path("tests/fixtures/transcript_cache")
    path.mkdir(exist_ok=True, parents=True)
    return str(path)
