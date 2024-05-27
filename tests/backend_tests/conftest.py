import pytest
import asyncio
from pathlib import Path
from tests.conftest import TEST_VOLUME_DIR
from trimit.backend.cut_transcript import (
    CutTranscriptLinearWorkflow,
    CutTranscriptLinearWorkflowStepOutput,
)
from trimit.backend.conf import CONF
from trimit.backend.transcription import Transcription
from trimit.backend.speaker_in_frame_detection import SpeakerInFrameDetection
from trimit.models import maybe_init_mongo, Video
from trimit.utils.video_utils import convert_video_to_audio

CONF["chunk_delay"] = 0
from trimit.backend.models import (
    Transcript,
    TranscriptChunk,
    Soundbites,
    SoundbitesChunk,
)


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


def load_soundbites(video_hash):
    return Soundbites.load_from_file(
        f"tests/fixtures/objects/soundbites_{video_hash}.p"
    )


def load_soundbites_chunk(video_hash, chunk):
    return SoundbitesChunk.load_from_file(
        f"tests/fixtures/objects/soundbites_chunk_{chunk}_{video_hash}.p"
    )


def transcribe_video(transcription: Transcription, video: Video):
    convert_video_to_audio(
        video.path(TEST_VOLUME_DIR), video.audio_path(TEST_VOLUME_DIR)
    )

    hash_to_transcription = transcription.transcribe_videos([video], with_cache=True)
    return hash_to_transcription[video.md5_hash]


async def video_with_transcription(transcription: Transcription, video: Video):
    video_hash = video.md5_hash
    if video.transcription is None:
        raw_transcript = transcribe_video(transcription, video)
        transcript = Transcript.from_video_transcription(raw_transcript)
        transcript.save(f"tests/fixtures/objects/transcript_{video_hash}.p")
        video.transcription = raw_transcript
        await video.save()
    return video


async def video_with_speakers_in_frame(
    speaker_in_frame_detection: SpeakerInFrameDetection, video: Video
):
    # TODO REMVOE
    video.speakers_in_frame = ["SPEAKER_00", "SPEAKER_01"]
    await video.save()
    if video.speakers_in_frame is None:
        await speaker_in_frame_detection.detect_speaker_in_frame_from_videos(
            [video], use_existing_output=True
        )
    return video


@pytest.fixture(scope="session")
async def video_15557970_with_transcription(transcription, video_15557970):
    return await video_with_transcription(transcription, video_15557970)


@pytest.fixture(scope="session")
async def video_3909774043_with_transcription(transcription, video_3909774043):
    return await video_with_transcription(transcription, video_3909774043)


@pytest.fixture(scope="session")
async def video_15557970_with_speakers_in_frame(
    speaker_in_frame_detection, video_15557970_with_transcription
):
    return await video_with_speakers_in_frame(
        speaker_in_frame_detection, video_15557970_with_transcription
    )


@pytest.fixture(scope="session")
async def video_3909774043_with_speakers_in_frame(
    speaker_in_frame_detection, video_3909774043_with_transcription
):
    return await video_with_speakers_in_frame(
        speaker_in_frame_detection, video_3909774043_with_transcription
    )


@pytest.fixture(scope="function")
def transcript_15557970(video_15557970_with_speakers_in_frame):
    return video_15557970_with_speakers_in_frame.transcription


@pytest.fixture(scope="function")
def transcript_3909774043(video_3909774043_with_speakers_in_frame):
    return video_3909774043_with_speakers_in_frame.transcription


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
def soundbites_3909774043():
    return load_soundbites("3909774043")


@pytest.fixture(scope="function")
def soundbites_3909774043_small():
    return load_soundbites("3909774043_small")


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


@pytest.fixture(scope="module")
def test_videos_output_dir():
    path = Path("tests/video_outputs/linear")
    path.mkdir(exist_ok=True, parents=True)
    return str(path)


@pytest.fixture(scope="module")
def test_videos_volume_dir():
    path = Path("tests/fixtures/volume")
    path.mkdir(exist_ok=True, parents=True)
    return str(path)


@pytest.fixture(scope="function")
async def workflow_3909774043_with_transcript(
    video_3909774043_with_speakers_in_frame,
    test_videos_output_dir,
    test_videos_volume_dir,
):
    loop = asyncio.get_running_loop()
    await maybe_init_mongo(io_loop=loop, reinitialize=True)
    return await CutTranscriptLinearWorkflow.from_video(
        video=video_3909774043_with_speakers_in_frame,
        timeline_name="3909774043_testimonial_test",
        output_folder=test_videos_output_dir,
        volume_dir=test_videos_volume_dir,
        new_state=True,
        length_seconds=120,
        nstages=2,
        first_pass_length=6 * 60,
        max_partial_transcript_words=800,
        max_word_extra_threshold=50,
        max_iterations=3,
        api_call_delay=0.5,
        with_provided_user_feedback=[],
        export_video=False,
    )


@pytest.fixture(scope="function")
async def workflow_3909774043_with_state_init(workflow_3909774043_with_transcript):
    workflow = workflow_3909774043_with_transcript
    output = None
    async for output, _ in workflow.step("make me a video"):
        pass
    assert isinstance(output, CutTranscriptLinearWorkflowStepOutput)
    assert len(workflow.raw_transcript.text) == 22855
    assert workflow.user_messages == ["make me a video"]
    return workflow


@pytest.fixture(scope="function")
async def workflow_15557970_with_transcript(
    video_15557970_with_speakers_in_frame,
    test_videos_output_dir,
    test_videos_volume_dir,
):
    loop = asyncio.get_running_loop()
    await maybe_init_mongo(io_loop=loop, reinitialize=True)
    return await CutTranscriptLinearWorkflow.from_video(
        video=video_15557970_with_speakers_in_frame,
        timeline_name="15557970_testimonial_test",
        output_folder=test_videos_output_dir,
        volume_dir=test_videos_volume_dir,
        new_state=True,
        length_seconds=120,
        nstages=2,
        first_pass_length=6 * 60,
        max_partial_transcript_words=800,
        max_word_extra_threshold=50,
        max_iterations=3,
        api_call_delay=0.5,
        with_provided_user_feedback=[],
        export_video=False,
    )


@pytest.fixture(scope="function")
async def workflow_15557970_with_state_init(workflow_15557970_with_transcript):
    workflow = workflow_15557970_with_transcript
    output = None
    async for output in workflow.step("make me a video"):
        pass
    assert isinstance(output, CutTranscriptLinearWorkflowStepOutput)
    assert workflow.user_messages == ["make me a video"]
    return workflow


@pytest.fixture(scope="function")
async def short_cut_transcript_15557970(transcript_15557970):
    transcript_15557970.split_in_chunks(500)
    transcript_15557970.erase_cuts()
    transcript_15557970.chunks[0].segments[0].cut(2, 4)
    transcript_15557970.chunks[0].segments[1].cut(1, 3)
    transcript_15557970.chunks[0].segments[5].cut(1, 2)
    transcript_15557970.chunks[4].segments[6].cut(4, 9)
    transcript_15557970.kept_segments |= set([0, 1, 5, 128])
    assert (
        transcript_15557970.text
        == "over here. then we go. create and surface products to"
    )
    return transcript_15557970
