import os
from pathlib import Path

os.environ["ENV"] = "test"
import diskcache as dc
from datetime import datetime
import pytest
import asyncio
import os
from trimit.backend.transcription import Transcription
from trimit.backend.speaker_in_frame_detection import SpeakerInFrameDetection
from trimit.backend.models import Soundbites
from trimit.models.models import User, Video
from trimit.models import get_upload_folder, maybe_init_mongo
from trimit.utils.model_utils import save_video_with_details
from trimit.backend.models import Transcript
from dotenv import load_dotenv
import shutil

load_dotenv(".local/.env")

auto_seed_mock_data = False
if os.getenv("AUTO_SEED_MOCK_DATA", "False") == "True":
    auto_seed_mock_data = True

from pytest_asyncio import is_async_test


def pytest_collection_modifyitems(items):
    pytest_asyncio_tests = (item for item in items if is_async_test(item))
    session_scope_marker = pytest.mark.asyncio(scope="session")
    for async_test in pytest_asyncio_tests:
        async_test.add_marker(session_scope_marker)


pytestmark = pytest.mark.asyncio(scope="session")

loop: asyncio.AbstractEventLoop


TEST_VOLUME_DIR = "tests/fixtures/volume"
TEST_MODEL_DIR = "tests/models"
TEST_CACHE_DIR = "tests/cache"
TEST_HIGH_RES_DIR = "tests/fixtures/high_res"

DAVE_EMAIL = "dave@hedhi.com"
DAVE_VIDEO_HIGH_RES_HASHES = [
    "2464358268",
    "2072360807",
    "569796697",
    "999941292",
    "167375884",
    "3975094682",
]
DAVE_VIDEO_LOW_RES_HASHES = [
    "159067417",
    "455765972",
    "3225395022",
    "1513485534",
    "1030593207",
    "2646261089",
]
DAVE_VIDEO_FOLDER = os.path.join("customer_videos", "hedhi", "small_sample")
DAVE_VIDEO_COMPRESSED_FOLDER = os.path.join("tests", "fixtures", "compressed")
DAVE_VIDEO_BASENAMES = [
    "longer conversation in kitchen.mp4",
    "lacrosse b roll.mp4",
    "interviewee and interviewer conversation.mp4",
    "multiple_kitchen_scenes.MP4",
    "interviewer.mp4",
    "interviewee.mp4",
]
DAVE_VIDEO_DATE = datetime(2024, 1, 1)
DAVE_UPLOADS_DIR = get_upload_folder(TEST_VOLUME_DIR, DAVE_EMAIL, DAVE_VIDEO_DATE)

DAVE_FULL_VIDEO_PATHS = [
    "tests/fixtures/volume/uploads/dave@hedhi.com/2024-01-01/15557970.mp4",
    "tests/fixtures/volume/uploads/dave@hedhi.com/2024-01-01/3909774043.mp4",
]

TIMELINE_NAME = "test_timeline"


async def create_user():
    user1 = User(
        name="brian armstrong", email="brian@coinbase.com", password="password"
    )
    await user1.insert()
    return user1


# TODO change md5_hashes to crc hashes in file names to get these tests to pass again
async def _seed_mock_data():
    dave_user = await User.find_one(User.email == DAVE_EMAIL)
    if dave_user is None:
        dave_user = User(name="dave brown", email=DAVE_EMAIL, password="password")
        await dave_user.insert()
    for high_res_hash, low_res_hash, basename in zip(
        DAVE_VIDEO_HIGH_RES_HASHES, DAVE_VIDEO_LOW_RES_HASHES, DAVE_VIDEO_BASENAMES
    ):
        video = await Video.find_one(Video.md5_hash == low_res_hash)
        if video is not None:
            continue
        compressed_filename = low_res_hash + ".mp4"
        compressed_path = os.path.join(
            DAVE_VIDEO_COMPRESSED_FOLDER, compressed_filename
        )
        upload_path = os.path.join(DAVE_UPLOADS_DIR, compressed_filename)
        if not os.path.exists(upload_path) or os.stat(upload_path).st_size == 0:
            shutil.copy(compressed_path, upload_path)

        await save_video_with_details(
            user_email=DAVE_EMAIL,
            timeline_name=TIMELINE_NAME,
            md5_hash=low_res_hash,
            ext=".mp4",
            upload_datetime=DAVE_VIDEO_DATE,
            high_res_user_file_path=os.path.join(DAVE_VIDEO_FOLDER, basename),
            volume_file_path=os.path.join(DAVE_UPLOADS_DIR, low_res_hash + ".mp4"),
            overwrite=False,
        )

    for path in DAVE_FULL_VIDEO_PATHS:
        video_hash = Path(path).stem
        video = await Video.find_one(Video.md5_hash == video_hash)
        if video is not None:
            continue

        await save_video_with_details(
            user_email=DAVE_EMAIL,
            timeline_name=TIMELINE_NAME,
            md5_hash=video_hash,
            ext=".mp4",
            upload_datetime=DAVE_VIDEO_DATE,
            high_res_user_file_path=path,
            volume_file_path=path,
            overwrite=False,
        )


#  @pytest.fixture(scope="session")
#  async def drop_collections():
#  await User.find().delete()
#  await Video.find().delete()
#  await Scene.find().delete()
#  await Frame.find().delete()


@pytest.fixture(scope="session")
async def mongo_connect():
    global loop
    loop = asyncio.get_running_loop()
    os.environ["MONGO_CERT_FILEPATH"] = ""
    os.environ["INIT_MONGO_WITH_INDEXES"] = "true"
    await maybe_init_mongo(io_loop=loop)


@pytest.fixture(scope="session")
async def seed_user(mongo_connect):
    return await create_user()


@pytest.fixture(autouse=auto_seed_mock_data, scope="session")
async def seed_mock_data(mongo_connect):
    await _seed_mock_data()


@pytest.fixture(scope="session")
async def test_videos():
    return await Video.find().to_list()


@pytest.fixture(scope="session")
async def dave_videos():
    return await Video.find(Video.user.email == DAVE_EMAIL).to_list()


@pytest.fixture(scope="session")
async def kitchen_conversation_video(seed_mock_data):
    return await Video.find_one(Video.md5_hash == "159067417")


@pytest.fixture(scope="session")
async def conversation_video(seed_mock_data):
    return await Video.find_one(Video.md5_hash == "3225395022")


@pytest.fixture(scope="session")
async def video_15557970(seed_mock_data):
    return await Video.find_one(Video.md5_hash == "15557970")


@pytest.fixture(scope="session")
async def video_3909774043(seed_mock_data):
    return await Video.find_one(Video.md5_hash == "3909774043")


@pytest.fixture(scope="session")
def transcription():
    return Transcription(
        TEST_MODEL_DIR, volume_dir=TEST_VOLUME_DIR, cache=dc.Cache(TEST_CACHE_DIR)
    )


@pytest.fixture(scope="session")
def speaker_in_frame_detection():
    return SpeakerInFrameDetection(
        cache=dc.Cache(TEST_CACHE_DIR), volume_dir=TEST_VOLUME_DIR
    )


def transcribe_video(transcription: Transcription, video: Video):
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
    transcript = Transcript.from_video_transcription(
        video_15557970_with_speakers_in_frame.transcription
    )
    transcript.split_in_chunks(600)
    return transcript


@pytest.fixture(scope="function")
def transcript_3909774043(video_3909774043_with_speakers_in_frame):
    transcript = Transcript.from_video_transcription(
        video_3909774043_with_speakers_in_frame.transcription
    )
    transcript.split_in_chunks(500)
    return transcript


@pytest.fixture(scope="function")
async def short_cut_transcript_3909774043(transcript_3909774043):
    transcript = transcript_3909774043
    transcript.split_in_chunks(500)
    transcript.erase_cuts()
    transcript.chunks[0].segments[0].cut(2, 4)
    transcript.chunks[0].segments[2].cut(1, 3)
    transcript.chunks[0].segments[6].cut(1, 2)
    transcript.chunks[4].segments[6].cut(4, 9)
    chunk_segments_to_keep = [(0, 0), (0, 2), (0, 6), (4, 6)]
    segments_to_keep = set()
    for chunk, chunk_seg in chunk_segments_to_keep:
        full_seg = transcript.chunks[chunk].chunk_segment_indexes[0] + chunk_seg
        segments_to_keep.add(full_seg)
    transcript.kept_segments |= segments_to_keep
    assert transcript.text == "start with with Surkana, like the data that you want"
    return transcript


@pytest.fixture(scope="function")
async def short_cut_transcript_15557970(transcript_15557970):
    transcript_15557970.split_in_chunks(500)
    transcript_15557970.erase_cuts()
    chunk_indexes = [0, 0, 1]
    segment_indexes = [6, 7, 10]
    cuts = [(3, 6), (0, 9), (0, 6)]
    for chunk_idx, seg_idx, cut in zip(chunk_indexes, segment_indexes, cuts):
        chunk = transcript_15557970.chunks[chunk_idx]
        chunk.segments[seg_idx].cut(*cut)
        transcript_15557970.kept_segments.add(
            chunk.full_segment_index_from_chunk_segment_index(seg_idx)
        )
    assert (
        transcript_15557970.text
        == "I'm Parbinder Darawal, I'm the VP GM over at CVS Media Exchange, We're number one in this category"
    )
    return transcript_15557970


def load_soundbites(video_hash):
    return Soundbites.load_from_file(
        f"tests/fixtures/objects/soundbites_{video_hash}.p"
    )


@pytest.fixture(scope="function")
def soundbites_3909774043():
    return load_soundbites("3909774043")


@pytest.fixture(scope="function")
def soundbites_15557970():
    return load_soundbites("15557970")
