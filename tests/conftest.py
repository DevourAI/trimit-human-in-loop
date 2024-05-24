import os

os.environ["ENV"] = "test"
from trimit.models import get_upload_folder, maybe_init_mongo
from datetime import datetime
import pytest
import asyncio
import os
from trimit.models.models import (
    User,
    Video,
    Scene,
    Frame,
    Timeline,
    TimelineVersion,
    Take,
    TakeItem,
)
from trimit.utils.model_utils import save_video_with_details
from dotenv import load_dotenv
import shutil

load_dotenv(".env.local")

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

BRIAN_USER_EMAIL = "brian@coinbase.com"
BRIAN_VIDEO_DATE = datetime(2024, 1, 1)
UPLOADS_DIR = get_upload_folder(TEST_VOLUME_DIR, BRIAN_USER_EMAIL, BRIAN_VIDEO_DATE)
# one speaker, on camera, one scene
HIGH_RES_BASENAME_1 = "roller skate queso.MOV"
HIGH_RES_HASH_1 = "a0b97e8bc9d2bef98eb290d26264e90e"
LOW_RES_HASH_1 = "bbb6141b26fb79cca20c7ecee899e9bb"

# two speakers, both off camera, one scene
HIGH_RES_BASENAME_2 = "maqlouba short 1 vert explosion.mp4"
HIGH_RES_HASH_2 = "5ee23e6311afe226b3e68a2f81856cb9"
LOW_RES_HASH_2 = "c4064248015b0ca75d95ee34f7cea8cb"

# one speaker, on camera, one scene
HIGH_RES_BASENAME_3 = "DSCF0064.MOV"
HIGH_RES_HASH_3 = "e50786f18157128ddc0148077dea1170"
LOW_RES_HASH_3 = "7a0382d3fa5a43fdf317bef7ca587d69"

# 3 speakers, both on and off camera, multiple scenes, multiple takes, speaker overlap
HIGH_RES_BASENAME_4 = "multiple_kitchen_scenes.MP4"
HIGH_RES_HASH_4 = "593565137133b5294645b8d88a211863"
LOW_RES_HASH_4 = "c5f4554db7b7cadaec59b5848c72b05f"


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

TIMELINE_NAME = "test_timeline"


async def create_user():
    user1 = User(
        name="brian armstrong", email="brian@coinbase.com", password="password"
    )
    await user1.insert()
    return user1


# TODO change md5_hashes to crc hashes in file names to get these tests to pass again
async def _seed_mock_data():
    user1 = await create_user()
    await save_video_with_details(
        user_email=user1.email,
        timeline_name=TIMELINE_NAME,
        md5_hash=LOW_RES_HASH_1,
        ext=".mov",
        upload_datetime=datetime(2024, 1, 1),
        high_res_user_file_path=os.path.join(TEST_HIGH_RES_DIR, HIGH_RES_BASENAME_1),
        high_res_user_file_hash=HIGH_RES_HASH_1,
        volume_file_path=os.path.join(UPLOADS_DIR, LOW_RES_HASH_1 + ".mov"),
    )
    await save_video_with_details(
        user_email=user1.email,
        timeline_name=TIMELINE_NAME,
        md5_hash=LOW_RES_HASH_2,
        ext=".mp4",
        upload_datetime=datetime(2024, 1, 1),
        high_res_user_file_path=os.path.join(TEST_HIGH_RES_DIR, HIGH_RES_BASENAME_2),
        high_res_user_file_hash=HIGH_RES_HASH_2,
        volume_file_path=os.path.join(UPLOADS_DIR, LOW_RES_HASH_2 + ".mp4"),
    )
    await save_video_with_details(
        user_email=user1.email,
        timeline_name=TIMELINE_NAME,
        md5_hash=LOW_RES_HASH_3,
        ext=".mov",
        upload_datetime=datetime(2024, 1, 1),
        high_res_user_file_path=os.path.join(TEST_HIGH_RES_DIR, HIGH_RES_BASENAME_3),
        high_res_user_file_hash=HIGH_RES_HASH_3,
        volume_file_path=os.path.join(UPLOADS_DIR, LOW_RES_HASH_3 + ".mov"),
    )
    await save_video_with_details(
        user_email=user1.email,
        timeline_name=TIMELINE_NAME,
        md5_hash=LOW_RES_HASH_4,
        ext=".mp4",
        upload_datetime=datetime(2024, 1, 1),
        high_res_user_file_path=os.path.join(TEST_HIGH_RES_DIR, HIGH_RES_BASENAME_4),
        high_res_user_file_hash=HIGH_RES_HASH_4,
        volume_file_path=os.path.join(UPLOADS_DIR, LOW_RES_HASH_4 + ".mp4"),
    )
    dave_user = User(name="dave brown", email=DAVE_EMAIL, password="password")
    await dave_user.insert()
    for high_res_hash, low_res_hash, basename in zip(
        DAVE_VIDEO_HIGH_RES_HASHES, DAVE_VIDEO_LOW_RES_HASHES, DAVE_VIDEO_BASENAMES
    ):
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
            upload_datetime=datetime(2024, 1, 1),
            high_res_user_file_path=os.path.join(DAVE_VIDEO_FOLDER, basename),
            high_res_user_file_hash=high_res_hash,
            volume_file_path=os.path.join(UPLOADS_DIR, low_res_hash + ".mp4"),
        )


@pytest.fixture(scope="session")
async def drop_collections():
    await User.find().delete()
    await Video.find().delete()
    await Scene.find().delete()
    await Frame.find().delete()
    await Timeline.find().delete()
    await TimelineVersion.find().delete()
    await Take.find().delete()
    await TakeItem.find().delete()


@pytest.fixture(scope="session")
async def mongo_connect():
    global loop
    loop = asyncio.get_running_loop()
    os.environ["MONGO_CERT_FILEPATH"] = ""
    os.environ["INIT_MONGO_WITH_INDEXES"] = "true"
    await maybe_init_mongo(io_loop=loop)


@pytest.fixture(scope="session")
async def seed_user(mongo_connect, drop_collections):
    return await create_user()


@pytest.fixture(autouse=auto_seed_mock_data, scope="session")
async def seed_mock_data(mongo_connect, drop_collections):
    await _seed_mock_data()


@pytest.fixture(scope="session")
async def brian_user():
    return await User.find_one(User.email == "brian@coinbase.com")


@pytest.fixture(scope="session")
async def test_videos():
    return await Video.find().to_list()


@pytest.fixture(scope="session")
async def test_video_1():
    video = await Video.find_one(Video.md5_hash == LOW_RES_HASH_1)
    assert video is not None
    return video


@pytest.fixture(scope="session")
async def test_video_2():
    video = await Video.find_one(Video.md5_hash == LOW_RES_HASH_2)
    assert video is not None
    return video


@pytest.fixture(scope="session")
async def test_video_3():
    video = await Video.find_one(Video.md5_hash == LOW_RES_HASH_3)
    assert video is not None
    return video


@pytest.fixture(scope="session")
async def test_video_4():
    video = await Video.find_one(Video.md5_hash == LOW_RES_HASH_4)
    assert video is not None
    return video


@pytest.fixture(scope="session")
async def dave_videos():
    return await Video.find(Video.user.email == DAVE_EMAIL).to_list()
