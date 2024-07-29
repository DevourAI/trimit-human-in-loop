from fastapi.testclient import TestClient
import shutil
import asyncio
from unittest.mock import patch
from unittest.mock import MagicMock

from datetime import datetime
from pathlib import Path
import pytest
import tempfile
from trimit.models import Video, maybe_init_mongo
from trimit.utils.video_utils import video_content_crc_hash_from_path
from trimit.api.index import web_app as original_web_app
import pytest_asyncio

pytestmark = pytest.mark.asyncio(scope="session")

loop: asyncio.AbstractEventLoop


@pytest_asyncio.fixture(scope="session")
async def client():
    global loop
    loop = asyncio.get_running_loop()
    with TestClient(original_web_app) as client:
        yield client


@patch("trimit.api.index.background_processor")
async def test_upload(background_processor, client, seed_user):
    from trimit.models import MONGO_INITIALIZED

    MONGO_INITIALIZED[0] = False
    user_email = seed_user.email
    background_processor.return_value = MagicMock()
    import trimit.app

    volume_dir = tempfile.TemporaryDirectory().name
    with patch("trimit.api.index.get_volume_dir", return_value=volume_dir):
        local_video_path = (
            "tests/fixtures/volume/uploads/dave@hedhi.com/2024-04-18/797079843.mp4"
        )
        local_video_hash = "797079843"
        local_filename = "797079843.mp4"
        video_hash_from_disk = await video_content_crc_hash_from_path(local_video_path)
        assert video_hash_from_disk == local_video_hash
        upload_date = datetime.now().date()
        video_on_volume_path = (
            Path(volume_dir)
            / "uploads"
            / user_email
            / str(upload_date)
            / local_filename
        )
        video_on_volume_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_video_path, video_on_volume_path)

        with open(local_video_path, "rb") as file:
            response = client.post(
                "/upload",
                files={"files": file},
                data={
                    "user_email": user_email,
                    "timeline_name": "test",
                    "high_res_user_file_paths": [local_filename],
                    "force": True,
                    "use_existing_output": False,
                    "reprocess": True,
                },
            )
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["result"] == "success"
        assert response_data["video_hashes"] == [video_hash_from_disk]
        await maybe_init_mongo(reinitialize=True)
        video = await Video.find_one(Video.md5_hash == video_hash_from_disk)
        assert video is not None
        video_path = Path(video.path(volume_dir))
        assert video_path.exists()
        assert video.upload_datetime.date() == upload_date
        assert video_path == video_on_volume_path
        assert video.ext == ".mp4"
        assert video.frame_rate == 24
        assert video.duration == 46.0
        assert video.details.frame_count == 1114
        assert video.details.width == 640
        assert video.details.height == 360
        assert "processing_call_id" in response_data


@patch("trimit.api.index.background_processor")
async def test_upload_weblink(background_processor, client, seed_user):
    from trimit.models import MONGO_INITIALIZED

    user_email = seed_user.email
    volume_dir = tempfile.TemporaryDirectory().name
    upload_date = datetime.now().date()
    local_filename = "649819196.mp4"
    video_on_volume_path = (
        Path(volume_dir) / "uploads" / user_email / str(upload_date) / local_filename
    )
    MONGO_INITIALIZED[0] = False
    background_processor.return_value = MagicMock()
    import trimit.app

    with patch("trimit.api.index.get_volume_dir", return_value=volume_dir):
        web_links = ["https://www.youtube.com/watch?v=L2cBvrP5Rjc"]
        response = client.post(
            "/upload",
            files={},
            data={
                "web_links": web_links,
                "user_email": user_email,
                "timeline_name": "test",
                "high_res_user_file_paths": [],
                "force": True,
                "use_existing_output": False,
                "reprocess": True,
            },
        )
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["result"] == "success"
        assert response_data["video_hashes"] == ["649819196"]
        await maybe_init_mongo(reinitialize=True)
        video = await Video.find_one(Video.md5_hash == "649819196")
        assert video is not None
        video_path = Path(video.path(volume_dir))
        assert video_path.exists()
        assert video.upload_datetime.date() == upload_date
        assert video_path == video_on_volume_path
        assert video.ext == ".mp4"
        assert video.frame_rate == 23.976023976023978
        assert video.duration == 665.373042
        assert video.details is not None
        assert video.details.frame_count == 15953
        assert video.details.width == 640
        assert video.details.height == 360
        assert "processing_call_id" in response_data
