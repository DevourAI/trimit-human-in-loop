from trimit.utils.fs_utils import save_file_to_volume_as_crc_hash
from pathlib import Path
from trimit.utils.video_utils import video_content_crc_hash_from_path
from fastapi.testclient import TestClient
from fastapi import UploadFile, File
import os
import pytest
import tempfile


@pytest.fixture()
def fastapi_app():
    from fastapi import FastAPI

    return FastAPI()


@pytest.fixture
def client(fastapi_app):
    with TestClient(fastapi_app) as client:
        yield client


pytestmark = pytest.mark.asyncio()


@pytest.mark.asyncio(scope="function")
async def test_save_file_to_volume_as_crc_hash(fastapi_app, client):
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(b"Sample file content")
        tmpfile_path = tmpfile.name

    with tempfile.TemporaryDirectory() as tmpdirname:

        @fastapi_app.post("/upload")
        async def upload(file: UploadFile = File(...)):
            path = await save_file_to_volume_as_crc_hash(file, tmpdirname)
            return {"filename": file.filename, "path": path}

        with open(tmpfile_path, "rb") as file:
            response = client.post("/upload", files={"file": file})

        video_hash_from_disk = await video_content_crc_hash_from_path(tmpfile_path)

        assert response.status_code == 200
        response_data = response.json()
        assert "path" in response_data
        path = response_data["path"]
        assert video_hash_from_disk == Path(path).stem
        assert os.path.exists(response_data["path"])
        assert os.stat(response_data["path"]).st_size > 0
        os.remove(response_data["path"])
