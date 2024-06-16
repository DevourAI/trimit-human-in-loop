from fastapi.testclient import TestClient
from trimit.models import Project, maybe_init_mongo, Video
from trimit.api.index import web_app as original_web_app
import pytest_asyncio
from ..conftest import DAVE_EMAIL, DAVE_VIDEO_LOW_RES_HASHES
import asyncio
import pytest

pytestmark = pytest.mark.asyncio(scope="session")

loop: asyncio.AbstractEventLoop


@pytest_asyncio.fixture(scope="session")
async def client():
    global loop
    loop = asyncio.get_running_loop()
    with TestClient(original_web_app) as client:
        yield client


async def test_create_project(client, seed_mock_data):
    from trimit.models import MONGO_INITIALIZED

    MONGO_INITIALIZED[0] = False

    name = "test_project"
    video_hash = DAVE_VIDEO_LOW_RES_HASHES[0]
    response = client.post(
        "/projects/new",
        data={
            "name": name,
            "video_hash": video_hash,
            "user_email": DAVE_EMAIL,
            "overwrite": "true",
            "raise_on_existing": "true",
        },
    )
    assert response.status_code == 200
    await maybe_init_mongo(reinitialize=True)
    project = await Project.find_one(
        Project.name == name, Project.user.email == DAVE_EMAIL, fetch_links=True
    )
    assert project is not None
    assert project.video.md5_hash == video_hash
