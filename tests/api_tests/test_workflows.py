from fastapi.testclient import TestClient
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


async def test_get_all_steps(client, workflow_15557970_with_transcript):
    from trimit.models import MONGO_INITIALIZED

    MONGO_INITIALIZED[0] = False

    response = client.get(
        "/all_steps",
        params={
            "video_hash": workflow_15557970_with_transcript.video.md5_hash,
            "user_email": workflow_15557970_with_transcript.user.email,
            "timeline_name": workflow_15557970_with_transcript.timeline_name,
            "length_seconds": workflow_15557970_with_transcript.length_seconds,
        },
    )
    assert response.status_code == 200
    assert len(response.json()["steps"]) == 5
