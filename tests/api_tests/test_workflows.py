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
    assert len(response.json()) == 5


async def test_get_latest_state(client, workflow_15557970_with_transcript):
    from trimit.models import MONGO_INITIALIZED

    MONGO_INITIALIZED[0] = False

    response = client.get(
        "/get_latest_state",
        params={
            "video_hash": workflow_15557970_with_transcript.video.md5_hash,
            "user_email": workflow_15557970_with_transcript.user.email,
            "timeline_name": workflow_15557970_with_transcript.timeline_name,
            "length_seconds": workflow_15557970_with_transcript.length_seconds,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["all_steps"]) == 5
    assert data["all_steps"][0]["substeps"][0]["substep_name"] == "init_state"
    assert (
        data["all_steps"][-1]["substeps"][-1]["substep_name"]
        == "modify_transcript_holistically"
    )
    assert data["last_step"] is None
    assert data["next_step"]["substep_name"] == "remove_off_screen_speakers"
    assert data["video_id"] is not None
    assert data["user_id"] is not None
    assert data["user_messages"] == []
    assert data["step_history_state"] == []


async def test_get_latest_state_after_step(client, workflow_15557970_after_first_step):
    workflow = workflow_15557970_after_first_step
    from trimit.models import MONGO_INITIALIZED

    MONGO_INITIALIZED[0] = False

    response = client.get(
        "/get_latest_state",
        params={
            "video_hash": workflow.video.md5_hash,
            "user_email": workflow.user.email,
            "timeline_name": workflow.timeline_name,
            "length_seconds": workflow.length_seconds,
            "export_video": workflow.export_video,
            "volume_dir": workflow.volume_dir,
            "output_folder": workflow.state.static_state.output_folder,
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert len(data["all_steps"]) == 5
    assert data["all_steps"][0]["substeps"][0]["substep_name"] == "init_state"
    assert (
        data["all_steps"][-1]["substeps"][-1]["substep_name"]
        == "modify_transcript_holistically"
    )
    assert data["last_step"]["substep_name"] == "remove_off_screen_speakers"
    assert data["next_step"]["substep_name"] == "generate_story"
    assert data["video_id"] is not None
    assert data["user_id"] is not None
    assert data["user_messages"] == ["make me a video"]
    assert data["step_history_state"] == [
        {
            "name": "preprocess_video",
            "substeps": ["init_state", "remove_off_screen_speakers"],
        }
    ]
