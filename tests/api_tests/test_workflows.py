from fastapi.testclient import TestClient
from trimit.api.index import web_app as original_web_app
import pytest_asyncio
from ..conftest import DAVE_EMAIL, DAVE_VIDEO_LOW_RES_HASHES
import asyncio
import pytest
from trimit.backend.models import CutTranscriptLinearWorkflowStepOutput

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


async def test_get_latest_state_init(client, workflow_15557970_with_transcript):
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
    assert data["outputs"] == []


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
    assert len(data["outputs"]) == 1
    output = data["outputs"][0]
    parsed = CutTranscriptLinearWorkflowStepOutput(**output)
    assert parsed.step_name == "preprocess_video"
    assert parsed.substep_name == "remove_off_screen_speakers"
    assert not parsed.done
    assert (
        parsed.user_feedback_request
        == "I identified these speakers as being on-screen: ['speaker_01']. \nDo you agree? Do you have modifications to make?"
    )
    assert parsed.step_outputs and list(parsed.step_outputs.keys()) == [
        "current_transcript_text",
        "current_transcript_state",
        "on_screen_speakers",
        "on_screen_transcript_text",
        "on_screen_transcript_state",
    ]
    assert parsed.export_result and list(parsed.export_result.keys()) == [
        "transcript",
        "transcript_text",
        "video_timeline",
        "speaker_tagging_clips",
    ]
    assert list(parsed.export_result["speaker_tagging_clips"].keys()) == [
        "SPEAKER_01",
        "SPEAKER_00",
    ]


async def test_get_output_for_name(client, workflow_15557970_after_second_step):
    workflow = workflow_15557970_after_second_step
    from trimit.models import MONGO_INITIALIZED

    MONGO_INITIALIZED[0] = False

    response = client.get(
        "/get_step_outputs",
        params={
            "video_hash": workflow.video.md5_hash,
            "user_email": workflow.user.email,
            "timeline_name": workflow.timeline_name,
            "length_seconds": workflow.length_seconds,
            "export_video": workflow.export_video,
            "volume_dir": workflow.volume_dir,
            "output_folder": workflow.state.static_state.output_folder,
            "step_names": "preprocess_video,generate_story",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert len(data["outputs"]) == 2
    assert data["outputs"][0]["step_name"] == "preprocess_video"
    assert data["outputs"][0]["substep_name"] == "remove_off_screen_speakers"
    assert data["outputs"][1]["step_name"] == "generate_story"
    assert data["outputs"][1]["substep_name"] == "generate_story"
