from fastapi.testclient import TestClient
from trimit.api.index import web_app as original_web_app
import pytest_asyncio

from trimit.models.models import (
    CutTranscriptLinearWorkflowState,
    FrontendWorkflowProjection,
    FrontendWorkflowState,
)
from ..conftest import DAVE_EMAIL, TEST_VOLUME_DIR, TEST_ASSET_DIR
import asyncio
import pytest
from trimit.backend.models import Message, Role
from trimit.models import FrontendStepOutput

pytestmark = pytest.mark.asyncio(scope="session")

loop: asyncio.AbstractEventLoop


@pytest_asyncio.fixture(scope="session")
async def client():
    global loop
    loop = asyncio.get_running_loop()
    with TestClient(original_web_app) as client:
        yield client


def assert_state_has_equivalent_creation_params(state, creation_params):
    assert state.static_state.user.email == creation_params["user_email"]
    assert state.static_state.video.md5_hash == creation_params["video_hash"]
    for k, v in creation_params.items():
        if k not in ["user_email", "video_hash"]:
            assert v == getattr(state.static_state, k)


async def test_new_workflow(client):
    from trimit.models import MONGO_INITIALIZED

    MONGO_INITIALIZED[0] = False
    workflow_creation_params = {
        "user_email": DAVE_EMAIL,
        "video_hash": "15557970",
        "timeline_name": "new timeline name",
        "length_seconds": 5000,
        "nstages": 3,
    }

    response = client.post("/workflows/new", data=workflow_creation_params)
    assert response.status_code == 200
    state = await CutTranscriptLinearWorkflowState.get(response.json())
    assert state is not None
    assert_state_has_equivalent_creation_params(state, workflow_creation_params)
    assert state.outputs == {}


async def test_recreate_workflow(client, workflow_15557970_with_state_init_no_export):
    from trimit.models import MONGO_INITIALIZED

    workflow = workflow_15557970_with_state_init_no_export

    MONGO_INITIALIZED[0] = False

    workflow_creation_params = {
        "user_email": workflow.state.static_state.user.email,
        "video_hash": workflow.state.static_state.video.md5_hash,
        "timeline_name": workflow.timeline_name,
        "length_seconds": workflow.length_seconds,
        "nstages": workflow.nstages,
    }

    response = client.post(
        "/workflows/new", data={"recreate": True, **workflow_creation_params}
    )
    assert response.status_code == 200
    state = await CutTranscriptLinearWorkflowState.get(response.json())
    assert state is not None
    assert_state_has_equivalent_creation_params(state, workflow_creation_params)
    assert state.outputs == {}


async def test_try_create_existing_workflow(
    client, workflow_15557970_with_state_init_no_export
):
    from trimit.models import MONGO_INITIALIZED

    workflow = workflow_15557970_with_state_init_no_export

    MONGO_INITIALIZED[0] = False

    workflow_creation_params = {
        "user_email": workflow.state.static_state.user.email,
        "video_hash": workflow.state.static_state.video.md5_hash,
        "timeline_name": workflow.timeline_name,
        "length_seconds": workflow.length_seconds,
        "nstages": workflow.nstages,
    }

    response = client.post(
        "/workflows/new", data={"recreate": False, **workflow_creation_params}
    )
    assert response.status_code == 200
    state = await CutTranscriptLinearWorkflowState.get(response.json())
    assert state is not None
    assert_state_has_equivalent_creation_params(state, workflow_creation_params)
    assert state.outputs == workflow.state.outputs


async def test_get_workflow_details(client, workflow_15557970_with_transcript):
    from trimit.models import MONGO_INITIALIZED

    workflow = workflow_15557970_with_transcript
    MONGO_INITIALIZED[0] = False

    response = client.get("/workflow", params={"workflow_id": workflow.id})
    assert response.status_code == 200
    details = FrontendWorkflowProjection(**response.json())
    assert details.timeline_name == workflow.state.static_state.timeline_name
    assert details.user_email == workflow.state.user.email
    assert details.video_hash == workflow.state.video.md5_hash
    assert details.length_seconds == workflow.state.static_state.length_seconds
    assert details.nstages == workflow.state.static_state.nstages
    assert details.id == str(workflow.state.id)


async def test_frontend_workflow_state(workflow_15557970_with_transcript):
    from trimit.models import MONGO_INITIALIZED

    workflow = workflow_15557970_with_transcript
    MONGO_INITIALIZED[0] = False
    state = await FrontendWorkflowState.from_workflow(
        workflow, TEST_VOLUME_DIR, TEST_ASSET_DIR
    )
    assert state.id == workflow.id
    assert state.model_dump()["id"] == str(workflow.id)


async def test_list_workflows(client, workflow_15557970_with_transcript):
    from trimit.models import MONGO_INITIALIZED

    workflow = workflow_15557970_with_transcript
    MONGO_INITIALIZED[0] = False

    response = client.get(
        "/workflows", params={"user_email": workflow.state.user.email}
    )
    assert response.status_code == 200
    workflows = response.json()
    assert len(workflows) == 1
    details = FrontendWorkflowProjection(**workflows[0])
    assert details.timeline_name == workflow.state.static_state.timeline_name
    assert details.user_email == workflow.state.user.email
    assert details.video_hash == workflow.state.video.md5_hash
    assert details.length_seconds == workflow.state.static_state.length_seconds
    assert details.nstages == workflow.state.static_state.nstages
    assert details.id == str(workflow.state.id)


async def test_get_all_steps(client, workflow_15557970_with_transcript):
    from trimit.models import MONGO_INITIALIZED

    MONGO_INITIALIZED[0] = False

    response = client.get(
        "/all_steps", params={"workflow_id": workflow_15557970_with_transcript.id}
    )
    assert response.status_code == 200
    assert len(response.json()) == 5


async def test_get_latest_state_init(client, workflow_15557970_with_transcript):
    from trimit.models import MONGO_INITIALIZED

    MONGO_INITIALIZED[0] = False

    response = client.get(
        "/get_latest_state",
        params={"workflow_id": workflow_15557970_with_transcript.id},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["all_steps"]) == 5
    assert data["all_steps"][0]["substeps"][0]["substep_name"] == "init_state"
    assert (
        data["all_steps"][-1]["substeps"][-1]["substep_name"]
        == "modify_transcript_holistically"
    )
    #  assert data["last_step"] is None
    #  assert data["next_step"]["substep_name"] == "remove_off_screen_speakers"
    assert data["static_state"]["video_id"] is not None
    assert data["static_state"]["user_id"] is not None
    # assert data["user_messages"] == []
    # assert data["step_history_state"] == []
    assert data["outputs"] == []


async def test_get_latest_state_after_step_with_retry(
    client, workflow_15557970_after_first_step_with_retry
):
    workflow = workflow_15557970_after_first_step_with_retry
    from trimit.models import MONGO_INITIALIZED

    MONGO_INITIALIZED[0] = False

    response = client.get("/get_latest_state", params={"workflow_id": workflow.id})
    assert response.status_code == 200
    data = response.json()

    assert len(data["all_steps"]) == 5
    assert data["all_steps"][0]["substeps"][0]["substep_name"] == "init_state"
    assert (
        data["all_steps"][-1]["substeps"][-1]["substep_name"]
        == "modify_transcript_holistically"
    )
    #  assert data["last_step"]["substep_name"] == "remove_off_screen_speakers"
    #  assert data["next_step"]["substep_name"] == "generate_story"
    assert data["static_state"]["video_id"] is not None
    assert data["static_state"]["user_id"] is not None
    #  assert data["step_history_state"] == [
    #  {
    #  "name": "preprocess_video",
    #  "substeps": ["init_state", "remove_off_screen_speakers"],
    #  }
    #  ]
    assert len(data["outputs"]) == 1
    output = data["outputs"][0]
    parsed = FrontendStepOutput(**output)
    assert parsed.step_name == "preprocess_video"
    assert parsed.substep_name == "remove_off_screen_speakers"
    assert not parsed.done
    assert parsed.full_conversation == [
        Message(role=Role.Human, value=""),
        Message(
            role=Role.AI,
            value="I identified these speakers as being on-screen: ['speaker_01']. \nDo you agree? Do you have modifications to make?",
        ),
        Message(role=Role.Human, value="Just try again"),
        Message(
            role=Role.AI,
            value="I identified these speakers as being on-screen: ['speaker_01']. \nDo you agree? Do you have modifications to make?",
        ),
    ]
    assert parsed.conversation == [
        Message(role=Role.Human, value="Just try again"),
        Message(
            role=Role.AI,
            value="I identified these speakers as being on-screen: ['speaker_01']. \nDo you agree? Do you have modifications to make?",
        ),
    ]

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
            "workflow_id": workflow.id,
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
