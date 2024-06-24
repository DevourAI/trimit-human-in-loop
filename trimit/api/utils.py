from numpy import test
from trimit.backend.conf import (
    LINEAR_WORKFLOW_OUTPUT_FOLDER,
    WORKFLOWS_DICT_NAME,
    RUNNING_WORKFLOWS_DICT_NAME,
)
from trimit.app import VOLUME_DIR
from modal import Dict, is_local
import time
import asyncio

if is_local():
    # TODO perhaps use redis for this locally
    workflows = {}
    running_workflows = {}
else:
    workflows = Dict.from_name(WORKFLOWS_DICT_NAME, create_if_missing=True)
    running_workflows = Dict.from_name(
        RUNNING_WORKFLOWS_DICT_NAME, create_if_missing=True
    )


async def load_or_create_workflow(
    timeline_name: str,
    length_seconds: int,
    nstages: int | None = None,
    user_email: str | None = None,
    video_hash: str | None = None,
    user_id: str | None = None,
    video_id: str | None = None,
    with_output: bool = False,
    wait_until_done_running: bool = False,
    timeout: float = 5,
    wait_interval: float = 0.1,
    force_restart: bool = False,
    export_video: bool = True,
    output_folder: str = LINEAR_WORKFLOW_OUTPUT_FOLDER,
    volume_dir: str = VOLUME_DIR,
):
    print("in load or create workflow")
    if not video_hash and not video_id:
        raise ValueError("video_hash or video_id must be provided")
    if not user_id and not user_email:
        raise ValueError("user_id or user_email must be provided")
    from trimit.backend.cut_transcript import CutTranscriptLinearWorkflow

    workflow_params = {
        "video_hash": video_hash,
        "timeline_name": timeline_name,
        "user_email": user_email,
        "length_seconds": length_seconds,
        "nstages": nstages,
        "output_folder": output_folder or LINEAR_WORKFLOW_OUTPUT_FOLDER,
        "volume_dir": volume_dir or VOLUME_DIR,
        "export_video": export_video,
    }
    print("workflow params:", workflow_params)

    workflow_id = await CutTranscriptLinearWorkflow.id_from_params(
        video_id=video_id or None, user_id=user_id or None, **workflow_params
    )
    print("got workflow id:", workflow_id)
    running = running_workflows.get(workflow_id, False)
    if running and wait_until_done_running:
        start_time = time.time()
        while running:
            await asyncio.sleep(wait_interval)
            running = running_workflows.get(workflow_id, False)
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout (workflow still running)")
    workflow = workflows.get(workflow_id, None)
    print("got workflow")
    if not workflow:
        if with_output:
            if video_id:
                if "video_hash" in workflow_params:
                    del workflow_params["video_hash"]
                if "user_email" in workflow_params:
                    del workflow_params["user_email"]
                workflow = await CutTranscriptLinearWorkflow.from_video_id(
                    video_id=video_id, new_state=force_restart, **workflow_params
                )
                print("created workflow from video_id")
            else:
                workflow = await CutTranscriptLinearWorkflow.from_video_hash(
                    new_state=force_restart, **workflow_params
                )
                print("created workflow from video_hash")
        else:
            workflow = await CutTranscriptLinearWorkflow.with_only_step_order(
                video_id=video_id or None, user_id=user_id or None, **workflow_params
            )
            print("created workflow with only step order")
        assert workflow_id == workflow.id
        workflows[workflow_id] = workflow
    else:
        if with_output:
            await workflow.load_state()
            print("loaded workflow state")
        else:
            await workflow.load_step_order()
            print("loaded workflow step order")

    return workflow
