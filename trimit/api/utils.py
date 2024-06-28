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


async def load_workflow(
    workflow_id: str,
    with_output: bool = False,
    wait_until_done_running: bool = False,
    timeout: float = 5,
    wait_interval: float = 0.1,
):
    from trimit.models import maybe_init_mongo
    from trimit.backend.cut_transcript import CutTranscriptLinearWorkflow

    await maybe_init_mongo()

    running = running_workflows.get(workflow_id, False)
    if running and wait_until_done_running:
        start_time = time.time()
        while running:
            await asyncio.sleep(wait_interval)
            running = running_workflows.get(workflow_id, False)
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout (workflow still running)")
    workflow = workflows.get(workflow_id, None)
    if not workflow:
        if with_output:
            workflow = await CutTranscriptLinearWorkflow.from_state_id(workflow_id)
        else:
            workflow = (
                await CutTranscriptLinearWorkflow.from_state_id_with_only_step_order(
                    workflow_id
                )
            )
        if workflow_id != str(workflow.id):
            raise ValueError(
                f"workflow_id {workflow_id} != workflow.id {str(workflow.id)}"
            )
        workflows[str(workflow_id)] = workflow
    else:
        if with_output:
            try:
                await workflow.load_state()
            except Exception as e:
                print("load workflow state error:", e, type(e), str(e))
                raise
        else:
            await workflow.load_step_order()

    return workflow
