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
    from trimit.backend.cut_transcript import CutTranscriptLinearWorkflow

    print("in load or create workflow")
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
            workflow = await CutTranscriptLinearWorkflow.from_state_id(workflow_id)
            print("created workflow from state")
        else:
            workflow = (
                await CutTranscriptLinearWorkflow.from_state_id_with_only_step_order(
                    workflow_id
                )
            )
            print("created workflow with only step order")
        assert workflow_id == str(workflow.id)
        workflows[str(workflow_id)] = workflow
    else:
        if with_output:
            await workflow.load_state()
            print("loaded workflow state")
        else:
            await workflow.load_step_order()
            print("loaded workflow step order")

    return workflow
