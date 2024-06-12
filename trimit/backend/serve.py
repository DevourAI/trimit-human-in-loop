from trimit.app import app, VOLUME_DIR
from trimit.backend.conf import (
    LINEAR_WORKFLOW_OUTPUT_FOLDER,
    WORKFLOWS_DICT_NAME,
    RUNNING_WORKFLOWS_DICT_NAME,
)
from .image import image
from modal import Dict
import asyncio

app_kwargs = dict(
    _allow_background_volume_commits=True,
    timeout=80000,
    image=image,
    container_idle_timeout=1200,
    _experimental_boost=True,
    _experimental_scheduler=True,
    cpu=12,
    memory=24000,
)

workflows = Dict.from_name(WORKFLOWS_DICT_NAME, create_if_missing=True)
running_workflows = Dict.from_name(RUNNING_WORKFLOWS_DICT_NAME, create_if_missing=True)


async def step_workflow_until_feedback_request(
    workflow: "CutTranscriptLinearWorkflow",
    user_input: str | None = None,
    load_state=True,
    save_state_to_db=True,
    async_export=True,
    retry_step=False,
):
    from trimit.backend.cut_transcript import CutTranscriptLinearWorkflowStepOutput
    from trimit.models import maybe_init_mongo

    await maybe_init_mongo()

    user_feedback_request = None
    first_time = True
    done = False
    while first_time or not user_feedback_request:
        result = None
        async for result, is_last in workflow.step(
            user_input or "",
            load_state=load_state,
            save_state_to_db=save_state_to_db,
            async_export=async_export,
            retry_step=retry_step and first_time,
        ):
            yield result, is_last

        if not isinstance(result, CutTranscriptLinearWorkflowStepOutput):
            yield CutTranscriptLinearWorkflowStepOutput(
                step_name="", substep_name="", done=False, error="No steps ran"
            ), True
            return

        assert isinstance(result, CutTranscriptLinearWorkflowStepOutput)
        done = result.done
        if done:
            break
        user_feedback_request = result.user_feedback_request
        first_time = False


@app.function(**app_kwargs)
async def step(
    user_email: str,
    timeline_name: str,
    video_hash: str,
    length_seconds: int,
    user_input: str | None = None,
    force_restart: bool = False,
    ignore_running_workflows: bool = False,
    timeout: float = 120,
    async_export: bool = True,
    retry_step: bool = False,
):
    from trimit.backend.cut_transcript import (
        CutTranscriptLinearWorkflow,
        CutTranscriptLinearWorkflowStepOutput,
    )
    from trimit.models import maybe_init_mongo

    await maybe_init_mongo()
    workflow = await CutTranscriptLinearWorkflow.from_video_hash(
        video_hash=video_hash,
        timeline_name=timeline_name,
        user_email=user_email,
        length_seconds=length_seconds,
        output_folder=LINEAR_WORKFLOW_OUTPUT_FOLDER,
        volume_dir=VOLUME_DIR,
        new_state=force_restart,
    )
    await workflow.load_state()
    id = workflow.id

    if not ignore_running_workflows:
        if workflow.id in workflows and not force_restart:
            if running_workflows.get(id, False):
                yield CutTranscriptLinearWorkflowStepOutput(
                    step_name="",
                    substep_name="",
                    done=False,
                    error="Workflow already running",
                ), True
                return
        else:
            workflows[id] = workflow
    running_workflows[id] = True
    gen = step_workflow_until_feedback_request(
        workflow, user_input, async_export=async_export, retry_step=retry_step
    )
    try:
        while True:
            output = await asyncio.wait_for(gen.__anext__(), timeout)
            running_workflows[id] = False
            yield output
    except asyncio.TimeoutError:
        running_workflows[id] = False
        yield CutTranscriptLinearWorkflowStepOutput(
            step_name="", substep_name="", done=False, error="Timeout"
        ), True
    except StopAsyncIteration:
        return


@app.local_entrypoint()
async def cut_transcript_cli(
    user_email: str,
    timeline_name: str,
    video_hash: str,
    length_seconds: int,
    user_input: str | None = None,
    force_restart: bool = False,
    ignore_running_workflows: bool = False,
    timeout: float = 120,
):
    from trimit.backend.cut_transcript import CutTranscriptLinearWorkflowStepOutput

    done = False
    i = 0
    while not done:
        steps_ran = []

        started_printing_feedback_request = False
        async for partial_output, is_last in step.remote_gen.aio(
            user_email=user_email,
            timeline_name=timeline_name,
            video_hash=video_hash,
            length_seconds=length_seconds,
            user_input=user_input,
            force_restart=False if i > 0 else force_restart,
            ignore_running_workflows=ignore_running_workflows,
            timeout=timeout,
        ):
            if is_last:
                assert isinstance(partial_output, CutTranscriptLinearWorkflowStepOutput)
                steps_ran.append(partial_output)
            else:
                if started_printing_feedback_request:
                    print(partial_output, end="", flush=True)
                else:
                    started_printing_feedback_request = True
                    print(f"Feedback request: {partial_output}", end="", flush=True)
        if len(steps_ran) == 0:
            print("No steps ran")
            done = True
            break
        last_step = steps_ran[-1]
        if last_step.done:
            done = True
            break
        if last_step.user_feedback_request:
            user_input = input(last_step.user_feedback_request)
        print(f"Step result: {steps_ran}")
        i += 1
    print("Workflow done")
