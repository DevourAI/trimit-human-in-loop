import time

from torch import save
from trimit.app import app, VOLUME_DIR
from trimit.backend.conf import (
    LINEAR_WORKFLOW_OUTPUT_FOLDER,
    WORKFLOWS_DICT_NAME,
    RUNNING_WORKFLOWS_DICT_NAME,
)
from trimit.backend.models import (
    PartialBackendOutput,
    PartialLLMOutput,
    StructuredUserInput,
    FinalLLMOutput,
)
from trimit.export.email import send_email_with_export_results
from trimit.models.models import StepNotYetReachedError
from trimit.utils.model_utils import get_dynamic_state_key
from .image import image
from modal import Dict, is_local
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
    structured_user_input: StructuredUserInput | None = None,
    load_state=True,
    save_state_to_db=True,
    async_export=True,
    retry_step=False,
    advance_until: str | None = None,
):
    from trimit.backend.cut_transcript import CutTranscriptLinearWorkflowStepOutput
    from trimit.models import maybe_init_mongo

    await maybe_init_mongo()
    if load_state:
        await workflow.load_state()

    user_feedback_request = None
    first_time = True
    done = False

    def get_existing_step_keys(workflow):
        return [
            get_dynamic_state_key(o.name, s)
            for o in workflow.state.dynamic_state_step_order
            for s in o.substeps
        ]

    i = 0
    while True:
        print(
            f"Stepping iter {i}, advance_until={advance_until}, existing_workflow_step_keys={get_existing_step_keys(workflow)}"
        )
        result = None
        async for result, is_last in workflow.step(
            user_input or "",
            structured_user_input=structured_user_input,
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
        print(
            f"Finished step for iter {i}, existing_workflow_step_keys={get_existing_step_keys(workflow)}, user_feedback_request={user_feedback_request}"
        )
        if user_feedback_request or (
            advance_until and advance_until in get_existing_step_keys(workflow)
        ):
            break
        i += 1


async def step_workflow_ignoring_feedback_request(
    workflow: "CutTranscriptLinearWorkflow",
    user_input: str | None = None,
    structured_user_input: StructuredUserInput | None = None,
    load_state=True,
    save_state_to_db=True,
    async_export=True,
    export_intermediate=False,
    stream_raw=False,
):
    from trimit.backend.cut_transcript import CutTranscriptLinearWorkflowStepOutput
    from trimit.models import maybe_init_mongo

    yield PartialBackendOutput(value="Initializing data"), False
    await maybe_init_mongo()
    if load_state:
        await workflow.load_state()

    if not export_intermediate:
        yield PartialBackendOutput(value="Setting up express mode"), False
        await workflow.set_all_export_to_false()
    while True:
        result = None
        step_user_input = ""
        next_substep = await workflow.get_next_substep(with_load_state=False)
        if next_substep and next_substep.name == "generate_story":
            step_user_input = user_input or ""
        human_readable_name = ""
        if next_substep and next_substep.step:
            human_readable_name = next_substep.step.human_readable_name
        yield PartialBackendOutput(value=f"Running step {human_readable_name}"), False
        async for result, is_last in workflow.step(
            step_user_input,
            structured_user_input=structured_user_input,
            load_state=load_state,
            save_state_to_db=save_state_to_db,
            async_export=async_export,
            retry_step=False,
        ):
            if not hasattr(result, "done") or not result.done:
                if isinstance(result, (PartialLLMOutput, FinalLLMOutput)):
                    if stream_raw:
                        yield result, is_last
                else:
                    yield result, is_last

        if not isinstance(result, CutTranscriptLinearWorkflowStepOutput):
            yield CutTranscriptLinearWorkflowStepOutput(
                step_name="", substep_name="", done=False, error="No steps ran"
            ), True
            return

        assert isinstance(result, CutTranscriptLinearWorkflowStepOutput)
        if result.done:
            break
    if not export_intermediate:
        # await workflow.revert_export_flags()
        print("setting all export to true")
        yield PartialBackendOutput(value="Exporting results"), False
        await workflow.set_all_export_to_true()
        output = None
        async for output in workflow.redo_export_results(local=True):
            yield output, False
        print("new export results", output)
        result.export_result = output
        send_email_with_export_results(workflow.user.email, output)
    yield result, True


@app.function(**app_kwargs)
async def step(
    workflow: "trimit.backend.cut_transcript.CutTranscriptLinearWorkflow",
    user_input: str | None = None,
    structured_user_input: StructuredUserInput | None = None,
    ignore_running_workflows: bool = False,
    timeout: float = 120,
    async_export: bool = True,
    retry_step: bool = False,
    step_name: str | None = None,
    substep_name: str | None = None,
    advance_until: int | None = None,
    load_state: bool = True,
    save_state_to_db=True,
):
    from trimit.backend.cut_transcript import CutTranscriptLinearWorkflowStepOutput
    from trimit.models import maybe_init_mongo

    await maybe_init_mongo()
    print("advance_until int", advance_until)
    if advance_until is not None and substep_name is None and step_name is None:
        step = workflow.steps[advance_until]
        step_name = step.name
        substep_name = step.substeps[0].name
        print(f"set step_name={step_name} substep_name={substep_name}")

    step_workflow_advance_until = None
    if step_name is not None:
        if substep_name is None:
            raise ValueError("substep_name must be provided if step_name is provided")
        try:
            await workflow.revert_step_to(step_name, substep_name)
            print(f"workflow reverted to {step_name}, {substep_name}")
        except StepNotYetReachedError:
            step_workflow_advance_until = get_dynamic_state_key(step_name, substep_name)
            print(f"set step_workflow_advance_until ={step_workflow_advance_until}")
    elif load_state:
        await workflow.load_state()
    print(f"step_workflow_advance_until ={step_workflow_advance_until}")
    id = str(workflow.id)

    if not ignore_running_workflows:
        if id in workflows:
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
        workflow=workflow,
        user_input=user_input,
        structured_user_input=structured_user_input,
        async_export=async_export,
        retry_step=retry_step,
        advance_until=step_workflow_advance_until,
        load_state=load_state,
        save_state_to_db=save_state_to_db,
    )

    async for output in gen:
        yield output
    running_workflows[id] = False

    # TODO this adds a timeout but is incredibly slow
    #  try:
    #  while True:
    #  output = await asyncio.wait_for(gen.__anext__(), timeout)
    #  running_workflows[id] = False
    #  yield output
    #  except asyncio.TimeoutError:
    #  running_workflows[id] = False
    #  yield CutTranscriptLinearWorkflowStepOutput(
    #  step_name="", substep_name="", done=False, error="Timeout"
    #  ), True
    #  except StopAsyncIteration:
    #  return


@app.function(**app_kwargs)
async def run(
    workflow: "trimit.backend.cut_transcript.CutTranscriptLinearWorkflow",
    user_input: str | None = None,
    structured_user_input: StructuredUserInput | None = None,
    ignore_running_workflows: bool = False,
    async_export: bool = True,
    load_state: bool = True,
    save_state_to_db: bool = True,
    export_intermediate: bool = False,
):
    print("serve.py run structured_user_input", structured_user_input)
    from trimit.backend.cut_transcript import CutTranscriptLinearWorkflowStepOutput
    from trimit.models import maybe_init_mongo

    await maybe_init_mongo()
    await workflow.restart_state()
    id = str(workflow.id)

    if not ignore_running_workflows:
        if id in workflows:
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
    gen = step_workflow_ignoring_feedback_request(
        workflow=workflow,
        user_input=user_input,
        structured_user_input=structured_user_input,
        async_export=async_export,
        load_state=load_state,
        save_state_to_db=save_state_to_db,
        export_intermediate=export_intermediate,
        stream_raw=False,
    )

    async for output in gen:
        yield output
    running_workflows[id] = False


@app.local_entrypoint()
async def cut_transcript_cli(
    user_email: str,
    timeline_name: str,
    video_hash: str,
    length_seconds: int,
    user_input: str | None = None,
    structured_user_input: StructuredUserInput | None = None,
    force_restart: bool = False,
    ignore_running_workflows: bool = False,
    timeout: float = 120,
):
    from trimit.backend.cut_transcript import (
        CutTranscriptLinearWorkflow,
        CutTranscriptLinearWorkflowStepOutput,
    )

    workflow = await CutTranscriptLinearWorkflow.from_video_hash(
        user_email=user_email,
        timeline_name=timeline_name,
        video_hash=video_hash,
        length_seconds=length_seconds,
        output_folder=LINEAR_WORKFLOW_OUTPUT_FOLDER,
        volume_dir=VOLUME_DIR,
        new_state=force_restart,
    )

    done = False
    i = 0
    while not done:
        steps_ran = []

        started_printing_feedback_request = False
        async for partial_output, is_last in step.remote_gen.aio(
            workflow=workflow,
            user_input=user_input,
            structured_user_input=structured_user_input,
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
