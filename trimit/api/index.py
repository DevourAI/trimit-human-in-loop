import os
import shutil
import uuid
from collections import defaultdict
import re
import asyncio
import json
from datetime import datetime
from pathlib import Path
from torch import export
import yaml

from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_random,
    RetryError,
    retry_if_exception_type,
)

from pydantic import EmailStr, BaseModel, Field
from modal.call_graph import InputStatus
from modal.functions import FunctionCall
from modal import asgi_app, is_local, Dict
from beanie import BulkWriter, PydanticObjectId
from beanie.operators import In
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import URL
from fastapi import (
    FastAPI,
    Form,
    UploadFile,
    File,
    Request,
    Response,
    Query,
    HTTPException,
    Depends,
    Body,
)
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
from fastapi.responses import StreamingResponse, FileResponse

from trimit.models.backend_models import (
    ExportResults,
    CallId,
    CallStatus,
    VideoProcessingStatus,
    UploadVideo,
    UploadedVideo,
    GetStepOutputs,
    CheckFunctionCallResults,
    CutTranscriptLinearWorkflowStepOutput,
    PartialBackendOutput,
    PartialLLMOutput,
    ExportableStepWrapper,
    StructuredUserInput,
)
from trimit.backend.utils import AGENT_OUTPUT_CACHE, export_results_wrapper
from trimit.utils.conf import SHARED_USER_EMAIL
from trimit.utils.async_utils import async_passthrough
from trimit.utils.fs_utils import (
    async_copy_to_s3,
    save_file_to_volume_as_crc_hash,
    save_weblink_to_volume_as_crc_hash,
    get_volume_file_path,
    get_s3_key,
    get_audio_file_path,
    convert_video_codec,
)
from trimit.utils.model_utils import save_video_with_details, check_existing_video
from trimit.utils.video_utils import convert_video_to_audio
from trimit.api.utils import load_workflow
from trimit.app import (
    app,
    get_volume_dir,
    TRIMIT_VIDEO_S3_CDN_BUCKET,
    CDN_ASSETS_PATH,
    volume,
)
from trimit.models import (
    start_transaction,
    maybe_init_mongo,
    Video,
    VideoHighResPathProjection,
    User,
    UploadedVideoProjection,
    FrontendWorkflowState,
    CutTranscriptLinearWorkflowStreamingOutput,
    CutTranscriptLinearWorkflowState,
    FrontendWorkflowProjection,
    Project,
)
from .image import image
from trimit.backend.conf import (
    VIDEO_PROCESSING_CALL_IDS_DICT_NAME,
    LINEAR_WORKFLOW_OUTPUT_FOLDER,
)
from trimit.backend.background_processor import BackgroundProcessor
from trimit.backend.cut_transcript import CutTranscriptLinearWorkflow

background_processor = None
if is_local():
    ASSETS_DIR = "tmp/assets"
    os.makedirs(ASSETS_DIR, exist_ok=True)
else:
    background_processor = BackgroundProcessor()
    ASSETS_DIR = CDN_ASSETS_PATH

TEMP_DIR = Path("/tmp/uploads")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
video_processing_call_ids = Dict.from_name(
    VIDEO_PROCESSING_CALL_IDS_DICT_NAME, create_if_missing=True
)
from trimit.utils.namegen import timeline_namegen, project_namegen


class DynamicCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Check if it's a preflight request
        if request.method == "OPTIONS":
            response = Response()
            self.apply_cors_headers(request, response)
            return response

        response = await call_next(request)
        self.apply_cors_headers(request, response)
        return response

    def apply_cors_headers(self, request: Request, response: Response):
        origin = request.headers.get("origin")
        # Regex to match allowed origins, e.g., any subdomain of trimit.vercel.app
        local_origins = ["http://127.0.0.1:3000", "http://localhost:3000"]
        allow_local = (
            origin
            and origin in local_origins
            and os.environ["ENV"] in ["dev", "staging"]
        )
        allow_remote = origin and origin in (
            "https://trimit-human-in-loop-git-deploy-prod-trimit.vercel.app",
            "https://app.trimit.ai",
            "https://app-staging.trimit.ai",
            "https://app-preview.trimit.ai",
        )
        origin = origin or ""
        if allow_local or allow_remote:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, PUT, DELETE, OPTIONS"
            )
            response.headers["Access-Control-Allow-Headers"] = (
                "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Content-Disposition"
            )
            response.headers["Access-Control-Expose-Headers"] = (
                "Content-Length,Content-Range,Content-Disposition"
            )
        return response


web_app = FastAPI()
web_app.add_middleware(SessionMiddleware, secret_key=os.environ["AUTH_SECRET_KEY"])
web_app.add_middleware(DynamicCORSMiddleware)
if not is_local():
    os.makedirs(ASSETS_DIR, exist_ok=True)


app_kwargs = dict(
    _allow_background_volume_commits=True,
    timeout=80000,
    image=image,
    container_idle_timeout=1200,
    _experimental_boost=True,
    _experimental_scheduler=True,
    keep_warm=1,
)


async def maybe_user_wrapper(user_email: EmailStr | None = None):
    if user_email is None:
        return None
    return await find_or_create_user(user_email)


@retry(stop=stop_after_attempt(10), wait=wait_fixed(0.1) + wait_random(0, 1))
async def copy_shared_video_to_user(video: Video, user: User):
    async with start_transaction() as session:
        if await check_existing_video(user.email, video.md5_hash, session=session):
            return
        new_video = await Video.from_user_email(
            user.email,
            md5_hash=video.md5_hash,
            ext=video.ext,
            upload_datetime=video.upload_datetime,
            details=video.details,
            high_res_user_file_path=video.high_res_user_file_path,
            high_res_user_file_hash=video.high_res_user_file_hash,
            transcription=video.transcription,
            transcription_text=video.transcription_text,
            video_llava_description=video.video_llava_description,
            summary=video.summary,
            speakers_in_frame=video.speakers_in_frame,
            session=session,
        )
        os.makedirs(os.path.dirname(new_video.path(get_volume_dir())), exist_ok=True)
        shutil.copyfile(video.path(get_volume_dir()), new_video.path(get_volume_dir()))
        await new_video.save(session=session)


async def copy_shared_videos_to_user(user: User):
    shared_videos = await Video.find(Video.user.email == SHARED_USER_EMAIL).to_list()
    for video in shared_videos:
        await copy_shared_video_to_user(video, user)


async def find_or_create_user(user_email: EmailStr):
    await maybe_init_mongo()
    user = await User.find_one(User.email == user_email)
    if user is None:
        user = User(email=user_email, name="")
        await user.save()
    # TODO indent this
    if user.email != SHARED_USER_EMAIL:
        await copy_shared_videos_to_user(user)
    return user


async def get_user_email(user_email: EmailStr = Form(...)):
    return user_email


async def form_user_dependency(user_email: EmailStr = Depends(get_user_email)):
    return await find_or_create_user(user_email)


async def get_current_workflow_frontend_state(workflow_id: str):
    await maybe_init_mongo()
    return await CutTranscriptLinearWorkflowState.find_one(
        CutTranscriptLinearWorkflowState.id == PydanticObjectId(workflow_id)
    ).project(FrontendWorkflowProjection)


async def get_current_workflow_or_none(
    workflow_id: str | None = Query(None),
    wait_until_done_running: bool = Query(
        False,
        description="If True, block returning a workflow until workflow is done running its current step",
    ),
    timeout: float = Query(
        5,
        description="Timeout until continuing without waiting if wait_until_done_running=True",
    ),
    wait_interval: float = Query(
        0.1,
        description="poll interval to wait for workflow step to finish if wait_until_done_running=True",
    ),
):
    if workflow_id is None or workflow_id == "":
        return None
    return await get_current_workflow(
        workflow_id=workflow_id,
        wait_until_done_running=wait_until_done_running,
        timeout=timeout,
        wait_interval=wait_interval,
    )


async def get_current_workflow(
    workflow_id: str,
    wait_until_done_running: bool = Query(
        False,
        description="If True, block returning a workflow until workflow is done running its current step",
    ),
    timeout: float = Query(
        5,
        description="Timeout until continuing without waiting if wait_until_done_running=True",
    ),
    wait_interval: float = Query(
        0.1,
        description="poll interval to wait for workflow step to finish if wait_until_done_running=True",
    ),
):
    try:
        return await load_workflow(
            workflow_id=workflow_id,
            with_output=True,
            wait_until_done_running=wait_until_done_running,
            timeout=timeout,
            wait_interval=wait_interval,
        )
    except Exception as e:
        print("load workflow exception", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


async def get_current_project(
    user_email: str, project_id: str | None = None, project_name: str | None = None
):
    if project_id:
        return await Project.get(project_id)
    try:
        return await Project.from_user_email(user_email=user_email, name=project_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.function(**app_kwargs)
@asgi_app()
def frontend_server():
    return web_app


@web_app.get("/openapi.yaml", include_in_schema=False)
async def get_openapi_yaml():
    openapi_schema = get_openapi(
        openapi_version="3.0.0",
        title="TrimIt API",
        version="0.0.1",
        description="API documentation",
        routes=web_app.routes,
    )
    yaml_schema = yaml.safe_dump(openapi_schema, sort_keys=False)
    return Response(content=yaml_schema, media_type="application/x-yaml")


@web_app.get(
    "/get_step_outputs",
    response_model=GetStepOutputs,
    tags=["Steps"],
    summary="Get specific outputs for a given workflow and list of steps",
    description="TODO",
)
async def get_step_outputs(
    step_keys: str | None = Query(
        None,
        description="Step keys in format `step_name.substep_name`, comma-separated",
    ),
    step_names: str | None = Query(
        None,
        description="Step names (no substeps), comma-separated. If provided instead of step_keys, the last substep of each step will be returned",
    ),
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
):
    if not workflow:
        raise HTTPException(status_code=400, detail="Workflow not found")

    if step_keys is not None:
        return GetStepOutputs(
            outputs=await workflow.get_output_for_keys(keys=step_keys.split(","))
        )
    elif step_names is not None:
        return GetStepOutputs(
            outputs=await workflow.get_output_for_names(names=step_names.split(","))
        )
    raise ValueError("one of step_keys or step_names must be provided")


@web_app.get(
    "/get_all_outputs",
    response_model=GetStepOutputs,
    tags=["Steps"],
    summary="Get all outputs, ordered earliest to latest in time, for a given workflow",
    description="TODO",
)
async def get_all_outputs(
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
):
    if not workflow:
        raise HTTPException(status_code=400, detail="Workflow not found")

    return GetStepOutputs(outputs=await workflow.get_all_outputs())


def find_fc_node_from_graph(graph, call_id):
    if graph.function_call_id == call_id:
        return graph
    for child in graph.children:
        matching_child_node = find_fc_node_from_graph(child, call_id)
        if matching_child_node:
            return matching_child_node
    return None


def check_call_status(modal_call_id, timeout: float = 0.5, with_result=False):
    try:
        fc = FunctionCall.from_id(modal_call_id)
        try:
            if with_result:
                output = fc.get(timeout=timeout)
                result = {"output": output["result"], "status": "done"}
            else:
                status = "error"
                matching_node = find_fc_node_from_graph(
                    fc.get_call_graph()[0], modal_call_id
                )
                if matching_node:
                    status_code = matching_node.status
                    if status_code == InputStatus.SUCCESS:
                        status = "done"
                    elif status_code == InputStatus.PENDING:
                        status = "pending"
                result = {"status": status}
        except TimeoutError as e:
            # result = {"status": "error", "error": "Timeout"}
            result = {"status": "pending"}
    except Exception as e:
        if "not found" in str(e):
            result = {"status": "done"}
        else:
            print("unknown error:", e)
            result = {"status": "error", "error": str(e)}
    return CallStatus(call_id=modal_call_id, **result)


# frontend should poll for this
# sometime in the future we can use kafka or pubsub to push to frontend
@web_app.get(
    "/get_video_processing_status",
    response_model=list[VideoProcessingStatus],
    tags=["FunctionCalls"],
    summary="Get status of video processing jobs",
    description="TODO",
)
async def get_video_processing_status(
    user: User = Depends(find_or_create_user),
    video_hashes: list[str] | None = None,
    timeout: float = 0,
):
    if video_hashes is None:
        await maybe_init_mongo()
        video_hashes = [
            video.md5_hash
            for video in await Video.find(Video.user.email == user.email)
            .project(VideoHighResPathProjection)
            .to_list()
        ]
    call_ids = [
        (
            video_processing_call_ids[(user.email, h)]
            if (user.email, h) in video_processing_call_ids
            else None
        )
        for h in video_hashes
    ]
    statuses = [
        check_call_status(call_id, timeout=timeout) if call_id else None
        for call_id in call_ids
    ]
    model_statuses = []
    for call_id, video_hash, status in zip(call_ids, video_hashes, statuses):
        if status is None:
            status = CallStatus(call_id=call_id, status="done")
        elif status.status == "done":
            if (user.email, video_hash) in video_processing_call_ids:
                try:
                    del video_processing_call_ids[(user.email, video_hash)]
                except KeyError:
                    # TODO not sure why this happens since we check the key on the previous line
                    video_processing_call_ids[(user.email, video_hash)] = None
        video_processing_status = VideoProcessingStatus.from_call_status(
            status, video_hash
        )
        model_statuses.append(video_processing_status)
    return model_statuses


# frontend should poll for this
# sometime in the future we can use kafka or pubsub to push to frontend
@web_app.get(
    "/check_function_call_results",
    response_model=CheckFunctionCallResults,
    tags=["FunctionCalls"],
    summary="Check the status of modal function calls",
    description="TODO",
)
async def check_function_call_results(modal_call_ids: str, timeout: float = 0):
    modal_call_ids_split = modal_call_ids.split(",")
    statuses = await asyncio.gather(
        *[
            async_passthrough(check_call_status(call_id, timeout))
            for call_id in modal_call_ids_split
        ]
    )
    print(statuses)
    return CheckFunctionCallResults(statuses=statuses)


class StepInput(BaseModel):
    user_input: str | None = Field(
        None,
        description="string conversational input from the user that will be provided to the underlying LLM",
    )
    streaming: bool = Field(
        True,
        description="If False, do not stream intermediate output and just provide the final workflow output after all substeps have ran",
    )
    ignore_running_workflows: bool = Field(
        False,
        description="If True, run this workflow's step even if it is already running. May lead to race conditions in underlying database",
    )
    retry_step: bool = Field(
        False,
        description="If True, indicates that the client desires to run the last step again, optionally with user feedback in user_input instructing the LLM how to modify its previous output",
    )
    structured_user_input: StructuredUserInput | None = Field(
        None,
        description="structured input that will be passed to a step to guide modification, separate from the LLM conversation. The particular structure is unique to each step. Only one of the subfields should be defined",
    )
    advance_until: int | None = Field(
        None,
        description="Advance steps until this step index is reached, or revert step to before this index and rerun step at this index",
    )
    export_intermediate: bool = Field(
        False,
        description="For run(), if True this will export the output of intermediate steps to disk",
    )


@web_app.post(
    "/step",
    response_model=CutTranscriptLinearWorkflowStreamingOutput,
    tags=["Steps"],
    summary="Run next step",
    description="""
This method returns a strongly typed `StreamingResponse`, where each chunk will be proper JSON.
The chunk JSONs are each "instances" of the `CutTranscriptLinearWorkflowStreamingOutput` wrapper class.
The wrapper class includes several type variants, only one of which will be non-null at a time.

The method will call the underlying `step()` method of the workflow until a step is run that needs user feedback to proceed.
Along the way, partial output will be streamed to the client.
These partial outputs include responses from the backend (`partial_backend_output: PartialBackendOutput`) and responses from the LLM (`partial_llm_output: PartialLLMOutput`).
While present in the wrapper class, the client should not expect to receive `FinalLLMOutput` on the frontend, as that is parsed by the backend for further processing.
The last output of every substep is of type `CutTranscriptLinearWorkflowStepOutput`,
which is also the type that is returned by the API in methods like `/get_step_outputs` and `/get_all_outputs`.
However, since some substeps do not request user feedback, some of these outputs will be streamed as `partial_step_output` to the client,
and the backend will continue on without waiting for feedback.
The last output this method produces is always `final_step_output`, which includes a request/prompt for user feedback (`response.final_step_output.user_feedback_request`)
""",
)
def step_endpoint(
    step_input: StepInput,
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
):
    if workflow is None:
        raise HTTPException(
            status_code=400, detail="necessary workflow params not provided"
        )
    step_params = {
        "workflow": workflow,
        "user_input": step_input.user_input,
        "structured_user_input": step_input.structured_user_input,
        "ignore_running_workflows": step_input.ignore_running_workflows,
        "async_export": os.environ.get("ASYNC_EXPORT", True),
        "retry_step": step_input.retry_step,
        "advance_until": step_input.advance_until,
    }
    print("step_params", step_params)
    from trimit.backend.serve import step as step_function

    print(f"Starting step with params: {step_params}")
    if not step_input.streaming:
        step_function.spawn(**step_params)
    else:

        async def streamer():
            yield CutTranscriptLinearWorkflowStreamingOutput(
                workflow_id=str(workflow.id),
                partial_backend_output=PartialBackendOutput(value="Running step"),
            ).model_dump_json()
            if is_local():
                method = step_function.local
            else:
                method = step_function.remote_gen.aio

            last_result = None
            async for partial_result, is_last in method(**step_params):
                if last_result is not None:
                    if last_result.user_feedback_request:
                        raise ValueError(
                            "Step loop should not continue after a feedback request"
                        )
                    yield CutTranscriptLinearWorkflowStreamingOutput(
                        workflow_id=str(workflow.id), partial_step_output=last_result
                    ).model_dump_json()
                    last_result = None
                if isinstance(partial_result, CutTranscriptLinearWorkflowStepOutput):
                    if not is_last:
                        raise ValueError(
                            "step result should be CutTranscriptLinearWorkflowStepOutput if is_last is True"
                        )
                    if partial_result.user_feedback_request:
                        last_result = partial_result
                elif isinstance(partial_result, PartialLLMOutput):
                    yield CutTranscriptLinearWorkflowStreamingOutput(
                        workflow_id=str(workflow.id), partial_llm_output=partial_result
                    ).model_dump_json()
                elif isinstance(partial_result, PartialBackendOutput):
                    yield CutTranscriptLinearWorkflowStreamingOutput(
                        workflow_id=str(workflow.id),
                        partial_backend_output=partial_result,
                    ).model_dump_json()
                elif isinstance(partial_result, ExportResults):
                    continue
                elif isinstance(partial_result, CallId):
                    continue
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Unparseable response from internal step function: {partial_result}",
                    )
                await asyncio.sleep(0)
            latest_state = await workflow.get_latest_frontend_state(
                volume_dir=get_volume_dir(),
                with_load_state=True,
                #  with_outputs=True,
                #  with_all_steps=True,
                #  only_last_substep_outputs=True,
            )
            yield CutTranscriptLinearWorkflowStreamingOutput(
                workflow_id=str(workflow.id), final_state=latest_state
            ).model_dump_json()

        return StreamingResponse(streamer(), media_type="text/event-stream")


class RunInput(StepInput):
    user_email: EmailStr | None = None
    length_seconds: int | None = None
    video_hash: str | None = None
    name: str | None = None
    video_type: str | None = None
    use_agent_cache: bool = False
    n_variations: int | None = None


async def run_streamer(workflow, **run_params):
    from trimit.backend.serve import run as run_function

    yield CutTranscriptLinearWorkflowStreamingOutput(
        workflow_id=str(workflow.id),
        partial_backend_output=PartialBackendOutput(value="Running step"),
    ).model_dump_json()
    if is_local():
        method = run_function.local
    else:
        method = run_function.remote_gen.aio

    last_result = None
    async for partial_result, is_last in method(workflow, **run_params):
        if last_result is not None:
            yield CutTranscriptLinearWorkflowStreamingOutput(
                workflow_id=str(workflow.id), partial_step_output=last_result
            ).model_dump_json()
            last_result = None
        if isinstance(partial_result, CutTranscriptLinearWorkflowStepOutput):
            if not is_last:
                raise ValueError(
                    "step result should be CutTranscriptLinearWorkflowStepOutput if is_last is True"
                )
            if partial_result.user_feedback_request:
                last_result = partial_result
        elif isinstance(partial_result, PartialLLMOutput):
            yield CutTranscriptLinearWorkflowStreamingOutput(
                workflow_id=str(workflow.id), partial_llm_output=partial_result
            ).model_dump_json()
        elif isinstance(partial_result, PartialBackendOutput):
            yield CutTranscriptLinearWorkflowStreamingOutput(
                workflow_id=str(workflow.id), partial_backend_output=partial_result
            ).model_dump_json()
        elif isinstance(partial_result, ExportResults):
            pass
        elif isinstance(partial_result, CallId):
            pass
        else:
            print("partial_result", partial_result)
            print("typeof partial_result", type(partial_result))
            print(f"Unparseable response from internal step function: {partial_result}")
            #  raise HTTPException(
            #  status_code=500,
            #  detail=f"Unparseable response from internal step function: {partial_result}",
            #  )
        await asyncio.sleep(0)
    print("loading last state")
    AGENT_OUTPUT_CACHE.close()
    try:
        await volume.reload()
    except TypeError:
        pass
    latest_state = await workflow.get_latest_frontend_state(
        volume_dir=get_volume_dir(),
        with_load_state=True,
        #  with_outputs=True,
        #  with_all_steps=True,
        #  only_last_substep_outputs=True,
    )
    yield CutTranscriptLinearWorkflowStreamingOutput(
        workflow_id=str(workflow.id), final_state=latest_state
    ).model_dump_json()


@web_app.post(
    "/run",
    response_model=CutTranscriptLinearWorkflowStreamingOutput,
    tags=["Steps"],
    summary="Run entire workflow start to finish",
    description="""
This method returns a strongly typed `StreamingResponse`, where each chunk will be proper JSON.
The chunk JSONs are each "instances" of the `CutTranscriptLinearWorkflowStreamingOutput` wrapper class.
The wrapper class includes several type variants, only one of which will be non-null at a time.

The method will call the underlying `step()` method of the workflow repeatedly until all steps have been run.
Along the way, partial output will be streamed to the client.
These partial outputs include responses from the backend (`partial_backend_output: PartialBackendOutput`) and responses from the LLM (`partial_llm_output: PartialLLMOutput`).
While present in the wrapper class, the client should not expect to receive `FinalLLMOutput` on the frontend, as that is parsed by the backend for further processing.
The last output of every substep is of type `CutTranscriptLinearWorkflowStepOutput`,
which is also the type that is returned by the API in methods like `/get_step_outputs` and `/get_all_outputs`.
The last output this method produces is always `FrontendWorkflowState`, which includes the entire state of the workflow.
""",
)
async def run(
    run_input: RunInput,
    project: Project | None = Depends(get_current_project),
    workflow: CutTranscriptLinearWorkflow | None = Depends(
        get_current_workflow_or_none
    ),
):
    if workflow is None:
        workflows = []
        for _ in range(run_input.n_variations or 1):
            state = await CutTranscriptLinearWorkflowState.find_or_create_from_video_hash(
                video_hash=run_input.video_hash,
                user_email=run_input.user_email,
                timeline_name=timeline_namegen(),
                video_type=run_input.video_type or "",
                volume_dir=get_volume_dir(),
                output_folder=LINEAR_WORKFLOW_OUTPUT_FOLDER,
                length_seconds=run_input.length_seconds,
                # nstages=nstages,
                project=project,
            )
            await state.save()
            workflow = CutTranscriptLinearWorkflow(state=state)
            workflows.append(workflow)
    else:
        assert workflow.state is not None
        if workflow.state.project is None:
            workflow.state.project = project
        elif workflow.state.project.id != project.id:
            raise HTTPException(
                status_code=400,
                detail="workflow already has a project, but it does not match provided project",
            )
        workflows = [workflow]
        for _ in range((run_input.n_variations or 1) - 1):
            workflows.append(await workflow.copy())

    run_params_list = [
        {
            "workflow": workflow,
            "user_input": run_input.user_input,
            "structured_user_input": run_input.structured_user_input,
            "ignore_running_workflows": run_input.ignore_running_workflows,
            "async_export": os.environ.get("ASYNC_EXPORT", True),
            "export_intermediate": run_input.export_intermediate,
            "use_agent_cache": run_input.use_agent_cache,
        }
        for workflow in workflows
    ]
    from trimit.backend.serve import run as run_function

    print(f"Running with params: {run_params_list[0]} ({len(workflows)} variations)")
    if not run_input.streaming:
        for run_params in run_params_list:
            run_function.spawn(**run_params)
    else:
        if len(workflows) > 1:
            for run_params in run_params_list[1:]:
                run_function.spawn(**run_params)
        return StreamingResponse(
            run_streamer(**run_params_list[0]), media_type="text/event-stream"
        )


@web_app.get(
    "/reset_workflow",
    tags=["Workflows"],
    summary="Reset a workflow to initial state",
    description="TODO",
)
async def reset_workflow(
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
):
    if workflow is None:
        raise HTTPException(status_code=400, detail="Workflow not found")
    print(f"Resetting workflow {workflow.id}")
    await workflow.restart_state()
    print(f"Workflow {workflow.id} reset")


@web_app.get(
    "/revert_workflow_step",
    tags=["Workflows"],
    summary="Revert a workflow one step",
    description="TODO",
)
async def revert_workflow_step(
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
    to_before_retries: bool = Query(
        False,
        description="If True, revert to before any retries. Otherwise, revert to the most recent retry",
    ),
):
    if workflow is None:
        raise HTTPException(status_code=400, detail="Workflow not found")
    print(f"Reverting workflow {workflow.id}")
    await workflow.revert_step(before_retries=to_before_retries)
    print(f"Workflow {workflow.id} reverted")


@web_app.get(
    "/revert_workflow_step_to",
    tags=["Workflows"],
    summary="Revert a workflow to a particular step/substep",
    description="TODO",
)
async def revert_workflow_step_to(
    step_name: str,
    substep_name: str | None = Query(
        None, description="if not provided, assume the first substep"
    ),
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
):
    if workflow is None:
        raise HTTPException(status_code=400, detail="Workflow not found")
    print(f"Reverting workflow {workflow.id}")
    matching = [s for s in workflow.steps if s.name == step_name]
    if len(matching) == 0:
        raise HTTPException(status_code=400, detail=f"step named {step_name} not found")
    substep_name = matching[0].substeps[0].name
    await workflow.revert_step_to_before(step_name, substep_name)
    print(f"Workflow {workflow.id} reverted to {step_name}.{substep_name}")


@web_app.get(
    "/workflows",
    tags=["Workflows"],
    response_model=list[FrontendWorkflowProjection],
    summary="List all workflows for a user, optionally filtered by video hashes",
    description="TODO",
)
async def workflows(
    user_email: str = Query(...), video_hashes: list[str] | None = Query(None)
):
    await maybe_init_mongo()
    filters = [CutTranscriptLinearWorkflowState.static_state.user.email == user_email]
    if video_hashes and len(video_hashes):
        filters.append(
            In(
                CutTranscriptLinearWorkflowState.static_state.video.md5_hash,
                video_hashes,
            )
        )
    return (
        await CutTranscriptLinearWorkflowState.find(*filters)
        .project(FrontendWorkflowProjection)
        .to_list()
    )


@web_app.post(
    "/workflows/new",
    tags=["Workflows"],
    response_model=str,
    summary="Create a new workflow, returning its id",
    description="TODO",
)
async def create_workflow(
    user_email: str = Form(...),
    video_hash: str = Form(...),
    project_name: str = Form(None),
    timeline_name: str = Form(None),
    length_seconds: int = Form(...),
    nstages: int = Form(2),
    video_type: str | None = Form(None),
    recreate: bool = Form(
        False,
        description="If True, recreate the workflow from scratch if it already exists",
    ),
):
    await maybe_init_mongo()
    method = CutTranscriptLinearWorkflowState.find_or_create_from_video_hash
    if recreate:
        method = CutTranscriptLinearWorkflowState.recreate_from_video_hash
    state = await method(
        video_hash=video_hash,
        user_email=user_email,
        project_name=project_name or project_namegen(),
        timeline_name=timeline_name or timeline_namegen(),
        volume_dir=get_volume_dir(),
        output_folder=LINEAR_WORKFLOW_OUTPUT_FOLDER,
        length_seconds=length_seconds,
        video_type=video_type or "",
        nstages=nstages,
    )
    await state.save()
    return str(state.id)


class WorkflowExists(BaseModel):
    exists: bool


@web_app.get(
    "/workflow_exists",
    tags=["Workflows"],
    response_model=WorkflowExists,
    summary="Return true if a workflow exists in db",
    description="TODO",
)
async def workflow_exists(
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
):
    return WorkflowExists(exists=workflow is not None)


@web_app.get(
    "/workflow",
    tags=["Workflows"],
    response_model=FrontendWorkflowProjection,
    summary="Return workflow details",
    description="TODO",
)
async def get_workflow_details(
    workflow_details: CutTranscriptLinearWorkflowState | None = Depends(
        get_current_workflow_frontend_state
    ),
):
    if workflow_details is None:
        raise HTTPException(status_code=400, detail="Workflow not found")
    if not isinstance(workflow_details, FrontendWorkflowProjection):
        raise HTTPException(
            status_code=500,
            detail=f"Expected FrontendWorkflowProjection, got {type(workflow_details)}",
        )
    return workflow_details


@web_app.get(
    "/all_steps",
    tags=["Workflows"],
    response_model=list[ExportableStepWrapper],
    summary="Get ordered, detailed description of each step for a workflow",
    description="These will be very similar for each workflow. The only current difference is in the number of stages.",
)
async def get_all_steps(
    workflow: CutTranscriptLinearWorkflow = Depends(get_current_workflow),
):
    return workflow.steps.to_exportable()


@web_app.get(
    "/get_latest_state",
    tags=["Workflows"],
    response_model=FrontendWorkflowState,
    summary="Get the latest state of a workflow",
    description="TODO",
)
async def get_latest_state(
    workflow: CutTranscriptLinearWorkflow = Depends(get_current_workflow),
    #  with_outputs: bool = Query(
    #  True, description="if True, include ordered list of step outputs"
    #  ),
    #  with_all_steps: bool = Query(
    #  True,
    #  description="if True, include the Steps object of all the workflow's steps",
    #  ),
    #  only_last_substep_outputs: bool = Query(
    #  True,
    #  description="If True, only return the output of the latest substep for each step",
    #  ),
):
    if workflow is None:
        raise HTTPException(status_code=400, detail="Workflow not found")
    AGENT_OUTPUT_CACHE.close()
    if not is_local():
        try:
            await volume.reload()
        except TypeError:
            pass

    return await workflow.get_latest_frontend_state(
        volume_dir=get_volume_dir(), with_load_state=False
    )
    #  with_outputs=with_outputs,
    #  with_all_steps=with_all_steps,
    #  only_last_substep_outputs=only_last_substep_outputs,
    #  )


@web_app.get(
    "/get_latest_export_results",
    response_model=ExportResults,
    response_model_exclude_unset=True,
)
async def get_latest_export_results(
    workflow: CutTranscriptLinearWorkflow | None = Depends(
        get_current_workflow_or_none
    ),
    step_name: str | None = None,
):
    if workflow is None:
        raise HTTPException(status_code=400, detail="Must provide workflow_id")
    assert workflow.state is not None

    print("os.listdir(ASSETS_DIR):", os.listdir(ASSETS_DIR))
    AGENT_OUTPUT_CACHE.close()
    try:
        await volume.reload()
    except TypeError:
        pass

    state = await workflow.get_latest_frontend_state(volume_dir=get_volume_dir())
    output_index = -1
    if state.outputs[-1].step_name == "end":
        output_index = -2
    if step_name is not None:
        matching_output_index = [
            i for i, o in enumerate(state.outputs) if o.step_name == step_name
        ]
        if len(matching_output_index) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"could not find output for step_name={step_name}",
            )
        output_index = matching_output_index[0]
    if output_index >= len(state.mapped_export_result):
        raise HTTPException(
            status_code=500,
            detail="mapped_export_result does not contain desired output",
        )
    return state.mapped_export_result[output_index]


@web_app.post("/redo_export_results", response_model=CallId)
async def redo_export_results(
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
    step_name: str | None = None,
    substep_name: str | None = None,
):
    if workflow is None:
        raise HTTPException(status_code=400, detail="Must provide workflow_id")
    output = None
    async for output in workflow.redo_export_results(
        step_name=step_name, substep_name=substep_name
    ):
        continue
    assert isinstance(output, CallId)
    assert isinstance(output.call_id, str)
    return output


@web_app.get("/download_transcript_text")
async def download_transcript_text(
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
    step_name: str | None = None,
    substep_name: str | None = None,
    stream: bool = False,
):

    if workflow is None:
        raise HTTPException(status_code=400, detail="Must provide workflow_id")

    if step_name is None:
        export_result = await workflow.most_recent_export_result(with_load_state=False)
    else:
        export_result = await workflow.export_result_for_step_substep_name(
            step_name=step_name, substep_name=substep_name, with_load_state=False
        )
    file_path = export_result.get("transcript_text")

    if file_path is None:
        raise HTTPException(status_code=500, detail="No transcript found")
    assert isinstance(file_path, str)
    #  if not os.path.exists(file_path):
    #  raise HTTPException(
    #  status_code=500, detail=f"Transcript not found at {file_path}"
    #  )

    return FileResponse(
        file_path, media_type="text/plain", filename=os.path.basename(file_path)
    )


@web_app.get("/download_soundbites_text")
async def download_soundbites_text(
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
    step_name: str | None = None,
    substep_name: str | None = None,
    stream: bool = False,
):

    if workflow is None:
        raise HTTPException(
            status_code=400, detail="Must provide timeline name and length_seconds"
        )
    if step_name is None:
        export_result = await workflow.most_recent_export_result(with_load_state=False)
    else:
        export_result = await workflow.export_result_for_step_substep_name(
            step_name=step_name, substep_name=substep_name, with_load_state=False
        )
    file_path = export_result.get("soundbites_text")

    if file_path is None:
        raise HTTPException(status_code=500, detail="No soundbites found")
    assert isinstance(file_path, str)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=500, detail=f"Soundbites not found at {file_path}"
        )

    return FileResponse(
        file_path, media_type="text/plain", filename=os.path.basename(file_path)
    )


@web_app.get("/download_soundbites_timeline")
async def download_soundbites_timeline(
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
    step_name: str | None = None,
    substep_name: str | None = None,
    stream: bool = False,
):

    print("workflow:", workflow)
    if workflow is None:
        raise HTTPException(
            status_code=400, detail="Must provide timeline name and length_seconds"
        )
    if step_name is None:
        export_result = await workflow.most_recent_export_result(with_load_state=False)
    else:
        export_result = await workflow.export_result_for_step_substep_name(
            step_name=step_name, substep_name=substep_name, with_load_state=False
        )
    print("export_result:", export_result)
    file_path = export_result.get("soundbites_timeline")
    print("file_path:", file_path)

    if file_path is None:
        print("no file path found")
        raise HTTPException(status_code=500, detail="No soundbites timeline found")

    if not isinstance(file_path, str):
        print("file path not a string")
        raise HTTPException(status_code=500, detail=f"file_path not a string")
    if not os.path.exists(file_path):
        print("file path doesn't exist")
        raise HTTPException(
            status_code=500, detail=f"Soundbites timeline not found at {file_path}"
        )

    return FileResponse(
        file_path, media_type="application/xml", filename=os.path.basename(file_path)
    )


@web_app.get("/download_timeline")
async def download_timeline(
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
    step_name: str | None = None,
    substep_name: str | None = None,
    stream: bool = False,
):

    if workflow is None:
        print("download_timeline: workflow is none")
        raise HTTPException(status_code=400, detail="Must provide workflow_id")
    if step_name is None:
        export_result = await workflow.most_recent_export_result(with_load_state=False)
    else:
        try:
            export_result = await workflow.export_result_for_step_substep_name(
                step_name=step_name, substep_name=substep_name, with_load_state=False
            )
        except Exception as e:
            print(e)
            raise

    timeline_path = export_result.get("video_timeline")

    if timeline_path is None:
        print("download_timeline: timeline_path is none")
        raise HTTPException(status_code=500, detail="No timeline found")
    if not isinstance(timeline_path, str):
        raise HTTPException(
            status_code=500, detail=f"Timeline path {timeline_path} not a string"
        )
    if not os.path.exists(timeline_path):
        print("download_timeline: timeline_path not found")
        raise HTTPException(
            status_code=500, detail=f"Timeline not found at {timeline_path}"
        )

    return FileResponse(
        timeline_path,
        media_type="application/xml",
        filename=os.path.basename(timeline_path),
    )


@web_app.get("/video")
async def stream_video(
    request: Request,
    video_path: str | None = None,
    workflow: CutTranscriptLinearWorkflow | None = Depends(
        get_current_workflow_or_none
    ),
    step_name: str | None = None,
    substep_name: str | None = None,
    stream: bool = False,
):
    if video_path is None and workflow is None:
        print("Must provide video path or workflow_id")
        raise HTTPException(
            status_code=400, detail="Must provide video path or workflow_id"
        )
    elif video_path is None:
        assert workflow is not None
        if step_name is None:
            export_result = await workflow.most_recent_export_result(
                with_load_state=False
            )
        else:
            export_result = await workflow.export_result_for_step_substep_name(
                step_name=step_name, substep_name=substep_name, with_load_state=False
            )

        video_path = export_result.get("video")
    if video_path is None:
        raise HTTPException(status_code=400, detail="No video found")
    assert isinstance(video_path, str)
    #  if not os.path.exists(video_path):
    #  print(f"Video not found at {video_path}")
    #  raise HTTPException(status_code=400, detail=f"Video not found at {video_path}")
    extension = os.path.splitext(video_path)[1]
    media_type = f"video/{extension[1:]}"
    if not stream:
        return FileResponse(
            video_path, media_type=media_type, filename=os.path.basename(video_path)
        )

    def iterfile():
        with open(video_path, mode="rb") as file_like:  # open the file in binary mode
            yield from file_like  # yield the binary data

    range_header = request.headers.get("range", None)
    file_size = os.path.getsize(video_path)
    if not range_header:
        return StreamingResponse(iterfile(), media_type=media_type)

    start, end = range_header.replace("bytes=", "").split("-")
    start = int(start)
    end = int(end) if end else file_size - 1

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
        "Content-Type": "video/mp4",
    }

    def video_stream():
        with open(video_path, "rb") as video:
            video.seek(start)
            while chunk := video.read(8192):
                yield chunk

    return StreamingResponse(video_stream(), status_code=206, headers=headers)


@web_app.get("/uploaded_high_res_video_paths")
async def uploaded_high_res_video_paths(
    user: User = Depends(find_or_create_user), md5_hashes: list[str] = Query(None)
):
    await maybe_init_mongo()
    video_filters = [Video.user.email == user.email]
    if md5_hashes:
        video_filters.append(In(Video.md5_hash, md5_hashes))

    return {
        video.high_res_user_file_path: video.md5_hash
        for video in await Video.find(*video_filters)
        .project(VideoHighResPathProjection)
        .to_list()
    }


@web_app.get("/uploaded_video_hashes")
async def uploaded_video_hashes(
    user: User = Depends(find_or_create_user),
    high_res_user_file_paths: list[str] = Query(None),
):
    await maybe_init_mongo()
    video_filters = [Video.user.email == user.email]
    if high_res_user_file_paths:
        video_filters.append(
            In(Video.high_res_user_file_path, high_res_user_file_paths)
        )
    return {
        video.high_res_user_file_path: video.md5_hash
        for video in await Video.find(*video_filters)
        .project(VideoHighResPathProjection)
        .to_list()
    }


def remote_video_stream_url_for_path(base_url, path):
    return f"{base_url}/video?video_path={path}"


@web_app.get(
    "/uploaded_videos",
    response_model=list[UploadedVideo],
    tags=["Videos"],
    summary="Get information about a user's uploaded videos",
    description="TODO",
)
async def uploaded_videos(request: Request, user: User = Depends(find_or_create_user)):
    await maybe_init_mongo()
    videos = (
        await Video.find(Video.user.email == user.email)
        .project(UploadedVideoProjection)
        .to_list()
    )
    asset_copy_tasks = []
    for video in videos:
        asset_copy_tasks.append(
            video.asset_path_with_fallback(get_volume_dir(), ASSETS_DIR)
        )
    asset_paths = await asyncio.gather(*asset_copy_tasks)
    data = [
        UploadedVideo(
            filename=video.high_res_user_file_path,
            video_hash=video.md5_hash,
            path=asset_path,
            remote_url=asset_path,
            duration=video.duration or 0,
            title=video.title or "",
        )
        for video, asset_path in zip(videos, asset_paths)
    ]
    return data


@web_app.post(
    "/upload",
    response_model=UploadVideo,
    tags=["UploadVideo"],
    summary="Upload a video",
    description="TODO",
)
async def upload_multiple_files(
    files: list[UploadFile] = File([]),
    user: User = Depends(form_user_dependency),
    high_res_user_file_paths: list[str] = Form([]),
    timeline_name: str = Form(...),
    web_links: list[str] = Form([]),
    overwrite: bool = Form(False),
    use_existing_output: bool = Form(True),
    reprocess: bool = Form(False),
):
    assert background_processor is not None

    print("in upload")
    print(
        "files",
        files,
        "user",
        user,
        "high_res_user_file_paths",
        high_res_user_file_paths,
        "timeline_name",
        timeline_name,
        "web_links",
        web_links,
        "overwrite",
        overwrite,
        "use_existing_output",
        use_existing_output,
        "reprocess",
        reprocess,
    )
    web_links = [l for l in web_links if l]
    await maybe_init_mongo()
    if not files and not web_links:
        raise HTTPException(
            status_code=400, detail="No files uploaded and no web links provided"
        )
    if files:
        if not high_res_user_file_paths:
            high_res_user_file_paths = [file.filename or "" for file in files]
    print(
        f"Received upload request for {len(files)} files and {len(web_links)} web links"
    )

    upload_datetime = datetime.now()

    video_details = []

    is_web_link_list = [False] * len(files) + [True] * len(web_links)
    # TODO allow same video for different users

    volume_dir = get_volume_dir()
    resp_msgs = []
    for i, (file, high_res_user_file_path) in enumerate(
        list(zip(files, high_res_user_file_paths)) + [(l, "") for l in web_links]
    ):
        is_web_link = is_web_link_list[i]
        volume_file_dir = Path(
            get_volume_file_path(user, upload_datetime, "temp", volume_dir=volume_dir)
        ).parent
        if is_web_link:
            assert isinstance(file, str)
            try:
                volume_file_path, high_res_path, tmp_audio_path, title = (
                    await save_weblink_to_volume_as_crc_hash(file, volume_file_dir)
                )
                # TODO here
            except NotImplementedError as e:
                raise HTTPException(status_code=400, detail=str(e))
        else:
            volume_file_path = await save_file_to_volume_as_crc_hash(
                file, file.filename or "", volume_file_dir
            )
            title = file.filename or ""
            high_res_path = None
            tmp_audio_path = None
        print(f"Saved file to {volume_file_path}")
        # TODO check if weird apple codec before converting
        # convert_video_codec(volume_file_path, codec="libx264")
        # print(f"Converted video codec to libx264")

        filename = Path(volume_file_path).name
        video_hash = Path(volume_file_path).stem
        ext = Path(volume_file_path).suffix

        video = await check_existing_video(
            user.email, video_hash, ignore_existing=overwrite
        )
        existing = False
        if video is not None:
            existing = True
            video_hash = video.md5_hash
            ext = video.ext
            if not reprocess:
                resp_msgs.append(
                    f"video {video_hash} ({high_res_user_file_path}) already exists"
                )
                print(
                    f"{video_hash} ({high_res_user_file_path}) existing on disk and not reprocess. continuing"
                )
                continue
            upload_datetime = video.upload_datetime
            volume_file_path = video.path(volume_dir)
            audio_file_path = video.audio_path(volume_dir)
        else:
            if not is_local():
                s3_key = get_s3_key(user, upload_datetime, Path(volume_file_path).name)
                print(f"Saving file to {TRIMIT_VIDEO_S3_CDN_BUCKET}/{s3_key}")
                await async_copy_to_s3(
                    TRIMIT_VIDEO_S3_CDN_BUCKET, str(volume_file_path), str(s3_key)
                )
            audio_file_path = get_audio_file_path(
                user, upload_datetime, filename, volume_dir=volume_dir
            )
            if tmp_audio_path:
                audio_file_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(tmp_audio_path, audio_file_path)
            else:
                convert_video_to_audio(str(volume_file_path), str(audio_file_path))

        video_details.append(
            {
                "volume_file_path": volume_file_path,
                "video_hash": video_hash,
                "ext": ext,
                "upload_datetime": upload_datetime,
                "high_res_user_file_path": high_res_user_file_path,
                "high_res_local_file_path": high_res_path,
                "title": title,
                "high_res_user_file_hash": "",
                "existing": existing,
            }
        )
    if len(video_details) == 0:
        return UploadVideo(result="success", messages=resp_msgs)

    to_process = []
    async with BulkWriter() as bulk_writer:
        for video_detail in video_details:
            if video_detail["existing"] and not overwrite:
                if reprocess:
                    to_process.append(video_detail["video_hash"])
                    continue
            to_process.append(video_detail["video_hash"])
            await save_video_with_details(
                user_email=user.email,
                timeline_name=timeline_name,
                md5_hash=video_detail["video_hash"],
                ext=video_detail["ext"],
                upload_datetime=video_detail["upload_datetime"],
                high_res_user_file_path=video_detail["high_res_user_file_path"],
                high_res_user_file_hash=video_detail["high_res_user_file_hash"],
                high_res_local_file_path=video_detail["high_res_local_file_path"],
                title=video_detail["title"],
                volume_file_path=video_detail["volume_file_path"],
                overwrite=overwrite,
            )

        await bulk_writer.commit()

    call = background_processor.process_videos_generic_from_video_hashes.spawn(
        user.email, to_process, use_existing_output=use_existing_output
    )
    for video_detail in video_details:
        video_processing_call_ids[(user.email, video_detail["video_hash"])] = (
            call.object_id
        )

    return UploadVideo(
        result="success",
        processing_call_id=call.object_id,
        video_hashes=[video_detail["video_hash"] for video_detail in video_details],
        filenames=[
            Path(video_detail["volume_file_path"]).name
            for video_detail in video_details
        ],
        titles=[video_detail["title"] for video_detail in video_details],
    )
