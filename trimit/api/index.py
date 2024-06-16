import os
import re
import asyncio
import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, EmailStr
from modal.functions import FunctionCall
from modal import asgi_app, is_local, Dict
from beanie import BulkWriter
from beanie.operators import In
from sqlalchemy import over
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import (
    FastAPI,
    Form,
    UploadFile,
    File,
    Request,
    Query,
    HTTPException,
    Depends,
)
from fastapi.responses import StreamingResponse, FileResponse

from trimit.utils import conf
from trimit.utils.async_utils import async_passthrough
from trimit.utils.fs_utils import (
    async_copy_to_s3,
    save_file_to_volume_as_crc_hash,
    get_volume_file_path,
    get_s3_key,
    get_audio_file_path,
)
from trimit.utils.model_utils import save_video_with_details, check_existing_video
from trimit.utils.video_utils import convert_video_to_audio
from trimit.api.utils import load_or_create_workflow
from trimit.app import app, get_volume_dir, S3_BUCKET
from trimit.models import (
    maybe_init_mongo,
    Video,
    VideoHighResPathProjection,
    User,
    VideoFileProjection,
)
from .image import image
from trimit.backend.conf import VIDEO_PROCESSING_CALL_IDS_DICT_NAME
from trimit.backend.background_processor import BackgroundProcessor
from trimit.backend.cut_transcript import CutTranscriptLinearWorkflow

background_processor = None
if not is_local():
    background_processor = BackgroundProcessor()

TEMP_DIR = Path("/tmp/uploads")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
video_processing_call_ids = Dict.from_name(
    VIDEO_PROCESSING_CALL_IDS_DICT_NAME, create_if_missing=True
)


class DynamicCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        origin = request.headers.get("origin")
        # Regex to match allowed origins, e.g., any subdomain of trimit.vercel.app
        local_origins = ["http://127.0.0.1:3000", "http://localhost:3000"]
        allow_local = (
            origin
            and origin in local_origins
            and os.environ["ENV"] in ["dev", "staging"]
        )
        allow_remote = origin and re.match(r"https?://.*-trimit\.vercel\.app", origin)
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


app_kwargs = dict(
    _allow_background_volume_commits=True,
    timeout=80000,
    image=image,
    container_idle_timeout=1200,
    _experimental_boost=True,
    _experimental_scheduler=True,
)


async def find_or_create_user(user_email: EmailStr):
    await maybe_init_mongo()
    user = await User.find_one(User.email == user_email)
    if user is None:
        user = User(email=user_email, name="")
        await user.save()
    return user


async def get_user_email(user_email: EmailStr = Form(...)):
    return user_email


async def form_user_dependency(user_email: EmailStr = Depends(get_user_email)):
    return await find_or_create_user(user_email)


async def get_current_workflow(
    timeline_name: str | None = None,
    length_seconds: int | None = None,
    user: User = Depends(find_or_create_user),
    video_hash: str | None = None,
    user_id: str | None = None,
    video_id: str | None = None,
    wait_until_done_running: bool = False,
    block_until: bool = False,
    timeout: float = 5,
    wait_interval: float = 0.1,
    force_restart: bool = False,
):
    if timeline_name is None or length_seconds is None:
        return None
    try:
        return await load_or_create_workflow(
            timeline_name=timeline_name,
            length_seconds=length_seconds,
            user_email=user.email,
            video_hash=video_hash,
            user_id=user_id,
            video_id=video_id,
            with_output=True,
            wait_until_done_running=wait_until_done_running,
            block_until=block_until,
            timeout=timeout,
            wait_interval=wait_interval,
            force_restart=force_restart,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.function(**app_kwargs)
@asgi_app()
def frontend_server():
    return web_app


@web_app.get("/get_step_outputs")
async def get_step_outputs(
    step_keys: str,
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
    latest_retry: bool = False,
):

    step_keys = step_keys.split(",")

    if not workflow:
        raise HTTPException(status_code=400, detail="Workflow not found")
    return {
        "outputs": await workflow.get_output_for_keys(
            step_keys, latest_retry=latest_retry
        )
    }


@web_app.get("/get_all_outputs")
async def get_all_outputs(
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
):
    if not workflow:
        raise HTTPException(status_code=400, detail="Workflow not found")

    return await workflow.get_all_outputs()


def check_call_status(modal_call_id, timeout: float = 0):
    try:
        fc = FunctionCall.from_id(modal_call_id)
        try:
            result = fc.get(timeout=timeout)
        except TimeoutError:
            result = {"status": "pending"}
    except Exception as e:
        if "not found" in str(e):
            result = {"status": "done"}
        else:
            result = {"status": "error", "error": str(e)}
    return result


# frontend should poll for this
# sometime in the future we can use kafka or pubsub to push to frontend
@web_app.get("/get_video_processing_status")
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
    expanded_statuses = []
    for video_hash, status in zip(video_hashes, statuses):
        if status is None:
            status = {"status": "done"}
        elif status["status"] == "done":
            if (user.email, video_hash) in video_processing_call_ids:
                try:
                    del video_processing_call_ids[(user.email, video_hash)]
                except KeyError:
                    # TODO not sure why this happens since we check the key on the previous line
                    video_processing_call_ids[(user.email, video_hash)] = None
        status["video_hash"] = video_hash
        expanded_statuses.append(status)
    return {"result": expanded_statuses}


# frontend should poll for this
# sometime in the future we can use kafka or pubsub to push to frontend
@web_app.get("/check_function_call_results")
async def check_function_call_results(modal_call_ids: list[str], timeout: float = 0):
    statuses = await asyncio.gather(
        *[
            async_passthrough(check_call_status(call_id, timeout))
            for call_id in modal_call_ids
        ]
    )
    return {"result": statuses}


@web_app.get("/step")
def step_endpoint(
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
    user_input: str | None = None,
    streaming: bool = False,
    ignore_running_workflows: bool = False,
    retry_step: bool = False,
):
    step_params = {
        "workflow": workflow,
        "user_input": user_input,
        "ignore_running_workflows": ignore_running_workflows,
        "async_export": os.environ.get("ASYNC_EXPORT", False),
        "retry_step": retry_step,
    }
    from trimit.backend.serve import step as step_function

    print(f"Starting step with params: {step_params}")
    if streaming:

        async def streamer():
            yield json.dumps({"message": "Running step...\n", "is_last": False}) + "\n"
            if is_local():
                method = step_function.local
            else:
                method = step_function.remote_gen.aio
            async for partial_result, is_last in method(**step_params):
                if isinstance(partial_result, BaseModel):
                    yield json.dumps(
                        {
                            "result": json.loads(partial_result.model_dump_json()),
                            "is_last": is_last,
                        }
                    ) + "\n"
                else:
                    yield json.dumps(
                        {"message": partial_result, "is_last": is_last}
                    ) + "\n"
                await asyncio.sleep(0)

        return StreamingResponse(streamer(), media_type="text/event-stream")

    else:
        step_function.spawn(**step_params)


@web_app.get("/reset_workflow")
async def reset_workflow(
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
):
    if workflow is None:
        raise HTTPException(status_code=400, detail="Workflow not found")
    print(f"Resetting workflow {workflow.id}")
    await workflow.restart_state()
    print(f"Workflow {workflow.id} reset")


@web_app.get("/revert_workflow_step")
async def revert_workflow_step(
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
    to_before_retries: bool = False,
):
    if workflow is None:
        raise HTTPException(status_code=400, detail="Workflow not found")
    print(f"Reverting workflow {workflow.id}")
    await workflow.revert_step(before_retries=to_before_retries)
    print(f"Workflow {workflow.id} reverted")


@web_app.get("/revert_workflow_step_to")
async def revert_workflow_step_to(
    step_name: str,
    substep_name: str,
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
):
    if workflow is None:
        raise HTTPException(status_code=400, detail="Workflow not found")
    print(f"Reverting workflow {workflow.id}")
    await workflow.revert_step_to_before(step_name, substep_name)
    print(f"Workflow {workflow.id} reverted to {step_name}.{substep_name}")


@web_app.get("/get_latest_state")
async def get_latest_state(
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
    with_output: bool = False,
):
    print("got workflow in get_latest_state")
    if workflow is None:
        raise HTTPException(status_code=400, detail="Workflow not found")

    last_step_obj = await workflow.get_last_substep_with_user_feedback(
        with_load_state=False
    )
    print("got last step obj")
    last_step_dict = last_step_obj.to_dict() if last_step_obj else None
    print("got last step dict")
    next_step_obj = await workflow.get_next_substep_with_user_feedback(
        with_load_state=False
    )
    print("got next step obj")
    next_step_dict = next_step_obj.to_dict() if next_step_obj else None
    print("got next step dict")

    return_dict = {
        "last_step": last_step_dict,
        "next_step": next_step_dict,
        "all_steps": workflow.serializable_steps,
        "video_id": str(workflow.video.id),
        "user_id": str(workflow.user.id),
        "user_messages": workflow.user_messages,
        "step_history_state": workflow.serializable_state_step_order,
    }
    print("got return dict")
    if with_output:
        return_dict["output"] = await workflow.get_last_output(with_load_state=False)
        print("got return dict output")
    return return_dict


@web_app.get("/download_transcript_text")
async def download_transcript_text(
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
        if substep_name is None:
            raise HTTPException(
                status_code=400,
                detail="Must provide both substep_name if step_name is provided",
            )

        export_result = await workflow.export_result_for_step_substep_name(
            step_name=step_name, substep_name=substep_name, with_load_state=False
        )
    file_path = export_result.get("transcript_text")

    if file_path is None:
        raise HTTPException(status_code=500, detail="No transcript found")
    assert isinstance(file_path, str)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=500, detail=f"Transcript not found at {file_path}"
        )

    return FileResponse(
        file_path, media_type="application/xml", filename=os.path.basename(file_path)
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
        if substep_name is None:
            raise HTTPException(
                status_code=400,
                detail="Must provide both substep_name if step_name is provided",
            )

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
        raise HTTPException(
            status_code=400, detail="Must provide timeline name and length_seconds"
        )
    if step_name is None:
        export_result = await workflow.most_recent_export_result(with_load_state=False)
    else:
        if substep_name is None:
            raise HTTPException(
                status_code=400,
                detail="Must provide both substep_name if step_name is provided",
            )

        export_result = await workflow.export_result_for_step_substep_name(
            step_name=step_name, substep_name=substep_name, with_load_state=False
        )

    timeline_path = export_result.get("video_timeline")

    if timeline_path is None:
        print("download_timeline: timeline_path is none")
        raise HTTPException(status_code=500, detail="No timeline found")
    assert isinstance(timeline_path, str)
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
    workflow: CutTranscriptLinearWorkflow | None = Depends(get_current_workflow),
    step_name: str | None = None,
    substep_name: str | None = None,
    stream: bool = False,
):
    if video_path is None and workflow is None:
        print("Must provide video path or timeline name and length_seconds")
        raise HTTPException(
            status_code=400,
            detail="Must provide video path or timeline name and length_seconds",
        )
    elif video_path is None:
        assert workflow is not None
        if step_name is None:
            export_result = await workflow.most_recent_export_result(
                with_load_state=False
            )
        else:
            if substep_name is None:
                raise HTTPException(
                    status_code=400,
                    detail="Must provide both substep_name if step_name is provided",
                )

            export_result = await workflow.export_result_for_step_substep_name(
                step_name=step_name, substep_name=substep_name, with_load_state=False
            )

        video_path = export_result.get("video")
    if video_path is None:
        raise HTTPException(status_code=400, detail="No video found")
    assert isinstance(video_path, str)
    if not os.path.exists(video_path):
        print(f"Video not found at {video_path}")
        raise HTTPException(status_code=400, detail=f"Video not found at {video_path}")
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


@web_app.get("/uploaded_videos")
async def uploaded_videos(user: User = Depends(find_or_create_user)):
    await maybe_init_mongo()
    return [
        {
            "filename": video.high_res_user_file_path,
            "video_hash": video.md5_hash,
            "path": video.path(get_volume_dir()),
        }
        for video in await Video.find(Video.user.email == user.email)
        .project(VideoFileProjection)
        .to_list()
    ]


@web_app.post("/upload")
async def upload_multiple_files(
    files: list[UploadFile] = File(...),
    user: User = Depends(form_user_dependency),
    high_res_user_file_paths: list[str] = Form(...),
    timeline_name: str = Form(...),
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
        "overwrite",
        overwrite,
        "use_existing_output",
        use_existing_output,
        "reprocess",
        reprocess,
    )
    await maybe_init_mongo()
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    print(f"Received upload request for {len(files)} files")
    print(f"high_res_user_file_paths: {high_res_user_file_paths}")

    upload_datetime = datetime.now()

    video_details = []

    # TODO allow same video for different users

    volume_dir = get_volume_dir()
    resp_msgs = []
    for file, high_res_user_file_path in zip(files, high_res_user_file_paths):
        volume_file_dir = Path(
            get_volume_file_path(user, upload_datetime, "temp", volume_dir=volume_dir)
        ).parent
        volume_file_path = await save_file_to_volume_as_crc_hash(file, volume_file_dir)
        print(f"Saved file to {volume_file_path}")

        filename = Path(volume_file_path).name
        video_hash = Path(volume_file_path).stem
        ext = Path(volume_file_path).suffix

        video = await check_existing_video(
            video_hash, high_res_user_file_path, ignore_existing=overwrite
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
                print(f"Saving file to {S3_BUCKET}/{s3_key}")
                await async_copy_to_s3(S3_BUCKET, str(volume_file_path), str(s3_key))
            audio_file_path = get_audio_file_path(
                user, upload_datetime, filename, volume_dir=volume_dir
            )
            convert_video_to_audio(str(volume_file_path), str(audio_file_path))

        video_details.append(
            {
                "volume_file_path": volume_file_path,
                "video_hash": video_hash,
                "ext": ext,
                "upload_datetime": upload_datetime,
                "high_res_user_file_path": high_res_user_file_path,
                "high_res_user_file_hash": "",
                "existing": existing,
            }
        )
    if len(video_details) == 0:
        return {"result": "success", "messages": resp_msgs}

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

    return {
        "result": "success",
        "processing_call_id": call.object_id,
        "video_hashes": [video_detail["video_hash"] for video_detail in video_details],
    }
