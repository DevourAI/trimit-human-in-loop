import os
import re
import asyncio
import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel
from modal import asgi_app, is_local
from beanie import BulkWriter
from beanie.operators import In
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import FastAPI, Form, UploadFile, File, Request, Query, HTTPException
from fastapi.responses import StreamingResponse, FileResponse

from trimit.utils import conf
from trimit.utils.fs_utils import (
    async_copy_to_s3,
    save_file_to_volume_as_crc_hash,
    get_volume_file_path,
    get_s3_key,
    get_audio_file_path,
)
from trimit.utils.model_utils import save_video_with_details, check_existing_video
from trimit.utils.video_utils import convert_video_to_audio
from trimit.api.utils import load_or_create_workflow, workflows
from trimit.app import app, get_volume_dir, S3_BUCKET, S3_VIDEO_PATH
from trimit.models import (
    maybe_init_mongo,
    Video,
    VideoHighResPathProjection,
    User,
    VideoFileProjection,
)
from .image import image
from trimit.backend.conf import LINEAR_WORKFLOW_OUTPUT_FOLDER
from trimit.backend.background_processor import BackgroundProcessor

background_processor = None
if not is_local():
    background_processor = BackgroundProcessor()

TEMP_DIR = Path("/tmp/uploads")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class DynamicCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        origin = request.headers.get("origin")
        # Regex to match allowed origins, e.g., any subdomain of trimit.vercel.app
        local_origins = ["http://127.0.0.1:3000", "http://localhost:3000"]
        allow_local = origin and origin in local_origins and os.environ["ENV"] == "dev"
        allow_remote = origin and re.match(r"https?://.*-trimit\.vercel\.app", origin)
        origin = origin or ""
        if allow_local or allow_remote:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, PUT, DELETE, OPTIONS"
            )
            response.headers["Access-Control-Allow-Headers"] = (
                "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range"
            )
            response.headers["Access-Control-Expose-Headers"] = (
                "Content-Length,Content-Range"
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


@app.function(**app_kwargs)
@asgi_app()
def frontend_server():
    return web_app


@web_app.get("/get_step_outputs")
async def get_step_outputs(
    user_email: str,
    timeline_name: str,
    video_hash: str,
    length_seconds: int,
    step_keys: str,
    latest_retry: bool = False,
):
    from trimit.backend.cut_transcript import CutTranscriptLinearWorkflow

    step_keys = step_keys.split(",")

    workflow = await CutTranscriptLinearWorkflow.from_video_hash(
        video_hash=video_hash,
        timeline_name=timeline_name,
        user_email=user_email,
        length_seconds=length_seconds,
        output_folder=LINEAR_WORKFLOW_OUTPUT_FOLDER,
        volume_dir=get_volume_dir(),
        new_state=False,
    )

    workflow = workflows.get(workflow.id, None)
    if not workflow:
        return {"error": "Workflow not found"}
    return {
        "outputs": await workflow.get_output_for_keys(
            step_keys, latest_retry=latest_retry
        )
    }


@web_app.get("/get_all_outputs")
async def get_all_outputs(
    user_email: str, timeline_name: str, video_hash: str, length_seconds: int
):
    from trimit.backend.cut_transcript import CutTranscriptLinearWorkflow

    workflow = await CutTranscriptLinearWorkflow.from_video_hash(
        video_hash=video_hash,
        timeline_name=timeline_name,
        user_email=user_email,
        length_seconds=length_seconds,
        output_folder=LINEAR_WORKFLOW_OUTPUT_FOLDER,
        volume_dir=get_volume_dir(),
        new_state=False,
    )

    workflow = workflows.get(workflow.id, None)
    if not workflow:
        return {"error": "Workflow not found"}
    return await workflow.get_all_outputs()


@web_app.get("/step")
def step_endpoint(
    user_email: str,
    timeline_name: str,
    video_hash: str,
    length_seconds: int,
    user_input: str | None = None,
    streaming: bool = False,
    force_restart: bool = False,
    ignore_running_workflows: bool = False,
):
    step_params = {
        "user_email": user_email,
        "timeline_name": timeline_name,
        "video_hash": video_hash,
        "length_seconds": length_seconds,
        "user_input": user_input,
        "force_restart": force_restart,
        "ignore_running_workflows": ignore_running_workflows,
    }
    from trimit.backend.serve import step as step_function

    print(f"Starting step with params: {step_params}")
    if streaming:

        async def streamer():
            yield json.dumps({"message": "Running step...\n", "is_last": False}) + "\n"
            async for partial_result, is_last in step_function.remote_gen.aio(
                **step_params
            ):
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
    timeline_name: str,
    length_seconds: int,
    user_email: str | None = None,
    video_hash: str | None = None,
    user_id: str | None = None,
    video_id: str | None = None,
):
    try:
        workflow = await load_or_create_workflow(
            timeline_name=timeline_name,
            length_seconds=length_seconds,
            user_email=user_email,
            video_hash=video_hash,
            user_id=user_id,
            video_id=video_id,
            with_output=False,
            wait_until_done_running=False,
        )
    except Exception as e:
        return {"error": str(e)}
    print(f"Resetting workflow {workflow.id}")
    await workflow.restart_state()
    print(f"Workflow {workflow.id} reset")


@web_app.get("/revert_workflow_step")
async def revert_workflow_step(
    timeline_name: str,
    length_seconds: int,
    user_email: str | None = None,
    video_hash: str | None = None,
    user_id: str | None = None,
    video_id: str | None = None,
    to_before_retries: bool = False,
):
    try:
        workflow = await load_or_create_workflow(
            timeline_name=timeline_name,
            length_seconds=length_seconds,
            user_email=user_email,
            video_hash=video_hash,
            user_id=user_id,
            video_id=video_id,
            with_output=False,
            wait_until_done_running=False,
        )
    except Exception as e:
        return {"error": str(e)}
    print(f"Reverting workflow {workflow.id}")
    await workflow.revert_step(before_retries=to_before_retries)
    print(f"Workflow {workflow.id} reverted")


@web_app.get("/get_latest_state")
async def get_latest_state(
    timeline_name: str,
    length_seconds: int,
    user_email: str | None = None,
    video_hash: str | None = None,
    user_id: str | None = None,
    video_id: str | None = None,
    with_output: bool = False,
    wait_until_done_running: bool = False,
    block_until: bool = False,
    timeout: float = 5,
    wait_interval: float = 0.1,
):
    try:
        workflow = await load_or_create_workflow(
            timeline_name=timeline_name,
            length_seconds=length_seconds,
            user_email=user_email,
            video_hash=video_hash,
            user_id=user_id,
            video_id=video_id,
            with_output=with_output,
            wait_until_done_running=wait_until_done_running,
            block_until=block_until,
            timeout=timeout,
            wait_interval=wait_interval,
        )
    except Exception as e:
        return {"error": str(e)}
    last_step_obj = await workflow.get_last_substep(with_load_state=False)
    last_step_dict = last_step_obj.to_dict() if last_step_obj else None
    next_step_obj = await workflow.get_next_substep(with_load_state=False)
    next_step_dict = next_step_obj.to_dict() if next_step_obj else None

    return_dict = {
        "last_step": last_step_dict,
        "next_step": next_step_dict,
        "all_steps": workflow.serializable_steps,
        "video_id": str(workflow.video.id),
        "user_id": str(workflow.user.id),
    }
    print("last step", last_step_dict.get("name") if last_step_dict else None)
    print("next step", next_step_dict.get("name") if next_step_dict else None)
    if with_output:
        return_dict["output"] = await workflow.get_last_output(with_load_state=False)
    return return_dict


@web_app.get("/download_timeline")
async def download_timeline(
    timeline_name: str,
    length_seconds: int,
    user_email: str | None = None,
    video_hash: str | None = None,
    user_id: str | None = None,
    video_id: str | None = None,
    wait_until_done_running: bool = False,
    block_until: bool = False,
    timeout: float = 5,
    wait_interval: float = 0.1,
):
    try:
        workflow = await load_or_create_workflow(
            timeline_name=timeline_name,
            length_seconds=length_seconds,
            user_email=user_email,
            video_hash=video_hash,
            user_id=user_id,
            video_id=video_id,
            with_output=True,
            wait_until_done_running=wait_until_done_running,
            block_until=block_until,
            timeout=timeout,
            wait_interval=wait_interval,
        )
    except Exception as e:
        return {"error": str(e)}

    most_recent_file = workflow.most_recent_timeline_path
    if most_recent_file is None:
        return {"error": "No timeline found"}
    if not os.path.exists(most_recent_file):
        return {"error": f"Timeline not found at {most_recent_file }"}

    return FileResponse(
        most_recent_file,
        media_type="application/xml",
        filename=os.path.basename(most_recent_file),
    )


@web_app.get("/video")
async def stream_video(
    request: Request,
    video_path: str | None = None,
    timeline_name: str | None = None,
    length_seconds: int | None = None,
    user_email: str | None = None,
    video_hash: str | None = None,
    user_id: str | None = None,
    video_id: str | None = None,
    stream: bool = False,
    wait_until_done_running: bool = False,
    block_until: bool = False,
    timeout: float = 5,
    wait_interval: float = 0.1,
):
    if video_path is None:
        if timeline_name is None or length_seconds is None:
            return {
                "error": "Must provide video path or timeline name and length_seconds"
            }
        try:
            workflow = await load_or_create_workflow(
                timeline_name=timeline_name,
                length_seconds=length_seconds,
                user_email=user_email,
                video_hash=video_hash,
                user_id=user_id,
                video_id=video_id,
                with_output=True,
                wait_until_done_running=wait_until_done_running,
                block_until=block_until,
                timeout=timeout,
                wait_interval=wait_interval,
            )
        except Exception as e:
            return {"error": str(e)}

        video_path = workflow.most_recent_video_path
    if video_path is None:
        return {"error": "No video found"}
    if not os.path.exists(video_path):
        return {"error": f"Video not found at {video_path}"}
    extension = os.path.splitext(video_path)[1]
    media_type = f"video/{extension[1:]}"
    if not stream:
        return FileResponse(
            video_path, media_type=media_type, filename=os.path.basename(video_path)
        )

    def iterfile():
        print("iterfile")
        with open(video_path, mode="rb") as file_like:  # open the file in binary mode
            print("opened")
            yield from file_like  # yield the binary data

    range_header = request.headers.get("range", None)
    file_size = os.path.getsize(video_path)
    print(f"range header: {range_header}")
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
            print(f"seeking to {start}")
            video.seek(start)
            while chunk := video.read(8192):
                yield chunk

    return StreamingResponse(video_stream(), status_code=206, headers=headers)


@web_app.get("/uploaded_high_res_video_paths")
async def uploaded_high_res_video_paths(
    user_email: str = Query(...), md5_hashes: list[str] = Query(None)
):
    await maybe_init_mongo()
    video_filters = [Video.user.email == user_email]
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
    user_email: str = Query(...), high_res_user_file_paths: list[str] = Query(None)
):
    await maybe_init_mongo()
    video_filters = [Video.user.email == user_email]
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
async def uploaded_videos(user_email: str = Query(...)):
    await maybe_init_mongo()
    return [
        {
            "filename": video.high_res_user_file_path,
            "video_hash": video.md5_hash,
            "path": video.path(get_volume_dir()),
        }
        for video in await Video.find(Video.user.email == user_email)
        .project(VideoFileProjection)
        .to_list()
    ]


@web_app.post("/upload")
async def upload_multiple_files(
    files: list[UploadFile] = File(...),
    user_email: str = Form(...),
    high_res_user_file_paths: list[str] = Form(...),
    timeline_name: str = Form(...),
    force: bool = Form(False),
    use_existing_output: bool = Form(True),
    reprocess: bool = Form(False),
):
    assert background_processor is not None

    await maybe_init_mongo()
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    print(f"Received upload request for {len(files)} files")
    print(f"high_res_user_file_paths: {high_res_user_file_paths}")

    current_user = await get_current_user_or_raise(user_email)
    upload_datetime = datetime.now()

    video_details = []

    # TODO allow same video for different users

    volume_dir = get_volume_dir()
    resp_msgs = []
    for file, high_res_user_file_path in zip(files, high_res_user_file_paths):
        volume_file_dir = Path(
            get_volume_file_path(
                current_user, upload_datetime, "temp", volume_dir=volume_dir
            )
        ).parent
        volume_file_path = await save_file_to_volume_as_crc_hash(file, volume_file_dir)
        print(f"Saved file to {volume_file_path}")

        filename = Path(volume_file_path).name
        video_hash = Path(volume_file_path).stem
        ext = Path(volume_file_path).suffix

        video = await check_existing_video(video_hash, high_res_user_file_path, force)
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
            s3_key = get_s3_key(
                current_user, upload_datetime, Path(volume_file_path).name
            )
            print(f"Saving file to {S3_BUCKET}/{s3_key}")
            await async_copy_to_s3(S3_BUCKET, str(volume_file_path), str(s3_key))
            audio_file_path = get_audio_file_path(
                current_user, upload_datetime, filename, volume_dir=volume_dir
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
            if video_detail["existing"]:
                if reprocess:
                    to_process.append(video_detail["video_hash"])
                    continue
            to_process.append(video_detail["video_hash"])
            await save_video_with_details(
                user_email=current_user.email,
                timeline_name=timeline_name,
                md5_hash=video_detail["video_hash"],
                ext=video_detail["ext"],
                upload_datetime=video_detail["upload_datetime"],
                high_res_user_file_path=video_detail["high_res_user_file_path"],
                high_res_user_file_hash=video_detail["high_res_user_file_hash"],
                volume_file_path=video_detail["volume_file_path"],
            )

        await bulk_writer.commit()

    print("USE_EXISTING_OUTPUT", use_existing_output)
    call = background_processor.process_videos_generic_from_video_hashes.spawn(
        current_user.email, to_process, use_existing_output=use_existing_output
    )

    return {
        "result": "success",
        "processing_call_id": call.object_id,
        "video_hashes": [video_detail["video_hash"] for video_detail in video_details],
    }


async def get_current_user_or_raise(user_email: str) -> User:
    current_user = await User.find_one(User.email == user_email)
    if current_user is None:
        raise HTTPException(status_code=400, detail="User not found")
    return current_user
