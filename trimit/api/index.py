import os
import re
import asyncio
import json
from pydantic import BaseModel
from pathlib import Path

import aiofiles

from modal import asgi_app
from fastapi.responses import StreamingResponse
from starlette.middleware.sessions import SessionMiddleware

from trimit.utils import conf
from trimit.api.utils import load_or_create_workflow, workflows
from trimit.app import app, VOLUME_DIR
from .image import image
from trimit.backend.conf import LINEAR_WORKFLOW_OUTPUT_FOLDER
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

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
        volume_dir=VOLUME_DIR,
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
        volume_dir=VOLUME_DIR,
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
                if not is_last:
                    yield json.dumps(
                        {"message": partial_result, "is_last": False}
                    ) + "\n"
                elif isinstance(partial_result, BaseModel):
                    yield json.dumps(
                        {
                            "result": json.loads(partial_result.model_dump_json()),
                            "is_last": True,
                        }
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
    last_step_obj = await workflow.get_last_step(with_load_state=False)
    last_step_dict = last_step_obj.to_dict() if last_step_obj else None
    next_step_obj = await workflow.get_next_step(with_load_state=False)
    next_step_dict = next_step_obj.to_dict() if next_step_obj else None

    return_dict = {
        "last_step": last_step_dict,
        "next_step": next_step_dict,
        "all_steps": workflow.serializable_steps,
        "video_id": str(workflow.video.id),
        "user_id": str(workflow.user.id),
    }
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

    headers = {
        "Content-Disposition": f"attachment; filename={os.path.basename(most_recent_file)}"
    }
    return StreamingResponse(
        await aiofiles.open(most_recent_file, mode="rb"),
        media_type="application/xml",
        headers=headers,
    )


@web_app.get("/video")
async def stream_video(
    request: Request,
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

    video_path = workflow.most_recent_video_path
    if video_path is None:
        return {"error": "No video found"}
    extension = os.path.splitext(video_path)[1]
    media_type = f"video/{extension[1:]}"

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
