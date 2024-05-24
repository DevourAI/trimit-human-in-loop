import aiohttp
import os

URL = os.environ["MODAL_FRONTEND_BASE_URL"]


def is_video_path(path):
    return any(
        path.lower().endswith(ext)
        for ext in [".mp4", ".mov", ".m4v", ".avi", ".mkv", ".flv"]
    )


async def upload_videos(
    user_email,
    timeline_name,
    local_folder=None,
    local_paths=None,
    recursive=True,
    force=False,
    reprocess=False,
    use_existing_output=True,
):
    if local_paths is None:
        local_paths = []
    if local_folder is not None:
        if recursive:
            local_paths.extend(
                [
                    os.path.join(root, file)
                    for root, _, files in os.walk(os.path.abspath(local_folder))
                    for file in files
                ]
            )
        else:
            local_paths.extend(
                [
                    os.path.join(local_folder, video_path)
                    for video_path in os.listdir(os.path.abspath(local_folder))
                ]
            )
    local_paths = sorted([p for p in set(local_paths) if is_video_path(p)])

    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        for path in local_paths:
            data.add_field(
                "files",
                open(path, "rb"),
                filename=os.path.basename(path),
                content_type="application/octet-stream",
            )
            data.add_field("high_res_user_file_paths", path)

        data.add_field("force", str(force))
        data.add_field("user_email", user_email)
        data.add_field("timeline_name", timeline_name)
        data.add_field("reprocess", str(reprocess))
        data.add_field("use_existing_output", str(use_existing_output))

        async with session.post(URL + "/upload", data=data) as response:
            error_text = await response.text()
            try:
                response.raise_for_status()
            except:
                print(f"Failed to upload videos, error text: {error_text}")
                raise

            if response.status == 200:
                print(f"Successfully uploaded {len(local_paths)} videos")
                resp_json = await response.json()
                print(f"Response: {resp_json}")
                resp_message = resp_json.get("result")
                if resp_message != "success":
                    raise ValueError(f"Unexpected response message: {resp_message}")
                return resp_json
