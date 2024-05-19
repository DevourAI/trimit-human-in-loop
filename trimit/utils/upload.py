import aiohttp
import asyncio
import os
from trimit.utils.video_utils import downsample_video, get_file_content_hash_with_cache
from tqdm.asyncio import tqdm
import diskcache as dc
import aiometer

CACHE_FILE = ".trimit_downsample_cache"
COMPRESSED_DIR = ".compressed_videos"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
URL = os.environ["MODAL_FRONTEND_BASE_URL"]


async def upload_videos(
    user_email,
    timeline_name,
    local_folder=None,
    local_paths=None,
    recursive=True,
    compressed_dir=None,
    cache_file=None,
    cache=None,
    force=False,
    reprocess=False,
    use_existing_output=True,
):
    cache = cache or dc.Cache(cache_file or CACHE_FILE)
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
    local_paths = sorted(list(set(local_paths)))

    high_res_hash_tasks = []
    video_paths = []

    for video_path in local_paths:
        if not any(
            video_path.lower().endswith(ext) for ext in [".mp4", ".mov", ".m4v"]
        ):
            continue
        high_res_hash_tasks.append(
            get_file_content_hash_with_cache(
                video_path, cache, prefix="high_res_user_file_hash/"
            )
        )
        video_paths.append(video_path)
    high_res_user_file_hashes = await tqdm.gather(
        *high_res_hash_tasks, desc="Getting high res user file hashes"
    )

    filename_locks = {}
    downsample_tasks = []
    video_details = []
    for high_res_user_file_hash, video_path in zip(
        high_res_user_file_hashes, video_paths
    ):
        video_filename = os.path.basename(video_path)
        filename_locks[video_filename] = filename_locks.get(
            video_filename, asyncio.Lock()
        )
        downsample_tasks.append(
            downsample_video(
                video_path,
                output_dir=compressed_dir or COMPRESSED_DIR,
                convert_filename_to_md5_hash=True,
                cache=cache,
                lock=filename_locks[video_filename],
            )
        )
        video_details.append(
            {
                "high_res_user_file_hash": high_res_user_file_hash,
                "high_res_user_file_path": video_path,
            }
        )

    if len(downsample_tasks) == 0:
        print("No videos to upload")
        return

    compressed_user_file_paths = await tqdm.gather(
        *downsample_tasks, desc="Downsampling videos"
    )

    if any(fp is None or not isinstance(fp, str) for fp in compressed_user_file_paths):
        raise ValueError(
            f"Failed to downsample some videos ({[hp for hp, lp in zip(compressed_user_file_paths, local_paths) if lp is None]}), exiting"
        )

    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        for video_detail, compressed_user_file_path in zip(
            video_details, compressed_user_file_paths
        ):
            data.add_field(
                "files",
                open(compressed_user_file_path, "rb"),
                filename=os.path.basename(compressed_user_file_path),
                content_type="application/octet-stream",
            )
            data.add_field(
                "high_res_user_file_hashes", video_detail["high_res_user_file_hash"]
            )
            data.add_field(
                "high_res_user_file_paths", video_detail["high_res_user_file_path"]
            )

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
                print(f"Successfully uploaded {len(video_details)} videos")
                resp_json = await response.json()
                print(f"Response: {resp_json}")
                resp_message = resp_json.get("result")
                if resp_message != "success":
                    raise ValueError(f"Unexpected response message: {resp_message}")
                return resp_json
