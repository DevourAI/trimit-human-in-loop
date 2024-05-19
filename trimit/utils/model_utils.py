import os
from datetime import datetime
from pathlib import Path


def filename_from_md5_hash(
    md5_hash: str,
    ext: str,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> str:
    if start_frame is not None and end_frame is not None:
        return f"{md5_hash}-{start_frame}-{end_frame}{ext}"
    elif start_frame is not None:
        return f"{md5_hash}-{start_frame}{ext}"
    elif end_frame is not None:
        return f"{md5_hash}-{end_frame}{ext}"
    else:
        return f"{md5_hash}{ext}"


def md5_hash_from_filename(filename: str) -> str:
    return Path(filename).stem


def md5_hash_ext_from_filename(filename: str) -> tuple[str, str]:
    return os.path.splitext(filename)


def format_date(dt: datetime) -> str:
    return dt.date().strftime("%Y-%m-%d")


def upload_date_from_filepath(filepath: str) -> datetime:
    date_str = filepath.split("/")[-2]
    return datetime.strptime(date_str, "%Y-%m-%d")


def get_upload_folder(volume_dir, user_email, upload_datetime):
    upload_date = format_date(upload_datetime)
    return Path(volume_dir) / "uploads" / user_email / upload_date


def get_generated_video_folder(volume_dir, user_email, timeline_name):
    return Path(volume_dir) / "generated_timeline_versions" / user_email / timeline_name


def get_transcript_cache_dir(volume_dir, user_email, timeline_name):
    return Path(volume_dir) / "transcript_cache" / user_email / timeline_name


def get_scene_folder(volume_dir, user_email, upload_datetime):
    upload_date = format_date(upload_datetime)
    return Path(volume_dir) / "scenes" / user_email / upload_date


def get_frame_folder(volume_dir, user_email, upload_datetime):
    upload_date = format_date(upload_datetime)
    return Path(volume_dir) / "frames" / user_email / upload_date


def get_audio_folder(volume_dir, user_email, upload_datetime):
    upload_date = format_date(upload_datetime)
    return Path(volume_dir) / "audio_extracts" / user_email / upload_date


async def save_video_with_details(
    user_email: str,
    timeline_name,
    md5_hash: str,
    ext: str,
    upload_datetime: datetime,
    high_res_user_file_path: str,
    high_res_user_file_hash: str,
    volume_file_path: str,
):
    from trimit.utils.video_utils import get_video_details
    from trimit.models.models import Video, VideoMetadata
    from fastapi import HTTPException

    try:
        details = await get_video_details(volume_file_path)
    except Exception as e:
        print(f"Error getting video details: {e}")
        details = VideoMetadata.from_default()

    try:
        return await Video.from_user_email_and_timeline_name(
            user_email=user_email,
            timeline_name=timeline_name,
            md5_hash=md5_hash,
            ext=ext,
            upload_datetime=upload_datetime,
            high_res_user_file_path=high_res_user_file_path,
            high_res_user_file_hash=high_res_user_file_hash,
            details=details,
        )
    except ValueError as e:
        print(f"Error saving video {md5_hash}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
