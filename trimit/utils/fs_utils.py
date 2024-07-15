import aioboto3
import aiofiles
from fastapi import UploadFile
from trimit.utils.video_utils import convert_video_to_audio
from trimit.utils.model_utils import get_upload_folder, get_audio_folder
from trimit.app import S3_BUCKET, VOLUME_DIR
from pathlib import Path
import os
import zlib
import uuid


async def s3_file_exists(bucket, key):
    session = aioboto3.Session()
    async with session.client("s3") as s3:
        try:
            await s3.head_object(Bucket=bucket, Key=key)
            return True
        except s3.exceptions.ClientError:
            return False


async def async_copy_from_s3(bucket, src, dst):
    session = aioboto3.Session()
    async with session.client("s3") as s3:
        print(f"Downloading {src} to {dst}")
        await s3.download_file(bucket, src, dst)


async def async_copy_to_s3(bucket, src, dst):
    session = aioboto3.Session()
    async with session.client("s3") as s3:
        await s3.upload_file(src, bucket, dst)


async def exists_in_bucket(bucket, s3_path):
    session = aioboto3.Session()
    async with session.client("s3") as s3:
        try:
            await s3.head_object(Bucket=bucket, Key=s3_path)
            return True
        except s3.exceptions.ClientError:
            return False


async def ensure_video_on_volume(
    volume_video_path: str, volume_dir: str = VOLUME_DIR, s3_bucket: str = S3_BUCKET
):
    if not os.path.exists(volume_video_path):
        Path(volume_video_path).parent.mkdir(parents=True, exist_ok=True)
        s3_path = volume_video_path.split(volume_dir, 1)[1]
        if s3_path.startswith("/"):
            s3_path = s3_path[1:]
        try:
            await async_copy_from_s3(s3_bucket, s3_path, volume_video_path)
        except:
            print(f"Failed to copy video to volume: {volume_video_path}")
            raise
        print(f"copied video to volume: {volume_video_path}")
    else:
        print(f"video already exists: {volume_video_path}")
    if os.stat(volume_video_path).st_size == 0:
        raise ValueError(f"Video is empty: {volume_video_path}")


async def ensure_audio_path_on_volume(
    video: "Video", volume_dir: str = VOLUME_DIR, s3_bucket: str = S3_BUCKET
):
    video_path = video.path(volume_dir)
    await ensure_video_on_volume(video_path, volume_dir=volume_dir, s3_bucket=s3_bucket)
    audio_path = video.audio_path(volume_dir)
    Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(audio_path):
        convert_video_to_audio(video_path, audio_path)
    else:
        print(f"audio_path already exists: {audio_path}")
    if os.stat(audio_path).st_size == 0:
        raise ValueError(f"Video is empty: {audio_path}")


async def save_file_to_volume(file: UploadFile, volume_file_path: Path | str):
    if os.path.exists(volume_file_path):
        os.remove(volume_file_path)
    else:
        Path(volume_file_path).parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(volume_file_path, "wb") as volume_buffer:
        while content := await file.read(1024):
            await volume_buffer.write(content)


async def save_file_to_volume_as_crc_hash(file: UploadFile, save_dir: Path | str):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix
    temp_file_name = str(uuid.uuid4())
    temp_path = Path(save_dir) / temp_file_name
    crc_value = 0
    async with aiofiles.open(temp_path, "wb") as buffer:
        while content := await file.read(1024):
            await buffer.write(content)
            crc_value = zlib.crc32(content, crc_value)
    final_hash = str(crc_value & 0xFFFFFFFF)
    final_path = Path(save_dir) / f"{final_hash}{ext}"
    os.rename(temp_path, final_path)
    return str(final_path)


def get_volume_file_path(
    current_user, upload_datetime, filename, volume_dir=VOLUME_DIR
):
    volume_upload_folder = get_upload_folder(
        volume_dir, current_user.email, upload_datetime
    )
    return volume_upload_folder / filename


def get_s3_mount_file_path(current_user, upload_datetime, filename):
    from trimit.app import S3_VIDEO_PATH

    s3_upload_mount_folder = get_upload_folder(
        S3_VIDEO_PATH, current_user.email, upload_datetime
    )
    return s3_upload_mount_folder / filename


def get_s3_key(current_user, upload_datetime, filename):
    s3_key_prefix = get_upload_folder("", current_user.email, upload_datetime)
    return s3_key_prefix / filename


def get_audio_file_path(current_user, upload_datetime, filename, volume_dir=VOLUME_DIR):
    audio_folder = get_audio_folder(volume_dir, current_user.email, upload_datetime)
    return audio_folder / filename


async def async_copy(src, dst):
    async with aiofiles.open(src, "rb") as src_file:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        async with aiofiles.open(dst, "wb") as dst_file:
            while True:
                data = await src_file.read(64 * 1024)  # Read in chunks of 64KB
                if not data:
                    break
                await dst_file.write(data)
