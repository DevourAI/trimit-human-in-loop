import aioboto3
from trimit.utils.video_utils import convert_video_to_audio
from trimit.app import S3_BUCKET, VOLUME_DIR
from pathlib import Path
import os


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
