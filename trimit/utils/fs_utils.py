import aioboto3
import asyncio
import aiofiles
from aiofiles.threadpool.binary import AsyncBufferedReader
from fastapi import UploadFile
from trimit.utils.video_utils import convert_video_to_audio
from trimit.utils.model_utils import get_upload_folder, get_audio_folder
from trimit.app import S3_BUCKET, VOLUME_DIR
from pathlib import Path
import os
import zlib
import uuid
from moviepy.editor import VideoFileClip
from pytubefix import YouTube
from pytubefix.cli import on_progress


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


async def async_copy_to_s3(bucket, src, dst, ignore_existing=False):
    session = aioboto3.Session()
    async with session.client("s3") as s3:
        if not ignore_existing:
            if await exists_in_bucket(bucket, dst):
                return
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


async def save_file_to_volume_as_crc_hash(
    file: AsyncBufferedReader | UploadFile, filename: str, save_dir: Path | str
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix
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


def is_youtube_link(weblink: str):
    return "youtube.com" in weblink or "youtu.be" in weblink


def parse_res(stream):
    try:
        return int(stream.resolution.split("p")[0])
    except (TypeError, AttributeError):
        return 0


async def download_youtube_video(weblink: str, save_dir: Path | str):
    yt = YouTube(weblink, on_progress_callback=on_progress)
    streams = [
        s
        for s in yt.streams.filter(mime_type="video/mp4").order_by("resolution").desc()
    ]
    if any(s for s in streams if parse_res(s) >= 360):
        streams = [s for s in streams if parse_res(s) >= 360]

    async def audio_only_task():
        ys = yt.streams.get_audio_only()
        return ys.download(mp3=True, filename_prefix="audio-", output_path=save_dir)

    async def low_res_task():
        return streams[-1].download(filename_prefix="low-res-", output_path=save_dir)

    async def hi_res_task():
        return streams[0].download(filename_prefix="hi-res-", output_path=save_dir)

    return await asyncio.gather(low_res_task(), hi_res_task(), audio_only_task())


async def convert_mp3_to_wav(audio_file_path, output_dir):
    ext = Path(audio_file_path).suffix
    output_file_path = os.path.join(
        output_dir, os.path.basename(audio_file_path).replace(ext, ".wav")
    )
    # Command to convert MP3 to WAV using FFmpeg
    command = [
        "ffmpeg",  # Command (FFmpeg)
        "-y",
        "-i",
        audio_file_path,  # Input file (the MP3 file)
        "-acodec",
        "pcm_s16le",  # Audio codec for WAV (pcm_s16le for 16-bit PCM)
        "-ar",
        "44100",  # Audio sample rate (44100Hz is a common sample rate)
        "-ac",
        "2",  # Number of audio channels (2 for stereo)
        output_file_path,  # Output file (the WAV file)
    ]

    # Execute the command asynchronously
    # Run the command with asyncio subprocess
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    # Wait for the subprocess to finish
    _, stderr = await process.communicate()

    if process.returncode == 0:
        return output_file_path
    else:
        raise ValueError(f"Error during conversion: {stderr.decode()}")


async def save_weblink_to_volume_as_crc_hash(weblink: str, save_dir: Path | str):
    if is_youtube_link(weblink):
        downloaded_low_res_path, downloaded_hi_res_path, downloaded_audio_path_mp3 = (
            await download_youtube_video(weblink, save_dir)
        )
        downloaded_audio_path = await convert_mp3_to_wav(
            downloaded_audio_path_mp3, save_dir
        )
    else:
        raise NotImplementedError("Only youtube links are supported")
    title = Path(downloaded_low_res_path).stem.replace("low-res-", "")
    async with aiofiles.open(downloaded_low_res_path, mode="rb") as file:
        low_res_volume_path = await save_file_to_volume_as_crc_hash(
            file, os.path.basename(downloaded_low_res_path), save_dir
        )
    return low_res_volume_path, downloaded_hi_res_path, downloaded_audio_path, title


def convert_video_codec(video_path, codec="libx264"):
    clip = VideoFileClip(str(video_path))
    ext = Path(video_path).suffix
    temp_file_name = str(uuid.uuid4()) + ext
    temp_path = Path(video_path).parent / temp_file_name
    clip.write_videofile(str(temp_path), codec=codec)
    os.rename(str(temp_path), str(video_path))


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
        if os.path.exists(dst):
            print("DST FILE ALREADY EXISTS, returning", dst)
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        async with aiofiles.open(dst, "wb") as dst_file:
            while True:
                data = await src_file.read(64 * 1024)  # Read in chunks of 64KB
                if not data:
                    break
                await dst_file.write(data)
