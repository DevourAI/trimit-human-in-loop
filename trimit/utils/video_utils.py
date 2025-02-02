from trimit.utils.model_utils import filename_from_hash
import re
from pathlib import Path, PosixPath
from dateutil import parser
from typing import IO
import asyncio
import os
import hashlib
import json
import datetime
from fractions import Fraction
import subprocess
import zlib
import aiofiles
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RUST_BINARIES_PATH = os.path.join(
    PROJECT_ROOT, "rust_src/target/release/trimit_rust_binaries"
)


def parse_timecode(timecode: str, prefix: str | None = None):
    if prefix is None:
        prefix = ""
    match = re.search(prefix + r"\d{2}):(\d{2}):(\d{2})\.(\d{2})", timecode)
    if not match:
        return None
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 100


def load_downsample_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return {}


def save_downsample_cache(new_cache, cache_file):
    with open(cache_file, "w") as f:
        json.dump(new_cache, f)


def is_video_file(file_path: str | PosixPath):
    return str(file_path).lower().endswith((".mp4", ".avi", ".mov", ".mkv"))


def video_content_md5_hash_from_path(file_path):
    with open(file_path, "rb") as f:
        md5_hash = video_content_md5_hash(f)
    return md5_hash


def video_content_md5_hash(video: IO[bytes]):
    data = video.read()
    md5_hash = hashlib.md5(data).hexdigest()
    video.seek(0)
    return md5_hash


async def video_content_crc_hash_from_path(file_path):
    crc_value = 0
    async with aiofiles.open(file_path, "rb") as file:
        file = await file.read()
        crc_value = zlib.crc32(file, crc_value)
    return str(crc_value & 0xFFFFFFFF)


async def get_file_content_hash_with_cache(file_path, cache, prefix):
    crc_hash = cache.get(prefix + file_path)
    if crc_hash:
        return crc_hash
    crc_hash = await video_content_crc_hash_from_path(file_path)
    cache[prefix + file_path] = crc_hash
    return crc_hash


async def downsample_video(
    file_path,
    output_dir,
    convert_filename_to_md5_hash=False,
    max_res=640,
    force=False,
    cache_file=None,
    cache=None,
    relative_cache_path=None,
    lock=None,
):
    import diskcache as dc

    if not is_video_file(file_path):
        print(f"Skipping non-video file: {file_path}")
        return

    cache_prefix = "downsample_cache/"
    cache = cache or dc.Cache(cache_file or "cache")
    rel_file_path = ""
    if relative_cache_path:
        rel_file_path = os.path.relpath(file_path, relative_cache_path)
    else:
        rel_file_path = file_path
    downsampled_file_path = cache.get(cache_prefix + rel_file_path)
    if (
        downsampled_file_path
        and isinstance(downsampled_file_path, str)
        and os.path.exists(downsampled_file_path)
        and not force
    ):
        print(f"Skipping existing file: {rel_file_path}")
        return downsampled_file_path

    filename, ext = os.path.splitext(os.path.basename(file_path))
    ext = ext.lower()
    filename = filename + ext

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    downsampled_file_path = os.path.join(output_dir, filename)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        file_path,
        "-vf",
        f"scale=min({max_res}\\,iw):-2",
        "-c:a",
        "copy",
        downsampled_file_path,
    ]

    async def run_command():
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return stdout, stderr, process.returncode

    if lock:
        async with lock:
            stdout, stderr, exit_code = await run_command()
    else:
        stdout, stderr, exit_code = await run_command()

    if exit_code != 0:
        # Handle errors
        print(f"Error downsampling video {file_path}: {stderr.decode()}")
        return None
    else:
        print(f"Video {file_path} downsampled successfully")

    if convert_filename_to_md5_hash:
        high_res_md5_hash = await get_file_content_hash_with_cache(
            downsampled_file_path, cache, prefix="downsampled_crc_hash/"
        )
        filename = filename_from_hash(high_res_md5_hash, ext)
        hashed_output_file_path = os.path.join(output_dir, filename)
        os.rename(downsampled_file_path, hashed_output_file_path)
        downsampled_file_path = hashed_output_file_path

    if cache:
        if relative_cache_path:
            rel_downsampled_file_path = os.path.relpath(
                downsampled_file_path, relative_cache_path
            )
        else:
            rel_downsampled_file_path = downsampled_file_path
        cache[cache_prefix + rel_file_path] = rel_downsampled_file_path
    return downsampled_file_path


def get_creation_date_unix(file_path):
    file_stats = os.stat(file_path)
    if hasattr(file_stats, "st_birthtime"):  # macOS and some Unix systems
        creation_time = datetime.datetime.fromtimestamp(file_stats.st_birthtime)
        print(f"Creation time of the file: {creation_time}")
        return creation_time
    else:
        print("Creation time not available. Falling back to modification time.")
        return get_st_ctime(file_path)


def get_file_creation_date(file_path):
    if os.name == "nt":
        return get_creation_date_windows(file_path)
    else:
        return get_creation_date_unix(file_path)


def get_creation_date_windows(file_path):
    return get_st_ctime(file_path)


def get_st_ctime(file_path):
    file_stats = os.stat(file_path)
    return datetime.datetime.fromtimestamp(file_stats.st_ctime)


def is_created_date_field(field):
    if "create" in field.lower() and "date" in field.lower():
        return True
    if "date" in field.lower() and "original" in field.lower():
        return True
    return False


async def get_exiftool_details(video_file_path):
    process = await asyncio.create_subprocess_exec(
        "exiftool",
        "-json",
        video_file_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if stderr:
        print("Error getting video frame count: " + stderr.decode())
    try:
        metadata = json.loads(stdout.decode("utf-8"))
    except json.JSONDecodeError as e:
        print(f"Failed to decode metadata: {e}")
        return {}
    else:
        return metadata[0]


async def get_duration(video_path):
    return (await get_video_info(video_path))[2]


async def get_video_info(video_path):
    from trimit.models import PydanticFraction

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        video_path,
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"ffprobe returned an error: {stderr.decode().strip()}")

    info = json.loads(stdout)
    stream_info = info["streams"][0]

    avg_frame_rate_raw = stream_info["avg_frame_rate"]
    try:
        avg_frame_rate = PydanticFraction.from_fraction(Fraction(avg_frame_rate_raw))
    except:
        print("Error getting video frame rate: " + stdout.decode())
        try:
            avg_frame_rate = eval(avg_frame_rate_raw)
        except:
            avg_frame_rate = None

    frame_count = try_cast_int(stream_info["nb_frames"])
    duration = try_cast_float(stream_info["duration"])

    return frame_count, avg_frame_rate, duration


def try_cast_int(val):
    if isinstance(val, int):
        return val
    removed = None
    try:
        removed = re.sub(r"[^0-9\-.]", "", val)
        return int(round(float(removed)))
    except Exception as e:
        print(f"Failed to cast {val} to int, re result: {removed}. Error: {e}")
        return None


def try_cast_float(val):
    removed = None
    try:
        removed = re.sub(r"[^0-9\-.]", "", val)
        return float(removed)
    except Exception as e:
        print(f"Failed to cast {val} to float, re result: {removed}. Error: {e}")
        return None


async def get_video_details(video_path):
    from trimit.models import VideoMetadata, PydanticFraction

    exiftool_details = await get_exiftool_details(video_path)
    frame_count, frame_rate_fraction, duration = await get_video_info(video_path)
    frame_rate = None
    if frame_rate_fraction and isinstance(frame_rate_fraction, PydanticFraction):
        frame_rate = frame_rate_fraction.numerator / frame_rate_fraction.denominator
    elif frame_rate_fraction:
        frame_rate = frame_rate_fraction
        frame_rate_fraction = None

    file_creation_date = get_file_creation_date(video_path)

    codec = exiftool_details.get("CompressorName", "")
    available_ffmpeg_codecs = [
        "libx264",
        "libx265" "libvpx-vp9",
        "libaom-av1",
        "libsvtav1",
        "mpeg4",
        "mpeg2video",
        "theora",
    ]
    for available_codec in available_ffmpeg_codecs:
        if available_codec in codec.lower():
            codec = available_codec
            break
    return VideoMetadata(
        frame_count=frame_count,
        frame_rate_fraction=frame_rate_fraction,
        frame_rate=frame_rate,
        mime_type=exiftool_details.get("MIMEType"),
        major_brand=exiftool_details.get("MajorBrand"),
        create_date=parser.parse(exiftool_details.get("CreateDate")),
        modify_date=parser.parse(exiftool_details.get("ModifyDate")),
        file_creation_date=file_creation_date,
        duration=duration,
        width=try_cast_int(exiftool_details.get("SourceImageWidth")),
        height=try_cast_int(exiftool_details.get("SourceImageHeight")),
        resolution_x=try_cast_int(exiftool_details.get("XResolution")),
        resolution_y=try_cast_int(exiftool_details.get("YResolution")),
        codec=codec,
        bit_depth=try_cast_int(exiftool_details.get("BitDepth")),
        audio_format=exiftool_details.get("AudioFormat"),
        audio_channels=try_cast_int(exiftool_details.get("AudioChannels")),
        audio_bits_per_sample=try_cast_int(exiftool_details.get("AudioBitsPerSample")),
        audio_sample_rate=try_cast_int(exiftool_details.get("AudioSampleRate")),
    )


def convert_video_to_audio(video_file_path, output_file_path):
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_file_path,
        "-vn",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-b:a",
        "192k",
        "-f",
        "wav",
        output_file_path,
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.communicate()
    if not os.path.exists(output_file_path):
        raise ValueError("Audio file could not be created.")
    if process.returncode != 0:
        raise ValueError("Error during audio extraction.")
    if os.stat(output_file_path).st_size == 0:
        raise ValueError("Empty audio file created.")
