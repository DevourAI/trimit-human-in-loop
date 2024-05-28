import subprocess
import io
import re
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path
import base64
from trimit.utils.video_utils import get_duration


def encode_image(image: io.BytesIO):
    return base64.b64encode(image.read()).decode("utf-8")


async def extract_frames(
    video_path,
    output_folder,
    max_frame_rate=30,
    output_to_buffer=False,
    output_filename="frames.png",
    total_duration=None,
    use_existing_output=True,
):
    # TODO if frames existing output_folder, use those
    # Step 1: Get the duration of the video in seconds
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    if total_duration is None:
        try:
            total_duration = await get_duration(video_path)
        except ValueError as e:
            print(f"Could not extract frames from the video: {e}")
            return None

    # Step 2: Calculate the interval for 10 frames
    # We subtract 1 from the total number of frames to ensure we get 10 frames starting from 0
    interval = total_duration / 9
    if 1 / interval > max_frame_rate:
        interval = 1 / max_frame_rate
    if interval == 0:
        print("Could not calculate the interval for frame extraction")
        return None

    # Step 3: Extract frames at regular intervals
    output_pattern = f"{output_folder}/frame_%d.jpg"
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"fps=1/{interval}",
        "-vsync",
        "vfr",
        output_pattern,
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        raise ValueError(f"Could not extract frames from the video: {e}")
    if output_to_buffer:
        output_buffer = io.BytesIO()
    else:
        output_buffer = os.path.join(output_folder, output_filename)
    plot_frames(output_folder, output_buffer)

    if output_to_buffer:
        output_buffer.seek(0)
    return output_buffer


def plot_frames(frames_folder, output_buffer: str | io.BytesIO):
    image_files = [
        os.path.abspath(os.path.join(frames_folder, fname))
        for fname in os.listdir(frames_folder)
    ][:10]
    images = [Image.open(img_file) for img_file in image_files]
    _, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
    axes = axes.flatten()
    for ax, img, title in zip(axes, images, range(1, 11)):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Frame {title}")
    plt.tight_layout()
    plt.savefig(output_buffer, format="jpg")
