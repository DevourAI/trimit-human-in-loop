from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from trimit.utils.video_utils import get_video_info


async def extract_scenes_to_disk(
    video_path: str,
    scenes: list["Scene"],
    output_dir: str,
    frame_rate: float | None = None,
    duration: float | None = None,
    codec: str | None = None,
    max_workers: int = 12,
) -> list[str]:
    print(f"Extracting {len(scenes)} scenes to {output_dir} from {video_path}")
    from moviepy.editor import VideoFileClip
    from trimit.models import PydanticFraction

    if frame_rate is None or duration is None:
        _, frame_rate_fraction, duration = await get_video_info(video_path)
        if isinstance(frame_rate_fraction, PydanticFraction):
            frame_rate = frame_rate_fraction.numerator / frame_rate_fraction.denominator
        elif isinstance(frame_rate_fraction, float):
            frame_rate = frame_rate_fraction
        if frame_rate is None:
            raise ValueError("Could not get frame rate")

    if codec is None or not codec.strip():
        print("codec not provided. Defaulting to 'libx264'.")
        codec = "libx264"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # TODO: if start/end == 0/total_frames, then just shutil.copyfile
    def writer_worker(scene, video_path, frame_rate, output_dir, codec):
        with VideoFileClip(str(video_path)) as video:
            start_frame = scene.start_frame
            end_frame = scene.end_frame
            if end_frame < start_frame:
                raise ValueError(
                    f"End frame {end_frame} is less than start frame {start_frame} for scene {scene.name}"
                )
            start_time = start_frame / frame_rate
            end_time = end_frame / frame_rate
            if end_time > duration:
                end_time = duration
            subclip = video.subclip(start_time, end_time)
            output_file = str(Path(output_dir) / scene.filename)
            subclip.write_videofile(
                output_file, audio_codec="aac", codec=codec, logger=None
            )
            return output_file

    paths = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                writer_worker, scene, video_path, frame_rate, output_dir, codec
            )
            for scene in scenes
        ]
        for scene, future in zip(scenes, futures):
            future.result()
            paths.append(str(Path(output_dir) / scene.filename))
    return paths


async def create_scenes_from_video(
    video: "Video", volume_dir: str, min_scene_len_seconds: float = 10
):
    raise NotImplementedError("need to copy from old repo")
