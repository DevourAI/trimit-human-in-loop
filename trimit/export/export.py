import os
from typing import Union
import opentimelineio as otio
from tqdm import tqdm
from trimit.models import Scene, Video
from trimit.utils.model_utils import get_scene_folder, get_generated_video_folder
from trimit.export.utils import get_new_integer_file_name_in_dir
from tqdm.asyncio import tqdm as tqdm_async


async def create_cut_video_from_transcript(
    video: Video,
    transcript: Union["Transcript", "Soundbites", "Soundbite"],
    timeline_name: str,
    volume_dir: str,
    output_dir: str,
    video_filename_suffix: str = "",
    clip_extra_trim_seconds: float = 0,
    prefix: str = "video_",
    clip_duration_safety_buffer: float = 0.01,
    clip_min_duration: float = 0.1,
    clip_gap_min_duration: float = 0.1,
):
    if video.duration is None:
        raise ValueError("Video duration must be provided")
    assert video.duration is not None
    timeline = []
    for cut in tqdm(transcript.iter_kept_cuts(), desc="Creating scenes"):
        start_time = max(0, float(cut.start) - clip_extra_trim_seconds)
        end_time = min(
            float(cut.end) + clip_extra_trim_seconds,
            video.duration - clip_duration_safety_buffer,
        )
        if end_time - start_time < clip_min_duration:
            continue

        if len(timeline) and start_time < timeline[-1].end + clip_gap_min_duration:
            timeline[-1].end = end_time
            continue
        scene = await Scene.from_video(
            video=video,
            start=start_time,
            end=end_time,
            save=False,
            check_existing=False,
        )
        timeline.append(scene)
    return await generate_video_from_timeline(
        video.user.email,
        timeline,
        volume_dir,
        timeline_name,
        output_dir=output_dir,
        video_filename_suffix=video_filename_suffix,
        prefix=prefix,
        frame_rate=video.frame_rate or 30,
    )


async def generate_video_from_timeline(
    user_email: str,
    timeline: list[Scene],
    volume_dir: str,
    timeline_name: str,
    output_dir: str | None = None,
    video_filename_suffix: str = "",
    prefix: str = "video_",
    frame_rate: float = 30,
):
    from moviepy.editor import VideoFileClip, concatenate_videoclips

    video_clips = {}
    scene_clips = []
    for scene in timeline:
        video_clip = video_clips.get(scene.video.md5_hash)
        if video_clip is None:
            video_clip = VideoFileClip(scene.video.path(volume_dir))
            video_clips[scene.video.md5_hash] = video_clip
        scene_clips.append(video_clip.subclip(scene.start, scene.end))

    final_clip = concatenate_videoclips(scene_clips, method="compose")
    if output_dir is None:
        output_dir = str(
            get_generated_video_folder(volume_dir, user_email, timeline_name)
        )

    local_output_file = get_new_integer_file_name_in_dir(
        output_dir, ".mp4", prefix=prefix, suffix=video_filename_suffix
    )
    final_clip.write_videofile(
        local_output_file,
        fps=frame_rate,
        codec="libx264",
        audio_codec="aac",
        threads=os.cpu_count(),
    )
    return local_output_file


def create_fcp_7_xml_from_single_video_transcript(
    video: Video,
    transcript: "Transcript",
    timeline_name: str,
    volume_dir: str,
    output_dir: str | None = None,
    user_email: str = "",
    clip_extra_trim_seconds: float = 0,
    use_high_res_path=False,
    use_full_path=False,
    output_width=1920,
    output_height=1080,
    prefix="timeline_",
):
    timeline = create_otio_timeline_from_single_video_transcript(
        video,
        transcript,
        timeline_name,
        volume_dir,
        clip_extra_trim_seconds=clip_extra_trim_seconds,
        use_high_res_path=use_high_res_path,
        use_full_path=use_full_path,
        output_width=output_width,
        output_height=output_height,
    )
    if output_dir is None:
        if user_email is None:
            raise ValueError("user_email must be provided if output_dir is None")
        output_dir = str(
            get_generated_video_folder(volume_dir, user_email, timeline_name)
        )

    return save_otio_timeline(
        timeline, output_dir, adapter="fcp_xml", ext=".xml", prefix=prefix
    )


def save_otio_timeline(
    timeline: "otio.schema.Timeline",
    output_dir: str,
    adapter="fcp_xml",
    ext=".xml",
    prefix="timeline_",
):
    file_name = os.path.abspath(
        get_new_integer_file_name_in_dir(output_dir, prefix=prefix, ext=ext)
    )

    otio.adapters.write_to_file(timeline, file_name, adapter_name=adapter)
    # TODO add audio tags:
    # <audio><channelcount>2</channelcount></audio>

    return file_name


def create_otio_timeline_from_single_video_transcript(
    video: Video,
    transcript: "Transcript",
    timeline_name: str,
    volume_dir: str,
    clip_extra_trim_seconds: float = 0,
    use_high_res_path=False,
    use_full_path=False,
    output_width=1920,
    output_height=1080,
):
    timeline = otio.schema.Timeline(name=timeline_name)
    video_track = otio.schema.Track(name="Main")
    timeline.tracks.append(video_track)
    audio_track = otio.schema.Track(name="Main", kind="Audio")
    timeline.tracks.append(audio_track)
    # TODO iteratoe over cut segments in each segment in create_cut_video fn

    filepath = video.path(volume_dir)
    if use_high_res_path:
        filepath = video.high_res_user_file_path
        if not use_full_path:
            filepath = os.path.basename(filepath)
    media = otio.schema.ExternalReference(
        target_url=filepath,
        available_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, video.frame_rate),
            duration=otio.opentime.RationalTime.from_seconds(
                float(video.duration), video.frame_rate
            ),
        ),
    )
    clip_idx = 1
    for cut in tqdm(transcript.iter_kept_cuts(), desc="Creating scenes"):
        start_time = otio.opentime.RationalTime(
            max(0, float(cut.start) - clip_extra_trim_seconds), 1.0
        ).rescaled_to(video.frame_rate)
        end_time_exclusive = otio.opentime.RationalTime(
            min(float(cut.end) + clip_extra_trim_seconds, video.duration), 1.0
        ).rescaled_to(video.frame_rate)

        video_clip = otio.schema.Clip(
            name=f"Clip{clip_idx}",
            media_reference=media,
            source_range=otio.opentime.range_from_start_end_time(
                start_time, end_time_exclusive
            ),
        )
        audio_clip = otio.schema.Clip(
            name=f"Clip{clip_idx}",
            media_reference=media,
            source_range=otio.opentime.range_from_start_end_time(
                start_time, end_time_exclusive
            ),
        )

        video_track.append(video_clip)
        audio_track.append(audio_clip)
        clip_idx += 1

    return timeline


def save_text_file(text, filename):
    with open(f"{filename}.txt", "w") as f:
        f.write(text)


def save_story_to_disk(output_dir, story):
    with open(os.path.join(output_dir, "story.txt"), "w") as f:
        f.write(story)


def save_transcript_to_disk(
    output_dir: str,
    transcript: Union["Transcript", "TranscriptChunk", "Soundbites", "SoundbitesChunk"],
    timeline_name: str = "",
    suffix: str = "",
    prefix="transcript_",
    save_text_file=True,
):
    if timeline_name:
        output_dir = os.path.join(output_dir, timeline_name)
    text_file = get_new_integer_file_name_in_dir(
        output_dir, ".txt", prefix=prefix, suffix=suffix
    )
    pickle_file = get_new_integer_file_name_in_dir(
        output_dir, ".p", prefix=prefix, suffix=suffix
    )

    transcript.save(pickle_file)
    if save_text_file:
        with open(text_file, "w") as f:
            f.write(transcript.text)
        return pickle_file, text_file
    return pickle_file, None


async def save_soundbites_videos_to_disk(
    output_dir: str,
    volume_dir: str,
    video: "Video",
    soundbites: Union["Soundbites", "SoundbitesChunk"],
    timeline_name: str = "",
    prefix: str = "soundbite_{}",
    clip_extra_trim_seconds: float = 0,
    clip_duration_safety_buffer: float = 0.01,
    clip_min_duration: float = 0.0,
    clip_gap_min_duration: float = 0.0,
):
    tasks = []
    for i, soundbite in enumerate(soundbites.soundbites):
        tasks.append(
            create_cut_video_from_transcript(
                video=video,
                transcript=soundbite,
                timeline_name=timeline_name,
                volume_dir=volume_dir,
                output_dir=output_dir,
                prefix=prefix.format(i),
                clip_extra_trim_seconds=clip_extra_trim_seconds,
                clip_duration_safety_buffer=clip_duration_safety_buffer,
                clip_min_duration=clip_min_duration,
                clip_gap_min_duration=clip_gap_min_duration,
            )
        )
    video_paths = await tqdm_async.gather(*tasks)
    return video_paths
