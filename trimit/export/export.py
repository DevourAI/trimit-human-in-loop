from numpy import clip
from typing import Union
import opentimelineio as otio
import os
from tqdm import tqdm
from trimit.models import Scene, maybe_init_mongo
from tqdm import tqdm
from beanie.operators import In
from collections import defaultdict
from trimit.utils.model_utils import get_scene_folder, get_generated_video_folder
from trimit.export.utils import get_new_integer_file_name_in_dir
import os


async def create_cut_video_from_transcript(
    video: "Video",
    transcript: "Transcript",
    timeline_name: str,
    volume_dir: str,
    output_dir: str,
    video_filename_suffix: str = "",
    max_workers: int = 12,
    clip_extra_trim_seconds: float = 0,
    prefix: str = "video_",
):
    from trimit.models import Scene, maybe_init_mongo, TimelineOutput, TimelineClip

    timeline_scenes = []
    try:
        await Scene.remove_all_for_video(video)
    except:
        await maybe_init_mongo()
        await Scene.remove_all_for_video(video)
    for cut in tqdm(transcript.iter_kept_cuts(), desc="Creating scenes"):
        start_time = max(0, float(cut.start) - clip_extra_trim_seconds)
        end_time = min(float(cut.end) + clip_extra_trim_seconds, video.duration)

        scene = await Scene.from_video(
            video=video, start=start_time, end=end_time, save=True
        )
        timeline_scenes.append(TimelineClip(scene_name=scene.simple_name))
    timeline = TimelineOutput(timeline=timeline_scenes)
    return await generate_video_from_timeline(
        video.user.email,
        timeline,
        volume_dir,
        timeline_name,
        output_dir=output_dir,
        video_filename_suffix=video_filename_suffix,
        max_workers=max_workers,
        prefix=prefix,
    )


async def generate_video_from_timeline(
    user_email: str,
    timeline: "TimelineOutput",
    volume_dir: str,
    timeline_name: str,
    scene_output_dir: str | None = None,
    output_dir: str | None = None,
    video_filename_suffix: str = "",
    max_workers: int = 12,
    prefix: str = "video_",
):
    from moviepy.editor import VideoFileClip, concatenate_videoclips
    from trimit.utils.scene_extraction import extract_scenes_to_disk

    await maybe_init_mongo()
    scene_names = [clip.scene_name for clip in timeline.timeline]

    scenes = await Scene.find(
        Scene.user.email == user_email, In(Scene.simple_name, scene_names)
    ).to_list()
    scenes_dict = {scene.simple_name: scene for scene in scenes}

    new_timeline_scenes = []
    for clip in tqdm(timeline.timeline, desc="Creating trimmed scenes"):
        existing_scene = scenes_dict[clip.scene_name]
        start_frame = existing_scene.start_frame
        end_frame = existing_scene.end_frame
        if hasattr(clip, "start_index") and clip.start_index is not None:
            word_index = 0
            start = existing_scene.transcription["segments"][0]["start"]
            for segment in existing_scene.transcription["segments"]:
                found_word = False
                for word in segment["words"]:
                    if word_index == clip.start_index:
                        start = word["start"]
                        found_word = True
                        break
                    word_index += 1
                if found_word:
                    break
            start_frame = int(round(existing_scene.video.frame_rate * start))

        if hasattr(clip, "end_index") and clip.end_index is not None:
            word_index = 0
            end = existing_scene.transcription["segments"][-1]["end"]
            for segment in existing_scene.transcription["segments"]:
                found_word = False
                for word in segment["words"]:
                    if word_index == clip.end_index:
                        end = word["end"]
                        found_word = True
                        break
                    word_index += 1
                if found_word:
                    break
            end_frame = int(round(existing_scene.video.frame_rate * end))
        if (
            start_frame != existing_scene.start_frame
            or end_frame != existing_scene.end_frame
        ):
            new_scene = await Scene.from_video(
                existing_scene.video, start_frame, end_frame, save=True
            )
            new_timeline_scenes.append(new_scene)
        else:
            new_timeline_scenes.append(existing_scene)

    video_to_scenes = defaultdict(list)
    md5_hash_to_video = {}
    for scene in new_timeline_scenes:
        md5_hash_to_video[scene.video.md5_hash] = scene.video
        video_to_scenes[scene.video.md5_hash].append(scene)
    scene_name_to_filepaths = {}
    for video_md5_hash, video_scenes in video_to_scenes.items():
        video = md5_hash_to_video[video_md5_hash]
        import datetime

        # TODO this was previously necessary
        if video.upload_datetime is None:
            video.upload_datetime = datetime.datetime(2024, 1, 1, 15, 40, 3, 970000)
        if scene_output_dir is None:
            scene_output_dir = get_scene_folder(
                volume_dir, video.user_email, video.upload_datetime
            )
        await extract_scenes_to_disk(
            video.path(volume_dir),
            video_scenes,
            scene_output_dir,
            frame_rate=video.frame_rate,
            codec=video.codec,
            max_workers=max_workers,
            # TODO turn on after recalculate duration
            # duration=video.duration,
        )
        for scene in video_scenes:
            scene_name_to_filepaths[scene.simple_name] = scene.path(volume_dir)
    filepaths = [
        scene_name_to_filepaths[scene.simple_name] for scene in new_timeline_scenes
    ]
    # TODO this was previously necessary
    # filepaths = [re.sub(r"2024-\d{2}-\d{2}", "2024-01-01", f) for f in filepaths]
    # TODO can do this from the video itself with VideoFileClip.subclip? not sure if better
    clips = [VideoFileClip(filepath) for filepath in filepaths]
    final_clip = concatenate_videoclips(clips, method="compose")
    # TODO this will change once we have a Timeline instance
    # TODO timeline versions
    if output_dir is None:
        output_dir = str(
            get_generated_video_folder(volume_dir, user_email, timeline_name)
        )

    local_output_file = get_new_integer_file_name_in_dir(
        output_dir, ".mp4", prefix=prefix, suffix=video_filename_suffix
    )
    final_clip.write_videofile(
        local_output_file, fps=30, codec="libx264", audio_codec="aac"
    )
    return local_output_file


# https://github.com/AcademySoftwareFoundation/OpenTimelineIO/blob/main/examples/shot_detect.py
def create_fcp_7_xml_from_single_video_transcript(
    video: "Video",
    transcript: "Transcript",
    timeline_name: str,
    volume_dir: str,
    output_dir: str | None = None,
    user_email: str = "",
    clip_extra_trim_seconds: float = 0,
    use_high_res_path=False,
    output_width=1920,
    output_height=1080,
):
    timeline = create_otio_timeline_from_single_video_transcript(
        video,
        transcript,
        timeline_name,
        volume_dir,
        clip_extra_trim_seconds=clip_extra_trim_seconds,
        use_high_res_path=use_high_res_path,
        output_width=output_width,
        output_height=output_height,
    )
    if output_dir is None:
        if user_email is None:
            raise ValueError("user_email must be provided if output_dir is None")
        output_dir = str(
            get_generated_video_folder(volume_dir, user_email, timeline_name)
        )

    return save_otio_timeline(timeline, output_dir, adapter="fcp_xml", ext=".xml")
    # return save_otio_timeline(timeline, output_dir, adapter="fcp_xml", ext=".aaf")


def save_otio_timeline(
    timeline: "otio.schema.Timeline", output_dir: str, adapter="fcp_xml", ext=".xml"
):
    file_name = os.path.abspath(
        get_new_integer_file_name_in_dir(output_dir, prefix="timeline_", ext=ext)
    )

    otio.adapters.write_to_file(timeline, file_name, adapter_name=adapter)
    # TODO add audio tags:
    # <audio><channelcount>2</channelcount></audio>

    return file_name


def create_otio_timeline_from_single_video_transcript(
    video: "Video",
    transcript: "Transcript",
    timeline_name: str,
    volume_dir: str,
    clip_extra_trim_seconds: float = 0,
    use_high_res_path=False,
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
    media = otio.schema.ExternalReference(
        target_url="file://" + filepath,
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
