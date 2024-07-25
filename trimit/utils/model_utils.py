import os
from typing import Union
import copy
import asyncio
from datetime import datetime
from pathlib import Path
from trimit.app.app import CDN_DOMAIN, TRIMIT_VIDEO_S3_CDN_BUCKET

from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_random,
    RetryError,
    retry_if_exception_type,
)


def filename_from_hash(
    hash: str, ext: str, start_frame: int | None = None, end_frame: int | None = None
) -> str:
    if start_frame is not None and end_frame is not None:
        return f"{hash}-{start_frame}-{end_frame}{ext}"
    elif start_frame is not None:
        return f"{hash}-{start_frame}{ext}"
    elif end_frame is not None:
        return f"{hash}-{end_frame}{ext}"
    else:
        return f"{hash}{ext}"


def hash_from_filename(filename: str) -> str:
    return Path(filename).stem


def hash_ext_from_filename(filename: str) -> tuple[str, str]:
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


def get_generated_soundbite_clips_folder(volume_dir, user_email, timeline_name):
    return Path(volume_dir) / "generated_soundbite_clips" / user_email / timeline_name


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
    volume_file_path: str,
    high_res_user_file_hash: str | None = None,
    overwrite: bool = False,
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
        return await Video.from_user_email(
            user_email=user_email,
            timeline_name=timeline_name,
            md5_hash=md5_hash,
            ext=ext,
            upload_datetime=upload_datetime,
            high_res_user_file_path=high_res_user_file_path,
            high_res_user_file_hash=high_res_user_file_hash,
            details=details,
            overwrite=overwrite,
        )
    except ValueError as e:
        print(f"Error saving video {md5_hash}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def check_existing_video(
    user_email: str, video_hash: str, ignore_existing: bool = False, session=None
):
    from trimit.models.models import Video
    from trimit.app import VOLUME_DIR

    existing_by_hash = await Video.find_one(
        Video.user.email == user_email, Video.md5_hash == video_hash, session=session
    )
    if existing_by_hash is not None and not ignore_existing:
        path = existing_by_hash.path(VOLUME_DIR)
        if os.path.exists(path) and os.stat(path).st_size > 0:
            return existing_by_hash


def scene_name_from_video(video, start_frame, end_frame):
    return f"{video.md5_hash}-{start_frame}-{end_frame}"


def get_dynamic_state_key(step_wrapper_name, substep_name):
    return f"{step_wrapper_name}.{substep_name}"


def get_step_substep_names_from_dynamic_state_key(key):
    return key.split(".")


def find_closest_subsegment_index_start(segment, start):
    prev_word_end = -1
    if start < segment["words"][0]["start"]:
        return 0
    for i, word in enumerate(segment["words"]):
        if start >= word["start"] and start < word["end"]:
            return i
        if start < word["start"] and start >= prev_word_end:
            return i
        prev_word_end = word["end"]
    return len(segment["words"])


def find_closest_subsegment_index_end(segment, end):
    prev_word_end = -1
    if end < segment["words"][0]["start"]:
        return 0
    for i, word in enumerate(segment["words"]):
        if end >= word["start"] and end < word["end"]:
            return i + 1
        if end < word["start"] and end >= prev_word_end:
            return i + 1
        prev_word_end = word["end"]
    return len(segment["words"])


def partial_transcription_indexes(video, start, end):
    prev_segment_end = -1
    segment_index_start = None
    segment_index_end = 0
    subsegment_index_start = None
    subsegment_index_end = None
    if len(video.transcription["segments"]) == 0:
        return 0, 0, 0, 0
    next_segment_end = video.transcription["segments"][0]["end"]
    for i, segment in enumerate(video.transcription["segments"]):
        if start >= prev_segment_end and start < segment["end"]:
            segment_index_start = i
            subsegment_index_start = find_closest_subsegment_index_start(segment, start)
            segment_index_end = segment_index_start + 1
            subsegment_index_end = find_closest_subsegment_index_end(segment, end)
        if i < len(video.transcription["segments"]) - 1:
            next_segment_end = video.transcription["segments"][i + 1]["end"]
        if segment_index_start is not None and end < next_segment_end:
            segment_index_end = i
            subsegment_index_end = find_closest_subsegment_index_end(segment, end)
            break
        prev_segment_end = segment["end"]
    if segment_index_start is None:
        segment_index_start = 0
        subsegment_index_start = 0
        subsegment_index_end = 0
    return (
        segment_index_start,
        segment_index_end,
        subsegment_index_start,
        subsegment_index_end,
    )


def partial_transcription_words(
    transcription,
    segment_index_start,
    segment_index_end,
    subsegment_index_start,
    subsegment_index_end,
):
    return [
        word["word"]
        for word in get_partial_transcription(
            transcription,
            segment_index_start,
            segment_index_end,
            subsegment_index_start,
            subsegment_index_end,
        )["word_segments"]
    ]


def get_partial_transcription(
    transcription,
    segment_index_start,
    segment_index_end,
    subsegment_index_start,
    subsegment_index_end,
):
    segments = transcription["segments"][segment_index_start : segment_index_end + 1]
    if len(segments) == 0:
        return {"segments": [], "word_segments": []}
    prior_subsegments = segments[0]["words"][:subsegment_index_start]
    prior_segments = transcription["segments"][:segment_index_start]
    start_segment_words = segments[0]["words"][subsegment_index_start:]
    end_segment_words = segments[-1]["words"][:subsegment_index_end]
    next_subsegments = segments[-1]["words"][subsegment_index_end:]
    next_segments = transcription["segments"][segment_index_end + 1 :]
    middle_segments = []
    if len(segments) > 2:
        middle_segments = segments[1:-1]

    # TODO handle case where start/end need to come from adjacent segments, not subsegments
    start_segment_start = None
    start_segment_end = None
    if len(start_segment_words):
        start_segment_start = start_segment_words[0]["start"]
        start_segment_end = start_segment_words[-1]["end"]
    elif len(prior_subsegments):
        start_segment_start = prior_subsegments[-1]["end"]
    elif "start" in segments[0]:
        start_segment_start = segments[0]["start"]
    elif len(prior_segments):
        start_segment_start = prior_segments[-1]["end"]

    if start_segment_end is None:
        if len(middle_segments):
            # TODO add buffer?
            start_segment_end = middle_segments[0]["start"]
        elif len(end_segment_words):
            start_segment_end = end_segment_words[0]["start"]
        elif len(next_subsegments):
            start_segment_end = next_subsegments[0]["start"]
        elif len(next_segments):
            start_segment_end = next_segments[0]["start"]

    end_segment_start = None
    end_segment_end = None
    if len(end_segment_words):
        end_segment_start = end_segment_words[0]["start"]
        end_segment_end = end_segment_words[-1]["end"]
    elif len(next_subsegments):
        end_segment_end = next_subsegments[0]["start"]
    elif "end" in segments[-1]:
        end_segment_end = segments[-1]["end"]
    elif len(next_segments):
        end_segment_end = next_segments[0]["start"]
    if end_segment_start is None:
        if len(middle_segments):
            end_segment_start = middle_segments[-1]["end"]
        elif len(start_segment_words):
            end_segment_start = start_segment_words[-1]["end"]
        elif len(prior_subsegments):
            end_segment_start = prior_subsegments[-1]["end"]
        elif len(prior_segments):
            end_segment_start = prior_segments[-1]["end"]

    start_speaker = None
    end_speaker = None
    if len(segments):
        start_speaker = segments[0].get("speaker")
        end_speaker = segments[-1].get("speaker")
    start_segment = {
        "start": start_segment_start,
        "end": start_segment_end,
        "words": start_segment_words,
        "speaker": start_speaker,
    }
    end_segment = {
        "start": end_segment_start,
        "end": end_segment_end,
        "words": end_segment_words,
        "speaker": end_speaker,
    }
    segments = [start_segment]
    if middle_segments:
        segments += middle_segments
    if segment_index_end > segment_index_start:
        segments.append(end_segment)
    return {
        "segments": [start_segment] + middle_segments + [end_segment],
        "word_segments": [
            word
            for word in transcription["word_segments"]
            if word["start"] >= start_segment["start"]
            and word["end"] <= end_segment["end"]
        ],
    }


def transcription_text(transcription):
    if "segments" in transcription:
        return "".join([segment["text"] for segment in transcription["segments"]])
    return ""


async def load_step_order(state_id):
    from trimit.models.models import CutTranscriptLinearWorkflowState, StepOrderMixin

    return await CutTranscriptLinearWorkflowState.find_one(
        CutTranscriptLinearWorkflowState.id == state_id, fetch_links=False
    ).project(StepOrderMixin)


def construct_retry_task_with_exception_type(exception_type):
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(0.1) + wait_random(0, 1),
        retry=retry_if_exception_type(exception_type),
    )
    async def retry_task(method, *args):
        return await method(*args)

    return retry_task


def form_cdn_url_from_path(path):
    return f"{CDN_DOMAIN}/{path}"


async def map_export_result_to_asset_path(
    export_result: Union[None, "trimit.models.backend_models.ExportResults"],
    volume_dir: str,
):
    from trimit.models.backend_models import ExportResults
    from trimit.utils.fs_utils import async_copy_to_s3

    # Note that this whole function could be done recursively for a much simpler function,
    # but we would lose the concurrency.
    # The way to get concurrency is to do the placement of each value (which is dependent on whether the path exists on disk)
    # into the ExportResults data structure
    # within each task. This gets dicey since we have a nested structure.
    # Perhaps one way to clean it up at only minor performance hit is to first check whether the file exists on disk serially,
    # and use this to determine the final data structure, and then assume we will succeed.
    # However, it seems like the volume nondeterministically returns whether a file exists or not- hence the retries- so this solution
    # may not be as accurate and may results in failed copies or successful copies but fallback placements.
    copy_tasks = []
    if export_result is None:
        return ExportResults()

    mapped_export_result = export_result.model_copy()
    retry_task = construct_retry_task_with_exception_type(FileNotFoundError)

    async def async_copy_with_placement(src, dest, placement_callback):
        path = form_cdn_url_from_path(dest)
        try:
            await retry_task(async_copy_to_s3, TRIMIT_VIDEO_S3_CDN_BUCKET, src, dest)
        except RetryError:
            path = src
        placement_callback(path)

    for filekey in mapped_export_result.model_fields_set:
        result = getattr(mapped_export_result, filekey)
        setattr(mapped_export_result, filekey, copy.deepcopy(result))

        if isinstance(result, str):
            asset_filepath = os.path.relpath(result, volume_dir)

            copy_tasks.append(
                async_copy_with_placement(
                    result,
                    asset_filepath,
                    # By using a default argument for filekey, we capture the current value of `filekey` during each iteration
                    # The only way to encapsulate the current value of filekey into the lambda is to use a default argument.
                    placement_callback=lambda p, filekey=filekey: setattr(
                        mapped_export_result, filekey, p
                    ),
                )
            )
        elif isinstance(result, list):

            def set_list_result_item(filekey, p, i):
                list_to_set = getattr(mapped_export_result, filekey)
                list_to_set[i] = p

            for i, filepath in enumerate(result):
                asset_filepath = os.path.relpath(filepath, volume_dir)

                copy_tasks.append(
                    async_copy_with_placement(
                        filepath,
                        asset_filepath,
                        # see above comment for why we need these default arguments.
                        placement_callback=lambda p, filekey=filekey, i=i: set_list_result_item(
                            filekey, p, i
                        ),
                    )
                )
        elif isinstance(result, dict):
            if isinstance(list(result.values())[0], list):

                def set_dict_list_result_item(filekey, p, k, i):
                    dict_to_set = getattr(mapped_export_result, filekey)
                    dict_to_set[k][i] = p

                for result_key, result_vals in result.items():
                    for i, filepath in enumerate(result_vals):
                        asset_filepath = os.path.relpath(filepath, volume_dir)

                        copy_tasks.append(
                            async_copy_with_placement(
                                filepath,
                                asset_filepath,
                                # see above comment for why we need these default arguments.
                                placement_callback=lambda p, filekey=filekey, result_key=result_key, i=i: set_dict_list_result_item(
                                    filekey, p, result_key, i
                                ),
                            )
                        )
            else:
                print(
                    f"dont know how to handle export result for filekey {filekey}: {result}"
                )
        else:
            print(
                f"dont know how to handle export result for filekey {filekey}: {result}"
            )
    await asyncio.gather(*copy_tasks)
    print("mapped_export_result", mapped_export_result)
    return mapped_export_result
