import datetime
from bson.dbref import DBRef
import json
import hashlib
import os
from trimit.utils.string import (
    strip_punctuation_case_whitespace,
    longest_contiguous_match,
)
from trimit.utils.model_utils import (
    filename_from_md5_hash,
    get_upload_folder,
    get_scene_folder,
    get_frame_folder,
    get_generated_video_folder,
    get_audio_folder,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential
from beanie.exceptions import DocumentWasNotSaved, StateNotSaved
from pymongo import IndexModel
import pymongo
from beanie import Document, Indexed, Link, PydanticObjectId
from beanie.operators import In
from pydantic import (
    validator,
    Field,
    BaseModel,
    field_validator,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
)

from typing import Optional
from typing import NamedTuple
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from typing import Optional, Any
from pyannote.core import Annotation, Segment
from typing_extensions import Annotated


def annotation_to_list(annotation: Annotation):
    return [
        ((seg.start, seg.end), track, speaker)
        for seg, track, speaker in annotation.itertracks(yield_label=True)
    ]


def scene_name_from_video_take_item(video, start_frame, end_frame, take_item):
    if take_item is None:
        take_item_name = "no-take-item"
    else:
        take_item_name = str(TakeItem.id)
    return f"{video.md5_hash}-{start_frame}-{end_frame}-{take_item_name}"


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


def find_segment_index_range_from_combined_range(
    segments,
    segment_index_start,
    segment_index_end,
    combined_matched_subsegment_start,
    combined_matched_subsegment_end,
):
    (
        matched_segment_index_start,
        matched_segment_index_end,
        matched_subsegment_start,
        matched_subsegment_end,
    ) = (None, None, None, None)
    combined_word_i = 0
    for segment_i, segment in enumerate(
        segments[segment_index_start : segment_index_end + 1]
    ):
        for subsegment_i in range(len(segment["words"])):
            if combined_word_i == combined_matched_subsegment_start:
                matched_segment_index_start = segment_index_start + segment_i
                matched_subsegment_start = subsegment_i
            if combined_word_i + 1 == combined_matched_subsegment_end:
                matched_segment_index_end = segment_index_start + segment_i
                matched_subsegment_end = subsegment_i + 1
            combined_word_i += 1
    if any(
        [
            matched_segment_index_start is None,
            matched_segment_index_end is None,
            matched_subsegment_start is None,
            matched_subsegment_end is None,
        ]
    ):
        raise ValueError("Combined match range not found in segments")
    return (
        matched_segment_index_start,
        matched_segment_index_end,
        matched_subsegment_start,
        matched_subsegment_end,
    )


def find_subsegment_start_end(
    take_item: "TakeItem",
    segment_index_start,
    segment_index_end,
    stripped_llm_words: list[str],
    matched_words_threshold: float = 0.5,
    subsegment_words_threshold: float = 0.5,
):
    matched_segments = take_item.transcription["segments"][
        segment_index_start : segment_index_end + 1
    ]
    matched_word_objs = [wo for ms in matched_segments for wo in ms["words"]]

    matched_words = [
        strip_punctuation_case_whitespace(o["word"]) for o in matched_word_objs
    ]
    print(f"LLM words: {stripped_llm_words}")
    print(f"Segment words: {matched_words}")
    (combined_matched_subsegment_start, combined_matched_subsegment_end) = (
        longest_contiguous_match(stripped_llm_words, matched_words)
    )
    subsegment_matched_word_objs = matched_word_objs[
        combined_matched_subsegment_start:combined_matched_subsegment_end
    ]
    subsegment_matched_words = [
        strip_punctuation_case_whitespace(o["word"])
        for o in subsegment_matched_word_objs
    ]

    subsegment_words_to_llm_words = len(subsegment_matched_words) / len(
        stripped_llm_words
    )
    subsegment_words_to_segment_words = len(subsegment_matched_words) / len(
        matched_words
    )

    (
        matched_segment_index_start,
        matched_segment_index_end,
        matched_subsegment_start,
        matched_subsegment_end,
    ) = find_segment_index_range_from_combined_range(
        take_item.transcription["segments"],
        segment_index_start,
        segment_index_end,
        combined_matched_subsegment_start,
        combined_matched_subsegment_end,
    )
    start = take_item.transcription["segments"][matched_segment_index_start]["words"][
        matched_subsegment_start
    ]["start"]
    end = take_item.transcription["segments"][matched_segment_index_end]["words"][
        matched_subsegment_end - 1
    ]["end"]

    if subsegment_words_to_llm_words < 1 and subsegment_words_to_segment_words > 0:
        expand_search_high = True
        expand_search_low = True
        expanded_segment_index_start = segment_index_start - 1
        expanded_segment_index_end = segment_index_end + 1
        if (
            matched_segment_index_start > segment_index_start
            or segment_index_start == 0
        ):
            expand_search_low = False
            expanded_segment_index_start = segment_index_start
        if (
            matched_segment_index_end < segment_index_end
            or segment_index_end == len(take_item.transcription["segments"]) - 1
        ):
            expand_search_high = False
            expanded_segment_index_end = segment_index_end

        if expand_search_high or expand_search_low:
            return find_subsegment_start_end(
                take_item,
                expanded_segment_index_start,
                expanded_segment_index_end,
                stripped_llm_words,
                matched_words_threshold,
                subsegment_words_threshold,
            )
        elif subsegment_words_to_llm_words < matched_words_threshold:
            raise ValueError("No match found")
        else:
            return (
                matched_segment_index_start,
                matched_segment_index_end,
                matched_subsegment_start,
                matched_subsegment_end,
                start,
                end,
                subsegment_matched_words,
                subsegment_words_to_llm_words,
            )

    elif subsegment_words_to_llm_words < matched_words_threshold:
        raise ValueError("No match found")
    else:
        return (
            matched_segment_index_start,
            matched_segment_index_end,
            matched_subsegment_start,
            matched_subsegment_end,
            start,
            end,
            subsegment_matched_words,
            subsegment_words_to_llm_words,
        )


class DocumentWithSaveRetry(Document):
    @retry(
        stop=stop_after_attempt(5), wait=wait_random_exponential(multiplier=1, max=60)
    )
    async def save_with_retry(self):
        try:
            await self.save_changes()
        except (DocumentWasNotSaved, StateNotSaved):
            await self.save()


class PathMixin:
    def path(self, volume_dir):
        upload_folder = get_upload_folder(
            volume_dir, self.user_email, self.upload_datetime
        )
        file_path = upload_folder / self.filename
        return str(file_path)


class User(DocumentWithSaveRetry):
    email: Indexed(str, unique=True)
    password: Optional[str] = Field(default=None, min_length=8)
    name: str
    authorized_with_google: bool = False

    class Settings:
        use_state_management = True

    def __repr__(self):
        return f"User(email={self.email}, name={self.name})"

    @validator("authorized_with_google", always=True, pre=True)
    def check_authentication_method(cls, v, values, **kwargs):
        # `values` is a dictionary containing the values of fields
        # validated before 'authorized_with_google'
        if "password" in values and values["password"] is not None:
            # If password is set, it doesn't matter if authorized_with_google is True or False
            return v
        if v:  # If authorized_with_google is True, it's valid
            return v
        raise ValueError("User must have a password set or be authorized with Google")

    @property
    async def timelines(self):
        return await Timeline.find(Timeline.user == self).to_list()

    @property
    async def videos(self):
        return await Video.find(Video.user == self).to_list()

    @property
    async def scenes(self):
        return await Scene.find(Scene.user == self).to_list()

    @property
    async def frames(self):
        return await Frame.find(Frame.user == self).to_list()


class SpeechSegment(BaseModel):
    start: float
    end: float


class DiarizationSegment(BaseModel):
    speaker: str
    start: float
    end: float


class _PyannoteAnotationAnnotation:
    segment_schema = core_schema.tuple_positional_schema(
        [core_schema.float_schema(), core_schema.float_schema()]
    )
    annotation_list_schema = core_schema.list_schema(
        core_schema.tuple_positional_schema(
            [
                segment_schema,
                core_schema.union_schema(
                    [core_schema.str_schema(), core_schema.int_schema()]
                ),
                core_schema.str_schema(),
            ]
        )
    )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def validate_from_list(
            value: list[tuple[tuple[float, float], str | int, str]]
        ) -> Annotation:
            result = Annotation()
            for (start, end), track, speaker in value:
                segment = Segment(start, end)
                result[segment, track] = speaker
            return result

        from_list_schema = core_schema.chain_schema(
            [
                cls.annotation_list_schema,
                core_schema.no_info_plain_validator_function(validate_from_list),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_list_schema,
            python_schema=core_schema.union_schema(
                [
                    # check if it's an instance first before doing any further work
                    core_schema.is_instance_schema(Annotation),
                    from_list_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: annotation_to_list(instance)
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return handler(cls.annotation_list_schema)


PydanticPyannoteAnnotation = Annotated[Annotation, _PyannoteAnotationAnnotation]


class PydanticFraction(NamedTuple):
    numerator: int
    denominator: int

    @classmethod
    def from_fraction(cls, fraction):
        return cls(numerator=fraction.numerator, denominator=fraction.denominator)


class VideoMetadata(BaseModel):
    frame_count: Optional[int]
    frame_rate_fraction: Optional[PydanticFraction]
    frame_rate: Optional[float]
    mime_type: Optional[str]
    major_brand: Optional[str]
    create_date: Optional[datetime.datetime]  # create_date returned by exiftool
    modify_date: Optional[datetime.datetime]  # modify_date returned by exiftool
    file_creation_date: Optional[datetime.datetime]  # file creation date returned by os
    duration: Optional[float]
    width: Optional[int]
    height: Optional[int]
    resolution_x: Optional[int]
    resolution_y: Optional[int]
    codec: Optional[str]
    bit_depth: Optional[int]
    audio_format: Optional[str]
    audio_channels: Optional[int]
    audio_bits_per_sample: Optional[int]
    audio_sample_rate: Optional[int]

    @classmethod
    def from_default(cls):
        return cls(
            frame_count=None,
            frame_rate_fraction=None,
            frame_rate=None,
            mime_type=None,
            major_brand=None,
            create_date=None,
            modify_date=None,
            file_creation_date=None,
            duration=None,
            width=None,
            height=None,
            resolution_x=None,
            resolution_y=None,
            codec=None,
            bit_depth=None,
            audio_format=None,
            audio_channels=None,
            audio_bits_per_sample=None,
            audio_sample_rate=None,
        )


#  class VideoTranscription(DocumentWithSaveRetry):
#  segments: list[dict]
#  word_segments: list[dict]
#  start: float | None = None
#  end: float | None = None
#  text: str | None = None


class Video(DocumentWithSaveRetry, PathMixin):
    md5_hash: str
    simple_name: Optional[str] = None
    ext: str
    upload_datetime: datetime.datetime
    details: Optional[VideoMetadata] = None
    high_res_user_file_path: str
    high_res_user_file_hash: str
    # transcription: Optional[Link[VideoTranscription]] = None
    # TODO
    transcription: Optional[dict] = None
    transcription_text: Optional[str] = None
    diarization: Optional[PydanticPyannoteAnnotation] = None
    speech_segments: Optional[list[SpeechSegment]] = None
    video_llava_description: Optional[str] = None
    user: User
    timelines: Optional[list[Link["Timeline"]]] = None
    summary: Optional[str] = None

    class Settings:
        name = "Video"
        use_state_management = True
        bson_encoders = {Annotation: annotation_to_list}
        indexes = [
            IndexModel([("md5_hash", pymongo.ASCENDING)], unique=True),
            [("simple_name", pymongo.ASCENDING)],
            [
                ("user", pymongo.ASCENDING),
                ("md5_hash", pymongo.ASCENDING),
                ("upload_datetime", pymongo.ASCENDING),
                ("project", pymongo.ASCENDING),
            ],
            IndexModel(
                [
                    ("user", pymongo.ASCENDING),
                    ("high_res_user_file_path", pymongo.ASCENDING),
                ],
                name="user_high_res_user_file_path_index",
                unique=True,
            ),
            [("high_res_user_file_path", pymongo.ASCENDING)],
            [("user", pymongo.ASCENDING)],
            [
                ("user", pymongo.ASCENDING),
                ("project", pymongo.ASCENDING),
                ("md5_hash", pymongo.ASCENDING),
            ],
            [("user", pymongo.ASCENDING), ("md5_hash", pymongo.ASCENDING)],
            [("user", pymongo.ASCENDING), ("upload_datetime", pymongo.ASCENDING)],
            [("user", pymongo.ASCENDING), ("recorded_date", pymongo.ASCENDING)],
            [
                ("user", pymongo.ASCENDING),
                ("md5_hash", pymongo.ASCENDING),
                ("diarization.speaker", pymongo.ASCENDING),
            ],
            [("user", pymongo.ASCENDING), ("diarization.speaker", pymongo.ASCENDING)],
            # This exists as a weighted index model because we can
            # add other text fields to the index in the future
            #  IndexModel(
            #  [("transcription.text", pymongo.TEXT)],
            #  weights={"transcription.text": 10},
            #  name="text_index",
            #  ),
            IndexModel(
                [("transcription_text", pymongo.TEXT)],
                weights={"transcription_text": 10},
                name="text_index",
            ),
        ]

    def __repr__(self):
        return f"Video(md5_hash={self.md5_hash}, user={self.user.email})"

    # TODO need to lock db for this to truly work
    @classmethod
    async def gen_simple_name(cls, user_email):
        num_videos = await Video.find(Video.user.email == user_email).count()
        return str(num_videos)

    @classmethod
    async def from_user_email(cls, user_email: str, md5_hash: str, **kwargs) -> "Video":
        user = await User.find_one(User.email == user_email)
        if user is None:
            raise ValueError(f"User not found: {user_email}")
        existing_video = await Video.find_one(
            Video.md5_hash == md5_hash, Video.user.email == user_email
        )
        if existing_video is not None:
            return existing_video
        if "timeline_name" not in kwargs and "timelines" not in kwargs:
            kwargs["timelines"] = []
        simple_name = await Video.gen_simple_name(user_email)
        video = Video(md5_hash=md5_hash, simple_name=simple_name, user=user, **kwargs)
        await video.save_with_retry()
        return video

    @classmethod
    async def from_user_email_and_timeline_name(
        cls, user_email: str, timeline_name: str, md5_hash: str, **kwargs
    ) -> "Video":
        timeline = await Timeline.find_or_create(user_email, timeline_name)
        existing_video = await Video.find_one(
            Video.md5_hash == md5_hash, Video.user.email == user_email, fetch_links=True
        )
        if existing_video is not None:
            save = False
            if "details" in kwargs:
                existing_video.details = kwargs["details"]
                save = True
            if timeline.id not in [tl.id for tl in existing_video.timelines]:
                existing_video.timelines.append(timeline)
                save = True
            if save:
                await existing_video.save_with_retry()
            return existing_video
        simple_name = await Video.gen_simple_name(user_email)
        video = Video(
            md5_hash=md5_hash,
            simple_name=simple_name,
            user=timeline.user,
            timelines=[timeline],
            **kwargs,
        )
        await video.save_with_retry()
        return video

    @classmethod
    async def find_all_from_user_email(cls, user_email: str):
        return await Video.find(Video.user.email == user_email).to_list()

    @classmethod
    async def find_one_from_user_email_high_res_path(
        cls, user_email: str, high_res_user_file_path: str
    ):
        return await Video.find_one(
            Video.user.email == user_email,
            Video.high_res_user_file_path == high_res_user_file_path,
        )

    @classmethod
    async def find_all_summaries_for_timeline(
        cls, user_email: str, timeline_name: str
    ) -> list[dict]:
        videos = (
            await Video.find(
                Video.user.email == user_email,
                # MAJOR TODO why isn't this working?
                # In(Video.timelines.name, [timeline_name]),
                fetch_links=True,
            )
            .project(VideoSummaryProjection)
            .to_list()
        )
        return [
            {"hash": video.md5_hash, "summary": video.summary}
            for video in videos
            # TODO remove
            if timeline_name in [t.name for t in video.timelines]
        ]

    @property
    def filename(self):
        return filename_from_md5_hash(self.md5_hash, self.ext)

    @property
    def frame_rate(self):
        return self.details.frame_rate if self.details else None

    @property
    def codec(self):
        return self.details.codec if self.details else None

    @property
    def duration(self):
        return self.details.duration if self.details else None

    @property
    def recorded_datetime(self):
        return self.details.create_date if self.details else None

    def path(self, volume_dir):
        upload_folder = get_upload_folder(
            volume_dir, self.user.email, self.upload_datetime
        )
        file_path = upload_folder / self.filename
        return str(file_path)

    def audio_path(self, volume_dir):
        audio_folder = get_audio_folder(
            volume_dir, self.user.email, self.upload_datetime
        )
        audio_filename = self.filename.rsplit(".", 1)[0] + ".wav"
        file_path = audio_folder / audio_filename
        return str(file_path)

    @property
    def project_name(self):
        return self.project.name

    @property
    def user_email(self):
        return self.user.email

    @property
    async def scenes(self):
        return await Scene.find(Scene.video.md5_hash == self.md5_hash).to_list()

    @property
    async def frames(self):
        return await Frame.find(Frame.video.md5_hash == self.md5_hash).to_list()


class VideoSummaryProjection(BaseModel):
    md5_hash: str
    summary: Optional[str] = None
    timelines: Optional[list[Link["Timeline"]]] = None


class UserEmailProjection(BaseModel):
    _id: PydanticObjectId
    email: str


class VideoFileProjection(BaseModel, PathMixin):
    md5_hash: str
    user: UserEmailProjection
    upload_datetime: datetime.datetime
    ext: str

    @property
    def filename(self):
        return filename_from_md5_hash(self.md5_hash, self.ext)

    @property
    def user_email(self):
        return self.user.email


class VideoUserHashProjection(BaseModel):
    _id: PydanticObjectId
    user: UserEmailProjection
    md5_hash: str


class SceneNameProjection(BaseModel):
    name: str


class SceneTextProjection(BaseModel):
    name: str
    user: User
    transcription_words: Optional[list[str]] = None


class Scene(DocumentWithSaveRetry):
    name: Indexed(str, unique=True)
    simple_name: Optional[str] = None
    start_frame: int
    end_frame: int
    start: float
    end: float
    video: Video
    user: User
    take_item: Optional["TakeItem"] = None
    transcription_words: Optional[list[str]] = None
    segment_index_start: Optional[int] = None
    segment_index_end: Optional[int] = None
    subsegment_index_start: Optional[int] = None
    subsegment_index_end: Optional[int] = None
    # these two are lists because it's difficult to construct a combined version of them after merging two adjacent scenes
    # so instead we just append
    llm_stripped_word_output: Optional[list[list[str]]] = None
    percentage_matched_words: Optional[list[float]] = None
    speaker_in_frame: Optional[bool] = None

    class Settings:
        name = "Scene"
        use_state_management = True
        indexes = [
            IndexModel([("name", pymongo.ASCENDING)], unique=True),
            [("simple_name", pymongo.ASCENDING)],
            [("user", pymongo.ASCENDING)],
            [("user.email", pymongo.ASCENDING)],
            [("video", pymongo.ASCENDING)],
            [("video.md5_hash", pymongo.ASCENDING)],
            [("user", pymongo.ASCENDING), ("video.md5_hash", pymongo.ASCENDING)],
            [("take_item", pymongo.ASCENDING), ("video.md5_hash", pymongo.ASCENDING)],
            [
                ("user", pymongo.ASCENDING),
                ("take_item", pymongo.ASCENDING),
                ("video.md5_hash", pymongo.ASCENDING),
            ],
            [("user", pymongo.ASCENDING), ("take_item", pymongo.ASCENDING)],
            [("take_item", pymongo.ASCENDING)],
            IndexModel(
                [("transcription_words", pymongo.TEXT)],
                weights={"transcription_text": 10},
                name="text_index",
            ),
        ]

    def __repr__(self):
        return (
            f"Scene(name={self.name}, "
            f"simple_name={self.simple_name}, "
            f"start_frame={self.start_frame}, "
            f"end_frame={self.end_frame}, "
            f"transcription_text={self.transcription_text})"
        )

    @property
    def transcription_text(self):
        return " ".join(self.transcription_words or [])

    @property
    def transcription(self):
        return get_partial_transcription(
            self.video.transcription,
            self.segment_index_start,
            self.segment_index_end,
            self.subsegment_index_start,
            self.subsegment_index_end,
        )

    @property
    def filename(self):
        return filename_from_md5_hash(
            self.video_hash,
            self.ext,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
        )

    def overlapping(self, other_scene: "Scene"):
        if self.video != other_scene.video:
            return False
        if self.end_frame + 1 < other_scene.start_frame:
            return False
        return True

    @classmethod
    async def gen_simple_name(cls, video):
        num_video_scenes = await cls.find(
            In(Scene.video.md5_hash, [video.md5_hash]), Scene.user == video.user
        ).count()
        return f"{video.simple_name}-{num_video_scenes}"

    async def merge_if_overlapping(
        self, other_scene: "Scene", save: bool = True, delete_other: bool = False
    ):
        if not self.overlapping(other_scene):
            return
        self.end_frame = other_scene.end_frame
        self.end = other_scene.end
        self.name = scene_name_from_video_take_item(
            self.video, self.start_frame, self.end_frame, self.take_item
        )
        self.segment_index_end = other_scene.segment_index_end
        self.subsegment_index_end = other_scene.subsegment_index_end
        # Note: we set these here rather than making them a @property
        # so that mongodb can create an index on them
        self.transcription_words = partial_transcription_words(
            self.video.transcription,
            self.segment_index_start,
            self.segment_index_end,
            self.subsegment_index_start,
            self.subsegment_index_end,
        )
        if not self.llm_stripped_word_output:
            self.llm_stripped_word_output = []
        self.llm_stripped_word_output.extend(other_scene.llm_stripped_word_output or [])
        if not self.percentage_matched_words:
            self.percentage_matched_words = []
        self.percentage_matched_words.extend(other_scene.percentage_matched_words or [])

        if save:
            await self.save_with_retry()
        if delete_other:
            await other_scene.delete()
        return True

    @classmethod
    async def from_video(
        cls,
        video: "Video",
        start_frame: int | None = None,
        end_frame: int | None = None,
        start: float | None = None,
        end: float | None = None,
        save: bool = True,
    ) -> Optional["Scene"]:
        if start is None and start_frame is None:
            raise ValueError("Must provide start or start_frame")
        if end is None and end_frame is None:
            raise ValueError("Must provide end or end_frame")

        frame_rate = video.frame_rate
        if frame_rate is None:
            print(f"Frame rate is None for video {video.md5_hash}. Defaulting to 30")
            frame_rate = 30
        if start is None:
            start = start_frame / frame_rate
        if start_frame is None:
            start_frame = int(start * frame_rate)
        if end is None:
            end = end_frame / frame_rate
        if end_frame is None:
            end_frame = int(end * frame_rate)
        name = scene_name_from_video_take_item(video, start_frame, end_frame, None)
        existing_scene = await Scene.find_one(
            Scene.name == name, Scene.user == video.user, Scene.take_item == None
        )
        if existing_scene is not None:
            return existing_scene

        (
            segment_index_start,
            segment_index_end,
            subsegment_index_start,
            subsegment_index_end,
        ) = (None, None, None, None)
        transcription_words = None
        if video.transcription is not None:
            (
                segment_index_start,
                segment_index_end,
                subsegment_index_start,
                subsegment_index_end,
            ) = partial_transcription_indexes(video, start, end)
            transcription_words = partial_transcription_words(
                video.transcription,
                segment_index_start,
                segment_index_end,
                subsegment_index_start,
                subsegment_index_end,
            )

        simple_name = await cls.gen_simple_name(video)
        scene = Scene(
            name=name,
            simple_name=simple_name,
            start_frame=start_frame,
            end_frame=end_frame,
            start=start,
            end=end,
            video=video,
            user=video.user,
            take_item=None,
            transcription_words=transcription_words,
            segment_index_start=segment_index_start,
            segment_index_end=segment_index_end,
            subsegment_index_start=subsegment_index_start,
            subsegment_index_end=subsegment_index_end,
        )
        if save:
            await scene.save_with_retry()
        return scene

    @classmethod
    async def from_take_item(
        cls,
        take_item: "TakeItem",
        segment_index: int,
        llm_words: str,
        matched_words_threshold: float = 0.5,
        subsegment_words_threshold: float = 0.5,
        save: bool = True,
    ) -> Optional["Scene"]:
        stripped_llm_words = [
            strip_punctuation_case_whitespace(w)
            for w in strip_punctuation_case_whitespace(llm_words).split()
        ]

        try:
            (
                segment_index_start,
                segment_index_end,
                subsegment_index_start,
                subsegment_index_end,
                start,
                end,
                matched_words,
                percentage_matched_words,
            ) = find_subsegment_start_end(
                take_item,
                segment_index,
                segment_index,
                stripped_llm_words,
                matched_words_threshold=matched_words_threshold,
                subsegment_words_threshold=subsegment_words_threshold,
            )
        except ValueError as e:
            print(f"Error finding subsegments: {e}")
            return None

        video = take_item.video

        frame_rate = video.frame_rate
        if frame_rate is None:
            print(f"Frame rate is None for video {video.md5_hash}. Defaulting to 30")
            frame_rate = 30
        start_frame = int(start * frame_rate)
        end_frame = int(end * frame_rate)
        name = scene_name_from_video_take_item(video, start_frame, end_frame, take_item)
        existing_scene = await Scene.find_one(
            Scene.name == name, Scene.user == video.user
        )
        if existing_scene is not None:
            return existing_scene

        scene = Scene(
            name=name,
            start_frame=start_frame,
            end_frame=end_frame,
            start=start,
            end=end,
            video=video,
            user=video.user,
            take_item=take_item,
            transcription_words=matched_words,
            segment_index_start=segment_index_start,
            segment_index_end=segment_index_end,
            subsegment_index_start=subsegment_index_start,
            subsegment_index_end=subsegment_index_end,
            llm_stripped_word_output=[stripped_llm_words],
            percentage_matched_words=[percentage_matched_words],
        )
        if save:
            await scene.save_with_retry()
        return scene

    @classmethod
    async def remove_all_for_video(cls, video):
        await Scene.find(
            Scene.video.md5_hash == video.md5_hash, Scene.user.email == video.user.email
        ).delete()

    def path(self, volume_dir):
        scene_folder = get_scene_folder(
            volume_dir, self.user_email, self.video.upload_datetime
        )
        file_path = scene_folder / self.filename
        return str(file_path)

    @property
    def video_hash(self):
        return self.video.md5_hash

    @property
    def ext(self):
        return self.video.ext

    @property
    def user_email(self):
        return self.user.email

    @property
    async def frames(self):
        return await Frame.find(Frame.scene.name == self.name).to_list()


class Timeline(DocumentWithSaveRetry):
    name: str
    user: User

    class Settings:
        use_state_management = True

    def __repr__(self):
        return f"Timeline(user={self.user}, name={self.name})"

    @property
    def user_email(self):
        return self.user.email

    @property
    async def timeline_versions(self):
        return await TimelineVersion.find(
            TimelineVersion.timeline == self, fetch_links=False
        ).to_list()

    @classmethod
    async def find_or_create(cls, user_email: str, name: str) -> "Timeline":
        timeline = await Timeline.find_one(
            Timeline.name == name, Timeline.user.email == user_email
        )
        if timeline is None:
            user = await User.find_one(User.email == user_email)
            if user is None:
                raise ValueError(f"User not found: {user_email}")
            timeline = Timeline(name=name, user=user)
            await timeline.save_with_retry()
        return timeline

    async def most_recent_version(
        self, fetch_links: bool = True, nesting_depths_per_field: dict | None = None
    ):
        mrv = (
            await TimelineVersion.find(
                TimelineVersion.timeline == self,
                fetch_links=fetch_links,
                nesting_depths_per_field=nesting_depths_per_field,
            )
            .sort(-TimelineVersion.version)
            .limit(1)
            .to_list()
        )
        if len(mrv) == 0:
            return None
        return mrv[0]

    async def new_version(
        self,
        resolution_x: int,
        resolution_y: int,
        scenes: list[Scene],
        story: str = None,
    ):
        mrv = await self.most_recent_version(fetch_links=False)
        if mrv is None:
            version = 1
        else:
            version = mrv.version + 1
        return TimelineVersion(
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            version=version,
            timeline=self,
            scenes=scenes,
            story=story,
        )


class TimelineVersion(DocumentWithSaveRetry):
    resolution_x: Optional[int] = 1920
    resolution_y: Optional[int] = 1080
    story: Optional[str] = None
    version: int
    timeline: Timeline
    scenes: list[Scene]

    class Settings:
        use_state_management = True

    def video_path(self, volume_dir):
        video_folder = get_generated_video_folder(
            volume_dir, self.timeline.user_email, self.timeline.name
        )
        file_path = video_folder / f"{self.version}.mp4"
        return str(file_path)


class Frame(DocumentWithSaveRetry):
    name: Indexed(str, unique=True)
    frame_number: int
    ext: str
    aesthetic_score: Optional[float] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    camera_shot_type: Optional[str] = None
    objects: Optional[list[str]] = None
    scene: Scene
    video: Video
    user: User

    class Settings:
        name = "Frame"
        use_state_management = True
        indexes = [
            IndexModel([("name", pymongo.ASCENDING)], unique=True),
            [("user", pymongo.ASCENDING)],
            [("user.email", pymongo.ASCENDING)],
            [("video", pymongo.ASCENDING)],
            [("video.md5_hash", pymongo.ASCENDING)],
            [("user", pymongo.ASCENDING), ("video.md5_hash", pymongo.ASCENDING)],
            [("user", pymongo.ASCENDING), ("video.md5_hash", pymongo.ASCENDING)],
        ]

    def __repr__(self):
        return f"Frame(name={self.name}, frame_number={self.frame_number}, frame_category={self.category})"

    @property
    def filename(self):
        return filename_from_md5_hash(
            self.video.md5_hash, self.ext, start_frame=self.frame_number
        )

    @classmethod
    async def from_scene(
        cls,
        scene: "Scene",
        frame_number: int,
        ext: str,
        aesthetic_score: float | None = None,
        save: bool = True,
    ) -> "Frame":
        if not ext.startswith("."):
            ext = f".{ext}"
        name = f"{scene.video_hash}-{frame_number}"
        existing_frame = await Frame.find_one(Frame.name == name)
        if existing_frame is not None:
            return existing_frame
        frame = Frame(
            name=name,
            frame_number=frame_number,
            ext=ext,
            aesthetic_score=aesthetic_score,
            scene=scene,
            video=scene.video,
            user=scene.video.user,
        )
        if save:
            await frame.save_with_retry()
        return frame

    @classmethod
    async def find_all_from_video_hash(cls, video_md5_hash: str) -> list["Frame"]:
        video = await Video.find_one(Video.md5_hash == video_md5_hash)
        if video is None:
            raise ValueError(f"Video not found: {video_md5_hash}")
        return await video.frames

    def path(self, volume_dir):
        frame_folder = get_frame_folder(
            volume_dir, self.video.user_email, self.video.upload_datetime
        )
        file_path = frame_folder / self.filename
        return str(file_path)

    def add_aesthetic_score(self, score):
        self.aesthetic_score = score

    async def save_aesthetic_score(self, score):
        self.add_aesthetic_score(score)
        await self.save_with_retry()

    @property
    def scene_name(self):
        return self.scene.name

    @property
    def scene_a_roll_b_roll(self):
        return self.scene.a_roll_b_roll_category

    @property
    def video_hash(self):
        return self.video.md5_hash

    @property
    def user_email(self):
        return self.user.email


class Take(DocumentWithSaveRetry):
    user: Link[User]

    # TODO i think this fails with the string class
    # take_items: BackLink['TakeItem']
    class Settings:
        name = "Take"
        use_state_management = True
        indexes = [[("user", pymongo.ASCENDING)]]


class TakeItem(DocumentWithSaveRetry):
    video: Link[Video]
    start: float
    end: float
    transcription: dict
    transcription_text: str
    take: Link[Take]

    class Settings:
        name = "TakeItem"
        use_state_management = True
        indexes = [
            [("take.user", pymongo.ASCENDING), ("video.md5_hash", pymongo.ASCENDING)],
            IndexModel(
                [("transcription_text", pymongo.TEXT)],
                weights={"transcription_text": 10},
                name="text_index",
            ),
        ]

    @classmethod
    def create(cls, transcription: dict, **kwargs):
        transcription_text = "".join(
            [segment["text"] for segment in transcription["segments"]]
        )
        return cls(
            transcription=transcription, transcription_text=transcription_text, **kwargs
        )

    def __repr__(self):
        return (
            f"TakeItem(id={self.id}, "
            f"video={self.video.md5_hash}, "
            f"start={self.start}, "
            f"end={self.end}, "
            f"transcription_text={self.transcription_text})"
        )


class CutTranscriptLinearWorkflowStaticState(DocumentWithSaveRetry):
    user: Link[User]
    timeline_name: str
    video: Link[Video]
    volume_dir: str
    output_folder: str
    length_seconds: int

    nstages: int = 2
    first_pass_length: int = 6 * 60
    max_partial_transcript_words: int = 800
    max_word_extra_threshold: int = 50
    clip_extra_trim_seconds: float = 0.1
    use_agent_output_cache: bool = True
    max_iterations: int = 3
    max_total_soundbites: int = 15
    # These default to false because we have them in MongoDB
    export_transcript_text: bool = False
    export_transcript: bool = False
    export_soundbites: bool = False
    export_soundbites_text: bool = False
    export_timeline: bool = True
    export_video: bool = True

    def create_object_id(self):
        dumped = self.model_dump(exclude=["user", "video"])
        user_ref = self.user.to_ref()
        if isinstance(user_ref, DBRef):
            user_ref = user_ref.id
        dumped["user"] = str(user_ref)
        video_ref = self.video.to_ref()
        if isinstance(video_ref, DBRef):
            video_ref = video_ref.id
        dumped["video"] = str(video_ref)
        self_as_str = json.dumps(dumped, sort_keys=True)
        md5_hash = hashlib.md5(self_as_str.encode()).hexdigest()
        return PydanticObjectId(md5_hash[:24])


class StepOrderMixin:
    def get_current_step_key_atomic(self):
        if len(self.dynamic_state_step_order):
            return self.dynamic_state_step_order[-1]
        return None

    def get_current_step_key_before_end(self):
        if len(self.dynamic_state_step_order):
            end = self.dynamic_state_step_order[-1]
            if end == "end":
                if len(self.dynamic_state_step_order) > 1:
                    return self.dynamic_state_step_order[-2]
                return None
        return None

    def current_step_retry(self):
        current_step_key = self.get_current_step_key_atomic()
        return self.dynamic_state_retries.get(current_step_key, False)

    @property
    def length_seconds(self):
        return self.static_state.length_seconds

    @property
    def video(self):
        return self.static_state.video

    @property
    def user(self):
        return self.static_state.user

    @property
    def timeline_name(self):
        return self.static_state.timeline_name

    @property
    def first_pass_length(self):
        return self.static_state.first_pass_length

    @property
    def nstages(self):
        return self.static_state.nstages

    @property
    def output_folder(self):
        return self.static_state.output_folder

    @property
    def volume_dir(self):
        return self.static_state.volume_dir

    @property
    def use_agent_output_cache(self):
        return self.static_state.use_agent_output_cache

    @property
    def max_partial_transcript_words(self):
        return self.static_state.max_partial_transcript_words

    @property
    def max_total_soundbites(self):
        return self.static_state.max_total_soundbites

    @property
    def max_word_extra_threshold(self):
        return self.static_state.max_word_extra_threshold

    @property
    def max_iterations(self):
        return self.static_state.max_iterations

    @property
    def export_transcript_text(self):
        return self.static_state.export_transcript_text

    @property
    def export_transcript(self):
        return self.static_state.export_transcript

    @property
    def export_soundbites(self):
        return self.static_state.export_soundbites

    @property
    def export_soundbites_text(self):
        return self.static_state.export_soundbites_text

    @property
    def export_timeline(self):
        return self.static_state.export_timeline

    @property
    def export_video(self):
        return self.static_state.export_video

    @property
    def clip_extra_trim_seconds(self):
        return self.static_state.clip_extra_trim_seconds


class CutTranscriptLinearWorkflowStepOrderProjection(BaseModel, StepOrderMixin):
    _id: PydanticObjectId
    static_state: CutTranscriptLinearWorkflowStaticState
    dynamic_state_step_order: list[str] = []
    dynamic_state_retries: dict = {}


async def load_step_order(state_id):
    return await CutTranscriptLinearWorkflowState.find_one(
        CutTranscriptLinearWorkflowState.id == state_id, fetch_links=False
    ).project(CutTranscriptLinearWorkflowStepOrderProjection)


class CutTranscriptLinearWorkflowState(DocumentWithSaveRetry, StepOrderMixin):
    id: PydanticObjectId | None = None
    static_state: CutTranscriptLinearWorkflowStaticState
    on_screen_transcript_state: dict | None = None
    original_soundbites_state: dict | None = None
    on_screen_speakers: list[str] | None = None
    story: str | None = None
    raw_transcript_state: dict | None = None
    current_transcript_state: dict | None = None
    current_soundbites_state: dict | None = None
    user_messages: list[str] = []
    dynamic_state: dict = {}
    dynamic_state_step_order: list[str] = []
    dynamic_state_retries: dict = {}
    run_output_dir: str | None = None
    soundbites_output_dir: str | None = None
    output_files: dict[str, str] | None = None

    class Settings:
        name = "CutTranscriptLinearWorkflowState"
        use_state_management = True
        use_revision = True

    def __init__(self, **data):
        super().__init__(**data)
        if not self.id:
            self.id = self.static_state.create_object_id()

    def set_state_val(self, name, value):
        if name not in self.model_dump().keys():
            try:
                json.dumps(value)
            except json.JSONDecodeError:
                raise ValueError(f"Value {value} is not JSON serializable")
            self.dynamic_state[name] = value
            return
        setattr(self, name, value)

    def __getattr__(self, name):
        if name in self.__fields__.keys():
            return super().__getattr__(name)
        elif name in self.dynamic_state:
            return self.dynamic_state[name]
        else:
            return super().__getattr__(name)

    @property
    def user_prompt(self):
        return self.user_messages[0] if len(self.user_messages) else None

    def stage_output_dir(self, stage):
        if self.run_output_dir is None:
            raise ValueError("run_output_dir is None")
        return os.path.join(self.run_output_dir, f"stage_{stage}")

    async def restart(self):
        self.on_screen_transcript_state = None
        self.soundbites_state = None
        self.story = None
        self.raw_transcript_state = None
        self.current_transcript_state = None
        self.user_messages = []
        self.dynamic_state = {}
        self.dynamic_state_retries = {}
        self.dynamic_state_step_order = []
        await self.save_with_retry()

    @classmethod
    async def find_or_create(
        cls, video, timeline_name, volume_dir, output_folder, length_seconds, **params
    ):
        print("find_or_create")
        static_state = CutTranscriptLinearWorkflowStaticState(
            user=video.user,
            timeline_name=timeline_name,
            video=video,
            volume_dir=volume_dir,
            output_folder=output_folder,
            length_seconds=length_seconds,
            **params,
        )
        obj = cls(static_state=static_state)
        existing = await cls.find_one(CutTranscriptLinearWorkflowState.id == obj.id)
        if existing is not None:
            return existing
        print("did not find, creating")
        await obj.save_with_retry()
        return obj

    @classmethod
    async def recreate(
        cls, video, timeline_name, volume_dir, output_folder, length_seconds, **params
    ):
        print("recreate")
        static_state = CutTranscriptLinearWorkflowStaticState(
            user=video.user,
            timeline_name=timeline_name,
            video=video,
            volume_dir=volume_dir,
            output_folder=output_folder,
            length_seconds=length_seconds,
            **params,
        )
        obj = cls(static_state=static_state)
        existing = await cls.find_one(CutTranscriptLinearWorkflowState.id == obj.id)
        if existing is not None:
            print("deleting")
            await existing.delete()
        await obj.save_with_retry()
        print("recreated")
        return obj

    async def set_current_step_output_atomic(self, name, results):
        from trimit.linear_workflow.utils import add_retry_suffix

        if name in self.dynamic_state:
            i = 1
            while True:
                name = add_retry_suffix(name, i)
                if name in self.dynamic_state:
                    i += 1
                else:
                    break
        self.dynamic_state[name] = results
        try:
            retry = results.retry
        except AttributeError:
            retry = results.get("retry", False)
        self.dynamic_state_retries[name] = retry
        self.dynamic_state_step_order.append(name)
        await self.save_with_retry()


class TimelineClip(BaseModel):
    scene_name: Optional[str] = Field(
        default=None,
        description="The name of the retrieved scene, representing a segment of a raw video",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @validator("scene_name")
    @classmethod
    def validate_scene_name(cls, scene_simple_name):
        if scene_simple_name is not None and not cls.scene_exists_in_db(
            scene_simple_name
        ):
            log = f"The last attempt to generate a timeline used the following nonexistent scene segment: {scene_simple_name}. Retry timeline generation with a valid scene segment name."
            raise ValueError(log)
        return scene_simple_name

    @classmethod
    def scene_exists_in_db(cls, scene_simple_name):
        from trimit.models import pymongo_conn

        db = pymongo_conn()
        scene_collection = db["Scene"]
        # TODO namespace by user
        return scene_collection.find_one({"simple_name": scene_simple_name}) is not None


class TimelineOutput(BaseModel):
    """Ordered list of clips to be concatenated into a timeline, and later made into a video"""

    timeline: list[TimelineClip]

    def merge(self, other: "TimelineOutput"):
        return TimelineOutput(timeline=self.timeline + other.timeline)


ALL_MODELS = [
    User,
    Video,
    Scene,
    Frame,
    Timeline,
    TimelineVersion,
    Take,
    TakeItem,
    CutTranscriptLinearWorkflowState,
]
