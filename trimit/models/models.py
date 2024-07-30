import datetime
import asyncio
from pathlib import Path
import json
import hashlib
import os
from typing import Optional, NamedTuple, Any
import copy

from tenacity import retry, stop_after_attempt, wait_fixed, wait_random, RetryError
from bson.dbref import DBRef
from pymongo import IndexModel
import pymongo
from beanie import Document, Link, BackLink, PydanticObjectId
from beanie.operators import In
from pydantic import Field, BaseModel

from trimit.app.app import TRIMIT_VIDEO_S3_CDN_BUCKET
from trimit.models.backend_models import (
    CutTranscriptLinearWorkflowStepOutput,
    StepKey,
    Message,
    ExportableStepWrapper,
    PartialLLMOutput,
    FinalLLMOutput,
    PartialBackendOutput,
    ExportResults,
)
from trimit.utils.model_utils import (
    filename_from_hash,
    get_upload_folder,
    get_scene_folder,
    get_frame_folder,
    get_audio_folder,
    get_dynamic_state_key,
    get_step_substep_names_from_dynamic_state_key,
    partial_transcription_indexes,
    partial_transcription_words,
    get_partial_transcription,
    scene_name_from_video,
    construct_retry_task_with_exception_type,
    map_export_result_to_asset_path,
    form_cdn_url_from_path,
)
from trimit.utils.namegen import project_namegen

from trimit.utils.fs_utils import async_copy_to_s3


class StepNotYetReachedError(ValueError):
    pass


class DocumentWithSaveRetry(Document):
    #  @retry(
    #  stop=stop_after_attempt(5), wait=wait_random_exponential(multiplier=1, max=60)
    #  )
    async def save_with_retry(self, session=None):
        try:
            await self.save(session=session)
        except Exception as e:
            print(f"Error saving document: {e}")
            try:
                await self.insert(session=session)
            except Exception as e:
                print(f"Error inserting document: {e}")
                raise
        #  try:
        #  await self.save_changes()
        #  print("Document saved changes")
        #  except (DocumentWasNotSaved, StateNotSaved):
        #  print("Document was not saved. Retrying...")
        #  await self.save()


class PathMixin:
    upload_datetime: datetime.datetime

    def path(self, volume_dir):
        upload_folder = get_upload_folder(
            volume_dir, self.user_email, self.upload_datetime
        )
        file_path = Path(upload_folder) / self.filename
        return str(file_path)

    @property
    def user_email(self) -> str: ...

    @property
    def filename(self) -> str: ...

    async def asset_path_with_copy(self, volume_dir, asset_dir):
        retry_task = construct_retry_task_with_exception_type(FileNotFoundError)
        volume_path = self.path(volume_dir)
        asset_path = self.path("")
        try:
            await retry_task(
                async_copy_to_s3, TRIMIT_VIDEO_S3_CDN_BUCKET, volume_path, asset_path
            )
        except RetryError:
            raise FileNotFoundError(f"Failed to copy video to asset: {volume_path}")
        return form_cdn_url_from_path(asset_path)

    async def asset_path_with_fallback(self, volume_dir, asset_dir):
        try:
            return await self.asset_path_with_copy(volume_dir, asset_dir)
        except FileNotFoundError:
            return self.path(volume_dir)


class User(DocumentWithSaveRetry):
    email: str
    password: Optional[str] = Field(default=None, min_length=8)
    name: Optional[str] = None
    authorized_with_google: bool = False

    class Settings:
        name = "User"
        indexes = [IndexModel([("email", pymongo.ASCENDING)], unique=True)]

    def __repr__(self):
        return f"User(email={self.email}, name={self.name})"

    @property
    async def videos(self):
        return await Video.find(Video.user == self).to_list()

    @property
    async def scenes(self):
        return await Scene.find(Scene.user == self).to_list()

    @property
    async def frames(self):
        return await Frame.find(Frame.user == self).to_list()


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


class Video(DocumentWithSaveRetry, PathMixin):
    md5_hash: str
    simple_name: Optional[str] = None
    ext: str
    upload_datetime: datetime.datetime
    details: Optional[VideoMetadata] = None
    high_res_user_file_path: str
    high_res_user_file_hash: Optional[str] = None
    high_res_local_file_path: Optional[str] = None
    transcription: Optional[dict] = None
    transcription_text: Optional[str] = None
    video_llava_description: Optional[str] = None
    user: User
    summary: Optional[str] = None
    speakers_in_frame: Optional[list[str]] = None
    title: Optional[str] = None

    class Settings:
        name = "Video"
        indexes = [
            IndexModel([("md5_hash", pymongo.ASCENDING)]),
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
            [("user", pymongo.ASCENDING), ("title", pymongo.ASCENDING)],
            [
                ("user", pymongo.ASCENDING),
                ("project", pymongo.ASCENDING),
                ("title", pymongo.ASCENDING),
            ],
            [("user", pymongo.ASCENDING)],
            [
                ("user", pymongo.ASCENDING),
                ("project", pymongo.ASCENDING),
                ("md5_hash", pymongo.ASCENDING),
            ],
            [("user", pymongo.ASCENDING), ("md5_hash", pymongo.ASCENDING)],
            [("user", pymongo.ASCENDING), ("upload_datetime", pymongo.ASCENDING)],
            [("user", pymongo.ASCENDING), ("recorded_date", pymongo.ASCENDING)],
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
    async def from_user_email(
        cls,
        user_email: str,
        md5_hash: str,
        overwrite: bool = False,
        session=None,
        **kwargs,
    ) -> "Video":
        user = await User.find_one(User.email == user_email, session=session)
        if user is None:
            raise ValueError(f"User not found: {user_email}")
        existing_video = await Video.find_one(
            Video.md5_hash == md5_hash, Video.user.email == user_email, session=session
        )
        if existing_video is not None:
            if not overwrite:
                return existing_video
            else:
                await existing_video.delete(session=session)
        simple_name = await Video.gen_simple_name(user_email)
        video = Video(md5_hash=md5_hash, simple_name=simple_name, user=user, **kwargs)
        await video.save_with_retry(session=session)
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

    @property
    def filename(self):
        return filename_from_hash(self.md5_hash, self.ext)

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


class VideoHighResPathProjection(BaseModel):
    md5_hash: str
    high_res_user_file_path: str


class UserEmailProjection(BaseModel):
    _id: PydanticObjectId
    email: str


class VideoFileProjection(BaseModel, PathMixin):
    md5_hash: str
    user: UserEmailProjection
    upload_datetime: datetime.datetime
    high_res_user_file_path: str
    ext: str
    title: str | None = None

    @property
    def filename(self):
        return filename_from_hash(self.md5_hash, self.ext)

    @property
    def user_email(self):
        return self.user.email


class UploadedVideoProjection(VideoFileProjection):
    details: Optional[dict[str, Any] | None] = None

    @property
    def duration(self):
        return self.details.get("duration") if self.details else 0


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
    name: str
    simple_name: Optional[str] = None
    start_frame: int
    end_frame: int
    start: float
    end: float
    video: Video
    user: User
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
        indexes = [
            IndexModel(
                [("name", pymongo.ASCENDING), ("user.email", pymongo.ASCENDING)],
                unique=True,
            ),
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
        return filename_from_hash(
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

    @classmethod
    async def from_video(
        cls,
        video: "Video",
        start_frame: int | None = None,
        end_frame: int | None = None,
        start: float | None = None,
        end: float | None = None,
        save: bool = True,
        check_existing: bool = True,
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
            assert start_frame is not None
            start = start_frame / frame_rate
        if start_frame is None:
            start_frame = int(start * frame_rate)
        if end is None:
            assert end_frame is not None
            end = end_frame / frame_rate
        if end_frame is None:
            end_frame = int(end * frame_rate)
        name = scene_name_from_video(video, start_frame, end_frame)
        if check_existing:
            existing_scene = await Scene.find_one(
                Scene.name == name, Scene.user.email == video.user.email
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


class Frame(DocumentWithSaveRetry):
    name: str
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
        return filename_from_hash(
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


class Project(DocumentWithSaveRetry):
    user: Link[User]
    name: str

    class Settings:
        name = "Project"
        indexes = [
            IndexModel(
                [("user.email", pymongo.ASCENDING), ("name", pymongo.ASCENDING)],
                unique=True,
            )
        ]

    @classmethod
    async def from_user_email(cls, user_email: str, name: str | None):
        user = await User.find_one(User.email == user_email)
        if user is None:
            raise ValueError(f"User not found: {user_email}")
        if not name:
            name = project_namegen()
            print("not name, generated", name)
        if not user_email:
            raise ValueError("User email not provided")
        existing_project = await Project.find_one(
            Project.user.email == user_email, Project.name == name, fetch_links=True
        )
        if existing_project is not None:
            return existing_project
        project = cls(user=user, name=name, workflow_states=[])
        await project.insert()
        return project

    @property
    async def get_workflows(self):
        from trimit.backend.cut_transcript import CutTranscriptLinearWorkflow

        await self.fetch_all_links()
        return [
            CutTranscriptLinearWorkflow(state=state) for state in self.workflow_states
        ]


class CutTranscriptLinearWorkflowStaticState(DocumentWithSaveRetry):
    user: Link[User]
    timeline_name: str
    project: Link[Project] | None = None
    video: Link[Video]
    volume_dir: str
    output_folder: str
    length_seconds: int
    video_type: str = "Customer testimonial"

    nstages: int = 2
    first_pass_length: int = 6 * 60
    max_partial_transcript_words: int = 800
    max_word_extra_threshold: int = 50
    clip_extra_trim_seconds: float = 0.1
    use_agent_output_cache: bool = True
    max_iterations: int = 6
    ask_user_for_feedback_every_iteration: bool = False
    max_total_soundbites: int = 15
    num_speaker_tagging_samples: int = 3
    export_transcript_text: bool = True
    export_transcript: bool = True
    export_soundbites: bool = True
    export_soundbites_text: bool = True
    export_timeline: bool = True
    export_video: bool = True
    export_speaker_tagging: bool = True

    def create_object_id(self):
        ref_keys = ["user", "video", "project"]
        dumped = self.model_dump(exclude=set(ref_keys))
        for key in ref_keys:
            ref = getattr(self, key).to_ref()
            if isinstance(ref, DBRef):
                ref = ref.id
            dumped[key] = str(ref)

        self_as_str = json.dumps(dumped, sort_keys=True)
        md5_hash = hashlib.md5(self_as_str.encode()).hexdigest()
        return PydanticObjectId(md5_hash[:24])


class FrontendWorkflowStaticState(CutTranscriptLinearWorkflowStaticState):
    user: Link[User] | None = None
    video: Link[Video] | None = None
    project: Link[Project] | None = None
    user_id: PydanticObjectId
    video_id: PydanticObjectId
    project_id: PydanticObjectId | None = None
    user_email: str
    video_hash: str
    project_name: str | None = None

    @classmethod
    def from_backend_static_state(cls, backend_state):
        return cls(
            video_id=backend_state.video.id,
            user_id=backend_state.user.id,
            project_id=backend_state.project.id if backend_state.project else None,
            video_hash=backend_state.video.md5_hash,
            user_email=backend_state.user.email,
            project_name=backend_state.project.name if backend_state.project else None,
            **backend_state.model_dump(exclude=["video", "user", "project"]),
        )


class StepOrderMixin(BaseModel):
    _id: PydanticObjectId
    static_state: CutTranscriptLinearWorkflowStaticState
    dynamic_state_step_order: list[StepKey] = []
    dynamic_state_retries: dict = {}

    @property
    def id(self):
        if not hasattr(self, "_id"):
            self._id = self.static_state.create_object_id()
        return self._id

    def get_current_step_key_atomic(self):
        if len(self.dynamic_state_step_order):
            current_step = self.dynamic_state_step_order[-1]
            current_substep = current_step.substeps[-1]
            return get_dynamic_state_key(current_step.name, current_substep)
        return None

    def get_step_key_before_end(self):
        if len(self.dynamic_state_step_order) < 1:
            return None
        last_step = self.dynamic_state_step_order[-1]
        if last_step.name == "end":
            if len(self.dynamic_state_step_order) < 2:
                return None
            last_step = self.dynamic_state_step_order[-2]
        return get_dynamic_state_key(last_step.name, last_step.substeps[-1])

    @property
    def length_seconds(self):
        return self.static_state.length_seconds

    @property
    def video(self):
        assert isinstance(self.static_state.video, Video), "Video not fetched from link"
        return self.static_state.video

    @property
    def user(self):
        assert isinstance(self.static_state.user, User), "User not fetched from link"
        return self.static_state.user

    @property
    def project(self):
        assert isinstance(
            self.static_state.project, Project
        ), "Project not fetched from link"
        return self.static_state.project

    @property
    def timeline_name(self):
        return self.static_state.timeline_name

    @property
    def first_pass_length(self):
        return self.static_state.first_pass_length

    @property
    def video_type(self):
        return self.static_state.video_type

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
    def ask_user_for_feedback_every_iteration(self):
        try:
            return self.static_state.ask_user_for_feedback_every_iteration
        except Exception as e:
            print(f"Error getting ask_user_for_feedback_every_iteration: {e}")
            return False

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
    def export_speaker_tagging(self):
        return self.static_state.export_speaker_tagging

    @property
    def num_speaker_tagging_samples(self):
        return self.static_state.num_speaker_tagging_samples

    @property
    def clip_extra_trim_seconds(self):
        return self.static_state.clip_extra_trim_seconds


class CutTranscriptLinearWorkflowState(DocumentWithSaveRetry, StepOrderMixin):
    id: PydanticObjectId | None = None
    on_screen_transcript_state: dict | None = None
    original_soundbites_state: dict | None = None
    on_screen_speakers: list[str] | None = None
    story: str | None = None
    raw_transcript_state: dict | None = None
    current_transcript_state: dict | None = None
    current_soundbites_state: dict | None = None
    run_output_dir: str | None = None
    soundbites_output_dir: str | None = None
    output_files: dict[str, str] | None = None
    dynamic_state_retries: dict[str, bool] = {}
    dynamic_state_step_order: list[StepKey] = []
    outputs: dict[str, list[CutTranscriptLinearWorkflowStepOutput]] = {}
    prev_export_flags: dict[str, bool] | None = None

    class Settings:
        name = "CutTranscriptLinearWorkflowState"
        indexes = [
            IndexModel(
                [
                    ("static_state.video.md5_hash", pymongo.ASCENDING),
                    ("static_state.user.email", pymongo.ASCENDING),
                    ("static_state.timeline_name", pymongo.ASCENDING),
                ],
                unique=True,
            ),
            IndexModel(
                [
                    ("static_state.user.email", pymongo.ASCENDING),
                    ("static_state.video_type", pymongo.ASCENDING),
                    ("static_state.video.length_seconds", pymongo.ASCENDING),
                    ("static_state.video.md5_hash", pymongo.ASCENDING),
                    ("static_state.timeline_name", pymongo.ASCENDING),
                ],
                unique=True,
            ),
            IndexModel(
                [
                    ("static_state.video.md5_hash", pymongo.ASCENDING),
                    ("static_state.user.email", pymongo.ASCENDING),
                    ("static_state.timeline_name", pymongo.ASCENDING),
                    ("static_state.project", pymongo.ASCENDING),
                ],
                unique=True,
            ),
            IndexModel(
                [
                    ("static_state.video.md5_hash", pymongo.ASCENDING),
                    ("static_state.user.email", pymongo.ASCENDING),
                    ("static_state.timeline_name", pymongo.ASCENDING),
                    ("static_state.project.name", pymongo.ASCENDING),
                ],
                unique=True,
            ),
            IndexModel([("static_state.project.name", pymongo.ASCENDING)]),
        ]

    def __init__(self, **data):
        super().__init__(**data)
        if not self.id:
            self.id = self.static_state.create_object_id()

    def merge(self, other: "CutTranscriptLinearWorkflowState"):
        if self.id != other.id:
            raise ValueError("Cannot merge states with different ids")
        fields_to_replace = [
            "on_screen_transcript_state",
            "original_soundbites_state",
            "on_screen_speakers",
            "story",
            "raw_transcript_state",
            "current_transcript_state",
            "current_soundbites_state",
            "run_output_dir",
            "soundbites_output_dir",
            "prev_export_flags",
        ]
        fields_to_merge = ["outputs", "output_files"]
        # TODO figure out what to do with dynamic_state_step_order and dynamic_state_retries
        for field_name in fields_to_replace:
            field_val = getattr(self, field_name)
            other_field_val = getattr(other, field_name)
            if other_field_val is not None:
                setattr(self, field_name, other_field_val)
        for field_name in fields_to_merge:
            field_val = getattr(self, field_name)
            other_field_val = getattr(other, field_name)
            if isinstance(field_val, dict) and isinstance(other_field_val, dict):
                setattr(self, field_name, {**field_val, **other_field_val})

    def set_state_val(self, key, value):
        if key not in self.model_dump().keys():
            raise ValueError(
                f"Attempt to set unknown key {key} with value {value} to state"
            )
        setattr(self, key, value)

    @property
    def export_flags(self):
        keys = [
            "export_transcript_text",
            "export_transcript",
            "export_soundbites",
            "export_soundbites_text",
            "export_timeline",
            "export_video",
            "export_speaker_tagging",
        ]
        return {k: getattr(self.static_state, k) for k in keys}

    async def set_all_export_to_false(self):
        for key, val in self.export_flags.items():
            setattr(self.static_state, key, False)
            if self.prev_export_flags is None:
                self.prev_export_flags = {}
            self.prev_export_flags[key] = val
        await self.save()

    async def set_all_export_to_true(self):
        for key, val in self.export_flags.items():
            setattr(self.static_state, key, True)
            if self.prev_export_flags is None:
                self.prev_export_flags = {}
            self.prev_export_flags[key] = val
        await self.save()

    async def revert_export_flags(self):
        if not self.prev_export_flags:
            print("prev export flags not set, returning")
            return
        await self.set_export_flags(**self.prev_export_flags)

    async def set_export_flags(self, **export_flags):
        for key in self.export_flags:
            val = export_flags.get(key)
            if not isinstance(val, bool):
                raise ValueError(f"export flag value must be boolean: {key}={val}")
            setattr(self.static_state, key, val)
        await self.save()

    @property
    def user_prompt(self):
        if len(self.dynamic_state_step_order) == 0:
            return None
        key = get_dynamic_state_key("generate_story", "generate_story")
        if len(self.outputs.get(key, [])) == 0:
            return None
        step_inputs = self.outputs[key][-1].step_inputs
        if step_inputs is not None:
            return step_inputs.user_prompt
        return ""

    def stage_output_dir(self, stage):
        if self.run_output_dir is None:
            raise ValueError("run_output_dir is None")
        return os.path.join(self.run_output_dir, f"stage_{stage}")

    def step_output_dir(self, step_name: str, substep_name: str):
        if self.run_output_dir is None:
            raise ValueError("run_output_dir is None")
        output_dir = Path(self.run_output_dir) / step_name / substep_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)

    @property
    def soundbites_video_dir(self):
        if self.run_output_dir is None:
            raise ValueError("run_output_dir is None")
        output_dir = Path(self.run_output_dir) / "soundbites_videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)

    @property
    def soundbites_timeline_dir(self):
        if self.run_output_dir is None:
            raise ValueError("run_output_dir is None")
        output_dir = Path(self.run_output_dir) / "soundbites_timeline"
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)

    async def revert_step_to(self, step_name: str, substep_name: str, save=True):
        if not self._already_in_dynamic_state_step_order(step_name, substep_name):
            raise StepNotYetReachedError(
                f"have not reached step {step_name}.{substep_name} yet"
            )
        state_key = get_dynamic_state_key(step_name, substep_name)
        latest_state_key = self.get_latest_dynamic_key()
        while latest_state_key != state_key:
            await self._revert_step()
            latest_state_key = self.get_latest_dynamic_key()
        if save:
            await self.save()

    async def revert_step(self):
        # TODO add back in option to revert just the latest retry
        if len(self.dynamic_state_step_order) == 0:
            print("no steps to revert")
            return
        await self._revert_step()
        await self.save()

    async def _revert_step(self):
        if len(self.dynamic_state_step_order) == 0:
            return
        last_step_wrapper = self.dynamic_state_step_order[-1]
        last_substep = last_step_wrapper.substeps.pop(-1)
        dynamic_state_key = get_dynamic_state_key(last_step_wrapper.name, last_substep)
        print(f"last step: {dynamic_state_key}")
        del self.outputs[dynamic_state_key]
        self.dynamic_state_retries = {
            k: v
            for k, v in self.dynamic_state_retries.items()
            if k != dynamic_state_key
        }
        if len(last_step_wrapper.substeps) == 0:
            self.dynamic_state_step_order.pop()
        if len(self.dynamic_state_step_order) == 0:
            return
        new_last_step_wrapper = self.dynamic_state_step_order[-1]
        new_last_substep = new_last_step_wrapper.substeps[-1]
        new_dynamic_state_key = get_dynamic_state_key(
            new_last_step_wrapper.name, new_last_substep
        )
        print("new last step", new_dynamic_state_key)
        new_last_step_outputs = self.outputs[new_dynamic_state_key][-1]
        if "current_transcript_state" in new_last_step_outputs:
            print("revert current transcript state")
            self.current_transcript_state = new_last_step_outputs[
                "current_transcript_state"
            ]
        if "current_soundbites_state" in new_last_step_outputs:
            print("revert current soundbites state")
            self.current_soundbites_state = new_last_step_outputs[
                "current_soundbites_state"
            ]

    async def restart(self, save=True):
        # TODO understand save_changes and why it doesnt work here
        self.on_screen_speakers = None
        self.run_output_dir = None
        self.on_screen_transcript_state = None
        self.current_soundbites_state = None
        self.story = None
        self.raw_transcript_state = None
        self.current_transcript_state = None
        self.outputs = {}
        self.dynamic_state_retries = {}
        self.dynamic_state_step_order = []
        if save:
            await self.save()

    @classmethod
    async def find_or_create_from_video_hash(
        cls,
        video_hash,
        user_email,
        timeline_name,
        volume_dir,
        output_folder,
        length_seconds,
        project=None,
        project_name=None,
        **params,
    ):
        video = await Video.find_one(
            Video.md5_hash == video_hash, Video.user.email == user_email
        )
        if video is None:
            raise ValueError(
                f"Video with hash {video_hash} and user_email {user_email} not found"
            )
        if project is None and project_name is not None:
            project = await Project.from_user_email(
                user_email=user_email, name=project_name
            )
        return await cls.find_or_create(
            video=video,
            timeline_name=timeline_name,
            volume_dir=volume_dir,
            output_folder=output_folder,
            length_seconds=length_seconds,
            project=project,
            **params,
        )

    @classmethod
    async def recreate_from_video_hash(
        cls,
        video_hash,
        user_email,
        timeline_name,
        volume_dir,
        output_folder,
        length_seconds,
        project=None,
        project_name=None,
        **params,
    ):
        video = await Video.find_one(
            Video.md5_hash == video_hash, Video.user.email == user_email
        )
        if video is None:
            raise ValueError(
                f"Video with hash {video_hash} and user_email {user_email} not found"
            )

        if project is None and project_name is not None:
            project = await Project.from_user_email(
                user_email=user_email, name=project_name
            )
        return await cls.recreate(
            video=video,
            timeline_name=timeline_name,
            volume_dir=volume_dir,
            output_folder=output_folder,
            length_seconds=length_seconds,
            project=project,
            **params,
        )

    @classmethod
    async def find_or_create(
        cls,
        video,
        timeline_name,
        volume_dir,
        output_folder,
        length_seconds,
        project,
        **params,
    ):
        print("find_or_create")
        static_state = CutTranscriptLinearWorkflowStaticState(
            user=video.user,
            timeline_name=timeline_name,
            video=video,
            volume_dir=volume_dir,
            output_folder=output_folder,
            length_seconds=length_seconds,
            project=project,
            **params,
        )
        obj = cls(static_state=static_state)
        existing = await cls.find_one(
            CutTranscriptLinearWorkflowState.static_state.user.email
            == video.user.email,
            CutTranscriptLinearWorkflowState.static_state.video.md5_hash
            == video.md5_hash,
            CutTranscriptLinearWorkflowState.static_state.project.name == project.name,
            CutTranscriptLinearWorkflowState.static_state.timeline_name
            == timeline_name,
        )

        if existing is not None:
            return existing
        print("did not find, creating")
        await obj.insert()
        return obj

    @classmethod
    async def recreate(
        cls,
        video,
        timeline_name,
        volume_dir,
        output_folder,
        length_seconds,
        project=None,
        **params,
    ):
        print("recreate")
        static_state = CutTranscriptLinearWorkflowStaticState(
            user=video.user,
            timeline_name=timeline_name,
            video=video,
            volume_dir=volume_dir,
            output_folder=output_folder,
            length_seconds=length_seconds,
            project=project,
            **params,
        )
        obj = cls(static_state=static_state)
        existing = await cls.find_one(
            CutTranscriptLinearWorkflowState.static_state.user.email
            == video.user.email,
            CutTranscriptLinearWorkflowState.static_state.video.md5_hash
            == video.md5_hash,
            CutTranscriptLinearWorkflowState.static_state.timeline_name
            == timeline_name,
        )

        if existing is not None:
            print("deleting")
            await existing.delete()
            await obj.insert()
        else:
            await obj.save()
        print("recreated")
        return obj

    def get_latest_dynamic_key(self):
        if len(self.dynamic_state_step_order) == 0:
            return None
        step = self.dynamic_state_step_order[-1]
        step_name = step.name
        if step_name == "end":
            step = self.dynamic_state_step_order[-2]
            step_name = step.name
        substep_name = step.substeps[-1]
        return get_dynamic_state_key(step_name, substep_name)

    def get_latest_dynamic_key_from(self, step_name, substep_name):
        key = get_dynamic_state_key(step_name, substep_name)
        for step_key in self.dynamic_state_step_order[::-1]:
            if step_key != step_name:
                continue
            for substep_key in step_key.substeps:
                if substep_key == substep_name:
                    key = get_dynamic_state_key(step_key, substep_key)
                    break
            if key:
                break
        return key

    def get_new_dynamic_key_with_retry_num(self, step_name, substep_name):
        dynamic_key = get_dynamic_state_key(step_name, substep_name)
        current_results = self.outputs.get(dynamic_key, [])
        retry_num = len(current_results)
        return dynamic_key, retry_num

    @retry(stop=stop_after_attempt(10), wait=wait_fixed(0.1) + wait_random(0, 1))
    async def set_current_step_output_atomic(
        self,
        dynamic_key,
        results: "CutTranscriptLinearWorkflowStepOutput",
        save_to_db: bool = True,
        use_session: bool = True,
    ):
        from trimit.models.backend_models import CutTranscriptLinearWorkflowStepOutput
        from trimit.models import start_transaction

        step_name, substep_name = get_step_substep_names_from_dynamic_state_key(
            dynamic_key
        )
        try:
            retry = results.retry
        except AttributeError:
            retry = results.get("retry", False)

        retry_num = results.retry_num

        # This copy is necessary because beanie overwrite self's attributes when we call .get()
        copied_self_dict = copy.deepcopy(self).model_dump()

        async def update_state(session):
            updated_self = None
            if save_to_db:
                updated_self = await CutTranscriptLinearWorkflowState.find_one(
                    CutTranscriptLinearWorkflowState.id == self.id, session=session
                )
            copied_self = CutTranscriptLinearWorkflowState(**copied_self_dict)
            for k, v in copied_self.outputs.items():
                if hasattr(v, "__iter__"):
                    for _retry_num, output in enumerate(v):
                        if isinstance(output, dict):
                            try:
                                copied_self.outputs[k][_retry_num] = (
                                    CutTranscriptLinearWorkflowStepOutput(**output)
                                )
                            except:
                                continue

            if updated_self is None:
                updated_self = copied_self
            updated_results = results
            new_step_outputs_list = updated_self.outputs.get(dynamic_key, [])
            existing_length = len(new_step_outputs_list)
            #  if retry_num == 1:
            #  breakpoint()
            if retry_num >= existing_length:
                new_step_outputs_list.extend(
                    [None for _ in range(1 + retry_num - existing_length)]
                )
            if (
                existing_length > retry_num
                and updated_self.outputs[dynamic_key][retry_num]
            ):
                #  if dynamic_key =="stage_1_generate_transcript.modify_transcript_holistically":
                #  breakpoint()
                updated_results = updated_self.outputs[dynamic_key][retry_num]
                if not isinstance(
                    updated_results, CutTranscriptLinearWorkflowStepOutput
                ):
                    assert isinstance(updated_results, dict)
                    updated_results = CutTranscriptLinearWorkflowStepOutput(
                        **updated_results
                    )
                updated_results.merge(results)
                new_step_outputs_list[retry_num] = updated_results
            else:
                new_step_outputs_list[retry_num] = results
            updated_self.outputs[dynamic_key] = new_step_outputs_list
            copied_self.outputs = {**copied_self.outputs, **updated_self.outputs}

            if dynamic_key not in updated_self.dynamic_state_retries:
                updated_self.dynamic_state_retries[dynamic_key] = retry
                copied_self.dynamic_state_retries[dynamic_key] = retry
            updated_self.add_to_dynamic_state_step_order(step_name, substep_name)
            copied_self.dynamic_state_step_order = updated_self.dynamic_state_step_order

            updated_self.merge(copied_self)
            if save_to_db:
                await updated_self.save(session=session)
            for field in self.model_fields:
                setattr(self, field, getattr(updated_self, field))

        if use_session:
            async with start_transaction() as session:
                await update_state(session)
        else:
            await update_state(None)

    def add_to_dynamic_state_step_order(self, step_name, substep_name):
        if len(self.dynamic_state_step_order) == 0:
            self.dynamic_state_step_order.append(StepKey(name=step_name, substeps=[]))
        if self._already_in_dynamic_state_step_order(step_name, substep_name):
            return

        current_step = self.dynamic_state_step_order[-1]
        if current_step.name != step_name:
            self.dynamic_state_step_order.append(StepKey(name=step_name, substeps=[]))
        self.dynamic_state_step_order[-1].substeps.append(substep_name)

    def _already_in_dynamic_state_step_order(self, step_name, substep_name):
        for existing_step in self.dynamic_state_step_order:
            for existing_substep in existing_step.substeps:
                if existing_step.name == step_name and existing_substep == substep_name:
                    return True
        return False


class FrontendWorkflowProjection(BaseModel):
    id: str
    timeline_name: str
    user_email: str
    video_hash: str
    video_filename: str
    length_seconds: int
    nstages: int
    project_name: str | None = None
    project_id: str | None = None
    video_type: str | None = None

    class Settings:
        projection = {
            "id": {"$toString": "$_id"},
            "video_hash": "$static_state.video.md5_hash",
            "video_filename": "$static_state.video.high_res_user_file_path",
            "video_type": "$static_state.video_type",
            "user_email": "$static_state.user.email",
            "timeline_name": "$static_state.timeline_name",
            "length_seconds": "$static_state.length_seconds",
            "nstages": "$static_state.nstages",
            "project_name": "$static_state.project.name",
            "project_id": {"$toString": "$static_state.project._id"},
        }


class FrontendStepOutput(CutTranscriptLinearWorkflowStepOutput):
    full_conversation: list[Message] = Field(default_factory=list)

    @classmethod
    def from_backend_outputs(cls, backend_outputs):
        return cls(
            full_conversation=[m for o in backend_outputs for m in o.conversation],
            **backend_outputs[-1].model_dump(),
        )


class FrontendWorkflowState(CutTranscriptLinearWorkflowState):
    outputs: list[FrontendStepOutput] = Field(
        [],
        description="list of step outputs, which have already ran, and that yield to the user.",
    )
    all_steps: list[ExportableStepWrapper]
    static_state: FrontendWorkflowStaticState
    mapped_export_result: list[ExportResults] = Field(
        [],
        description="export result file paths in web server's asset directory, available to be served",
    )

    @classmethod
    async def from_workflow(cls, workflow, volume_dir):
        backend_state = workflow.state
        frontend_outputs = []
        mapped_export_result_tasks = []
        for step in backend_state.dynamic_state_step_order:
            substep = step.substeps[-1]
            step_key = get_dynamic_state_key(step.name, substep)
            outputs = backend_state.outputs[step_key]
            frontend_output = FrontendStepOutput.from_backend_outputs(outputs)
            frontend_outputs.append(frontend_output)
            mapped_export_result_tasks.append(
                map_export_result_to_asset_path(
                    frontend_output.export_result, volume_dir
                )
            )
        mapped_export_results = await asyncio.gather(*mapped_export_result_tasks)

        return cls(
            outputs=frontend_outputs,
            all_steps=workflow.steps.to_exportable(),
            static_state=FrontendWorkflowStaticState.from_backend_static_state(
                backend_state.static_state
            ),
            mapped_export_result=mapped_export_results,
            **backend_state.model_dump(exclude=["static_state", "outputs"]),
        )


class CutTranscriptLinearWorkflowStreamingOutput(BaseModel):
    workflow_id: str
    partial_llm_output: PartialLLMOutput | None = Field(
        None, description="Chunk of output from the LLM"
    )
    final_llm_output: FinalLLMOutput | None = Field(
        None, description="Full output from the LLM, not currently send to frontend"
    )
    partial_backend_output: PartialBackendOutput | None = Field(
        None, description="Text output with metadata from the backend"
    )
    partial_step_output: CutTranscriptLinearWorkflowStepOutput | None = Field(
        None,
        description="An output of an intermediary substep that did not require user feedback",
    )
    final_state: FrontendWorkflowState | None = Field(
        None,
        description="The full final state, same as calling /get_latest_state. This will always be the last item returned.",
    )


class TimelineClip(BaseModel):
    scene: Link[Scene]


class TimelineOutput(BaseModel):
    """Ordered list of clips to be concatenated into a timeline, and later made into a video"""

    timeline: list[TimelineClip]

    def merge(self, other: "TimelineOutput"):
        return TimelineOutput(timeline=self.timeline + other.timeline)


ALL_MODELS = [User, Video, Scene, Frame, CutTranscriptLinearWorkflowState, Project]
