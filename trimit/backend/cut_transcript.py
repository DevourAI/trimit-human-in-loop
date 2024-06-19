import aiostream
import asyncio
import json
from pathlib import Path
import datetime
import os
from collections import defaultdict
import random

from tqdm.asyncio import tqdm as tqdm_async
from beanie import Link, PydanticObjectId
from schema import Schema
from griptape.utils import PromptStack
from bson.dbref import DBRef
from modal import is_local

import trimit.utils.conf
from trimit.backend.utils import (
    match_output_to_actual_transcript_fast,
    remove_boundary_tags,
    get_soundbite_rule,
    remove_off_screen_speakers,
    desired_words_from_length,
    remove_soundbites,
    parse_partials_to_redo_from_agent_output,
    parse_relevant_user_feedback_list_from_agent_output,
    add_complete_format,
    remove_retry_suffix,
    Message,
    parse_stage_num_from_step_name,
    stage_key_for_step_name,
    get_agent_output_modal_or_local,
    export_results_wrapper,
)
from trimit.export.utils import get_new_integer_file_name_in_dir
from trimit.utils.async_utils import async_passthrough_gen
from trimit.utils.prompt_engineering import (
    load_prompt_template_as_string,
    parse_prompt_template,
    render_jinja_string,
)
from trimit.utils.model_utils import (
    get_generated_video_folder,
    get_dynamic_state_key,
    get_step_substep_names_from_dynamic_state_key,
    load_step_order,
)
from trimit.models import (
    Video,
    User,
    CutTranscriptLinearWorkflowState,
    StepOrderMixin,
    maybe_init_mongo,
    CutTranscriptLinearWorkflowStaticState,
)
from trimit.export import (
    create_cut_video_from_transcript,
    create_fcp_7_xml_from_single_video_transcript,
    save_transcript_to_disk,
    save_soundbites_videos_to_disk,
)
from trimit.backend.models import (
    PartialBackendOutput,
    PartialLLMOutput,
    FinalLLMOutput,
    Transcript,
    TranscriptChunk,
    SoundbitesChunk,
    Soundbites,
    PartialFeedback,
    CutTranscriptLinearWorkflowStepInput,
    CutTranscriptLinearWorkflowStepOutput,
    StepWrapper,
    CurrentStepInfo,
    CutTranscriptLinearWorkflowStepResults,
    Steps,
)


END_STEP_NAME = "end"
VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"]


class CutTranscriptLinearWorkflow:
    #### INITIALIZERS ####
    def __init__(
        self,
        state: CutTranscriptLinearWorkflowState | None = None,
        step_order: StepOrderMixin | None = None,
    ):
        if state is None and step_order is None:
            raise ValueError("Either state or step_order must be provided")
        self.state = state
        if state is not None:
            self.step_order = state
        else:
            assert step_order is not None
            self.step_order: StepOrderMixin = step_order

    @classmethod
    async def id_from_params(
        cls,
        timeline_name: str,
        volume_dir: str,
        output_folder: str,
        length_seconds: int,
        user_email: str | None = None,
        video_hash: str | None = None,
        user_id: str | PydanticObjectId | None = None,
        video_id: str | PydanticObjectId | None = None,
        **cut_transcript_linear_workflow_static_state_params,
    ):
        await maybe_init_mongo()
        if not video_id:
            if user_email is None:
                raise ValueError(
                    "user_email must be provided if video_id is not provided"
                )
            if video_hash is None:
                raise ValueError(
                    "video_hash must be provided if video_id is not provided"
                )
            video = await Video.find_one(
                Video.md5_hash == video_hash,
                Video.user.email == user_email,
                fetch_links=False,
            )
            if video is None:
                raise ValueError(
                    f"Video with hash {video_hash} and user_email {user_email} not found"
                )
            video_id = video.id
            user_id = video.user.id
        assert user_id is not None
        assert video_id is not None
        if isinstance(user_id, str):
            user_id = PydanticObjectId(user_id)
        if isinstance(video_id, str):
            video_id = PydanticObjectId(video_id)
        user_collection_name = User.get_collection_name()
        assert isinstance(user_collection_name, str)
        video_collection_name = Video.get_collection_name()
        assert isinstance(video_collection_name, str)
        user_ref = DBRef(id=user_id, collection=user_collection_name)
        video_ref = DBRef(id=video_id, collection=video_collection_name)
        state = CutTranscriptLinearWorkflowStaticState(
            user=Link(user_ref, User),
            video=Link(video_ref, Video),
            timeline_name=timeline_name,
            volume_dir=volume_dir,
            output_folder=output_folder,
            length_seconds=length_seconds,
            **cut_transcript_linear_workflow_static_state_params,
        )
        obj_id = state.create_object_id()
        return obj_id

    @classmethod
    async def from_video_id(
        cls,
        video_id: str,
        timeline_name: str,
        length_seconds: int,
        output_folder: str,
        volume_dir: str,
        video_local_upload_date: datetime.datetime | None = None,
        new_state: bool = False,
        **init_kwargs,
    ):
        video = await Video.get(video_id)
        if video is None:
            raise ValueError(f"Video with id {video_id} not found")
        if video_local_upload_date:
            video.upload_datetime = video_local_upload_date
        return await cls.from_video(
            video=video,
            timeline_name=timeline_name,
            length_seconds=length_seconds,
            output_folder=output_folder,
            volume_dir=volume_dir,
            new_state=new_state,
            **init_kwargs,
        )

    @classmethod
    async def from_video_hash(
        cls,
        video_hash: str,
        user_email: str,
        timeline_name: str,
        length_seconds: int,
        output_folder: str,
        volume_dir: str,
        video_local_upload_date: datetime.datetime | None = None,
        new_state: bool = False,
        **init_kwargs,
    ):
        await maybe_init_mongo()
        video = await Video.find_one(
            Video.md5_hash == video_hash, Video.user.email == user_email
        )
        if video is None:
            raise ValueError(f"Video with hash {video_hash} not found")
        if video_local_upload_date:
            video.upload_datetime = video_local_upload_date
        return await cls.from_video(
            video=video,
            timeline_name=timeline_name,
            length_seconds=length_seconds,
            output_folder=output_folder,
            volume_dir=volume_dir,
            new_state=new_state,
            **init_kwargs,
        )

    @classmethod
    async def with_only_step_order(
        cls,
        timeline_name: str,
        length_seconds: int,
        output_folder: str,
        volume_dir: str,
        video_id: str | None = None,
        user_id: str | None = None,
        video_hash: str | None = None,
        user_email: str | None = None,
        **init_kwargs,
    ):
        workflow_id = await CutTranscriptLinearWorkflow.id_from_params(
            video_id=video_id or None,
            user_id=user_id or None,
            video_hash=video_hash,
            timeline_name=timeline_name,
            user_email=user_email,
            length_seconds=length_seconds,
            output_folder=output_folder,
            volume_dir=volume_dir,
            **init_kwargs,
        )
        step_order = await load_step_order(workflow_id)
        return CutTranscriptLinearWorkflow(state=None, step_order=step_order)

    @classmethod
    async def from_video(
        cls,
        video: Video,
        timeline_name: str,
        length_seconds: int,
        output_folder: str,
        volume_dir: str,
        new_state: bool = False,
        **init_kwargs,
    ):

        if new_state:
            state = await CutTranscriptLinearWorkflowState.recreate(
                video=video,
                timeline_name=timeline_name,
                length_seconds=length_seconds,
                output_folder=output_folder,
                volume_dir=volume_dir,
                **init_kwargs,
            )
        else:
            state = await CutTranscriptLinearWorkflowState.find_or_create(
                video=video,
                timeline_name=timeline_name,
                length_seconds=length_seconds,
                output_folder=output_folder,
                volume_dir=volume_dir,
                **init_kwargs,
            )

        return CutTranscriptLinearWorkflow(state=state)

    #### PROPERTIES ####
    @property
    def id(self):
        return self.state.id

    @property
    def run_output_dir(self):
        assert self.state is not None
        return self.state.run_output_dir

    @property
    def raw_transcript(self):
        assert self.state is not None
        return Transcript.load_from_state(self.state.raw_transcript_state)

    @property
    def on_screen_transcript(self):
        assert self.state is not None
        if self.state.on_screen_transcript_state is not None:
            return Transcript.load_from_state(self.state.on_screen_transcript_state)

    @property
    def soundbites(self):
        assert self.state is not None
        if self.state.original_soundbites_state is not None:
            return Soundbites.load_from_state(self.state.original_soundbites_state)

    @property
    def current_transcript(self):
        assert self.state is not None
        if self.state.current_transcript_state:
            return Transcript.load_from_state(self.state.current_transcript_state)

    @property
    def current_soundbites(self):
        assert self.state is not None
        if self.state.current_soundbites_state:
            return Soundbites.load_from_state(self.state.current_soundbites_state)

    @property
    def base_user_feedback_prompt(self):
        formatted_inner_base_prompt = add_complete_format(
            "\n\nDo you have any feedback to provide the AI assistant?",
            ["bold", "yellow"],
        )
        return "{{ prefix }}" + formatted_inner_base_prompt

    @property
    def output_dir(self):
        assert self.state is not None
        return get_generated_video_folder(
            self.state.output_folder, self.video.user.email, self.timeline_name
        )

    @property
    def volume_dir(self):
        return self.step_order.volume_dir

    @property
    def video(self):
        return self.step_order.video

    @property
    def user(self):
        return self.step_order.user

    @property
    def timeline_name(self):
        return self.step_order.timeline_name

    @property
    def length_seconds(self):
        return self.step_order.length_seconds

    @property
    def first_pass_length(self):
        return self.step_order.first_pass_length

    @property
    def nstages(self):
        return self.step_order.nstages

    @property
    def use_agent_output_cache(self):
        return self.step_order.use_agent_output_cache

    @property
    def max_partial_transcript_words(self):
        return self.step_order.max_partial_transcript_words

    @property
    def max_total_soundbites(self):
        return self.step_order.max_total_soundbites

    @property
    def export_transcript_text(self):
        return self.step_order.export_transcript_text

    @property
    def export_transcript(self):
        return self.step_order.export_transcript

    @property
    def export_soundbites(self):
        return self.step_order.export_soundbites

    @property
    def export_soundbites_text(self):
        return self.step_order.export_soundbites_text

    @property
    def export_timeline(self):
        return self.step_order.export_timeline

    @property
    def export_video(self):
        return self.step_order.export_video

    @property
    def export_speaker_tagging(self):
        return self.step_order.export_speaker_tagging

    @property
    def num_speaker_tagging_samples(self):
        return self.step_order.num_speaker_tagging_samples

    @property
    def clip_extra_trim_seconds(self):
        return self.step_order.clip_extra_trim_seconds

    @property
    def max_iterations(self):
        return self.step_order.max_iterations

    @property
    def ask_user_for_feedback_every_iteration(self):
        return self.step_order.ask_user_for_feedback_every_iteration

    @property
    def max_word_extra_threshold(self):
        return self.step_order.max_word_extra_threshold

    @property
    def user_prompt(self):
        assert self.state is not None
        return self.state.user_prompt

    @property
    def user_messages(self):
        assert self.state is not None
        return self.state.user_messages

    @property
    def serializable_state_step_order(self):
        assert self.state is not None
        return self.state.dynamic_state_step_order

    @property
    def story(self):
        assert self.state is not None
        return self.state.story

    def step_output_dir(self, step_name, substep_name):
        assert self.state is not None
        return self.state.step_output_dir(step_name, substep_name)

    @property
    def soundbites_video_dir(self):
        assert self.state is not None
        return self.state.soundbites_video_dir

    @property
    def soundbites_timeline_dir(self):
        assert self.state is not None
        return self.state.soundbites_timeline_dir

    @property
    def last_step_name(self):
        return self.steps[-1].name

    @property
    def stage_lengths(self):
        if self.length_seconds == self.first_pass_length:
            if self.nstages != 1:
                raise ValueError(
                    "If length_seconds == first_pass_length, nstages must be 1"
                )
            return [self.length_seconds]
        stage_lengths = list(
            range(
                self.first_pass_length,
                self.length_seconds,
                (self.length_seconds - self.first_pass_length) // self.nstages,
            )
        )
        stage_lengths[-1] = self.length_seconds
        return stage_lengths

    async def most_recent_export_result(self, with_load_state=True):
        last_output = await self.get_last_output_before_end(
            with_load_state=with_load_state
        )
        if last_output is None:
            return {}
        return last_output.export_result or {}

    async def export_result_for_step_substep_name(
        self, step_name: str, substep_name: str, with_load_state=True
    ):
        if with_load_state:
            await self.load_state()
        assert self.state is not None

        state_key = get_dynamic_state_key(step_name, substep_name)
        output = self._get_output_for_key(state_key)
        if output is None:
            return {}
        return output.export_result or {}

    @property
    def steps(self):
        _steps = [
            StepWrapper(
                name="preprocess_video",
                human_readable_name="Remove Off-Screen Speakers",
                substeps=[
                    CurrentStepInfo(
                        name="init_state",
                        method=self.init_state,
                        user_feedback=False,
                        export_transcript=False,
                        export_soundbites=False,
                        export_video=False,
                        export_timeline=False,
                        export_speaker_tagging=False,
                    ),
                    CurrentStepInfo(
                        name="remove_off_screen_speakers",
                        method=self.remove_off_screen_speakers,
                        user_feedback=True,
                        export_transcript=True,
                        export_soundbites=False,
                        export_video=True,
                        export_timeline=True,
                        export_speaker_tagging=True,
                    ),
                ],
            ),
            StepWrapper(
                name="generate_story",
                human_readable_name="Generate Narrative Story",
                substeps=[
                    CurrentStepInfo(
                        name="generate_story",
                        method=self.generate_story,
                        user_feedback=True,
                        export_transcript=False,
                        export_soundbites=False,
                        export_video=False,
                        export_timeline=False,
                        export_speaker_tagging=False,
                    )
                ],
            ),
            StepWrapper(
                name="identify_key_soundbites",
                human_readable_name="Identify Key Selects",
                substeps=[
                    CurrentStepInfo(
                        name="identify_key_soundbites",
                        method=self.identify_key_soundbites,
                        user_feedback=True,
                        chunked_feedback=True,
                        export_transcript=False,
                        export_soundbites=True,
                        export_video=False,
                        export_timeline=False,
                        export_speaker_tagging=False,
                    )
                ],
            ),
        ]
        for stage_num, _ in enumerate(self.stage_lengths):
            step_wrapper = StepWrapper(
                name=stage_key_for_step_name("generate_transcript", stage_num),
                human_readable_name=f"Generate Transcript, Stage {stage_num}",
                substeps=[
                    CurrentStepInfo(
                        name="cut_partial_transcripts_with_critiques",
                        method=self.cut_transcript_with_critiques,
                        user_feedback=False,
                        chunked_feedback=True,
                        export_transcript=False,
                        export_soundbites=False,
                        export_video=False,
                        export_timeline=False,
                        export_speaker_tagging=False,
                    ),
                    CurrentStepInfo(
                        name="modify_transcript_holistically",
                        method=self.modify_transcript_holistically,
                        user_feedback=True,
                        chunked_feedback=False,
                        export_transcript=True,
                        export_soundbites=False,
                        export_video=True,
                        export_timeline=True,
                        export_speaker_tagging=False,
                    ),
                ],
            )
            _steps.append(step_wrapper)
        return Steps(steps=_steps)

    @property
    def serializable_steps(self):
        return [
            {
                "name": step.name,
                "substeps": [
                    {
                        "user_feedback": substep.user_feedback,
                        "chunked_feedback": substep.chunked_feedback,
                        "name": substep.name,
                    }
                    for substep in step.substeps
                ],
            }
            for step in self.steps
        ]

    #### READ STATE/STEP ####

    async def load_state(self):
        assert self.state is not None
        await self.state.sync()
        await self.state.fetch_all_links()
        await self.state.video.sync()
        self.step_order = self.state

    async def load_step_order(self):
        assert self.step_order is not None
        step_order = await load_step_order(self.step_order.id)
        assert step_order is not None
        self.step_order = step_order

    # TODO all these methods need a major refactor
    async def get_last_step(self, with_load_state=True):
        if with_load_state:
            await self.load_step_order()
        return self._get_last_step_with_index()[1:]

    async def get_last_substep(self, with_load_state=True):
        if with_load_state:
            await self.load_step_order()
        try:
            last_step_index, step, substep_index = self._get_last_step_with_index()
        except ValueError:
            return None
        if step is not None:
            if substep_index >= len(step.substeps):
                assert last_step_index >= len(self.steps)
                return step.substeps[-1]
            return step.substeps[substep_index]
        return None

    async def get_last_substep_with_user_feedback(self, with_load_state=True):
        if with_load_state:
            await self.load_step_order()
        try:
            step_index, step, substep_index = self._get_last_step_with_index()
        except ValueError:
            return None
        while True:
            if step is None or step_index is None:
                return None
            else:
                assert substep_index is not None
                assert step_index is not None
                if substep_index >= len(step.substeps):
                    assert step_index >= len(self.steps)
                    substep_index = len(step.substeps) - 1
                    step_index = len(self.steps) - 1
                step = self.steps[step_index]
                substep = step.substeps[substep_index]
                if substep.user_feedback:
                    return substep
                try:
                    step_index, substep_index = self.steps.last_step_index(
                        step_index, substep_index
                    )
                except ValueError:
                    return None

    async def get_next_step(self, with_load_state=True):
        if with_load_state:
            await self.load_step_order()
        try:
            return self._get_next_step_with_index()[1:]
        except ValueError:
            return None

    async def get_next_substep(self, with_load_state=True):
        if with_load_state:
            await self.load_step_order()
        try:
            step, substep_index = self._get_next_step_with_index()[1:]
        except ValueError:
            return None
        if step is not None:
            assert substep_index is not None
            return step.substeps[substep_index]
        return None

    # TODO this function name is overloaded with _get_next_step_with_user_feedback
    # which has a different meaning and is used in a different context
    async def get_next_substep_with_user_feedback(self, with_load_state=True):
        if with_load_state:
            await self.load_step_order()
        try:
            step_index, step, substep_index = self._get_next_step_with_index()
        except ValueError:
            return None
        while True:
            if step is not None:
                assert substep_index is not None
                substep = step.substeps[substep_index]
                if substep.user_feedback:
                    return substep
                try:
                    step_index, substep_index = self.steps.next_step_index(
                        step_index, substep_index
                    )
                except ValueError:
                    return None
                step = self.steps[step_index]
            else:
                return None

    async def get_last_output(self, with_load_state=True):
        if with_load_state:
            await self.load_step_order()
        assert self.state is not None
        key = self.state.get_current_step_key_atomic()
        if key is None:
            return None
        return self._get_output_for_key(key)

    async def get_last_output_before_end(self, with_load_state=True):
        if with_load_state:
            await self.load_state()
        assert self.state is not None
        key = self.state.get_step_key_before_end()
        if key is None:
            return None
        return self._get_output_for_key(key)

    async def get_output_for_keys(
        self, keys: list[str], with_load_state=True, latest_retry=False
    ):
        if with_load_state:
            await self.load_state()
        assert self.state is not None
        return [
            self._get_output_for_name(
                *get_step_substep_names_from_dynamic_state_key(key),
                latest_retry=latest_retry,
            )
            for key in keys
        ]

    async def get_output_for_names(
        self, names: list[str], with_load_state=True, latest_retry=False
    ):
        if with_load_state:
            await self.load_state()
        assert self.state is not None
        return [
            self._get_output_for_name(
                name, substep_name=None, latest_retry=latest_retry
            )
            for name in names
        ]

    async def get_all_outputs(self, with_load_state=True):
        if with_load_state:
            await self.load_state()
        assert self.state is not None
        outputs = []
        for step_key in self.state.dynamic_state_step_order:
            step_output = {"name": step_key.name, "substeps": []}
            for substep_name in step_key.substeps:
                step_output["substeps"].append(
                    self._get_output_for_name(step_key.name, substep_name)
                )
            outputs.append(step_output)
        return outputs

    def get_step_by_name(self, step_name: str, substep_name: str):
        substep_name = remove_retry_suffix(substep_name)
        for step in self.steps:
            if step.name == step_name:
                for substep in step.substeps:
                    if substep.name == substep_name:
                        return step, substep
        raise ValueError(f"Step {step_name}.{substep_name} not found")

    #### WRITE STATE/STEP ####

    async def restart_state(self):
        await self.load_state()
        if self.state is not None:
            await self.state.restart()

    async def revert_step(self, before_retries: bool = False):
        await self.load_state()
        if self.state is not None:
            await self.state.revert_step(before_retries=before_retries)

    async def revert_step_to_before(self, step_name: str, substep_name: str):
        await self.load_state()
        if self.state is not None:
            await self.state.revert_step_to_before(step_name, substep_name)

    async def delete(self):
        if self.state is not None:
            await self.state.delete()

    async def _save_and_export_step(
        self,
        result,
        current_step,
        current_substep_index,
        save_state_to_db=True,
        async_export=True,
    ):
        assert self.state is not None
        current_substep = current_step.substeps[current_substep_index]
        # This needs to come before the export result task,
        # because it sets run_output_dir to the state from the init_state method outputs
        # which is used by export_results_wrapper
        # It's okay that it's not saved to mongo by the time export_reuslts_wrapper is called
        # because we pass the version of workflow/workflow.state that has it to the method
        await self._save_raw_step_result(result)

        state_save_key = self.state.get_new_dynamic_key_with_retry(
            current_step.name, current_substep.name
        )
        export_result = {}
        task = None
        export_call_id = None
        if is_local() and async_export:
            task = asyncio.create_task(
                export_results_wrapper.local(self, state_save_key, current_substep)
            )
        elif not async_export:
            export_result, is_last = None, False
            async for export_result, is_last in self.export_results(
                current_substep.input
            ):
                continue
            assert export_result is not None and is_last
        else:
            call = export_results_wrapper.spawn(self, state_save_key, current_substep)
            export_call_id = call.object_id

        step_output_parsed = await self._parse_step_output_from_step_result(
            current_step,
            current_substep_index,
            result,
            export_result=export_result,
            export_call_id=export_call_id,
        )

        await self.state.set_current_step_output_atomic(
            state_save_key,
            step_output_parsed,
            save_to_db=save_state_to_db,
            use_session=async_export,
        )

        if is_local() and async_export:
            assert isinstance(task, asyncio.Task)
            await task
        print(f"Saved state for step {current_step.name}.{current_substep.name}")
        return step_output_parsed

    async def step(
        self,
        user_feedback: str = "",
        load_state=True,
        save_state_to_db=True,
        async_export=True,
        retry_step=False,
    ):
        if load_state:
            await self.load_state()
        assert self.state is not None
        current_step, current_substep_index = (
            await self._get_next_step_with_user_feedback(user_feedback, retry_step)
        )
        if not current_step:
            step_result = CutTranscriptLinearWorkflowStepOutput(
                step_name=END_STEP_NAME, substep_name=END_STEP_NAME, done=True
            )
            state_save_key = self.state.get_new_dynamic_key_with_retry(
                END_STEP_NAME, END_STEP_NAME
            )

            await self.state.set_current_step_output_atomic(
                state_save_key,
                step_result,
                save_to_db=save_state_to_db,
                use_session=async_export,
            )
            yield step_result, True
            return
        assert isinstance(current_substep_index, int)

        current_substep = current_step.substeps[current_substep_index]

        assert isinstance(
            current_substep, CurrentStepInfo
        ), f"current_substep: {current_substep}"

        yield PartialBackendOutput(
            value="Retrieved current substep",
            current_substep=current_substep.to_exportable(),
        ), False

        assert current_substep.input is not None

        step_output_parsed = None
        result = None
        print(f"Running step {current_step.name}.{current_substep.name}")
        async for result, is_last in current_substep.method(current_substep.input):
            if not is_last:
                yield result, False
        assert result is not None

        step_output_parsed = await self._save_and_export_step(
            result,
            current_step,
            current_substep_index,
            save_state_to_db=save_state_to_db,
            async_export=async_export,
        )
        yield step_output_parsed, True
        print(f"Yielded results for step {current_step.name}.{current_substep.name}")

    #### STEP METHODS ####

    async def init_state(self, step_input: CutTranscriptLinearWorkflowStepInput):
        assert self.state is not None
        if self.state.raw_transcript_state is None:
            assert self.video.transcription is not None
            current_transcript = Transcript.from_video_transcription(
                self.video.transcription
            )
            self.state.raw_transcript_state = current_transcript.state
        else:
            current_transcript = Transcript.load_from_state(
                self.state.raw_transcript_state
            )
        self.state.user_messages = [step_input.user_prompt or ""]

        run_output_dir = get_new_integer_file_name_in_dir(
            self.output_dir, ext="", prefix="run_"
        )
        Path(run_output_dir).mkdir(parents=True, exist_ok=True)
        yield CutTranscriptLinearWorkflowStepResults(
            outputs={
                "current_transcript": current_transcript,
                "run_output_dir": run_output_dir,
            }
        ), True

    async def remove_off_screen_speakers(
        self, step_input: CutTranscriptLinearWorkflowStepInput | None
    ):
        user_prompt = step_input.user_prompt if step_input else ""

        on_screen_transcript = self.raw_transcript
        if self.video.speakers_in_frame:
            on_screen_transcript = remove_off_screen_speakers(
                self.video.speakers_in_frame, self.raw_transcript
            )
        on_screen_speakers = []
        async for output, is_last in self._id_on_screen_speakers(
            transcript=on_screen_transcript,
            user_prompt=user_prompt,
            use_agent_output_cache=self.use_agent_output_cache,
        ):
            if is_last:
                on_screen_speakers = output
                break
            else:
                yield output, is_last
        if not on_screen_speakers:
            current_transcript = self.raw_transcript.copy()
            current_transcript.erase_cuts()
            yield CutTranscriptLinearWorkflowStepResults(
                outputs={
                    "current_transcript": current_transcript,
                    "on_screen_transcript": current_transcript,
                    "on_screen_speakers": on_screen_speakers,
                },
                user_feedback_request="I could not find any on-screen speakers. Maybe try another video? Or enter additional information to help me find them.",
            ), True
            return

        self.video.speakers_in_frame = on_screen_speakers
        await self.video.save()
        assert isinstance(on_screen_speakers, list)

        on_screen_transcript = remove_off_screen_speakers(
            on_screen_speakers, self.raw_transcript
        )
        yield CutTranscriptLinearWorkflowStepResults(
            outputs={
                "current_transcript": on_screen_transcript,
                "on_screen_speakers": on_screen_speakers,
                "on_screen_transcript": on_screen_transcript,
            },
            user_feedback_request=f"I identified these speakers as being on-screen: {on_screen_speakers}. \nDo you agree? Do you have modifications to make?",
        ), True

    async def generate_story(
        self, user_feedback: CutTranscriptLinearWorkflowStepInput | None
    ):
        assert self.state is not None
        assert isinstance(self.on_screen_transcript, Transcript), "Transcript not set"

        prompt = load_prompt_template_as_string("linear_workflow_story")
        user_messages = self.state.user_messages[:]
        # TODO llm_modified_prompt?
        if user_feedback and user_feedback.user_prompt:
            user_messages.append(user_feedback.user_prompt)

        output = ""
        async for output, is_last in get_agent_output_modal_or_local(
            prompt,
            transcript=self.on_screen_transcript.text,
            length_seconds=self.length_seconds,
            total_words=desired_words_from_length(self.length_seconds),
            user_prompt=user_messages[0],
            from_cache=self.use_agent_output_cache,
            user_feedback_messages=user_messages[1:],
        ):
            if not is_last:
                yield output, is_last
        assert isinstance(output, FinalLLMOutput) and output.str_value is not None
        story = output.str_value
        story = remove_boundary_tags("story", story)
        yield CutTranscriptLinearWorkflowStepResults(
            user_feedback_request=render_jinja_string(
                self.base_user_feedback_prompt,
                prefix="I generated a story from the transcript. ",
            ),
            outputs={"story": story},
        ), True

    async def identify_key_soundbites(
        self, step_input: CutTranscriptLinearWorkflowStepInput | None
    ):
        assert self.state is not None
        # figure out how to load from file if need be for testing
        #  if self.soundbites is not None:
        #  return self.soundbites
        #  if self.soundbites_file is not None:
        #  return Soundbites.load_from_file(self.soundbites_file)
        if self.run_output_dir is None:
            raise ValueError("run_output_dir must be set before calling this function")
        if self.on_screen_transcript is None:
            raise ValueError(
                "on_screen_transcript must be set before calling this function"
            )
        self.state.soundbites_output_dir = os.path.join(
            self.run_output_dir, "soundbites"
        )

        existing_soundbites = self.current_soundbites

        partials_to_redo = None
        relevant_user_feedback_list = None
        if (
            step_input is not None
            and step_input.llm_modified_partial_feedback is not None
        ):
            partials_to_redo = step_input.llm_modified_partial_feedback.partials_to_redo
            relevant_user_feedback_list = (
                step_input.llm_modified_partial_feedback.relevant_user_feedback_list
            )

        soundbite_tasks = []

        partial_transcripts = self.on_screen_transcript.copy().split_in_chunks(
            self.max_partial_transcript_words
        )
        max_soundbites_per_chunk = self.max_total_soundbites // len(partial_transcripts)
        for i, partial_transcript in enumerate(partial_transcripts):
            if not partials_to_redo or partials_to_redo[i]:
                existing_soundbite = None
                relevant_user_feedback = None
                if partials_to_redo:
                    if not existing_soundbites or len(existing_soundbites.chunks) <= i:
                        raise ValueError(
                            f"Existing soundbites not provided for partial {i}"
                        )
                    existing_soundbite = existing_soundbites.chunks[i]
                    if (
                        not relevant_user_feedback_list
                        or len(relevant_user_feedback_list) <= i
                    ):
                        raise ValueError(
                            f"Relevant user feedback not provided for partial {i}"
                        )
                    relevant_user_feedback = relevant_user_feedback_list[i]
                soundbite_tasks.append(
                    self._identify_key_soundbites_partial(
                        partial_transcript=partial_transcript,
                        existing_soundbite=existing_soundbite,
                        user_feedback=relevant_user_feedback or "",
                        max_soundbites=max_soundbites_per_chunk,
                        use_agent_output_cache=self.use_agent_output_cache,
                    )
                )
            elif existing_soundbites and len(existing_soundbites.chunks) > i:
                soundbite_tasks.append(
                    async_passthrough_gen(existing_soundbites.chunks[i])
                )
            else:
                raise ValueError(f"No existing soundbite provided for partial {i}")
        merged_stream = aiostream.stream.merge(*soundbite_tasks)
        all_soundbites = []
        async for output, is_last in merged_stream:
            if not is_last:
                yield output, is_last
            else:
                all_soundbites.append(output)
        all_soundbites = sorted(all_soundbites, key=lambda x: x.chunk_index)

        soundbites_merged = Soundbites.merge(*all_soundbites)
        soundbites_formatted = [
            (i, add_complete_format(soundbite, ["bold", "green"]))
            for i, soundbite in soundbites_merged.iter_text()
        ]
        user_feedback_prompt = parse_prompt_template(
            "soundbite_user_feedback_prompt", soundbites=soundbites_formatted
        )
        yield CutTranscriptLinearWorkflowStepResults(
            outputs={
                "current_soundbites": soundbites_merged,
                "original_soundbites": soundbites_merged,
            },
            user_feedback_request=user_feedback_prompt,
        ), True

    async def cut_transcript_with_critiques(
        self, step_input: CutTranscriptLinearWorkflowStepInput
    ):
        step_name = step_input.step_name
        stage_num = parse_stage_num_from_step_name(step_name)
        if stage_num is None:
            raise ValueError(f"Step name {step_name} does not contain a stage number")

        # TODO decide whether to use user feedback for this method at all
        user_feedback = step_input.user_prompt or ""
        assert self.current_transcript is not None
        partials_to_redo = [True] * len(self.current_transcript.chunks)
        user_feedback_list = [user_feedback or ""] * len(self.current_transcript.chunks)
        if step_input.llm_modified_partial_feedback:
            partials_to_redo = step_input.llm_modified_partial_feedback.partials_to_redo
            user_feedback_list = (
                step_input.llm_modified_partial_feedback.relevant_user_feedback_list
            )
        assert (
            len(user_feedback_list)
            == len(partials_to_redo)
            == len(self.current_transcript.chunks)
        )

        input_transcript = self.current_transcript
        assert isinstance(input_transcript, Transcript)

        existing_cut_transcript = input_transcript.copy()
        existing_cut_transcript.erase_cuts()
        final_transcript = input_transcript if stage_num > 0 else None
        kept_soundbites = self.current_soundbites
        assert kept_soundbites is not None, "Soundbites must be provided for each stage"

        partial_transcript_fresh = input_transcript.copy()
        if not step_input.is_retry:
            partial_transcript_fresh.chunks = []
            partial_transcripts = partial_transcript_fresh.split_in_chunks(
                self.max_partial_transcript_words
            )
            kept_soundbites.align_to_transcript_chunks(partial_transcript_fresh)
        else:
            partial_transcripts = partial_transcript_fresh.chunks

        yield PartialBackendOutput(
            value=f"Cutting and critiquing partial transcripts for stage {stage_num}\n"
        ), False
        cut_partial_transcript_with_critiques_jobs = []

        passthroughs = []
        for i, partial_transcript in enumerate(partial_transcripts):
            print("Cutting partial transcript")
            kept_soundbites_chunk = kept_soundbites.chunks[
                partial_transcript.chunk_index
            ]

            if len(partials_to_redo) <= i or partials_to_redo[i]:
                cut_partial_transcript_with_critiques_jobs.append(
                    self._cut_partial_transcript_with_critiques(
                        stage_num=stage_num,
                        partial_transcript=partial_transcript.copy(),
                        existing_cut_transcript=existing_cut_transcript.copy(),
                        prev_final_transcript=(
                            final_transcript.copy() if final_transcript else None
                        ),
                        kept_soundbites_chunk=kept_soundbites_chunk.copy(),
                        user_feedback=(
                            user_feedback_list[i] or ""
                            if len(user_feedback_list) > i
                            else user_feedback
                        ),
                    )
                )
            else:
                passthroughs.append(i)
                cut_partial_transcript_with_critiques_jobs.append(
                    async_passthrough_gen(
                        (partial_transcript.copy(), kept_soundbites_chunk.copy())
                    )
                )

        new_kept_soundbites_chunks = []
        new_cut_transcript_chunks = []
        final_transcript = existing_cut_transcript

        merged_stream = aiostream.stream.merge(
            *cut_partial_transcript_with_critiques_jobs
        )
        async for output, is_last in merged_stream:
            if not is_last:
                yield output, is_last
            else:
                new_cut_transcript_chunk, kept_soundbites_chunk = output
                new_kept_soundbites_chunks.append(kept_soundbites_chunk)
                new_cut_transcript_chunks.append(new_cut_transcript_chunk)

        assert all(
            [s is not None for s in new_kept_soundbites_chunks]
        ), f"Some soundbites came back None: {[i for i,s in enumerate(new_kept_soundbites_chunks) if s is None]}. Passthroughs: {passthroughs}"
        assert all(
            [t is not None for t in new_cut_transcript_chunks]
        ), f"Some partials came back None: {[i for i,s in enumerate(new_cut_transcript_chunks) if s is None]}. Passthroughs: {passthroughs}"
        assert all([s.chunk_index is not None for s in new_kept_soundbites_chunks])
        assert all([t.chunk_index is not None for t in new_cut_transcript_chunks])
        new_kept_soundbites_chunks = sorted(
            new_kept_soundbites_chunks, key=lambda x: x.chunk_index
        )
        new_cut_transcript_chunks = sorted(
            new_cut_transcript_chunks, key=lambda x: x.chunk_index
        )

        yield PartialBackendOutput(
            value="Merging cut transcripts and soundbites\n"
        ), False
        if new_cut_transcript_chunks:
            final_transcript = Transcript.merge(*new_cut_transcript_chunks)
        if new_kept_soundbites_chunks:
            kept_soundbites = Soundbites.merge(*new_kept_soundbites_chunks)
        # TODO figure out some way to include this output
        #  yield "Here is the new cut transcript from this stage before modifying holistically\n", False
        #  for cut in final_transcript.iter_kept_cuts():
        #  yield f"```json\n{cut.model_dump_json()}\n```", False

        yield CutTranscriptLinearWorkflowStepResults(
            outputs={
                "current_soundbites": kept_soundbites,
                "current_transcript": final_transcript,
            }
        ), True

    async def modify_transcript_holistically(
        self, step_input: CutTranscriptLinearWorkflowStepInput
    ):
        output = None
        modified_transcript = None

        iterations = max(1, self.max_iterations)
        if self.ask_user_for_feedback_every_iteration:
            iterations = 1
        for i in range(max(1, iterations)):
            # TODO _modify_transcript_holistically_single_iteration is returning same thing for each retry_num
            # likely from cache, which means prompt isn't changing
            async for (
                output,
                is_last,
            ) in self._modify_transcript_holistically_single_iteration(
                step_input, transcript=modified_transcript, retry_num=i
            ):
                if not is_last:
                    yield output, is_last
            assert isinstance(output, CutTranscriptLinearWorkflowStepResults)
            if output.outputs:
                modified_transcript = output.outputs["current_transcript"]
                stage_num = parse_stage_num_from_step_name(step_input.step_name)
                print(
                    "Checking word count excess, desired words:",
                    self._desired_words_for_stage(stage_num),
                )
                if not self._word_count_excess(
                    modified_transcript, self._desired_words_for_stage(stage_num)
                ):
                    output.outputs["retries"] = i
                    yield output, True
                    return
            else:
                print("No output from modify_transcript_holistically")
        assert isinstance(output, CutTranscriptLinearWorkflowStepResults)
        yield output, True

    async def export_results(self, step_input: CutTranscriptLinearWorkflowStepInput):
        await maybe_init_mongo()
        substep = self._substep_for_step_input(step_input)
        export_transcript = (
            self.export_transcript
            and substep.export_transcript
            and self.current_transcript is not None
        )
        export_soundbites = (
            self.export_soundbites
            and substep.export_soundbites
            and self.current_soundbites is not None
        )
        export_timeline = (
            self.export_timeline
            and substep.export_timeline
            and self.current_transcript is not None
        )
        export_video = (
            self.export_video
            and substep.export_video
            and self.current_transcript is not None
        )
        export_speaker_tagging = (
            self.export_speaker_tagging
            and substep.export_speaker_tagging
            and self.raw_transcript is not None
        )

        output_dir = self.step_output_dir(step_input.step_name, step_input.substep_name)
        output_files = {}
        prefix = f"{Path(self.video.high_res_user_file_path).stem}_{step_input.substep_name}_"
        if export_transcript:
            yield "Exporting transcript", False
            transcript_file, transcript_text_file = save_transcript_to_disk(
                output_dir=output_dir,
                transcript=self.current_transcript,
                save_text_file=self.export_transcript_text,
                prefix=f"{prefix}transcript_",
            )
            output_files["transcript"] = transcript_file
            if self.export_transcript_text:
                output_files["transcript_text"] = transcript_text_file

        if export_soundbites:
            yield "Exporting soundbites", False
            soundbites_file, soundbites_text_file = save_transcript_to_disk(
                output_dir=output_dir,
                transcript=self.current_soundbites,
                save_text_file=self.export_soundbites_text,
                suffix="_soundbites",
                prefix=f"{prefix}transcript_",
            )
            soundbites_video_files = await save_soundbites_videos_to_disk(
                video=self.video,
                output_dir=self.soundbites_video_dir,
                volume_dir=self.volume_dir,
                soundbites=self.current_soundbites,
                clip_extra_trim_seconds=self.clip_extra_trim_seconds,
                prefix=f"{prefix}_" + "{}",
                timeline_name=self.timeline_name,
            )
            soundbites_timeline_file = create_fcp_7_xml_from_single_video_transcript(
                video=self.video,
                transcript=self.current_soundbites,
                timeline_name=self.timeline_name,
                volume_dir=self.volume_dir,
                output_dir=self.soundbites_timeline_dir,
                clip_extra_trim_seconds=self.clip_extra_trim_seconds,
                use_high_res_path=True,
                prefix=f"{prefix}soundbites_timeline_",
            )

            output_files["soundbites_transcript"] = soundbites_file
            output_files["soundbites_videos"] = soundbites_video_files
            output_files["soundbites_timeline"] = soundbites_timeline_file
            if self.export_transcript_text:
                output_files["soundbites_text"] = soundbites_text_file

        if export_timeline:
            yield "Exporting timeline", False
            # TODO can make this async gen and pass partial output
            video_timeline_file = create_fcp_7_xml_from_single_video_transcript(
                video=self.video,
                transcript=self.current_transcript,
                timeline_name=self.timeline_name,
                volume_dir=self.volume_dir,
                output_dir=output_dir,
                clip_extra_trim_seconds=self.clip_extra_trim_seconds,
                use_high_res_path=True,
                prefix=f"{prefix}timeline_",
            )
            output_files["video_timeline"] = video_timeline_file

        if export_video:
            yield "Exporting video", False
            # TODO can make this async gen and pass partial output
            cut_video_path = await create_cut_video_from_transcript(
                video=self.video,
                transcript=self.current_transcript,
                timeline_name=self.timeline_name,
                volume_dir=self.volume_dir,
                output_dir=output_dir,
                clip_extra_trim_seconds=self.clip_extra_trim_seconds,
                prefix=f"{prefix}video_",
            )
            output_files["video"] = cut_video_path

        if export_speaker_tagging:
            output_files["speaker_tagging_clips"] = (
                await self._export_speaker_tagging_samples(output_dir, prefix)
            )
        yield output_files, True

    #### HELPER FUNCTIONS ####

    async def _export_speaker_tagging_samples(self, output_dir, prefix):
        speaker_tagging_clips = defaultdict(list)
        speakers_to_segments = defaultdict(list)
        for i, segment in enumerate(self.raw_transcript.segments):
            speakers_to_segments[segment.speaker].append((i, segment))
        clip_creation_tasks = []
        flat_speakers = []
        for speaker, segments in speakers_to_segments.items():
            random_segments = segments[:]
            random.shuffle(random_segments)
            sampled_segments = random_segments[: self.num_speaker_tagging_samples]
            for segment_index, segment in sampled_segments:
                segment_transcript = self.raw_transcript.copy()
                segment_transcript.kept_segments = set([segment_index])
                clip_creation_tasks.append(
                    create_cut_video_from_transcript(
                        video=self.video,
                        transcript=segment_transcript,
                        timeline_name=self.timeline_name,
                        volume_dir=self.volume_dir,
                        output_dir=output_dir,
                        clip_extra_trim_seconds=self.clip_extra_trim_seconds,
                        prefix=f"{prefix}_segment_{segment_index}_",
                        create_new_if_existing=False,
                    )
                )
                flat_speakers.append(speaker)
        clip_paths = await tqdm_async.gather(*clip_creation_tasks)
        for speaker, path in zip(flat_speakers, clip_paths):
            speaker_tagging_clips[speaker].append(path)
        return speaker_tagging_clips

    def _step_output_val_for_stage(self, stage_num, step_output_key):
        assert self.state is not None
        for step_name in self.state.dynamic_state_step_order[::-1]:
            if f"stage_{stage_num}" in step_name.substeps[-1]:
                step_state = self.state.dynamic_state.get(step_name, {})
                step_outputs = step_state.get("step_outputs", {})
                if step_outputs:
                    output = step_outputs.get(step_output_key)
                    if output:
                        return output

    def _get_last_step_with_index(self):
        # TODO use get_step_by_name
        key = self.step_order.get_current_step_key_atomic()
        if key is None:
            return -1, None, -1
        last_step_name, last_substep_name_with_retry = (
            get_step_substep_names_from_dynamic_state_key(key)
        )

        assert last_substep_name_with_retry is not None
        last_substep_name = remove_retry_suffix(last_substep_name_with_retry)
        if last_step_name == END_STEP_NAME:
            return len(self.steps), self.steps[-1], len(self.steps[-1].substeps)
        last_steps_list = [
            (i, s, j)
            for i, s in enumerate(self.steps)
            for j, ss in enumerate(s.substeps)
            if s.name == last_step_name and ss.name == last_substep_name
        ]
        if len(last_steps_list) == 0:
            raise ValueError(
                f"Last step {last_step_name}.{last_substep_name_with_retry} not found in steps"
            )
        last_step_index, last_step, last_substep_index = last_steps_list[0]
        last_step.substeps[last_substep_index].name = last_substep_name_with_retry
        return last_step_index, last_step, last_substep_index

    def _get_next_step_with_index(self):
        # TODO use get_step_by_name
        last_step_index, _, last_substep_index = self._get_last_step_with_index()
        if last_step_index < 0:
            return 0, self.steps[0], 0
        next_step_index, next_substep_index = self.steps.next_step_index(
            last_step_index, last_substep_index
        )
        if next_step_index is None:
            return len(self.steps), None, len(self.steps[-1].substeps)
        assert last_substep_index is not None
        next_step = self.steps[next_step_index]
        return next_step_index, next_step, next_substep_index

    async def _get_next_step_with_user_feedback(
        self, user_feedback: str | None = None, retry_step: bool = False
    ):
        # TODO use get_step_by_name
        last_step_index, last_step, last_substep_index = (
            self._get_last_step_with_index()
        )
        if last_step_index == -1 and last_substep_index == -1:
            first_step = self.steps[0]
            first_substep = first_step.substeps[0]
            first_substep.input = CutTranscriptLinearWorkflowStepInput(
                user_prompt=user_feedback,
                is_retry=False,
                step_name=first_step.name,
                substep_name=first_step.substeps[0].name,
            )
            return first_step, 0

        # we've already done all steps once, repeat the last step as a retry
        force_retry = retry_step
        if last_step_index >= len(self.steps):
            last_step_index -= 1
            last_substep_index -= 1
            force_retry = True

        assert isinstance(last_step, StepWrapper)
        assert last_substep_index is not None
        last_substep = last_step.substeps[last_substep_index]

        retry, retry_input = await self._decide_retry(
            last_step.name, last_substep.name, user_feedback, force_retry=force_retry
        )
        if retry:
            print("retrying", retry_input)
            last_substep.input = retry_input or CutTranscriptLinearWorkflowStepInput(
                user_prompt=user_feedback, is_retry=True
            )
            last_substep.input.step_name = last_step.name
            last_substep.input.substep_name = last_substep.name
            return last_step, last_substep_index

        next_step_index, next_substep_index = self.steps.next_step_index(
            last_step_index, last_substep_index
        )
        print(
            "next_step_index", next_step_index, "next_substep_index", next_substep_index
        )

        if next_step_index is None:
            return None, None

        next_step = self.steps[next_step_index]
        next_substep = next_step.substeps[next_substep_index]
        next_substep.input = CutTranscriptLinearWorkflowStepInput(
            # We don't want to pass the user_feedback to the next step,
            # user feedback is currently only for retry (and the first step)
            user_prompt=None,
            is_retry=False,
            step_name=next_step.name,
            substep_name=next_substep.name,
        )
        # TODO next_substep should reference next_step so we can just return next_substep
        # instead of needing both of these
        return next_step, next_substep_index

    async def _save_raw_step_result(self, step_result_raw):
        assert self.state is not None
        state_class_parsers = {
            Transcript: (lambda x: ("{}_state", x.state),),
            Soundbites: (lambda x: ("{}_state", x.state),),
        }
        if step_result_raw is not None and step_result_raw.outputs is not None:
            for name, value in step_result_raw.outputs.items():
                if type(value) in state_class_parsers:
                    for parser in state_class_parsers[type(value)]:
                        format_name_str, state_val = parser(value)
                        self.state.set_state_val(
                            format_name_str.format(name), state_val
                        )
                else:
                    self.state.set_state_val(name, value)

    async def _parse_step_output_from_step_result(
        self,
        current_step,
        current_substep_index,
        result,
        export_result=None,
        export_call_id=None,
    ):
        step_outputs = {}
        output_class_parsers = {
            Transcript: (
                lambda x: ("{}_text", x.text),
                lambda x: ("{}_state", x.state),
            ),
            Soundbites: (
                lambda x: ("{}_iter_text", x.iter_text_list()),
                lambda x: ("{}_state", x.state),
            ),
        }
        if result.outputs is not None:
            for name, value in result.outputs.items():
                output_val = value
                if type(value) in output_class_parsers:
                    for parser in output_class_parsers[type(value)]:
                        format_name_str, output_val = parser(value)
                        step_outputs[format_name_str.format(name)] = output_val
                else:
                    step_outputs[name] = output_val
        current_substep = current_step.substeps[current_substep_index]
        return CutTranscriptLinearWorkflowStepOutput(
            step_name=current_step.name,
            substep_name=current_substep.name,
            done=False,
            user_feedback_request=result.user_feedback_request,
            step_inputs=current_substep.input,
            step_outputs=step_outputs,
            export_result=export_result or {},
            export_call_id=export_call_id,
            retry=result.retry,
        )

    async def _save_export_result_to_step_output(self, state_save_key, export_result):
        assert self.state is not None
        output = self._get_output_for_key(state_save_key)
        output.export_result = export_result
        await self.state.set_current_step_output_atomic(state_save_key, output)

    def _get_output_for_name(
        self, step_name: str, substep_name: str | None = None, latest_retry=False
    ):
        assert self.state is not None
        if substep_name is None:
            matching = [step for step in self.steps if step.name == step_name]
            if not matching:
                raise ValueError(f"Unknown step_name: {step_name}")
            step = matching[0]
            substep_name = step.substeps[-1].name
        key = get_dynamic_state_key(step_name, substep_name)
        if latest_retry:
            key = self.state.get_latest_dynamic_key_with_retry_from(
                step_name, substep_name
            )
        return self._get_output_for_key(key)

    def _get_output_for_key(self, key: str):
        assert self.state is not None
        results = self.state.dynamic_state.get(key, None)
        if results is None:
            return CutTranscriptLinearWorkflowStepOutput(
                step_name="",
                substep_name="",
                done=False,
                user_feedback_request="",
                step_outputs={},
            )

        if not isinstance(results, CutTranscriptLinearWorkflowStepOutput):
            results = CutTranscriptLinearWorkflowStepOutput(**results)
        return results

    async def _decide_retry(
        self,
        step_name: str,
        substep_name: str,
        user_prompt: str | None,
        force_retry=False,
    ):
        if not user_prompt or substep_name == "init_state":
            return force_retry, None

        assert self.state is not None
        _, substep = self.get_step_by_name(step_name, substep_name)
        dynamic_key = self.state.get_latest_dynamic_key_with_retry_from(
            step_name, substep_name
        )
        if self.state.dynamic_state_retries.get(dynamic_key):
            print("Latest state was retry")
            return True, CutTranscriptLinearWorkflowStepInput(
                user_prompt=user_prompt, is_retry=True
            )

        if substep.chunked_feedback:
            if "identify_key_soundbites" in step_name:
                if self.current_soundbites is None:
                    raise ValueError(
                        f"Soundbites must be already set if step has chunked_feedback (step_name: {step_name})"
                    )
                gen = self._identify_partial_soundbites_to_redo_from_user_feedback(
                    user_feedback=user_prompt, soundbites=self.current_soundbites
                )
            else:
                if self.current_transcript is None:
                    raise ValueError(
                        f"Transcript must be already set if step has chunked_feedback (step_name: {step_name})"
                    )
                gen = (
                    self._identify_partial_transcript_chunks_to_redo_from_user_feedback(
                        user_feedback=user_prompt, transcript=self.current_transcript
                    )
                )
            output = None
            # TODO this can be an async generator too
            async for output, is_last in gen:
                if is_last:
                    break
            assert isinstance(output, tuple) and len(output) == 2
            partials_to_redo, relevant_user_feedback_list = output
            try:
                assert isinstance(relevant_user_feedback_list, list) and all(
                    [isinstance(s, str) for s in relevant_user_feedback_list]
                )
            except:
                print(
                    f"relevant_user_feedback_list incorrect types: {relevant_user_feedback_list}, types={list(map(type, relevant_user_feedback_list))}"
                )
                relevant_user_feedback_list = [user_prompt] * len(partials_to_redo)
            return force_retry or any(
                partials_to_redo
            ), CutTranscriptLinearWorkflowStepInput(
                user_prompt=user_prompt,
                llm_modified_partial_feedback=PartialFeedback(
                    partials_to_redo=partials_to_redo,
                    relevant_user_feedback_list=relevant_user_feedback_list,
                ),
                is_retry=True,
            )

        else:
            if self.current_transcript is None:
                raise ValueError(
                    f"Transcript must be already set if step desires feedback (step_name: {step_name})"
                )
            method = self._ask_llm_to_parse_user_prompt_for_transcript_retry
            if "story" in step_name:
                if self.story is None:
                    raise ValueError(
                        f"Story must be already set if step has chunked_feedback (step_name: {step_name})"
                    )
                method = self._ask_llm_to_parse_user_prompt_for_story_retry
            elif "off_screen_speakers" in step_name:
                method = self._ask_llm_to_parse_user_prompt_for_speaker_id_retry
            output = None
            async for output, is_last in method(user_feedback=user_prompt):
                if is_last:
                    break
            assert isinstance(output, bool)
            print(
                f"retry step: {step_name}, retry_output: {output}, user_prompt: {user_prompt}"
            )

            return force_retry or output, CutTranscriptLinearWorkflowStepInput(
                user_prompt=user_prompt, is_retry=True
            )

    def _create_user_feedback_prompt_from_modify_final_transcript(
        self, final_transcript: Transcript, desired_words: int, stage_num: int
    ):
        user_feedback_prefix_list = []
        user_feedback_prefix_list.append(
            add_complete_format(
                "\nTrimIt created the following transcript:", ["bold", "yellow"]
            )
        )
        user_feedback_prefix_list.append(
            "\n".join(
                [
                    add_complete_format(p.strip(), ["bold", "green"])
                    for p in final_transcript.text.split(".")
                ]
            )
        )
        is_excess = self._word_count_excess(final_transcript, desired_words)
        if is_excess:
            excess_words = final_transcript.kept_word_count - desired_words
            stage_length_seconds = self._get_stage_length_seconds(stage_num)
            user_feedback_prefix_list.append(
                f"Final transcript was too long by {excess_words} words for this stage with desired length "
                f"{stage_length_seconds} ({desired_words} words) and we will work to cut it down further."
                "Before we cut it down, do you have additional feedback to provide the AI assistant (and redo the creation using this feedback)?"
            )
        return render_jinja_string(
            self.base_user_feedback_prompt, prefix="\n".join(user_feedback_prefix_list)
        )

    async def _id_on_screen_speakers(
        self,
        transcript: Transcript,
        user_prompt: str | None = None,
        use_agent_output_cache: bool = True,
    ):
        possible_speakers = set(
            [segment.speaker.lower() for segment in transcript.segments]
        )
        schema = Schema({"on_screen_speakers": [str]}).json_schema(
            "OnScreenSpeakerIdentification"
        )
        prompt = load_prompt_template_as_string("speaker_id")
        output = None
        async for output, is_last in get_agent_output_modal_or_local(
            prompt,
            user_prompt=user_prompt,
            json_mode=True,
            schema=schema,
            transcript=transcript.text_with_speaker_tags,
            from_cache=use_agent_output_cache,
        ):
            if not is_last:
                yield output, is_last
            else:
                break
        assert (
            isinstance(output, FinalLLMOutput) and output.json_value is not None
        ), f"Bad output from speaker identification: {output}"

        matched_possible_speakers = set()
        for speaker in output.json_value.get("on_screen_speakers", []):
            if speaker not in possible_speakers:
                print(f"Speaker {speaker} not found in possible speakers")
                continue
            matched_possible_speakers.add(speaker)
        yield list(matched_possible_speakers), True

    async def _critique_cut_transcript(
        self,
        assistant_cut_transcript_chunk: TranscriptChunk,
        prev_critiques="",
        existing_cut_transcript: Transcript | None = None,
        stage_num: int = 0,
        prev_final_transcript: Transcript | None = None,
        key_soundbites: SoundbitesChunk | None = None,
        user_feedback="",
        use_agent_output_cache=True,
    ):
        async for output, is_last in self._get_output_for_shared_cut_partial(
            partial_on_screen_transcript=assistant_cut_transcript_chunk,
            partial_on_screen_transcript_text=assistant_cut_transcript_chunk.text_with_keep_and_remove_tags,
            task_description_prompt=parse_prompt_template(
                "critique_cut_detailed_task_description_no_remove",
                num_partial_transcripts=len(
                    assistant_cut_transcript_chunk.transcript.chunks
                ),
            ),
            existing_cut_transcript=existing_cut_transcript,
            stage_num=stage_num,
            critiques=prev_critiques,
            user_feedback=user_feedback,
            prev_final_transcript=prev_final_transcript,
            key_soundbites=key_soundbites,
            use_agent_output_cache=use_agent_output_cache,
        ):
            yield output, is_last

    async def _cut_partial_transcript(
        self,
        partial_on_screen_transcript: TranscriptChunk,
        previous_cut_transcript_portion: TranscriptChunk | None = None,
        critiques="",
        existing_cut_transcript: Transcript | None = None,
        stage_num: int = 0,
        prev_final_transcript: Transcript | None = None,
        key_soundbites: SoundbitesChunk | None = None,
        user_feedback="",
        use_agent_output_cache=True,
    ):
        async for output, is_last in self._get_output_for_shared_cut_partial(
            partial_on_screen_transcript=partial_on_screen_transcript,
            partial_on_screen_transcript_text=partial_on_screen_transcript.text,
            task_description_prompt=parse_prompt_template(
                "cut_partial_detailed_task_description",
                num_partial_transcripts=len(
                    partial_on_screen_transcript.transcript.chunks
                ),
            ),
            existing_cut_transcript=existing_cut_transcript,
            stage_num=stage_num,
            previous_cut_transcript_portion=previous_cut_transcript_portion,
            critiques=critiques,
            user_feedback=user_feedback,
            prev_final_transcript=prev_final_transcript,
            key_soundbites=key_soundbites,
            use_agent_output_cache=use_agent_output_cache,
        ):
            yield output, is_last

    async def _get_output_for_shared_cut_partial(
        self,
        task_description_prompt: str,
        partial_on_screen_transcript: TranscriptChunk,
        partial_on_screen_transcript_text: str,
        existing_cut_transcript: Transcript | None = None,
        stage_num=0,
        previous_cut_transcript_portion: TranscriptChunk | None = None,
        critiques="",
        user_feedback="",
        prev_final_transcript: Transcript | None = None,
        key_soundbites: SoundbitesChunk | None = None,
        use_agent_output_cache=True,
    ):
        length_seconds = self._get_stage_length_seconds(stage_num)
        prev_length_seconds = None
        prev_desired_words = None
        if stage_num > 0:
            prev_length_seconds = self._get_stage_length_seconds(stage_num - 1)
            prev_desired_words = desired_words_from_length(prev_length_seconds)
        existing_cut_transcript_text = (
            existing_cut_transcript.text if existing_cut_transcript else ""
        )
        previous_cut_transcript_portion_text = (
            previous_cut_transcript_portion.text
            if previous_cut_transcript_portion
            else ""
        )
        prev_final_transcript_text = (
            prev_final_transcript.text if prev_final_transcript else ""
        )

        preamble = parse_prompt_template(
            "cut_partial_high_level_preamble",
            partial_video_length_seconds=partial_on_screen_transcript.length_seconds,
            full_video_length_seconds=partial_on_screen_transcript.transcript.length_seconds,
            length_seconds=length_seconds,
            total_words=desired_words_from_length(length_seconds),
            user_prompt=self.user_prompt,
            narrative_story=self.story,
            num_partial_transcripts=len(partial_on_screen_transcript.transcript.chunks),
        )
        output_reqs = parse_prompt_template("shared_output_requirements_no_remove")
        example = parse_prompt_template("shared_example_no_remove")
        extra_notes = parse_prompt_template(
            "cut_partial_extra_notes",
            prev_final_transcript_words=(
                prev_final_transcript.kept_word_count if prev_final_transcript else None
            ),
            prev_length_seconds=prev_length_seconds,
            prev_desired_words=prev_desired_words,
            desired_words=desired_words_from_length(length_seconds),
            num_partial_transcripts=len(partial_on_screen_transcript.transcript.chunks),
            prev_final_transcript=prev_final_transcript_text,
            existing_cut_transcript=existing_cut_transcript_text,
            previous_cut_transcript_portion=previous_cut_transcript_portion_text,
            critiques=critiques,
            user_feedback=user_feedback,
        )

        important_summary = parse_prompt_template("shared_important_summary_no_remove")
        prompt = load_prompt_template_as_string("cut_partial_transcript")
        output = ""
        async for output, is_last in get_agent_output_modal_or_local(
            prompt,
            partial_transcript=partial_on_screen_transcript_text,
            current_transcript_num=partial_on_screen_transcript.chunk_index,
            num_partial_transcripts=len(partial_on_screen_transcript.transcript.chunks),
            high_level_preamble=preamble,
            detailed_task_description=task_description_prompt,
            output_requirements=output_reqs,
            example=example,
            important_summary=important_summary,
            extra_notes=extra_notes,
            rules=(
                [get_soundbite_rule(key_soundbites)]
                if key_soundbites is not None and len(key_soundbites.soundbites)
                else None
            ),
            from_cache=use_agent_output_cache,
        ):
            if not is_last:
                yield output, is_last
        assert isinstance(output, FinalLLMOutput) and isinstance(output.str_value, str)
        new_cut_transcript = match_output_to_actual_transcript_fast(
            partial_on_screen_transcript, output.str_value
        )
        kept_soundbites = (
            key_soundbites.keep_only_in_transcript(new_cut_transcript)
            if key_soundbites is not None
            else None
        )
        yield (new_cut_transcript, kept_soundbites), True

    async def _identify_partial_soundbites_to_redo_from_user_feedback(
        self, user_feedback, soundbites: "Soundbites"
    ):
        schema = Schema(
            {"chunks_to_redo": [bool], "relevant_user_feedback_list": [str]}
        ).json_schema("PartialChunkRetryID")

        prompt = load_prompt_template_as_string(
            "identify_partial_soundbites_to_redo_from_user_feedback"
        )

        nchunks = len(soundbites.chunks)

        output = None
        async for output, is_last in get_agent_output_modal_or_local(
            prompt,
            schema=schema,
            json_mode=True,
            user_feedback=user_feedback,
            from_cache=self.use_agent_output_cache,
            soundbites=[
                [(i, text) for i, text in soundbites_chunk.iter_text()]
                for soundbites_chunk in soundbites.chunks
            ],
        ):
            if not is_last:
                yield output, is_last
        assert isinstance(output, FinalLLMOutput) and output.json_value is not None

        partials_to_redo = parse_partials_to_redo_from_agent_output(
            output.json_value, n=nchunks
        )
        relevant_user_feedback_list = []
        if len(partials_to_redo):
            relevant_user_feedback_list = (
                parse_relevant_user_feedback_list_from_agent_output(
                    output.json_value,
                    n=len(partials_to_redo),
                    user_feedback=user_feedback,
                )
            )
        yield (partials_to_redo, relevant_user_feedback_list), True

    async def _identify_partial_transcript_chunks_to_redo_from_user_feedback(
        self, user_feedback, transcript: "Transcript"
    ):
        schema = Schema(
            {"chunks_to_redo": [bool], "relevant_user_feedback_list": [str]}
        ).json_schema("PartialChunkRetryID")

        prompt = load_prompt_template_as_string(
            "identify_partial_transcript_chunks_to_redo_from_user_feedback"
        )

        nchunks = len(transcript.chunks)

        output = None
        async for output, is_last in get_agent_output_modal_or_local(
            prompt,
            schema=schema,
            json_mode=True,
            user_feedback=user_feedback,
            from_cache=self.use_agent_output_cache,
            transcript=[chunk.text for chunk in transcript.chunks],
        ):
            if not is_last:
                yield output, is_last
        assert isinstance(output, FinalLLMOutput) and output.json_value is not None

        print(
            "user feedback in _identify_partial_transcript_chunks_to_redo_from_user_feedback",
            user_feedback,
        )
        partials_to_redo = parse_partials_to_redo_from_agent_output(
            output.json_value, n=nchunks
        )
        relevant_user_feedback_list = []
        if len(partials_to_redo):
            relevant_user_feedback_list = (
                parse_relevant_user_feedback_list_from_agent_output(
                    output.json_value,
                    n=len(partials_to_redo),
                    user_feedback=user_feedback,
                )
            )
            print(
                "relevant_user_feedback_list in if stmt in _identify_partial_transcript_chunks_to_redo_from_user_feedback",
                relevant_user_feedback_list,
            )
        print(
            "relevant_user_feedback_list outside if stmt in _identify_partial_transcript_chunks_to_redo_from_user_feedback",
            relevant_user_feedback_list,
        )
        yield (partials_to_redo, relevant_user_feedback_list), True

    async def _ask_llm_to_parse_user_prompt_for_story_retry(self, user_feedback):
        schema = Schema({"retry": bool}).json_schema("Retry")

        prompt = load_prompt_template_as_string("parse_user_prompt_for_story_retry")

        output = None
        async for output, is_last in get_agent_output_modal_or_local(
            prompt,
            schema=schema,
            json_mode=True,
            user_feedback=user_feedback,
            from_cache=self.use_agent_output_cache,
            story=self.story,
        ):
            if not is_last:
                yield output, is_last
        assert isinstance(output, FinalLLMOutput) and output.json_value is not None

        yield list(output.json_value.values())[0], True

    async def _ask_llm_to_parse_user_prompt_for_speaker_id_retry(self, user_feedback):
        assert self.state is not None
        schema = Schema({"retry": bool}).json_schema("Retry")

        original_prompt = parse_prompt_template(
            "speaker_id",
            user_prompt=self.user_prompt,
            transcript=self.raw_transcript.text_with_speaker_tags,
        )
        # TODO deal with multiple user inputs and multiple system outputs
        conversation = [
            Message(role=PromptStack.SYSTEM_ROLE, message=original_prompt),
            Message(
                role=PromptStack.ASSISTANT_ROLE,
                message=json.dumps(
                    {"on_screen_speakers": self.state.on_screen_speakers}
                ),
            ),
            Message(role=PromptStack.USER_ROLE, message=user_feedback),
            Message(
                role=PromptStack.SYSTEM_ROLE,
                message=load_prompt_template_as_string("generic_json_bool_response"),
            ),
        ]

        output = None
        async for output, is_last in get_agent_output_modal_or_local(
            conversation=conversation,
            schema=schema,
            json_mode=True,
            from_cache=self.use_agent_output_cache,
        ):
            if not is_last:
                yield output, is_last
        assert isinstance(output, FinalLLMOutput) and output.json_value is not None

        yield list(output.json_value.values())[0], True

    async def _ask_llm_to_parse_user_prompt_for_transcript_retry(self, user_feedback):
        assert self.current_transcript is not None, "Transcript must be provided"
        schema = Schema({"retry": bool}).json_schema("Retry")

        prompt = load_prompt_template_as_string(
            "parse_user_prompt_for_transcript_retry"
        )

        output = None
        async for output, is_last in get_agent_output_modal_or_local(
            prompt,
            schema=schema,
            json_mode=True,
            user_feedback=user_feedback,
            from_cache=self.use_agent_output_cache,
            transcript=self.current_transcript.text,
        ):
            if not is_last:
                yield output, is_last
        assert isinstance(output, FinalLLMOutput) and output.json_value is not None
        yield list(output.json_value.values())[0], True

    async def _identify_key_soundbites_partial(
        self,
        partial_transcript: TranscriptChunk,
        existing_soundbite: SoundbitesChunk | None = None,
        user_feedback: str = "",
        max_soundbites: int = 5,
        use_agent_output_cache=True,
    ):
        chunk_soundbites = []
        if existing_soundbite:
            chunk_soundbites = [
                (i, soundbite) for i, soundbite in existing_soundbite.iter_text()
            ]
        prompt = load_prompt_template_as_string("identify_key_soundbites")
        output = None
        async for output, is_last in get_agent_output_modal_or_local(
            prompt,
            partial_transcript=partial_transcript.text,
            narrative_story=self.story,
            user_feedback=user_feedback,
            existing_soundbites=chunk_soundbites,
            user_prompt=self.user_prompt,
            max_soundbites=max_soundbites,
            from_cache=use_agent_output_cache,
        ):
            if not is_last:
                output.chunk = partial_transcript.chunk_index
                yield output, is_last
        assert isinstance(output, FinalLLMOutput) and output.str_value is not None
        soundbites = await SoundbitesChunk.from_keep_tags(
            partial_transcript, output.str_value
        )
        assert isinstance(
            soundbites, SoundbitesChunk
        ), "Expected result to be SoundbitesChunk"
        if len(soundbites.soundbites) > max_soundbites:
            print(
                f"Expected at most {max_soundbites} soundbites, but got {len(soundbites.soundbites)}. Asking agent to remove some."
            )
            async for output, is_last in remove_soundbites(soundbites, max_soundbites):
                if not is_last:
                    output.chunk = partial_transcript.chunk_index
                    yield output, is_last
            assert isinstance(output, FinalLLMOutput) and output.str_value is not None
            soundbites = await SoundbitesChunk.from_keep_tags(
                partial_transcript, output.str_value
            )
        yield soundbites, True

    # TODO figure out how to make this an async generator
    # that can still be called in parallel
    async def _cut_partial_transcript_with_critiques(
        self,
        stage_num: int,
        user_feedback: str,  # TODO turn into list of messages
        partial_transcript: TranscriptChunk,
        kept_soundbites_chunk: SoundbitesChunk,
        existing_cut_transcript: Transcript,
        prev_final_transcript: Transcript | None,
    ):

        # TODO if we are already under the length, skip the whole stage
        common_kwargs = dict(
            stage_num=stage_num,
            existing_cut_transcript=existing_cut_transcript,
            prev_final_transcript=prev_final_transcript,
            user_feedback=user_feedback,
            use_agent_output_cache=self.use_agent_output_cache,
        )
        # chunk_index = partial_transcript.chunk_index
        # yield f"Cutting partial transcript {chunk_index}\n", False
        new_cut_transcript_chunk = None
        async for output, is_last in self._cut_partial_transcript(
            partial_on_screen_transcript=partial_transcript,
            key_soundbites=kept_soundbites_chunk,
            **common_kwargs,
        ):
            if not is_last:
                assert isinstance(output, PartialLLMOutput)
                output.chunk = partial_transcript.chunk_index
                output.calling_method_name = "cut_partial_transcript"
                yield output, False
            else:
                assert isinstance(output, tuple) and len(output) == 2
                assert isinstance(output[0], TranscriptChunk)
                assert isinstance(output[1], SoundbitesChunk)
                new_cut_transcript_chunk, kept_soundbites_chunk = output[0], output[1]
        assert isinstance(new_cut_transcript_chunk, TranscriptChunk)
        assert isinstance(kept_soundbites_chunk, SoundbitesChunk)

        # yield f"Critiquing partial transcript {chunk_index}\n", False
        async for output, is_last in self._critique_cut_transcript(
            assistant_cut_transcript_chunk=new_cut_transcript_chunk,
            key_soundbites=kept_soundbites_chunk,
            **common_kwargs,
        ):
            if not is_last:
                assert isinstance(output, PartialLLMOutput)
                output.chunk = partial_transcript.chunk_index
                output.calling_method_name = "critique_cut_transcript"
                yield output, False
            else:
                assert isinstance(output[0], TranscriptChunk)
                assert isinstance(output[1], SoundbitesChunk)
                new_cut_transcript_chunk, kept_soundbites_chunk = output[0], output[1]
        assert isinstance(new_cut_transcript_chunk, TranscriptChunk)
        yield (new_cut_transcript_chunk, kept_soundbites_chunk), True

    async def _modify_transcript_holistically_single_iteration(
        self,
        step_input: CutTranscriptLinearWorkflowStepInput,
        transcript: Transcript | None = None,
        retry_num: int = 0,
    ):

        user_prompt = step_input.user_prompt or ""
        # TODO
        # llm_modified_prompt = user_feedback.llm_modified_prompt if user_feedback else ""
        # is_retry = user_feedback.is_retry if user_feedback else False
        stage_num = parse_stage_num_from_step_name(step_input.step_name)
        assert stage_num is not None
        desired_words = self._desired_words_for_stage(stage_num)
        first_round = stage_num == 0 and not step_input.is_retry
        stage_num = parse_stage_num_from_step_name(step_input.step_name)
        assert isinstance(stage_num, int)
        transcript = transcript if transcript is not None else self.current_transcript
        assert isinstance(transcript, Transcript)
        soundbites = self.current_soundbites
        assert isinstance(soundbites, Soundbites)
        output = None
        async for (
            output,
            is_last,
        ) in self._agent_interaction_for_modify_transcript_holistically(
            stage_num=stage_num,
            transcript=transcript,
            key_soundbites=soundbites,
            user_feedback=user_prompt,
            allow_reordering=True,  # TODO
            is_first_round=first_round,
            human_retry=step_input.is_retry,
            agent_word_length_retry_num=retry_num,
        ):
            if not is_last:
                yield output, is_last
        assert isinstance(output, tuple) and len(output) == 2

        (final_transcript, kept_soundbites) = output
        assert isinstance(kept_soundbites, Soundbites)
        assert isinstance(final_transcript, Transcript)
        user_feedback_prompt = (
            self._create_user_feedback_prompt_from_modify_final_transcript(
                final_transcript, desired_words, stage_num
            )
        )

        # TODO if user_feedback_prompt in output is None,
        # then we rerun the step with the same user_prompt
        # for max_iterations

        yield CutTranscriptLinearWorkflowStepResults(
            outputs={
                "current_soundbites": kept_soundbites,
                "current_transcript": final_transcript,
            },
            user_feedback_request=user_feedback_prompt,
            retry=self._word_count_excess(final_transcript, desired_words),
        ), True

    def _get_stage_length_seconds(self, stage_num: int):
        return self.stage_lengths[stage_num]

    def _desired_words_for_stage(self, stage_num: int):
        return desired_words_from_length(self._get_stage_length_seconds(stage_num))

    def _word_count_excess(self, transcript, desired_words):
        return (
            transcript.kept_word_count > desired_words + self.max_word_extra_threshold
        )

    async def _agent_interaction_for_modify_transcript_holistically(
        self,
        stage_num: int,
        transcript: Transcript | TranscriptChunk,
        key_soundbites: Soundbites | None,
        allow_reordering: bool = True,
        user_feedback: str = "",
        is_first_round: bool = True,
        human_retry: bool = False,
        agent_word_length_retry_num: int = 0,
    ):
        # TODO based on user_feedback, isntead of looping through the whole thing, have a new prompt which decides which portion of the transcript to work on
        # then grab that portion of the transcript in a secondary prompt to regenerate
        # this is where an agent/tools workflow could come in handy. The tool provided should be to grab a portion of the transcript from user's feedback (e.g. beg/middle/end or something a bit more refined)
        # this is especially useful for the critic, it would allow it to use parts of transcript out of order. And if we know it requested a particular part, we can search through that part in the matching algo

        transcript = transcript.copy()
        stage_length_seconds = self._get_stage_length_seconds(stage_num)
        desired_words = self._desired_words_for_stage(stage_num)
        prev_length_seconds = None
        prev_desired_words = None
        if human_retry or agent_word_length_retry_num > 0:
            prev_length_seconds = stage_length_seconds
            prev_desired_words = desired_words
        elif stage_num > 0:
            prev_length_seconds = self._get_stage_length_seconds(stage_num - 1)
            prev_desired_words = desired_words_from_length(prev_length_seconds)
        else:
            assert is_first_round

        # return transcript, True, user_feedback
        #  print(
        #  add_complete_format(
        #  '\nUsing transcript expansion tool to search for previously removed scenes that match "introduce; my name is"\n',
        #  ["bold", "yellow"],
        #  )
        #  )
        # TODO
        #  tools = [
        #  # this tool allows the agent to find previously removed segments of the transcript
        #  # by passing in phrases of the kept segments nearby
        #  create_transcript_expansion_lookup_tool(transcriptu)
        #  ]
        preamble = parse_prompt_template(
            "modify_high_level_preamble",
            length_seconds=stage_length_seconds,
            total_words=desired_words,
            user_prompt=self.user_prompt,
            narrative_story=self.story,
        )
        output_reqs = parse_prompt_template("shared_output_requirements_no_remove")
        example = parse_prompt_template("shared_example_no_remove")
        extra_notes = parse_prompt_template(
            "modify_extra_notes",
            transcript_nwords=transcript.kept_word_count,
            desired_words=desired_words,
            prev_length_seconds=prev_length_seconds,
            prev_desired_words=prev_desired_words,
            user_feedback=user_feedback,
            is_retry=human_retry or not is_first_round,
            agent_word_length_retry_num=agent_word_length_retry_num,
        )
        task_description = parse_prompt_template(
            "modify_detailed_task_description",
            transcript_nwords=transcript.kept_word_count,
            total_words=desired_words,
        )
        important_summary = parse_prompt_template("shared_important_summary_no_remove")
        prompt = load_prompt_template_as_string("modify_holistically")
        output = None
        async for output, is_last in get_agent_output_modal_or_local(
            prompt=prompt,
            # tools=tools,
            # allow_reordering=allow_reordering,
            transcript=transcript.text,
            high_level_preamble=preamble,
            detailed_task_description=task_description,
            output_requirements=output_reqs,
            example=example,
            important_summary=important_summary,
            extra_notes=extra_notes,
            rules=(
                [get_soundbite_rule(key_soundbites)]
                if key_soundbites is not None and len(key_soundbites.soundbites)
                else None
            ),
            from_cache=self.use_agent_output_cache,
        ):
            if not is_last:
                yield output, False

        assert isinstance(output, FinalLLMOutput) and isinstance(output.str_value, str)

        transcript = match_output_to_actual_transcript_fast(
            transcript, output.str_value
        )

        kept_soundbites = (
            key_soundbites.keep_only_in_transcript(transcript)
            if key_soundbites
            else None
        )
        yield (transcript, kept_soundbites), True

    def _substep_for_step_input(self, step_input):
        step_name = step_input.step_name
        substep_name = step_input.substep_name
        _, substep = self.get_step_by_name(step_name, substep_name)
        return substep
