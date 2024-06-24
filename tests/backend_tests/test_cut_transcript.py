from unittest.mock import patch
from unittest.mock import AsyncMock
from pathlib import Path
from trimit.backend.cut_transcript import (
    CutTranscriptLinearWorkflowStepResults,
    CutTranscriptLinearWorkflowStepInput,
    CutTranscriptLinearWorkflowStepOutput,
)
from trimit.backend.models import (
    Role,
    PartialBackendOutput,
    Soundbite,
    CurrentStepInfo,
    Soundbites,
    Transcript,
    FinalLLMOutput,
    PartialLLMOutput,
    StructuredUserInput,
    RemoveOffScreenSpeakersInput,
    SpeakerTaggingSegmentModification,
)

from trimit.backend.serve import step_workflow_until_feedback_request, step

import os
import pytest

pytestmark = pytest.mark.asyncio()


@pytest.mark.asyncio(scope="function")
async def test_init_state(
    workflow_3909774043_with_transcript,
):  # , video_15557970_with_speakers_in_frame):
    workflow = workflow_3909774043_with_transcript
    step_input = CutTranscriptLinearWorkflowStepInput(user_prompt="make me a video")
    expected_transcript = Transcript.from_video_transcription(
        workflow.video.transcription
    )
    async for output, is_last in workflow.init_state(step_input):
        assert isinstance(output, CutTranscriptLinearWorkflowStepResults)
        assert output.user_feedback_request is None
        assert not output.retry
        assert output.outputs is not None
        assert isinstance(output.outputs.get("current_transcript"), Transcript)
        assert output.outputs["current_transcript"].text == expected_transcript.text
        assert is_last
    assert len(workflow.raw_transcript.text) == 22855
    assert workflow.user_messages == ["make me a video"]


async def test_remove_off_screen_speakers(workflow_3909774043_with_state_init):
    workflow = workflow_3909774043_with_state_init
    stream_outputs = []
    final_outputs = []
    current_substep = None
    async for output, is_last in workflow.step():
        if not is_last:
            if isinstance(output, CurrentStepInfo):
                current_substep = output
            else:
                assert isinstance(output, str)
                stream_outputs.append(output)
        else:
            final_outputs.append(output)
    assert (
        current_substep is not None
        and current_substep.name == "remove_off_screen_speakers"
    )
    assert len(final_outputs) == 1
    final_output = final_outputs[0]
    assert isinstance(final_output, CutTranscriptLinearWorkflowStepOutput)
    assert len(stream_outputs) == 6
    assert workflow.state.on_screen_speakers == ["speaker_01"]
    assert all(
        workflow.on_screen_transcript.segments[s_idx].speaker.lower() == "speaker_01"
        for s_idx in workflow.on_screen_transcript.kept_segments
    )
    assert final_output.export_result is not None
    assert (
        Path(final_output.export_result["video_timeline"]).name
        == "3909774043_remove_off_screen_speakers_timeline_0.xml"
    )


async def test_decide_retry_speaker_id_false(
    workflow_3909774043_with_state_init, soundbites_3909774043
):
    workflow = workflow_3909774043_with_state_init
    step_name = "remove_off_screen_speakers"
    workflow.state.on_screen_speakers = ["speaker_01"]
    workflow.state.dynamic_state_step_order = [step_name]
    workflow.state.outputs[step_name] = CutTranscriptLinearWorkflowStepOutput(
        step_name=step_name, done=False, user_feedback_request=""
    )
    retry, _ = await workflow._decide_retry(step_name, user_prompt="Good job")
    assert not retry


async def test_decide_retry_speaker_id_true(
    workflow_3909774043_with_state_init, soundbites_3909774043
):
    workflow = workflow_3909774043_with_state_init
    step_name = "remove_off_screen_speakers"
    workflow.state.on_screen_speakers = ["speaker_01", "speaker_02"]
    workflow.state.dynamic_state_step_order = [step_name]
    workflow.state.outputs[step_name] = CutTranscriptLinearWorkflowStepOutput(
        step_name=step_name, done=False, user_feedback_request=""
    )
    retry, _ = await workflow._decide_retry(
        step_name, user_prompt="Actually I don't think we should include speaker_02"
    )
    assert retry


async def test_decide_retry_story_true(
    workflow_3909774043_with_state_init, story_3909774043
):
    workflow = workflow_3909774043_with_state_init
    step_name = "generate_story"
    workflow.state.on_screen_speakers = ["speaker_01"]
    workflow.state.dynamic_state_step_order = [step_name]
    workflow.state.outputs[step_name] = CutTranscriptLinearWorkflowStepOutput(
        step_name=step_name, done=False, user_feedback_request=""
    )
    workflow.state.story = story_3909774043
    retry, _ = await workflow._decide_retry(
        step_name, user_prompt="Actually can you make it 5 chapters?"
    )
    assert retry


async def test_decide_retry_story_false(
    workflow_3909774043_with_state_init, story_3909774043
):
    workflow = workflow_3909774043_with_state_init
    step_name = "generate_story"
    workflow.state.on_screen_speakers = ["speaker_01"]
    workflow.state.dynamic_state_step_order = [step_name]
    workflow.state.outputs[step_name] = CutTranscriptLinearWorkflowStepOutput(
        step_name=step_name, done=False, user_feedback_request=""
    )
    workflow.state.story = story_3909774043
    retry, _ = await workflow._decide_retry(
        step_name, user_prompt="What a compelling narrative!"
    )
    assert not retry


async def test_decide_retry_soundbites(
    workflow_3909774043_with_state_init, soundbites_3909774043
):
    # TODO test the actual retry handling of just one chunk
    workflow = workflow_3909774043_with_state_init
    workflow.state.current_soundbites_state = soundbites_3909774043.state
    step_name = "identify_key_soundbites"
    workflow.state.dynamic_state_step_order = [step_name]
    workflow.state.outputs[step_name] = CutTranscriptLinearWorkflowStepOutput(
        step_name=step_name, done=False, user_feedback_request=""
    )
    retry, step_input = await workflow._decide_retry(
        step_name, user_prompt="remove soundbite 6"
    )
    assert retry
    assert step_input.llm_modified_partial_feedback.partials_to_redo == [
        False,
        False,
        True,
        False,
    ]
    assert [
        (x or "").lower()
        for x in step_input.llm_modified_partial_feedback.relevant_user_feedback_list
    ] == ["", "", "remove soundbite 6", ""]


async def test_retry_soundbites_step(workflow_3909774043_with_state_init):
    workflow = workflow_3909774043_with_state_init
    step_name = "identify_key_soundbites"
    output = None
    while workflow._get_last_step_with_index()[1].name != step_name:
        async for output, _ in workflow.step():
            pass
    assert workflow.current_soundbites is not None
    n_soundbites_before = len(workflow.current_soundbites.soundbites)
    all_other_sbs = [sb for i, sb in workflow.current_soundbites.iter_text() if i != 6]
    async for output, _ in workflow.step("remove soundbite 6"):
        pass
    assert isinstance(output, CutTranscriptLinearWorkflowStepOutput)
    n_soundbites_after = len(workflow.current_soundbites.soundbites)
    assert n_soundbites_after == n_soundbites_before - 1
    assert [sb for _, sb in workflow.current_soundbites.iter_text()] == all_other_sbs
    assert output.step_name == step_name


async def test_retry_remove_off_screen_speakers_with_structured_user_input(
    workflow_3909774043_with_state_init_no_export,
):
    workflow = workflow_3909774043_with_state_init_no_export
    step_name = "remove_off_screen_speakers"
    output = None
    while (await workflow.get_last_substep(with_load_state=False)).name != step_name:
        async for output, _ in workflow.step(
            load_state=False, save_state_to_db=False, async_export=False
        ):
            pass
    structured_user_input = StructuredUserInput(
        remove_off_screen_speakers=RemoveOffScreenSpeakersInput(
            segments=[
                SpeakerTaggingSegmentModification(
                    index=1, speaker="Ruta Gupta", on_screen=True
                ),
                SpeakerTaggingSegmentModification(
                    index=4, speaker="Interviewer", on_screen=False
                ),
                SpeakerTaggingSegmentModification(
                    index=18, speaker="Interviewer", on_screen=False
                ),
            ]
        )
    )
    old_transcript = workflow.on_screen_transcript.copy()
    async for output, _ in workflow.step(
        load_state=False,
        save_state_to_db=False,
        async_export=False,
        user_feedback="",
        structured_user_input=structured_user_input,
        retry_step=True,
    ):
        pass
    assert isinstance(output, CutTranscriptLinearWorkflowStepOutput)
    assert workflow.video.speakers_in_frame == ["Ruta Gupta"]
    new_transcript = workflow.on_screen_transcript
    assert new_transcript.kept_segments == old_transcript.kept_segments - {18}
    for i, (old_segment, new_segment) in enumerate(
        zip(old_transcript.segments, new_transcript.segments)
    ):
        if i == 18:
            assert new_segment.speaker == "Interviewer"
        elif old_segment.speaker == "SPEAKER_00":
            assert new_segment.speaker == "Ruta Gupta"
        elif old_segment.speaker == "SPEAKER_01":
            assert new_segment.speaker == "Interviewer"
        else:
            assert False


async def test_decide_retry_transcript_chunks(
    workflow_3909774043_with_state_init, soundbites_3909774043
):
    # TODO test the actual retry handling of just one chunk
    workflow = workflow_3909774043_with_state_init
    transcript = Transcript.load_from_state(workflow.state.current_transcript_state)
    transcript.split_in_chunks(workflow.max_partial_transcript_words)
    workflow.state.current_transcript_state = transcript.state
    step_name = "stage_0_cut_partial_transcripts_with_critiques"
    workflow.state.dynamic_state_step_order = [step_name]
    workflow.state.outputs[step_name] = CutTranscriptLinearWorkflowStepOutput(
        step_name=step_name, done=False, user_feedback_request=""
    )
    retry, step_input = await workflow._decide_retry(
        step_name, user_prompt="drop everything from the last chunk"
    )
    assert retry
    assert step_input.llm_modified_partial_feedback.partials_to_redo == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert [
        (x or "").lower()
        for x in step_input.llm_modified_partial_feedback.relevant_user_feedback_list
    ] == ["", "", "", "", "drop everything from the last chunk"]


async def test_retry_partial_transcript_step(
    workflow_3909774043_with_state_init_no_export,
):
    workflow = workflow_3909774043_with_state_init_no_export
    substep_name = "modify_transcript_holistically"
    output = None
    while (await workflow.get_last_substep(with_load_state=False)).name != substep_name:
        async for output, _ in workflow.step(async_export=False):
            pass
    assert workflow.current_transcript is not None
    chunks_before = workflow.current_transcript.chunks
    assert len(chunks_before) == 4
    third_chunk = workflow.current_transcript.chunks[2]
    assert len(third_chunk.text) > 0
    async for output, _ in workflow.step(
        "drop everything from the third chunk, even the text that aligns with the narrative story",
        async_export=False,
        retry_step=True,
    ):
        pass
    assert isinstance(output, CutTranscriptLinearWorkflowStepOutput)
    assert len(workflow.current_transcript.chunks) == 4
    for i, (before_chunk, after_chunk) in enumerate(
        zip(chunks_before, workflow.current_transcript.chunks)
    ):
        assert len(after_chunk.text) <= len(before_chunk.text)
        #  # TODO this isn't working anymore
        #  if i == 2:
        #  assert after_chunk.text == ""
        #  else:
        #  assert before_chunk.text == after_chunk.text
    assert output.step_name == "stage_0_generate_transcript"
    assert output.substep_name == substep_name
    assert output.retry_num == 1

    assert len(output.conversation) == 2
    assert output.conversation[0].role == Role.Human
    assert output.conversation[1].role == Role.AI
    assert (
        output.conversation[0].value
        == "drop everything from the third chunk, even the text that aligns with the narrative story"
    )
    assert (
        "Do you have any feedback to provide the AI assistant?"
        in output.conversation[1].value
    )

    saved_outputs = [
        o
        for o in workflow.state.outputs[
            "stage_0_generate_transcript.modify_transcript_holistically"
        ]
    ]
    assert len(saved_outputs) == 2
    assert saved_outputs[0].retry_num == 0
    assert saved_outputs[1].retry_num == 1
    prior_output_transcript = Transcript.load_from_state(
        saved_outputs[0].step_outputs["current_transcript_state"]
    )
    assert prior_output_transcript.chunks[-1].text == chunks_before[-1].text

    assert len(saved_outputs[0].conversation) == 2
    assert len(saved_outputs[1].conversation) == 2

    assert saved_outputs[0].conversation[0].role == Role.Human
    assert saved_outputs[0].conversation[1].role == Role.AI
    assert saved_outputs[0].conversation[0].value == ""
    assert (
        "Do you have any feedback to provide the AI assistant?"
        in saved_outputs[0].conversation[1].value
    )
    assert saved_outputs[0].step_inputs.prior_conversation == []

    assert saved_outputs[1].conversation[0].role == Role.Human
    assert saved_outputs[1].conversation[1].role == Role.AI
    assert (
        saved_outputs[1].conversation[0].value
        == "drop everything from the third chunk, even the text that aligns with the narrative story"
    )
    assert (
        "Do you have any feedback to provide the AI assistant?"
        in saved_outputs[1].conversation[1].value
    )
    assert (
        saved_outputs[1].step_inputs.prior_conversation == saved_outputs[0].conversation
    )


def assert_under_word_count_threshold(workflow, transcript, stage):
    desired_words = workflow._desired_words_for_stage(stage)
    assert (
        transcript.kept_word_count < desired_words + workflow.max_word_extra_threshold
    )


async def test_retry_transcript_no_chunks(workflow_3909774043_with_state_init):
    workflow = workflow_3909774043_with_state_init
    step_name = "stage_0_modify_transcript_holistically"
    output = None
    while workflow._get_last_step_with_index()[1].name != step_name:
        async for output, _ in workflow.step():
            pass
    assert workflow.current_transcript is not None
    async for output in workflow.step(
        "Remove all segments that mention Circana (or mispelled like Surkana). Do not alter text to remove the mentions- just remove the entire segment if it mentions Circana"
    ):
        pass
    assert isinstance(output, CutTranscriptLinearWorkflowStepOutput)
    lowered_transcript_text = workflow.current_transcript.text.lower()
    assert "circana" not in lowered_transcript_text
    assert "cercana" not in lowered_transcript_text
    assert "surkana" not in lowered_transcript_text
    assert "sorkana" not in lowered_transcript_text
    assert output.step_name == step_name


async def test_retry_modify_transcript_holistically_due_to_word_count(
    workflow_3909774043_with_transcript,
):
    workflow = workflow_3909774043_with_transcript

    workflow.state.static_state.export_video = False
    workflow.state.static_state.export_timeline = False
    workflow.state.static_state.export_transcript = False
    workflow.state.static_state.export_transcript_text = False
    workflow.state.static_state.export_soundbites = False
    workflow.state.static_state.export_soundbites_text = False
    workflow.state.static_state.export_speaker_tagging = False
    while True:
        output = None
        is_last = False
        async for output, is_last in workflow.step(
            "", load_state=False, save_state_to_db=False, async_export=False
        ):
            pass
        assert output is not None and is_last
        if output.substep_name == "cut_partial_transcripts_with_critiques":
            break
    current_step, current_substep_index = (
        await workflow._get_next_step_with_user_feedback("")
    )

    current_substep = current_step.substeps[current_substep_index]

    original_modify_method = workflow._modify_transcript_holistically_single_iteration
    nretries = [0]
    previous_transcript = [None]

    async def modify_spy(
        step_input: CutTranscriptLinearWorkflowStepInput,
        transcript: Transcript | None = None,
        retry_num: int = 0,
    ):
        if retry_num == 0:
            previous_transcript[0] = workflow.current_transcript
        if retry_num == 1:
            transcript = previous_transcript[0]
        output = None
        async for output, is_last in original_modify_method(
            step_input, transcript=transcript, retry_num=retry_num
        ):
            yield output, is_last
        assert isinstance(output, CutTranscriptLinearWorkflowStepResults)
        assert isinstance(output.outputs, dict)
        nretries[0] += 1

    workflow._modify_transcript_holistically_single_iteration = modify_spy
    async for output, is_last in workflow.modify_transcript_holistically(
        current_substep.input
    ):
        pass
    assert nretries[0] > 1
    assert output is not None and isinstance(output.outputs, dict)
    modified_transcript = output.outputs["current_transcript"]
    assert not workflow._word_count_excess(
        modified_transcript, workflow._desired_words_for_stage(0)
    )


async def test_step_until_finish_no_db_save(workflow_3909774043_with_transcript):
    workflow = workflow_3909774043_with_transcript

    user_inputs = ["make me a video", "", "", "", "", "", "", ""]
    expected_step_names = [
        "init_state",
        "remove_off_screen_speakers",
        "generate_story",
        "identify_key_soundbites",
        "cut_partial_transcripts_with_critiques",
        "modify_transcript_holistically",
        "cut_partial_transcripts_with_critiques",
        "modify_transcript_holistically",
        "end",
    ]

    step_outputs = []
    str_outputs_by_step = {}
    step_info_outputs_by_step = {}
    i = 0
    while True:
        assert i < len(user_inputs)
        str_outputs = []
        step_info_outputs = []
        final_llm_outputs = []
        async for output, is_last in step_workflow_until_feedback_request(
            workflow,
            user_inputs[i],
            load_state=False,
            save_state_to_db=False,
            async_export=False,
        ):
            if is_last:
                assert isinstance(output, CutTranscriptLinearWorkflowStepOutput)
                step_outputs.append(output)
                step_key = f"{output.step_name}.{output.substep_name}"
                str_outputs_by_step[step_key] = str_outputs
                step_info_outputs_by_step[step_key] = step_info_outputs
                str_outputs = []
                step_info_outputs = []
            else:
                assert isinstance(
                    output, (PartialLLMOutput, FinalLLMOutput, PartialBackendOutput)
                )
                if (
                    isinstance(output, PartialBackendOutput)
                    and output.current_substep is not None
                ):
                    step_info_outputs.append(output.current_substep)
                elif isinstance(output, PartialLLMOutput):
                    str_outputs.append(output)
                elif isinstance(output, FinalLLMOutput):
                    final_llm_outputs.append(output)
        assert len(step_outputs)

        if len(step_outputs) < len(expected_step_names):
            assert (
                await workflow.get_last_substep(with_load_state=False)
            ).name == expected_step_names[len(step_outputs) - 1]
            if len(step_outputs) < len(expected_step_names) - 1:
                assert (
                    await workflow.get_next_substep(with_load_state=False)
                ).name == expected_step_names[len(step_outputs)]
            else:
                assert await workflow.get_next_substep(with_load_state=False) is None
        else:
            # last step will never be "end", we always return the last real step we ran
            assert (
                await workflow.get_last_substep(with_load_state=False)
            ).name == expected_step_names[-2]
            assert await workflow.get_next_substep(with_load_state=False) is None
        if step_outputs[-1].done:
            break
        i += 1
    assert all(isinstance(o.value, str) for o in str_outputs)
    assert all(f"{o.name}" == e for o, e in zip(step_info_outputs, expected_step_names))
    assert len(final_llm_outputs) == 0
    assert len(step_outputs) == len(expected_step_names)
    assert [s.substep_name == e for s, e in zip(step_outputs, expected_step_names)]

    # try after we finish
    result = None
    async for result, is_last in workflow.step(
        "", load_state=False, save_state_to_db=False, async_export=False
    ):
        pass
    assert result is not None
    assert result.step_name == "stage_1_generate_transcript"
    assert result.substep_name == "modify_transcript_holistically"
    step_outputs.append(result)

    speaker_tagging_output = (
        await workflow.get_output_for_keys(
            ["preprocess_video.remove_off_screen_speakers"], with_load_state=False
        )
    )[0]
    speaker_tagging_clips = speaker_tagging_output.export_result[
        "speaker_tagging_clips"
    ]
    assert len(speaker_tagging_clips) == 2
    assert "SPEAKER_00" in speaker_tagging_clips
    assert "SPEAKER_01" in speaker_tagging_clips

    soundbites_output = (
        await workflow.get_output_for_keys(
            ["identify_key_soundbites.identify_key_soundbites"], with_load_state=False
        )
    )[0]
    assert "hypothetical" not in " ".join(
        [
            s.value
            for s in str_outputs_by_step[
                "identify_key_soundbites.identify_key_soundbites"
            ]
        ]
    )
    output_files = soundbites_output.export_result
    for file_key in [
        "soundbites_transcript",
        "soundbites_text",
        "soundbites_videos",
        "soundbites_timeline",
    ]:
        assert file_key in output_files
        if isinstance(output_files[file_key], list):
            for f in output_files[file_key]:
                assert os.stat(f).st_size > 0
        else:
            assert os.stat(output_files[file_key]).st_size > 0

    output = await workflow.get_last_output_before_end(with_load_state=False)
    output_files = output.export_result
    for file_key in ["video_timeline", "transcript", "transcript_text"]:
        assert file_key in output_files
        if isinstance(output_files[file_key], list):
            for f in output_files[file_key]:
                assert os.stat(f).st_size > 0
        else:
            assert os.stat(output_files[file_key]).st_size > 0

    all_outputs = await workflow.get_all_outputs(
        only_last_substep=False, with_load_state=False
    )
    assert all(
        "video_timeline" in substep_output.export_result
        for substep_output in all_outputs
        if substep_output.substep_name
        in ("remove_off_screen_speakers", "modify_transcript_holistically")
    )
    assert all(
        "video_timeline" not in substep_output.export_result
        for substep_output in all_outputs
        if substep_output.substep_name
        not in ("end", "remove_off_screen_speakers", "modify_transcript_holistically")
    )

    output_for_steps = await workflow.get_output_for_keys(
        [
            "stage_0_generate_transcript.modify_transcript_holistically",
            "stage_1_generate_transcript.cut_partial_transcripts_with_critiques",
            "stage_1_generate_transcript.modify_transcript_holistically",
        ],
        with_load_state=False,
    )
    step0_tl_file = output_for_steps[0].export_result.get("video_timeline")
    assert step0_tl_file and os.stat(step0_tl_file).st_size > 0

    first_stage_transcript = Transcript.load_from_state(
        output_for_steps[0].step_outputs["current_transcript_state"]
    )
    assert_under_word_count_threshold(workflow, first_stage_transcript, 0)
    second_stage_transcript = Transcript.load_from_state(
        output_for_steps[2].step_outputs["current_transcript_state"]
    )
    assert_under_word_count_threshold(workflow, second_stage_transcript, 1)

    assert len(workflow.story) == 1977
    assert workflow.story == step_outputs[2].step_outputs["story"]

    for output_idx, step_key in zip(
        [3, 5, 7],
        [
            "identify_key_soundbites.identify_key_soundbites",
            "stage_0_generate_transcript.modify_transcript_holistically",
            "stage_1_generate_transcript.modify_transcript_holistically",
        ],
    ):
        parent_obj = Soundbites.load_from_state(
            step_outputs[output_idx].step_outputs["current_soundbites_state"]
        )
        assert [
            (
                Soundbite.from_dict_with_parent_obj(parent_obj=parent_obj, **sb)
                if isinstance(sb, dict)
                else Soundbite.from_dict_with_parent_obj(
                    parent_obj=parent_obj,
                    start_segment_index=sb.start_segment_index,
                    start_word_index=sb.start_word_index,
                    end_segment_index=sb.end_segment_index,
                    end_word_index=sb.end_word_index,
                )
            )
            for sb in workflow.state.outputs[step_key][-1].step_outputs[
                "current_soundbites_state"
            ]["soundbites"]
        ] == parent_obj.soundbites

    for output_idx, retry_num, step_key in zip(
        [5, 7, 9],
        [-1, 0, -1],
        [
            "stage_0_generate_transcript.modify_transcript_holistically",
            "stage_1_generate_transcript.modify_transcript_holistically",
        ],
    ):
        assert (
            Transcript.load_from_state(
                workflow.state.outputs[step_key][retry_num].step_outputs[
                    "current_transcript_state"
                ]
            ).kept_word_count
            == Transcript.load_from_state(
                step_outputs[output_idx].step_outputs["current_transcript_state"]
            ).kept_word_count
        )


async def test_step_until_finish_with_db_save(workflow_3909774043_with_transcript):
    workflow = workflow_3909774043_with_transcript

    user_inputs = ["make me a video", "", "", "", "", "", "", ""]
    expected_step_names = [
        "init_state",
        "remove_off_screen_speakers",
        "generate_story",
        "identify_key_soundbites",
        "cut_partial_transcripts_with_critiques",
        "modify_transcript_holistically",
        "cut_partial_transcripts_with_critiques",
        "modify_transcript_holistically",
        "end",
    ]

    step_outputs = []
    str_outputs = []
    i = 0
    while True:
        assert i < len(user_inputs)
        if "cut_partial_transcripts_with_critiques" in [
            s.substep_name for s in step_outputs
        ]:
            workflow.state.static_state.export_video = True
            await workflow.state.save()
        async for output, is_last in step_workflow_until_feedback_request(
            workflow,
            user_inputs[i],
            load_state=True,
            save_state_to_db=True,
            async_export=True,
        ):
            if is_last:
                assert isinstance(output, CutTranscriptLinearWorkflowStepOutput)
                step_outputs.append(output)
            else:
                str_outputs.append(output)
        assert len(step_outputs)

        if len(step_outputs) < len(expected_step_names):
            assert (
                await workflow.get_last_substep(with_load_state=False)
            ).name == expected_step_names[len(step_outputs) - 1]
            if len(step_outputs) < len(expected_step_names) - 1:
                assert (
                    await workflow.get_next_substep(with_load_state=False)
                ).name == expected_step_names[len(step_outputs)]
            else:
                assert await workflow.get_next_substep(with_load_state=False) is None
        else:
            # last step will never be "end", we always return the last real step we ran
            assert (
                await workflow.get_last_substep(with_load_state=False)
            ).name == expected_step_names[-2]
            assert await workflow.get_next_substep(with_load_state=False) is None
        if step_outputs[-1].done:
            break
        i += 1
    assert len(step_outputs) == len(expected_step_names)
    assert [s.substep_name == e for s, e in zip(step_outputs, expected_step_names)]

    # try after we finish
    result = None
    async for result, is_last in workflow.step(
        "", load_state=True, save_state_to_db=True, async_export=True
    ):
        pass
    assert result is not None
    assert result.step_name == "stage_1_generate_transcript"
    assert result.substep_name == "modify_transcript_holistically"
    step_outputs.append(result)

    soundbites_output = (
        await workflow.get_output_for_keys(
            ["identify_key_soundbites.identify_key_soundbites"], with_load_state=False
        )
    )[0]
    output_files = soundbites_output.export_result
    for file_key in [
        "soundbites_transcript",
        "soundbites_text",
        "soundbites_videos",
        "soundbites_timeline",
    ]:
        assert file_key in output_files
        if isinstance(output_files[file_key], list):
            for f in output_files[file_key]:
                assert os.stat(f).st_size > 0
        else:
            assert os.stat(output_files[file_key]).st_size > 0

    last_output = await workflow.get_last_output_before_end(with_load_state=False)
    output_files = last_output.export_result
    for file_key in ["video_timeline", "video", "transcript", "transcript_text"]:
        assert file_key in output_files
        if isinstance(output_files[file_key], list):
            for f in output_files[file_key]:
                assert os.stat(f).st_size > 0
        else:
            assert os.stat(output_files[file_key]).st_size > 0
    assert (
        Path(output_files["video"]).name
        == "3909774043_modify_transcript_holistically_video_1.mp4"
    )

    all_outputs = await workflow.get_all_outputs(
        only_last_substep=False, with_load_state=False
    )
    assert all(
        "video_timeline" in substep_output.export_result
        for substep_output in all_outputs
        if substep_output.substep_name
        in ("remove_off_screen_speakers", "modify_transcript_holistically")
    )
    assert all(
        "video_timeline" not in substep_output.export_result
        for substep_output in all_outputs
        if substep_output.substep_name
        not in ("end", "remove_off_screen_speakers", "modify_transcript_holistically")
    )

    output_for_steps = await workflow.get_output_for_keys(
        [
            "stage_0_generate_transcript.modify_transcript_holistically",
            "stage_1_generate_transcript.cut_partial_transcripts_with_critiques",
            "stage_1_generate_transcript.modify_transcript_holistically",
        ],
        with_load_state=False,
    )
    step0_tl_file = output_for_steps[0].export_result.get("video_timeline")
    assert step0_tl_file and os.stat(step0_tl_file).st_size > 0

    first_stage_transcript = Transcript.load_from_state(
        output_for_steps[0].step_outputs["current_transcript_state"]
    )
    assert_under_word_count_threshold(workflow, first_stage_transcript, 0)
    second_stage_transcript = Transcript.load_from_state(
        output_for_steps[2].step_outputs["current_transcript_state"]
    )
    assert_under_word_count_threshold(workflow, second_stage_transcript, 1)

    assert len(workflow.story) == 1977
    assert workflow.story == step_outputs[2].step_outputs["story"]

    for output_idx, step_key in zip(
        [3, 5, 7],
        [
            "identify_key_soundbites.identify_key_soundbites",
            "stage_0_generate_transcript.modify_transcript_holistically",
            "stage_1_generate_transcript.modify_transcript_holistically",
        ],
    ):
        parent_obj = Soundbites.load_from_state(
            step_outputs[output_idx].step_outputs["current_soundbites_state"]
        )
        assert [
            Soundbite.from_dict_with_parent_obj(parent_obj=parent_obj, **sb)
            for sb in workflow.state.outputs[step_key][-1].step_outputs[
                "current_soundbites_state"
            ]["soundbites"]
        ] == parent_obj.soundbites

    for output_idx, retry_num, step_key in zip(
        [5, 7, 9],
        [-1, 0, -1],
        [
            "stage_0_generate_transcript.modify_transcript_holistically",
            "stage_1_generate_transcript.modify_transcript_holistically",
        ],
    ):
        assert (
            Transcript.load_from_state(
                workflow.state.outputs[step_key][retry_num].step_outputs[
                    "current_transcript_state"
                ]
            ).kept_word_count
            == Transcript.load_from_state(
                step_outputs[output_idx].step_outputs["current_transcript_state"]
            ).kept_word_count
        )


async def test_step_by_name(workflow_3909774043_with_transcript):
    workflow = workflow_3909774043_with_transcript

    given_step_order = [
        "preprocess_video.remove_off_screen_speakers",
        "identify_key_soundbites.identify_key_soundbites",
        "stage_0_generate_transcript.modify_transcript_holistically",
        "preprocess_video.remove_off_screen_speakers",
        "stage_1_generate_transcript.modify_transcript_holistically",
    ]
    expected_steps_ran = [
        "preprocess_video.init_state",
        "preprocess_video.remove_off_screen_speakers",
        "generate_story.generate_story",
        "identify_key_soundbites.identify_key_soundbites",
        "stage_0_generate_transcript.cut_partial_transcripts_with_critiques",
        "stage_0_generate_transcript.modify_transcript_holistically",
        "preprocess_video.init_state",
        "preprocess_video.remove_off_screen_speakers",
        "generate_story.generate_story",
        "identify_key_soundbites.identify_key_soundbites",
        "stage_0_generate_transcript.cut_partial_transcripts_with_critiques",
        "stage_0_generate_transcript.modify_transcript_holistically",
        "stage_1_generate_transcript.cut_partial_transcripts_with_critiques",
        "stage_1_generate_transcript.modify_transcript_holistically",
    ]
    standard_step_order = [
        "preprocess_video.init_state",
        "preprocess_video.remove_off_screen_speakers",
        "generate_story.generate_story",
        "identify_key_soundbites.identify_key_soundbites",
        "stage_0_generate_transcript.cut_partial_transcripts_with_critiques",
        "stage_0_generate_transcript.modify_transcript_holistically",
        "stage_1_generate_transcript.cut_partial_transcripts_with_critiques",
        "stage_1_generate_transcript.modify_transcript_holistically",
    ]

    step_outputs = []
    for step_name in given_step_order:
        async for output, is_last in workflow.step_by_key(
            step_name, "", load_state=False, save_state_to_db=False, async_export=False
        ):
            if is_last:
                assert isinstance(output, CutTranscriptLinearWorkflowStepOutput)
                step_outputs.append(output)

    assert [
        f"{output.step_name}.{output.name}" for output in step_outputs
    ] == expected_steps_ran
    actual_step_order = []
    for step in workflow.state.dynamic_state_step_order:
        for substep in step.substeps:
            actual_step_order.append(f"{step.name}.{substep}")
    assert actual_step_order == standard_step_order


async def test_step_workflow_until_feedback_request_advance_until(
    workflow_3909774043_with_transcript_no_export,
):
    workflow = workflow_3909774043_with_transcript_no_export
    async for _ in step_workflow_until_feedback_request(
        workflow,
        "",
        load_state=False,
        save_state_to_db=False,
        async_export=False,
        advance_until="preprocess_video.remove_off_screen_speakers",
    ):
        pass
    assert (
        await workflow.get_last_substep(with_load_state=False)
    ).name == "remove_off_screen_speakers"
    async for _ in step_workflow_until_feedback_request(
        workflow,
        "",
        load_state=False,
        save_state_to_db=False,
        async_export=False,
        advance_until="identify_key_soundbites.identify_key_soundbites",
    ):
        pass
    assert (
        await workflow.get_last_substep(with_load_state=False)
    ).name == "identify_key_soundbites"


async def test_step_advance_until_int(workflow_3909774043_with_transcript_no_export):
    workflow = workflow_3909774043_with_transcript_no_export
    async for _ in step.local(
        workflow,
        "",
        load_state=False,
        save_state_to_db=False,
        async_export=False,
        advance_until=2,
        ignore_running_workflows=True,
    ):
        pass
    assert (
        await workflow.get_last_substep(with_load_state=False)
    ).name == "identify_key_soundbites"

    async for _ in step.local(
        workflow,
        "",
        load_state=False,
        save_state_to_db=False,
        async_export=False,
        advance_until=1,
        ignore_running_workflows=True,
    ):
        pass
    assert (
        await workflow.get_last_substep(with_load_state=False)
    ).name == "generate_story"

    async for _ in step.local(
        workflow,
        "new_input",
        load_state=False,
        save_state_to_db=False,
        async_export=False,
        advance_until=0,
        ignore_running_workflows=True,
    ):
        pass
    assert (
        await workflow.get_last_substep(with_load_state=False)
    ).name == "remove_off_screen_speakers"
    assert (
        workflow.state.outputs["preprocess_video.remove_off_screen_speakers"][
            0
        ].step_inputs.user_prompt
        == "new_input"
    )
