from trimit.backend.cut_transcript import (
    CutTranscriptLinearWorkflowStepResults,
    CutTranscriptLinearWorkflowStepInput,
    CutTranscriptLinearWorkflowStepOutput,
)
from trimit.backend.models import Soundbite
from trimit.backend.serve import step_workflow_until_feedback_request
from trimit.backend.models import Transcript

import os
import pytest

pytestmark = pytest.mark.asyncio()


@pytest.mark.asyncio(scope="function")
async def test_init_state(mongo_connect, workflow_3909774043_with_transcript_small):
    workflow = workflow_3909774043_with_transcript_small
    step_input = CutTranscriptLinearWorkflowStepInput(user_prompt="make me a video")
    async for output in workflow.init_state(step_input):
        assert output == CutTranscriptLinearWorkflowStepResults()
    assert len(workflow.raw_transcript.text) == 22861
    assert workflow.user_messages == ["make me a video"]


async def test_remove_off_screen_speakers(workflow_3909774043_with_state_init):
    workflow = workflow_3909774043_with_state_init
    stream_outputs = []
    final_outputs = []
    async for output in workflow.remove_off_screen_speakers(
        CutTranscriptLinearWorkflowStepInput(
            user_prompt="", llm_modified_prompt="", is_retry=False
        )
    ):
        if isinstance(output, str):
            stream_outputs.append(output)
        else:
            final_outputs.append(output)
    assert len(final_outputs) == 1
    final_output = final_outputs[0]
    assert isinstance(final_output, CutTranscriptLinearWorkflowStepResults)
    # assert len(stream_outputs) == 10
    # assert workflow.state.on_screen_speakers == ["speaker_01"]
    assert [
        workflow.on_screen_transcript.segments[s_idx].speaker == "speaker_01"
        for s_idx in workflow.on_screen_transcript.kept_segments
    ]


async def test_decide_retry_speaker_id_false(
    workflow_3909774043_with_state_init, soundbites_3909774043
):
    workflow = workflow_3909774043_with_state_init
    step_name = "remove_off_screen_speakers"
    workflow.state.on_screen_speakers = ["speaker_01"]
    workflow.state.dynamic_state_step_order = [step_name]
    workflow.state.dynamic_state[step_name] = CutTranscriptLinearWorkflowStepOutput(
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
    workflow.state.dynamic_state[step_name] = CutTranscriptLinearWorkflowStepOutput(
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
    workflow.state.dynamic_state[step_name] = CutTranscriptLinearWorkflowStepOutput(
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
    workflow.state.dynamic_state[step_name] = CutTranscriptLinearWorkflowStepOutput(
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
    workflow.state.dynamic_state[step_name] = CutTranscriptLinearWorkflowStepOutput(
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
        async for output in workflow.step():
            pass
    assert workflow.current_soundbites is not None
    n_soundbites_before = len(workflow.current_soundbites.soundbites)
    all_other_sbs = [sb for i, sb in workflow.current_soundbites.iter_text() if i != 6]
    async for output in workflow.step("remove soundbite 6"):
        pass
    assert isinstance(output, CutTranscriptLinearWorkflowStepOutput)
    n_soundbites_after = len(workflow.current_soundbites.soundbites)
    assert n_soundbites_after == n_soundbites_before - 1
    assert [sb for _, sb in workflow.current_soundbites.iter_text()] == all_other_sbs
    assert output.step_name == step_name


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
    workflow.state.dynamic_state[step_name] = CutTranscriptLinearWorkflowStepOutput(
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


async def test_retry_partial_transcript_step(workflow_3909774043_with_state_init):
    workflow = workflow_3909774043_with_state_init
    step_name = "stage_0_cut_partial_transcripts_with_critiques"
    output = None
    while workflow._get_last_step_with_index()[1].name != step_name:
        async for output in workflow.step():
            pass
    assert workflow.current_transcript is not None
    chunks_before = workflow.current_transcript.chunks
    assert len(chunks_before) == 5
    last_chunk = workflow.current_transcript.chunks[-1]
    assert len(last_chunk.text) > 0
    async for output in workflow.step(
        "drop everything from the last chunk, even the text that aligns with the narrative story"
    ):
        pass
    assert isinstance(output, CutTranscriptLinearWorkflowStepOutput)
    assert len(workflow.current_transcript.chunks) == 5
    for i, (before_chunk, after_chunk) in enumerate(
        zip(chunks_before, workflow.current_transcript.chunks)
    ):
        if i < 4:
            assert before_chunk.text == after_chunk.text
        else:
            assert after_chunk.text == ""
    assert output.step_name == step_name


async def test_retry_transcript_no_chunks(workflow_3909774043_with_state_init):
    workflow = workflow_3909774043_with_state_init
    step_name = "stage_0_modify_transcript_holistically"
    output = None
    while workflow._get_last_step_with_index()[1].name != step_name:
        async for output in workflow.step():
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


async def test_step_until_finish(workflow_3909774043_with_transcript):
    workflow = workflow_3909774043_with_transcript

    user_inputs = ["make me a video", "", "", "", "", "", "", ""]
    expected_step_names = [
        "init_state",
        "remove_off_screen_speakers",
        "generate_story",
        "identify_key_soundbites",
        "stage_0_cut_partial_transcripts_with_critiques",
        "stage_0_modify_transcript_holistically",
        "stage_0_export_results",
        "stage_1_cut_partial_transcripts_with_critiques",
        "stage_1_modify_transcript_holistically",
        "stage_1_export_results",
        "end",
    ]

    step_outputs = []
    str_outputs = []
    i = 0
    while True:
        assert i < len(user_inputs)
        async for output in step_workflow_until_feedback_request(
            workflow, user_inputs[i]
        ):
            if isinstance(output, CutTranscriptLinearWorkflowStepOutput):
                step_outputs.append(output)
            else:
                str_outputs.append(output)
        assert len(step_outputs)

        if len(step_outputs) < len(expected_step_names):
            assert (await workflow.get_last_step()).name == expected_step_names[
                len(step_outputs) - 1
            ]
            if len(step_outputs) < len(expected_step_names) - 1:
                assert (await workflow.get_next_step()).name == expected_step_names[
                    len(step_outputs)
                ]
            else:
                assert await workflow.get_next_step() is None
        else:
            # last step will never be "end", we always return the last real step we ran
            assert (await workflow.get_last_step()).name == expected_step_names[-2]
            assert await workflow.get_next_step() is None
        if step_outputs[-1].done:
            break
        i += 1
    assert len(step_outputs) == len(expected_step_names)
    assert [s.step_name == e for s, e in zip(step_outputs, expected_step_names)]

    output = await workflow.get_last_output_before_end()
    output_files = output.step_outputs["output_files"]
    assert "video_timeline" in output_files
    assert os.stat(output_files["video_timeline"]).st_size > 0
    assert len(workflow.story) == 1658
    assert workflow.story == step_outputs[2].step_outputs["story"]
    assert [
        Soundbite(**sb)
        for sb in workflow.state.dynamic_state["identify_key_soundbites"][
            "step_outputs"
        ]["current_soundbites_state"]["soundbites"]
    ] == step_outputs[3].step_outputs["current_soundbites_state"]["soundbites"]
    assert (
        workflow.soundbites_for_stage(0).soundbites
        == step_outputs[5].step_outputs["current_soundbites_state"]["soundbites"]
    )
    assert (
        workflow.transcript_for_stage(0).kept_word_count
        == Transcript.load_from_state(
            step_outputs[5].step_outputs["current_transcript_state"]
        ).kept_word_count
    )
    assert "video_timeline" in step_outputs[6].step_outputs["output_files"]

    assert (
        workflow.soundbites_for_stage(1).soundbites
        == step_outputs[8].step_outputs["current_soundbites_state"]["soundbites"]
    )
    assert (
        workflow.transcript_for_stage(1).kept_word_count
        == Transcript.load_from_state(
            step_outputs[8].step_outputs["current_transcript_state"]
        ).kept_word_count
    )
    assert "video_timeline" in step_outputs[6].step_outputs["output_files"]
