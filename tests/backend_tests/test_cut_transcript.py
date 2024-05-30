from pathlib import Path
from trimit.backend.cut_transcript import (
    CutTranscriptLinearWorkflowStepResults,
    CutTranscriptLinearWorkflowStepInput,
    CutTranscriptLinearWorkflowStepOutput,
)
from trimit.backend.models import Soundbite, CurrentStepInfo
from trimit.backend.serve import step_workflow_until_feedback_request
from trimit.backend.models import Transcript

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
        async for output, _ in workflow.step():
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


async def test_step_until_finish(workflow_3909774043_with_transcript):
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
            workflow, user_inputs[i]
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

    output = await workflow.get_last_output_before_end()
    output_files = output.export_result
    for file_key in [
        "video_timeline",
        "video",
        "soundbites",
        "soundbites_text",
        "transcript",
        "transcript_text",
    ]:
        assert file_key in output_files
        assert os.stat(output_files[file_key]).st_size > 0
    assert (
        Path(output_files["video"]).name
        == "3909774043_modify_transcript_holistically_video_0.mp4"
    )

    all_outputs = await workflow.get_all_outputs(with_load_state=False)
    assert all(
        "video_timeline" in substep_output.export_result
        for output in all_outputs
        for substep_output in output["substeps"]
        if substep_output.substep_name not in ("init_state", "end")
    )

    output_for_steps = await workflow.get_output_for_keys(
        [
            "stage_0_generate_transcript.modify_transcript_holistically",
            "stage_1_generate_transcript.cut_partial_transcripts_with_critiques",
        ]
    )
    step0_tl_file = output_for_steps[0].export_result.get("video_timeline")
    assert step0_tl_file and os.stat(step0_tl_file).st_size > 0
    step1_tl_file = output_for_steps[1].export_result.get("video_timeline")
    assert step1_tl_file and os.stat(step1_tl_file).st_size > 0

    assert len(workflow.story) == 1977
    assert workflow.story == step_outputs[2].step_outputs["story"]
    assert [
        Soundbite(**sb)
        for sb in workflow.state.dynamic_state[
            "identify_key_soundbites.identify_key_soundbites"
        ]["step_outputs"]["current_soundbites_state"]["soundbites"]
    ] == step_outputs[3].step_outputs["current_soundbites_state"]["soundbites"]

    assert [
        Soundbite(**sb)
        for sb in workflow.state.dynamic_state[
            "stage_0_generate_transcript.modify_transcript_holistically"
        ]["step_outputs"]["current_soundbites_state"]["soundbites"]
    ] == step_outputs[5].step_outputs["current_soundbites_state"]["soundbites"]

    assert (
        Transcript.load_from_state(
            workflow.state.dynamic_state[
                "stage_0_generate_transcript.modify_transcript_holistically"
            ]["step_outputs"]["current_transcript_state"]
        ).kept_word_count
        == Transcript.load_from_state(
            step_outputs[5].step_outputs["current_transcript_state"]
        ).kept_word_count
    )

    assert [
        Soundbite(**sb)
        for sb in workflow.state.dynamic_state[
            "stage_1_generate_transcript.modify_transcript_holistically"
        ]["step_outputs"]["current_soundbites_state"]["soundbites"]
    ] == step_outputs[7].step_outputs["current_soundbites_state"]["soundbites"]

    assert (
        Transcript.load_from_state(
            workflow.state.dynamic_state[
                "stage_1_generate_transcript.modify_transcript_holistically"
            ]["step_outputs"]["current_transcript_state"]
        ).kept_word_count
        == Transcript.load_from_state(
            step_outputs[7].step_outputs["current_transcript_state"]
        ).kept_word_count
    )
