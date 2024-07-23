from trimit.backend.cut_transcript import CutTranscriptLinearWorkflow
from trimit.models.backend_models import CutTranscriptLinearWorkflowStepOutput
import pytest

pytestmark = pytest.mark.asyncio()


async def test_workflow_id_from_params(workflow_3909774043_with_transcript):
    workflow = workflow_3909774043_with_transcript
    workflow_id = await CutTranscriptLinearWorkflow.id_from_params(
        project_name=workflow.project.name,
        timeline_name=workflow.timeline_name,
        video_hash=workflow.video.md5_hash,
        user_email=workflow.video.user.email,
        length_seconds=workflow.length_seconds,
        output_folder=workflow.state.output_folder,
        volume_dir=workflow.volume_dir,
        nstages=workflow.nstages,
        first_pass_length=workflow.first_pass_length,
        max_partial_transcript_words=workflow.max_partial_transcript_words,
        max_word_extra_threshold=workflow.max_word_extra_threshold,
        max_iterations=workflow.max_iterations,
        export_video=workflow.export_video,
    )
    assert workflow.id == workflow_id


async def test_workflow_with_only_step_order(workflow_3909774043_with_transcript):
    workflow = workflow_3909774043_with_transcript
    await workflow.state.set_current_step_output_atomic(
        "preprocess_video.remove_off_screen_speakers",
        CutTranscriptLinearWorkflowStepOutput(
            step_name="preprocess_video", substep_name="remove_off_screen_speakers"
        ),
        save_to_db=True,
        use_session=False,
    )
    workflow_with_only_step_order = (
        await CutTranscriptLinearWorkflow.with_only_step_order(
            project_name=workflow.project.name,
            timeline_name=workflow.timeline_name,
            video_hash=workflow.video.md5_hash,
            user_email=workflow.video.user.email,
            length_seconds=workflow.length_seconds,
            output_folder=workflow.state.output_folder,
            volume_dir=workflow.volume_dir,
            nstages=workflow.nstages,
            first_pass_length=workflow.first_pass_length,
            max_partial_transcript_words=workflow.max_partial_transcript_words,
            max_word_extra_threshold=workflow.max_word_extra_threshold,
            max_iterations=workflow.max_iterations,
            export_video=workflow.export_video,
        )
    )
    assert workflow_with_only_step_order.state is None
    next_step_with_state, next_step_with_state_substep_index = (
        await workflow.get_next_step(with_load_state=False)
    )
    next_step_with_step_order, next_step_with_step_order_substep_index = (
        await workflow_with_only_step_order.get_next_step(with_load_state=False)
    )
    assert next_step_with_state is not None
    assert next_step_with_step_order is not None
    assert next_step_with_state.name == next_step_with_step_order.name
    assert next_step_with_state_substep_index == next_step_with_step_order_substep_index
    last_step_with_state, last_step_with_state_substep_index = (
        await workflow.get_last_step(with_load_state=False)
    )
    assert last_step_with_state is not None
    last_step_with_step_order, last_step_with_step_order_substep_index = (
        await workflow_with_only_step_order.get_last_step(with_load_state=False)
    )
    assert last_step_with_step_order is not None
    assert last_step_with_state.name == last_step_with_step_order.name
    assert last_step_with_state_substep_index == last_step_with_step_order_substep_index


async def test_steps_object(workflow_3909774043_with_transcript):
    steps = workflow_3909774043_with_transcript.steps
    assert len([s for s in steps]) == len(steps.steps) == 5
    assert steps[0].name == "preprocess_video"
    assert steps.next_step_index(0, 0) == (0, 1)
    assert steps.next_step_index(0, 1) == (1, 0)
    assert steps.next_step_index(1, 0) == (2, 0)
    assert steps.next_step_index(2, 0) == (3, 0)
    assert steps.next_step_index(3, 0) == (3, 1)
    assert steps.next_step_index(3, 1) == (4, 0)
    assert steps.next_step_index(4, 0) == (4, 1)
    assert steps.next_step_index(4, 1) == (None, None)
    with pytest.raises(ValueError):
        steps.next_step_index(4, 2)
        steps.next_step_index(5, 0)

    assert steps.last_step_index(0, 0) == (None, None)
    assert steps.last_step_index(0, 1) == (0, 0)
    assert steps.last_step_index(1, 0) == (0, 1)
    assert steps.last_step_index(2, 0) == (1, 0)
    assert steps.last_step_index(3, 0) == (2, 0)
    assert steps.last_step_index(3, 1) == (3, 0)
    assert steps.last_step_index(4, 0) == (3, 1)
    assert steps.last_step_index(4, 1) == (4, 0)
    with pytest.raises(ValueError):
        steps.last_step_index(4, 2)
        steps.next_step_index(5, 0)


async def test_get_next_step_with_user_feedback(workflow_3909774043_with_transcript):
    workflow = workflow_3909774043_with_transcript
    next_substep = await workflow.get_next_substep_with_user_feedback(
        with_load_state=False
    )
    assert next_substep.name == "remove_off_screen_speakers"
    assert next_substep.step.name == "preprocess_video"

    await workflow.state.set_current_step_output_atomic(
        "preprocess_video.remove_off_screen_speakers",
        {},
        save_to_db=False,
        use_session=False,
    )
    next_substep = await workflow.get_next_substep_with_user_feedback(
        with_load_state=False
    )
    assert next_substep.name == "generate_story"
    assert next_substep.step.name == "generate_story"

    await workflow.state.set_current_step_output_atomic(
        "generate_story.generate_story", {}, save_to_db=False, use_session=False
    )
    next_substep = await workflow.get_next_substep_with_user_feedback(
        with_load_state=False
    )
    assert next_substep.name == "identify_key_soundbites"
    assert next_substep.step.name == "identify_key_soundbites"

    await workflow.state.set_current_step_output_atomic(
        "identify_key_soundbites.identify_key_soundbites",
        {},
        save_to_db=False,
        use_session=False,
    )
    next_substep = await workflow.get_next_substep_with_user_feedback(
        with_load_state=False
    )
    assert next_substep.name == "modify_transcript_holistically"
    assert next_substep.step.name == "stage_0_generate_transcript"

    await workflow.state.set_current_step_output_atomic(
        "stage_0_generate_transcript.cut_partial_transcripts_with_critiques",
        {},
        save_to_db=False,
        use_session=False,
    )
    next_substep = await workflow.get_next_substep_with_user_feedback(
        with_load_state=False
    )
    assert next_substep.name == "modify_transcript_holistically"
    assert next_substep.step.name == "stage_0_generate_transcript"

    await workflow.state.set_current_step_output_atomic(
        "stage_0_generate_transcript.modify_transcript_holistically",
        {},
        save_to_db=False,
        use_session=False,
    )
    next_substep = await workflow.get_next_substep_with_user_feedback(
        with_load_state=False
    )
    assert next_substep.name == "modify_transcript_holistically"
    assert next_substep.step.name == "stage_1_generate_transcript"

    await workflow.state.set_current_step_output_atomic(
        "stage_1_generate_transcript.cut_partial_transcripts_with_critiques",
        {},
        save_to_db=False,
        use_session=False,
    )
    next_substep = await workflow.get_next_substep_with_user_feedback(
        with_load_state=False
    )
    assert next_substep.name == "modify_transcript_holistically"
    assert next_substep.step.name == "stage_1_generate_transcript"

    await workflow.state.set_current_step_output_atomic(
        "stage_1_generate_transcript.modify_transcript_holistically",
        {},
        save_to_db=False,
        use_session=False,
    )
    next_substep = await workflow.get_next_substep_with_user_feedback(
        with_load_state=False
    )
    assert next_substep is None

    await workflow.state.set_current_step_output_atomic(
        "end.end", {}, save_to_db=False, use_session=False
    )
    next_substep = await workflow.get_next_substep_with_user_feedback(
        with_load_state=False
    )
    assert next_substep is None


async def test_get_last_step_with_user_feedback(workflow_3909774043_with_transcript):
    workflow = workflow_3909774043_with_transcript
    await workflow.state.set_current_step_output_atomic(
        "end.end", {}, save_to_db=False, use_session=False
    )
    last_substep = await workflow.get_last_substep_with_user_feedback(
        with_load_state=False
    )
    assert last_substep.name == "modify_transcript_holistically"
    assert last_substep.step.name == "stage_1_generate_transcript"

    await workflow.state.set_current_step_output_atomic(
        "stage_1_generate_transcript.modify_transcript_holistically",
        {},
        save_to_db=False,
        use_session=False,
    )
    last_substep = await workflow.get_last_substep_with_user_feedback(
        with_load_state=False
    )
    assert last_substep.name == "modify_transcript_holistically"
    assert last_substep.step.name == "stage_1_generate_transcript"

    await workflow.state.set_current_step_output_atomic(
        "stage_1_generate_transcript.cut_partial_transcripts_with_critiques",
        {},
        save_to_db=False,
        use_session=False,
    )
    last_substep = await workflow.get_last_substep_with_user_feedback(
        with_load_state=False
    )
    assert last_substep.name == "modify_transcript_holistically"
    assert last_substep.step.name == "stage_0_generate_transcript"

    await workflow.state.set_current_step_output_atomic(
        "stage_0_generate_transcript.modify_transcript_holistically",
        {},
        save_to_db=False,
        use_session=False,
    )
    last_substep = await workflow.get_last_substep_with_user_feedback(
        with_load_state=False
    )
    assert last_substep.name == "modify_transcript_holistically"
    assert last_substep.step.name == "stage_0_generate_transcript"

    await workflow.state.set_current_step_output_atomic(
        "stage_0_generate_transcript.cut_partial_transcripts_with_critiques",
        {},
        save_to_db=False,
        use_session=False,
    )
    last_substep = await workflow.get_last_substep_with_user_feedback(
        with_load_state=False
    )
    assert last_substep.name == "identify_key_soundbites"
    assert last_substep.step.name == "identify_key_soundbites"

    await workflow.state.set_current_step_output_atomic(
        "identify_key_soundbites.identify_key_soundbites",
        {},
        save_to_db=False,
        use_session=False,
    )

    last_substep = await workflow.get_last_substep_with_user_feedback(
        with_load_state=False
    )
    assert last_substep.name == "identify_key_soundbites"
    assert last_substep.step.name == "identify_key_soundbites"

    await workflow.state.set_current_step_output_atomic(
        "generate_story.generate_story", {}, save_to_db=False, use_session=False
    )
    last_substep = await workflow.get_last_substep_with_user_feedback(
        with_load_state=False
    )
    assert last_substep.name == "generate_story"
    assert last_substep.step.name == "generate_story"

    await workflow.state.set_current_step_output_atomic(
        "preprocess_video.remove_off_screen_speakers",
        {},
        save_to_db=False,
        use_session=False,
    )
    last_substep = await workflow.get_last_substep_with_user_feedback(
        with_load_state=False
    )
    assert last_substep.name == "remove_off_screen_speakers"

    await workflow.state.set_current_step_output_atomic(
        "preprocess_video.init_state", {}, save_to_db=False, use_session=False
    )
    last_substep = await workflow.get_last_substep_with_user_feedback(
        with_load_state=False
    )
    assert last_substep is None


async def set_state_to(state, step_key):
    await state.restart(save=False)
    steps = [
        "preprocess_video.init_state",
        "preprocess_video.remove_off_screen_speakers",
        "generate_story.generate_story",
        "identify_key_soundbites.identify_key_soundbites",
        "stage_0_generate_transcript.cut_partial_transcripts_with_critiques",
        "stage_0_generate_transcript.modify_transcript_holistically",
        "stage_1_generate_transcript.cut_partial_transcripts_with_critiques",
        "stage_1_generate_transcript.modify_transcript_holistically",
    ]
    for _step_key in steps:
        await state.set_current_step_output_atomic(
            _step_key, {}, save_to_db=False, use_session=False
        )
        if _step_key == step_key:
            break


async def test_revert_step_to_before(workflow_3909774043_with_transcript):
    workflow = workflow_3909774043_with_transcript
    await set_state_to(workflow.state, "preprocess_video.remove_off_screen_speakers")
    await workflow.state.revert_step_to_before(
        "preprocess_video", "remove_off_screen_speakers", save=False
    )
    last_substep = await workflow.get_last_substep(with_load_state=False)
    assert last_substep.name == "init_state"
    assert last_substep.step.name == "preprocess_video"

    await set_state_to(workflow.state, "preprocess_video.remove_off_screen_speakers")
    await workflow.state.revert_step_to_before(
        "preprocess_video", "init_state", save=False
    )
    last_substep = await workflow.get_last_substep(with_load_state=False)
    assert last_substep is None

    await set_state_to(
        workflow.state,
        "stage_0_generate_transcript.cut_partial_transcripts_with_critiques",
    )
    await workflow.state.revert_step_to_before(
        "preprocess_video", "remove_off_screen_speakers", save=False
    )
    last_substep = await workflow.get_last_substep(with_load_state=False)
    assert last_substep.name == "init_state"
    assert last_substep.step.name == "preprocess_video"

    await set_state_to(
        workflow.state, "stage_1_generate_transcript.modify_transcript_holistically"
    )
    await workflow.state.revert_step_to_before(
        "stage_0_generate_transcript",
        "cut_partial_transcripts_with_critiques",
        save=False,
    )
    last_substep = await workflow.get_last_substep(with_load_state=False)
    assert last_substep.name == "identify_key_soundbites"
    assert last_substep.step.name == "identify_key_soundbites"


async def test_kept_cuts_with_start_end(soundbites_15557970):
    transcript = soundbites_15557970.transcript
    cuts = transcript.segments[6].cuts_with_start_end(3, 5)
    assert len(cuts) == 3
    assert cuts[0].text == "Yep, so hey,"
    assert cuts[1].text == "I'm Parbinder"
    assert cuts[2].text == "Darawal, or Parbs, everybody calls me."


async def test_soundbites_iter_kept_cuts(soundbites_3909774043):
    cuts = [c for c in soundbites_3909774043.iter_kept_cuts()]
    cuts_text = " ".join([c.text for c in cuts])
    assert cuts_text == soundbites_3909774043.text.replace("\n", " ")
