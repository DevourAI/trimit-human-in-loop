from trimit.backend.cut_transcript import CutTranscriptLinearWorkflow
import pytest

pytestmark = pytest.mark.asyncio()


async def test_workflow_id_from_params(workflow_3909774043_with_transcript):
    workflow = workflow_3909774043_with_transcript
    workflow_id = await CutTranscriptLinearWorkflow.id_from_params(
        video_hash=workflow.video.md5_hash,
        timeline_name=workflow.timeline_name,
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
        "remove_off_screen_speakers", {}
    )
    await workflow.state.set_current_step_output_atomic(
        "remove_off_screen_speakers", {}
    )
    workflow_with_only_step_order = (
        await CutTranscriptLinearWorkflow.with_only_step_order(
            video_hash=workflow.video.md5_hash,
            timeline_name=workflow.timeline_name,
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
    next_step_with_state = await workflow.get_next_step(with_load_state=False)
    next_step_with_step_order = await workflow_with_only_step_order.get_next_step(
        with_load_state=False
    )
    assert next_step_with_state.name == next_step_with_step_order.name
    last_step_with_state = await workflow.get_last_step(with_load_state=False)
    last_step_with_step_order = await workflow_with_only_step_order.get_last_step(
        with_load_state=False
    )
    assert last_step_with_state.name == last_step_with_step_order.name
