from trimit.export import (
    create_fcp_7_xml_from_single_video_transcript,
    create_cut_video_from_transcript,
    save_soundbites_videos_to_disk,
)
from trimit.export.email import send_email_with_export_results
from trimit.backend.utils import match_output_to_actual_transcript_fast
from trimit.models.backend_models import ExportResults
from trimit.models import maybe_init_mongo
from trimit.utils.model_utils import (
    get_generated_video_folder,
    get_generated_soundbite_clips_folder,
    map_export_result_to_asset_path,
)
import os
import pytest
import asyncio

pytestmark = pytest.mark.asyncio()


@pytest.fixture(autouse=True)
async def mongo_init():
    loop = asyncio.get_running_loop()
    await maybe_init_mongo(io_loop=loop, reinitialize=True)


async def test_create_fcp_7_xml_from_transcript(
    video_15557970, short_cut_transcript_15557970
):
    video_15557970.high_res_user_file_path = video_15557970.path(
        os.path.abspath("tests/fixtures/volume")
    )
    video_timeline_file = create_fcp_7_xml_from_single_video_transcript(
        video=video_15557970,
        transcript=short_cut_transcript_15557970,
        timeline_name="test_timeline",
        volume_dir="tests/fixtures/volume",
        output_dir="tests/timeline_outputs/linear",
        clip_extra_trim_seconds=0.05,
        use_high_res_path=True,
        use_full_path=True,
    )
    assert os.stat(video_timeline_file).st_size > 0


async def test_create_fcp_7_xml_from_soundbites(
    video_3909774043, soundbites_3909774043
):
    video_3909774043.high_res_user_file_path = video_3909774043.path(
        os.path.abspath("tests/fixtures/volume")
    )
    video_timeline_file = create_fcp_7_xml_from_single_video_transcript(
        video=video_3909774043,
        transcript=soundbites_3909774043,
        timeline_name="test_soundbites_timeline",
        volume_dir="tests/fixtures/volume",
        output_dir="tests/timeline_outputs/linear",
        prefix="soundbites_",
        clip_extra_trim_seconds=0.05,
        use_high_res_path=True,
        use_full_path=True,
    )
    assert os.stat(video_timeline_file).st_size > 0


async def test_create_cut_video_from_transcript(
    video_3909774043, short_cut_transcript_3909774043
):
    timeline_name = "test_timeline"
    output_dir = get_generated_video_folder(
        "tests/video_outputs/linear/generated_videos",
        video_3909774043.user.email,
        timeline_name,
    )

    cut_video_path = await create_cut_video_from_transcript(
        video=video_3909774043,
        transcript=short_cut_transcript_3909774043,
        timeline_name=timeline_name,
        volume_dir="tests/fixtures/volume",
        output_dir=output_dir,
    )
    assert os.stat(cut_video_path).st_size > 0


async def test_create_cut_video_from_soundbite(video_3909774043, soundbites_3909774043):
    timeline_name = "test_timeline"
    video_3909774043.high_res_user_file_path = video_3909774043.path(
        os.path.abspath("tests/fixtures/volume")
    )
    output_dir = get_generated_video_folder(
        "tests/video_outputs/linear/generated_videos",
        video_3909774043.user.email,
        timeline_name,
    )

    cut_video_path = await create_cut_video_from_transcript(
        video=video_3909774043,
        transcript=soundbites_3909774043.soundbites[0],
        timeline_name=timeline_name,
        volume_dir="tests/fixtures/volume",
        output_dir=output_dir,
        prefix="soundbite_0_",
    )
    assert os.stat(cut_video_path).st_size > 0


async def test_save_soundbites_videos_to_disk(video_3909774043, soundbites_3909774043):
    timeline_name = "test_timeline"
    video_3909774043.high_res_user_file_path = video_3909774043.path(
        os.path.abspath("tests/fixtures/volume")
    )
    output_dir = get_generated_soundbite_clips_folder(
        "tests/video_outputs/linear/generated_videos",
        video_3909774043.user.email,
        timeline_name,
    )

    video_paths = await save_soundbites_videos_to_disk(
        output_dir=output_dir,
        volume_dir="tests/fixtures/volume",
        video=video_3909774043,
        soundbites=soundbites_3909774043,
        timeline_name=timeline_name,
    )
    assert len(video_paths) == len(soundbites_3909774043.soundbites)
    assert all([os.stat(vp).st_size > 0 for vp in video_paths])


async def test_create_longer_cut_video_from_transcript(
    mongo_connect, video_15557970, transcript_15557970
):
    transcript = "<transcript><keep>I'm the VP GM over at CVS Media Exchange, which is, well, CMX as we like to refer to it. We're the retail media network for CVS Health.  We are here at ramp-up and we're also announcing here at ramp-up that we're building a coalition and partnership in collaboration with Pinterest.  multitude of different channels. We continue to see growth as we see more brands lean in from a clean room standpoint and more specifically It also enables a number of different types of use cases. We think about partnership growth, and we think about enabling other partners with our platform, with our audiences. Attribution is a of measurement, and LiveRamp really helps us enable that.  My next question is, what's next for CMX? How are you planning to continue to evolve and innovate? Measurement is incredibly important to the industry. It's also really how retail media networks are and what we're predicated upon. I've been working tirelessly around bringing standardization around measurement and metrics through the IAB. We're a real driver of that and the work that the IAB is doing in order to bring that capabilities but also the way in which the industry should operate is really around transparency. Transparency in measurement, transparency from a retail media network is gonna continue to see the growth that we expect, with other partners as well. As we continue to take learnings from our Pinterest partnership, building that with other beauty, health, and wellness consumers, that's where we will also lean in as well. of what we're doing as we think about growth within retail media networks, but CMX is leading the charge around that.  of everything that we do, both from a digital perspective, but also from a physical perspective. We will continue to create experiences for consumers.</keep></transcript>"
    leftover_transcript_chunk = match_output_to_actual_transcript_fast(
        transcript_15557970, transcript
    )
    timeline_name = "test_timeline"
    output_dir = get_generated_video_folder(
        "tests/video_outputs/linear/generated_videos",
        video_15557970.user.email,
        timeline_name,
    )

    cut_video_path = await create_cut_video_from_transcript(
        video=video_15557970,
        transcript=leftover_transcript_chunk,
        timeline_name=timeline_name,
        volume_dir="tests/fixtures/volume",
        output_dir=output_dir,
    )
    assert os.stat(cut_video_path).st_size > 0


async def test_redo_export_results(workflow_15557970_after_first_step):
    workflow = workflow_15557970_after_first_step
    output = await workflow.get_last_output(with_load_state=True)
    key = f"{output.step_name}.{output.substep_name}"
    workflow.state.outputs[key][-1].export_result = {}
    workflow.state.save()

    output = await workflow.get_last_output(with_load_state=True)
    assert output.export_result == {}
    new_output = None
    async for new_output in workflow.redo_export_results():
        continue
    assert isinstance(new_output, ExportResults)
    output = await workflow.get_last_output(with_load_state=True)
    assert output.export_result != {}


def test_send_email_with_export_results():
    export_results = ExportResults(
        video_timeline="tests/fixtures/export_results/timeline.xml",
        video="tests/fixtures/export_results/video.mp4",
        transcript="tests/fixtures/export_results/transcript.p",
        transcript_text="tests/fixtures/export_results/transcript.txt",
        soundbites_videos=["1", "2"],
    )
    response = send_email_with_export_results(
        "86461740d2a347bea5acc0c7", "benjaminschreck93@gmail.com", export_results
    )
    assert response.status_code == 202
    assert response.body == b""


async def test_map_export_result_to_asset_path():
    export_results = ExportResults(
        video_timeline="tests/fixtures/export_results/timeline.xml",
        transcript_text="nonexistant",
        soundbites_videos=[
            "nonexistant",
            "tests/fixtures/export_results/transcript.txt",
        ],
        speaker_tagging_clips={
            "1": ["tests/fixtures/export_results/transcript.txt", "nonexistant"]
        },
    )
    mapped = await map_export_result_to_asset_path(
        export_results, "tests/fixtures/export_results"
    )
    assert mapped.video_timeline == "https://d1i4vf5igkqj3d.cloudfront.net/timeline.xml"
    assert mapped.transcript_text == "nonexistant"
    assert mapped.soundbites_videos == [
        "nonexistant",
        "https://d1i4vf5igkqj3d.cloudfront.net/transcript.txt",
    ]
    assert mapped.speaker_tagging_clips == {
        "1": ["https://d1i4vf5igkqj3d.cloudfront.net/transcript.txt", "nonexistant"]
    }

    assert export_results.video_timeline == "tests/fixtures/export_results/timeline.xml"
    assert export_results.transcript_text == "nonexistant"
    assert export_results.soundbites_videos == [
        "nonexistant",
        "tests/fixtures/export_results/transcript.txt",
    ]
    assert export_results.speaker_tagging_clips == {
        "1": ["tests/fixtures/export_results/transcript.txt", "nonexistant"]
    }
