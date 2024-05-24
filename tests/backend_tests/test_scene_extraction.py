import pytest
import os
from trimit.utils.scene_extraction import create_scenes_from_video
from trimit.utils.model_utils import get_scene_folder
from trimit.models.models import Scene, scene_name_from_video_take_item
from ..conftest import TEST_VOLUME_DIR
from pathlib import Path

pytestmark = pytest.mark.asyncio(scope="session")


# TODO: include a short version that mocks detect_cuts and/or moviepy.VideoFileClip.write_videofile(), or just the entire extract_scenes_to_disk
# TODO test partial_transcription_indexes, and partial_transcription_words


# TODO min length of scene (10 seconds)
@pytest.mark.long
async def test_extract_scenes_from_single_video(test_video_2):
    video = test_video_2
    scene_dir = get_scene_folder(
        TEST_VOLUME_DIR, video.user_email, video.upload_datetime
    )
    scene_dir.mkdir(parents=True, exist_ok=True)
    extracted_scenes = await create_scenes_from_video(video, TEST_VOLUME_DIR)
    assert len(extracted_scenes) == 1
    expected_start_frame = 0
    expected_end_frame = 945
    assert extracted_scenes[0].video.md5_hash == video.md5_hash
    assert extracted_scenes[0].start_frame == expected_start_frame
    assert extracted_scenes[0].end_frame == expected_end_frame
    assert extracted_scenes[0].ext == video.ext
    assert extracted_scenes[0].name == scene_name_from_video_take_item(
        video, expected_start_frame, expected_end_frame, None
    )
    assert Path(extracted_scenes[0].path(TEST_VOLUME_DIR)).parent == get_scene_folder(
        TEST_VOLUME_DIR, video.user_email, video.upload_datetime
    )
    assert [s.dict() for s in await video.scenes] == [
        s.dict() for s in extracted_scenes
    ]
    scene_from_db = await Scene.find_one(Scene.name == extracted_scenes[0].name)
    assert scene_from_db.dict() == extracted_scenes[0].dict()


# TODO Im unsure whether we actually should split scenes like this.
# using AdaptiveDetector is more correct in figuring out that it's shot from a continuous take
# however, the camera moves quite a bit, so picking a sampled frame will result in very different views across the duration of the shot
@pytest.mark.long
async def test_extract_scenes_from_multi_video(test_video_4):
    video = test_video_4
    scene_dir = get_scene_folder(
        TEST_VOLUME_DIR, video.user_email, video.upload_datetime
    )
    scene_dir.mkdir(parents=True, exist_ok=True)
    extracted_scenes = await create_scenes_from_video(video, TEST_VOLUME_DIR)
    assert len(extracted_scenes) == 6
    expected_start_end_frames = [
        (0, 262),
        (262, 704),
        (704, 1107),
        (1107, 1739),
        (1739, 2095),
        (2095, 2196),
    ]
    for i, scene in enumerate(extracted_scenes):
        expected_start_frame, expected_end_frame = expected_start_end_frames[i]
        assert scene.video_hash == scene.video.md5_hash == video.md5_hash
        assert scene.user_email == video.user.email == video.user_email
        assert scene.start_frame == expected_start_frame
        assert scene.end_frame == expected_end_frame
        assert scene.ext == video.ext
        assert scene.name == scene_name_from_video_take_item(
            video, expected_start_frame, expected_end_frame, None
        )
        assert os.path.exists(scene.path(TEST_VOLUME_DIR))
        assert os.stat(scene.path(TEST_VOLUME_DIR)).st_size > 0
    assert [s.dict() for s in await video.scenes] == [
        s.dict() for s in extracted_scenes
    ]
    scenes_from_db = [
        await Scene.find_one(Scene.name == s.name) for s in extracted_scenes
    ]
    assert [
        s_db.dict() == s.dict() for s_db, s in zip(scenes_from_db, extracted_scenes)
    ]
