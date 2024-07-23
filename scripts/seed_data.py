import asyncio
import os
import fire
import requests
from trimit.utils.conf import SHARED_USER_EMAIL
from trimit.models import User, Video, maybe_init_mongo, VideoHighResPathProjection

LOCAL_VIDEO_FILEPATHS = ["elon_interview.mp4"]


async def seed_shared_user():
    shared_user = await User.find_one(User.email == SHARED_USER_EMAIL)
    if shared_user is None:
        print("Seeding shared user")
        shared_user = User(email=SHARED_USER_EMAIL, name="shared", password=None)
        await shared_user.save()


async def seed_shared_vids():
    base_url = os.environ["MODAL_FRONTEND_BASE_URL"]
    shared_vids = (
        await Video.find(Video.user.email == SHARED_USER_EMAIL)
        .project(VideoHighResPathProjection)
        .to_list()
    )
    shared_vid_filepaths = [vid.high_res_user_file_path for vid in shared_vids]
    files = []
    for file_path in LOCAL_VIDEO_FILEPATHS:
        if os.path.basename(file_path) not in shared_vid_filepaths:
            files.append(("files", (open(file_path, "rb"))))
    if len(files) == 0:
        return
    print(f"Uploading {len(files)} files")
    data = {
        "user_email": SHARED_USER_EMAIL,
        "high_res_user_file_paths": [
            os.path.basename(fp) for fp in LOCAL_VIDEO_FILEPATHS
        ],
        "timeline_name": "shared",
        "reprocess": True,
        "use_existing_output": False,
    }

    resp = requests.post(f"{base_url}/upload", data=data, files=files)
    assert resp.status_code == 200


async def seed_data():
    await maybe_init_mongo()
    await seed_shared_user()
    await seed_shared_vids()


def main():
    asyncio.run(seed_data())


if __name__ == "__main__":
    fire.Fire(main)
