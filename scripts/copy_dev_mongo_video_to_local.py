from dotenv import load_dotenv
import os
from trimit.models import maybe_init_mongo, Video


async def get_video_from_dev_mongo(video_hash, user_email):
    load_dotenv(".dev/.env")
    os.environ["CONFIG_IS_SET"] = "true"
    await maybe_init_mongo()
    return await Video.find_one(
        Video.md5_hash == video_hash, Video.user.email == user_email
    )


async def save_video_to_local_mongo(video: Video):
    load_dotenv(".local/.env", override=True)
    os.environ["CONFIG_IS_SET"] = "true"
    await maybe_init_mongo(reinitialize=True)
    await video.save()


async def copy_dev_mongo_video_to_local(video_hash, user_email):
    video = await get_video_from_dev_mongo(video_hash, user_email)
    if video is None:
        print(f"Video with hash {video_hash} not found in dev mongo")
        return
    await save_video_to_local_mongo(video)
    return video


def main(video_hash: str, user_email: str):
    import asyncio

    asyncio.run(copy_dev_mongo_video_to_local(str(video_hash), user_email))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
