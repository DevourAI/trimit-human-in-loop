import fire
from trimit.utils.conf import ENV
from trimit.utils.upload import upload_videos
import asyncio


def main(
    user_email,
    timeline_name,
    local_folder: str | None = None,
    local_paths: str | None = None,
    force=False,
    recursive=True,
    reprocess=False,
    use_existing_output=True,
):
    print(f"Env is {ENV}")
    if local_paths is not None:
        local_paths = local_paths.split(",")
    uploaded = asyncio.run(
        upload_videos(
            user_email,
            timeline_name=timeline_name,
            local_folder=local_folder,
            local_paths=local_paths,
            force=force,
            recursive=recursive,
            reprocess=reprocess,
            use_existing_output=use_existing_output,
        )
    )
    print(f"Uploaded videos: {uploaded}")


if __name__ == "__main__":
    fire.Fire(main)
