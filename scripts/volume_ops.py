import shutil
import os
from pathlib import Path
import aiofiles
from modal import Volume, Image
from modal.volume import FileEntryType
from trimit.app import app, VOLUME_NAME, VOLUME_DIR
from tqdm.asyncio import tqdm as tqdm_async

image = Image.debian_slim(python_version="3.11").pip_install("aiofiles", "tqdm")

app_kwargs = dict(
    _allow_background_volume_commits=True,
    image=image,
    _experimental_boost=True,
    _experimental_scheduler=True,
)


@app.function(**app_kwargs)
def copy_files(source_dir, dest_dir):
    shutil.copytree(
        os.path.join(VOLUME_DIR, source_dir), os.path.join(VOLUME_DIR, dest_dir)
    )


@app.function(**app_kwargs)
def move_files(source_dir, dest_dir):
    os.rename(os.path.join(VOLUME_DIR, source_dir), os.path.join(VOLUME_DIR, dest_dir))


async def download_file(vol, source, dest):
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(dest, "wb") as file:
        print(f"Downloading {source} to {dest}")
        try:
            remote = vol.read_file(source)
        except:
            return
        else:
            for chunk in remote:
                await file.write(chunk)


async def download_files(source_dir, dest_dir):
    vol = Volume.lookup(VOLUME_NAME)
    tasks = []
    for path in vol.listdir(source_dir, recursive=True):
        if path.type != FileEntryType.FILE:
            continue
        tasks.append(
            download_file(
                vol,
                path.path,
                os.path.join(dest_dir, os.path.relpath(path.path, source_dir)),
            )
        )
    await tqdm_async.gather(*tasks)


@app.local_entrypoint()
async def volume_ops_cli(
    cmd: str,
    source_dir: str | None = None,
    dest_dir: str | None = None,
    source_file: str | None = None,
    dest_file: str | None = None,
):
    if source_dir is not None:
        if cmd == "copy":
            copy_files.remote(source_dir, dest_dir)
        elif cmd == "move":
            move_files.remote(source_dir, dest_dir)
        elif cmd == "download":
            await download_files(source_dir, dest_dir)
        else:
            raise ValueError(f"Unknown command: {cmd}")
    elif source_file is not None:
        if dest_file is None:
            raise ValueError("dest_file must be provided if source_file is provided")
        if cmd == "copy":
            copy_files.remote(source_file, dest_file)
        elif cmd == "move":
            move_files.remote(source_file, dest_file)
        elif cmd == "download":
            await download_file(Volume.lookup(VOLUME_NAME), source_file, dest_file)
        else:
            raise ValueError(f"Unknown command: {cmd}")
    else:
        raise ValueError("source_dir or source_file must be provided")
