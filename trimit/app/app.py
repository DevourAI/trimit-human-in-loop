from modal import Volume, App, Mount, Secret, CloudBucketMount, is_local
import os
from trimit.utils.conf import DOTENV_PATH
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
APP_NAME = "trimit-human-in-loop"
SCENE_TRANSCRIPT_PINECONE_INDEX_NAME = os.environ.get("SCENE_TRANSCRIPT_PINECONE_INDEX")
REPO_HOME = "/app"
LOCAL_VOLUME_DIR = "local_volume"
REMOTE_VOLUME_DIR = "/volume"


def get_volume_dir():
    if is_local():
        return LOCAL_VOLUME_DIR
    return REMOTE_VOLUME_DIR


VOLUME_DIR = get_volume_dir()


LOCAL_AGENT_OUTPUT_CACHE_DIR = os.environ.get("LOCAL_AGENT_OUTPUT_CACHE_DIR", ".cache")
AGENT_OUTPUT_CACHE_DIR = os.environ.get(
    "AGENT_OUTPUT_CACHE_DIR", "/volume/agent_output_cache"
)
HF_HOME = str(Path(get_volume_dir()) / "huggingface")
S3_VIDEO_PATH = "/s3-videos"
S3_BUCKET = os.environ.get("TRIMIT_VIDEO_S3_BUCKET", "")
VOLUME_NAME = "trimit-human-in-loop-volume"
LOCAL_CERT_PATH = str(ROOT_DIR / os.environ.get("MONGO_CERT_FILENAME", ""))
CERT_PATH = str(Path(REPO_HOME) / os.environ.get("MONGO_CERT_FILENAME", ""))
mounts = [Mount.from_local_dir("./trimit", remote_path=REPO_HOME)]
volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)
volumes = {
    REMOTE_VOLUME_DIR: volume,
    S3_VIDEO_PATH: CloudBucketMount(
        # the get is only necessary because during the image creation process,
        # there are some points where this gets called before the .env file is loaded
        os.environ.get("TRIMIT_VIDEO_S3_BUCKET", ""),
        secret=Secret.from_dotenv(path=DOTENV_PATH),
    ),
}
EXTRA_ENV = {"MONGO_CERT_FILEPATH": CERT_PATH, "HF_HOME": HF_HOME}

app = App(
    APP_NAME,
    mounts=mounts,
    volumes=volumes,
    secrets=[Secret.from_dotenv(path=DOTENV_PATH)],
)
