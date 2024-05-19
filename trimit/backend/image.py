from modal import Image
from trimit.app import LOCAL_CERT_PATH, CERT_PATH, EXTRA_ENV

image = (
    Image.from_registry("ubuntu:22.04", add_python="3.11")
    .apt_install("git", "curl", "libimage-exiftool-perl", "ffmpeg")
    .pip_install(
        "wheel",
        "aioboto3",
        "aiofiles",
        "jinja2",
        "tenacity",
        "beanie",
        "pymongo[srv]",
        "pydantic==2.6.1",
        "python-dotenv",
        "fastapi==0.109.2",
        "Starlette>=0.36.3,<0.37.0",
        "schema",
        "tqdm",
        "griptape[drivers-vector-pinecone]>=0.25.1",
        "rapidfuzz",
        "diskcache",
        "opentimelineio",
    )
    .pip_install("git+https://github.com/bschreck/pyannote-audio.git#save-clusters")
    .copy_local_file(LOCAL_CERT_PATH, CERT_PATH)
    .env(EXTRA_ENV)
    .pip_install("moviepy")
)
