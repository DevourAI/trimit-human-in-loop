from modal import Image
from trimit.app import LOCAL_CERT_PATH, CERT_PATH, EXTRA_ENV

image = (
    Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "git",
        "curl",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "clang",
        "libopenmpi-dev",
        "libimage-exiftool-perl",
        gpu="any",
    )
    .pip_install("git+https://github.com/m-bain/whisperX.git", gpu="any")
    .pip_install(
        "wheel>=0.43.0,<1.0.0",
        "python-dotenv>=1.0.1,<2.0.0",
        "numpy>=1.26.4,<2.0.0",
        "aioboto3>=12.3.0,<13.0.0",
        "beanie>=1.25.0,<2.0.0",
        "tenacity>=8.2.3,<9.0.0",
        "jinja2>=3.1.3,<4.0.0",
        "moviepy>=1.0.3,<2.0.0",
        "langchain-core>=0.2.0,<1.0.0",
        "langchain-openai>=0.1.7,<1.0.0",
        "langchain-pinecone>=0.1.1,<1.0.0",
        "langchain-text-splitters>=0.2.0,<1.0.0",
        "git+https://github.com/bschreck/pyannote-audio.git#save-clusters",
        "aiofiles>=23.2.1,<24.0.0",
        "pyannote-core>=5.0.0,<6.0.0",
        "aiometer>=0.5.0,<1.0.0",
        "pydantic>=2.7.0,<3.0.0",
        "rapidfuzz>=3.9.0,<4.0.0",
        "opentimelineio>=0.16.0,<1.0.0",
        "griptape[drivers-vector-pinecone]>=0.25.1",
        "tiktoken>=0.7.0,<1.0.0",
        "aiostream>=0.5.2,<0.6.0",
        "fastapi>=0.111.0,<1.0.0",
        "passlib>=1.7.4,<2.0.0",
        "bcrypt>=4.1.3,<5.0.0",
        "python-jose>=3.3.0,<4.0.0",
        "authlib>=1.3.0,<2.0.0",
        "itsdangerous>=2.2.0,<3.0.0",
        "google-cloud-storage",
        "python-dateutil>=2.9.0.post0,<3.0.0",
        "pinecone-client>=3,<4",
        "torch>=2.3.0,<3.0.0",
        "diskcache>=5.6.3,<6.0.0",
        "pymongo[srv]>=4.7.2,<5.0.0",
        "torchaudio>=2.3.0,<3.0.0",
        "speechbrain>=1.0.0,<2.0.0",
        "scenedetect>=0.6.3,<1.0.0",
        "opencv-python>=4.9.0.80,<5.0.0.0",
    )
    .pip_install("git+https://github.com/bschreck/pyannote-audio.git#save-clusters")
    .copy_local_file(LOCAL_CERT_PATH, CERT_PATH)
    .env(EXTRA_ENV)
)
