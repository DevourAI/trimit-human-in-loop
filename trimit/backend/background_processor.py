import os
import asyncio

from beanie.operators import In, Set, Not
from beanie import BulkWriter
from modal import method, enter
from tenacity import retry, retry_if_exception_type, wait_fixed, stop_after_delay

from trimit.backend.image import image, MODEL_DIR
from trimit.backend.speaker_in_frame_detection import SpeakerInFrameDetection
from trimit.app import app, volume, VOLUME_DIR
from trimit.models import maybe_init_mongo, Video, User
from trimit.utils.model_utils import transcription_text
from trimit.backend.transcription import Transcription
from trimit.utils.fs_utils import ensure_audio_path_on_volume


CACHE_DIR = os.path.join(VOLUME_DIR, ".background_processor_cache")
VOLUME_RELOAD_TIMEOUT = 10

app_kwargs = dict(
    _allow_background_volume_commits=True,
    timeout=80000,
    image=image,
    concurrency_limit=100,
    container_idle_timeout=1200,
    _experimental_boost=True,
    _experimental_scheduler=True,
)


@retry(
    wait=wait_fixed(0.5),
    stop=stop_after_delay(VOLUME_RELOAD_TIMEOUT),
    retry=retry_if_exception_type(RuntimeError),
)
def volume_reload_with_catch():
    print("Reloading volume...")
    volume.reload()


@app.cls(gpu="any", **app_kwargs)
class BackgroundProcessor:
    @enter()
    def load_models(self):
        from openai import OpenAI
        import diskcache as dc

        # volume_reload_with_catch()
        self.cache = dc.Cache(CACHE_DIR)
        self._transcription = None
        self.speaker_in_frame_detection = SpeakerInFrameDetection(
            cache=self.cache, volume_dir=VOLUME_DIR
        )
        self._lock = asyncio.Lock()

    @property
    async def transcription(self):
        async with self._lock:
            if self._transcription is None:
                self._transcription = Transcription(
                    MODEL_DIR, cache=self.cache, volume_dir=VOLUME_DIR
                )
        return self._transcription

    @method()
    async def process_videos_generic_from_video_hashes(
        self, user_email: str, video_hashes: list[str], use_existing_output=True
    ):
        return await self._process_videos_generic_from_video_hashes(
            user_email, video_hashes, use_existing_output=use_existing_output
        )

    async def _process_videos_generic_from_video_hashes(
        self, user_email: str, video_hashes: list[str], use_existing_output=True
    ):
        videos = await self._process_audio_from_video_hashes(
            user_email, video_hashes, use_existing_output=use_existing_output
        )

        scenes = await self._detect_speaker_in_frame(
            videos, use_existing_output=use_existing_output
        )
        return scenes

    @method()
    async def process_audio_from_video_hashes(
        self, user_email: str, video_hashes: list[str], use_existing_output=True
    ):
        return await self._process_audio_from_video_hashes(
            user_email, video_hashes, use_existing_output=use_existing_output
        )

    async def _process_audio_from_video_hashes(
        self, user_email: str, video_hashes: list[str], use_existing_output=True
    ):
        await maybe_init_mongo()
        user = await User.find_one(User.email == user_email)
        if user is None:
            raise ValueError(f"User not found: {user_email}")

        videos = await Video.find(
            In(Video.md5_hash, video_hashes),
            Video.user.email == user.email,
            fetch_links=True,
        ).to_list()

        print(f"Processing audio for {len(videos)} videos")

        videos = await self._transcribe_videos(
            user_email, videos, use_existing_output=use_existing_output
        )
        return videos

    @method()
    async def transcribe_videos_from_video_hashes(
        self, user_email: str, video_hashes: list[str], use_existing_output: bool = True
    ):
        return await self._transcribe_videos_from_video_hashes(
            user_email, video_hashes, use_existing_output=use_existing_output
        )

    async def _transcribe_videos_from_video_hashes(
        self, user_email: str, video_hashes: list[str], use_existing_output: bool = True
    ):
        await maybe_init_mongo()
        videos = await Video.find(
            In(Video.md5_hash, video_hashes), Video.user.email == user_email
        ).to_list()

        return await self._transcribe_videos(
            user_email=user_email,
            videos=videos,
            use_existing_output=use_existing_output,
        )

    async def _transcribe_videos(
        self, user_email: str, videos: list[Video], use_existing_output: bool = True
    ):
        print(
            f"transcribing videos: {videos}, use_existing_output: {use_existing_output}"
        )
        if use_existing_output:
            missing_transcript_videos = {
                video.md5_hash: video for video in videos if not video.transcription
            }
            print(
                f"missing transcript video hashes: {list(missing_transcript_videos.keys())}"
            )
        else:
            missing_transcript_videos = {video.md5_hash: video for video in videos}

        volume_download_tasks = [ensure_audio_path_on_volume(video) for video in videos]
        await asyncio.gather(*volume_download_tasks)
        missing_transcriptions = (await self.transcription).transcribe_videos(
            videos, with_cache=use_existing_output
        )

        async with BulkWriter() as bulk_writer:
            for video_hash, transcription in missing_transcriptions.items():
                video = missing_transcript_videos.get(video_hash)
                if video is None:
                    continue
                video.transcription = transcription
                await Video.find_one(
                    Video.md5_hash == video_hash, Video.user.email == user_email
                ).update(
                    Set(
                        {
                            Video.transcription: transcription,
                            Video.transcription_text: transcription_text(transcription),
                        }
                    )
                )
            await bulk_writer.commit()
            print("wrote missing transcriptions")
        return videos

    @method()
    async def transcribe_video(self, video: Video, use_existing_output: bool = True):
        return await self._transcribe_videos(
            video.user.email, [video], use_existing_output=use_existing_output
        )

    @method()
    async def detect_speaker_in_frame_from_hashes(
        self, user_email: str, video_hashes: list[str], use_existing_output=True
    ):
        await maybe_init_mongo()
        videos = await Video.find(
            Video.user.email == user_email, In(Video.md5_hash, video_hashes)
        ).to_list()
        return await self._detect_speaker_in_frame(
            videos, use_existing_output=use_existing_output
        )

    @method()
    async def detect_speaker_in_frame(
        self, videos: list[Video], use_existing_output=True
    ):
        await maybe_init_mongo()
        return await self._detect_speaker_in_frame(
            videos, use_existing_output=use_existing_output
        )

    async def _detect_speaker_in_frame(
        self, videos: list[Video], use_existing_output=True
    ):
        return (
            await self.speaker_in_frame_detection.detect_speaker_in_frame_from_videos(
                videos, use_existing_output=use_existing_output
            )
        )
