import fire
from trimit.utils import conf
from trimit.app import APP_NAME
from modal import Cls


def main(user_email: str, video_hashes: str, use_existing_output: bool = False):
    video_hashes = str(video_hashes).split(",")
    processor = Cls.lookup(APP_NAME, "BackgroundProcessor")()
    speakers = processor.detect_speaker_in_frame_from_hashes.remote(
        user_email=user_email,
        video_hashes=video_hashes,
        use_existing_output=use_existing_output,
    )
    print(speakers)


if __name__ == "__main__":
    fire.Fire(main)
