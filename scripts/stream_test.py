import os
import requests
import fire
from storybook.utils import conf


def run_against_webapp(
    user_email: str,
    timeline_name: str,
    video_hash: str,
    length_seconds: int,
    user_input: str | None = None,
    force_restart: bool = False,
):
    base_url = os.environ["MODAL_LINEAR_WORKFLOW_SERVER_BASE_URL"]
    params = {
        "user_email": user_email,
        "timeline_name": timeline_name,
        "video_hash": video_hash,
        "length_seconds": length_seconds,
        "user_input": user_input,
        "streaming": True,
        "force_restart": force_restart,
    }

    streaming_response = requests.get(f"{base_url}/step", params=params, stream=True)
    for chunk in streaming_response.iter_content(
        chunk_size=1024
    ):  # Loop over the streaming data
        print(chunk)
    output_response = requests.get(f"{base_url}/get_latest_output", params=params)
    print(output_response.json())


if __name__ == "__main__":
    fire.Fire(run_against_webapp)
