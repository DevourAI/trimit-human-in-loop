import os
from pathlib import Path
import modal

ENV = os.environ.get("ENV", "dev")
CONFIG_IS_SET = os.environ.get("CONFIG_IS_SET", "")
DOTENV_PATH = Path(__file__).parent.parent.parent / f".{ENV}"

if not CONFIG_IS_SET and modal.is_local():
    print("Loading .env file")
    from dotenv import load_dotenv

    load_dotenv(DOTENV_PATH / ".env")
    os.environ["CONFIG_IS_SET"] = "true"

SHARED_USER_EMAIL = "shared@shared.com"
