from .models import *
from .models import ALL_MODELS
from modal import is_local
from beanie import init_beanie
from beanie.odm.utils.init import Initializer
import os

global MONGO_INITIALIZED
MONGO_INITIALIZED = [False]


class IndexlessBeaineInitializer(Initializer):
    """
    Beanie initializer subclass to skip indexes operations
    """

    async def init_indexes(self, cls, allow_index_dropping: bool = False):
        pass


async def init_beanie_without_indexes(*args, **kwargs):
    await IndexlessBeaineInitializer(*args, **kwargs)


async def maybe_init_mongo(
    cert_path: str | None = None, reinitialize: bool = False, **motor_kwargs
):
    global MONGO_INITIALIZED
    if reinitialize:
        MONGO_INITIALIZED[0] = False
    if MONGO_INITIALIZED[0]:
        return
    if not cert_path:
        if is_local() and not os.environ["ENV"] == "test":
            os.environ["MONGO_CERT_FILEPATH"] = os.environ["MONGO_CERT_FILENAME"]
        cert_path = os.environ["MONGO_CERT_FILEPATH"] or None
    from motor.motor_asyncio import AsyncIOMotorClient

    mongo_url = os.environ["MONGO_URL"]
    client = AsyncIOMotorClient(
        mongo_url, tlsCertificateKeyFile=cert_path, **motor_kwargs
    )
    if os.getenv("INIT_MONGO_WITH_INDEXES", "true") == "true":
        print("Initializing mongo with indexes")
        await init_beanie(database=client.db_name, document_models=ALL_MODELS)
    else:
        print("Initializing mongo without indexes")
        await init_beanie_without_indexes(
            database=client.db_name, document_models=ALL_MODELS
        )
    MONGO_INITIALIZED[0] = True


def pymongo_conn(db: str | None = None, cert_path: str | None = None):
    if not cert_path:
        cert_path = os.environ["MONGO_CERT_FILEPATH"] or None
    from pymongo import MongoClient

    mongo_url = os.environ["MONGO_URL"]
    client = MongoClient(mongo_url, tlsCertificateKeyFile=cert_path)
    if db:
        return client[db]
    else:
        return client["db_name"]
