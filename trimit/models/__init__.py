from contextlib import asynccontextmanager
import asyncio
from .models import *
from .models import ALL_MODELS
from modal import is_local
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from beanie.odm.utils.init import Initializer
from pymongo import MongoClient
from pymongo.client_session import ClientSession
import os

global MONGO_INITIALIZED
MONGO_INITIALIZED = [False]
global MONGO_CLIENT
MONGO_CLIENT = None


class IndexlessBeanieInitializer(Initializer):
    """
    Beanie initializer subclass to skip indexes operations
    """

    async def init_indexes(self, cls, allow_index_dropping: bool = False):
        pass


async def init_beanie_without_indexes(*args, **kwargs):
    await IndexlessBeanieInitializer(*args, **kwargs)


async def maybe_init_mongo(
    cert_path: str | None = None, reinitialize: bool = False, **motor_kwargs
):
    global MONGO_INITIALIZED
    global MONGO_CLIENT
    if reinitialize:
        MONGO_INITIALIZED[0] = False
        MONGO_CLIENT = None
    if MONGO_INITIALIZED[0]:
        return
    if MONGO_CLIENT is None:
        MONGO_CLIENT = get_motor_client(cert_path, **motor_kwargs)
    assert MONGO_CLIENT is not None
    if os.getenv("INIT_MONGO_WITH_INDEXES", "true") == "true":
        await init_beanie(database=MONGO_CLIENT.db_name, document_models=ALL_MODELS)
    else:
        print("Initializing mongo without indexes")
        await init_beanie_without_indexes(
            database=MONGO_CLIENT.db_name, document_models=ALL_MODELS
        )
    MONGO_INITIALIZED[0] = True


def pymongo_conn(db: str | None = None, cert_path: str | None = None):
    if not cert_path:
        cert_path = os.environ["MONGO_CERT_FILEPATH"] or None

    mongo_url = os.environ["MONGO_URL"]
    client = MongoClient(mongo_url, tlsCertificateKeyFile=cert_path)
    if db:
        return client[db]
    else:
        return client["db_name"]


def get_motor_client(cert_path: str | None = None, **motor_kwargs):
    if not cert_path:
        if is_local() and not os.environ["ENV"] in ["test", "local"]:
            os.environ["MONGO_CERT_FILEPATH"] = os.environ["MONGO_CERT_FILENAME"]
        cert_path = os.environ["MONGO_CERT_FILEPATH"] or None

    mongo_url = os.environ["MONGO_URL"]
    client = AsyncIOMotorClient(
        mongo_url, tlsCertificateKeyFile=cert_path, **motor_kwargs
    )
    client.get_io_loop = asyncio.get_running_loop
    return client


@asynccontextmanager
async def start_transaction():
    global MONGO_CLIENT
    if MONGO_CLIENT is None:
        raise ValueError("Mongo client not initialized")
    async with await MONGO_CLIENT.start_session() as session:
        async with session.start_transaction():
            yield session
