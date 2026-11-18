from functools import lru_cache

import certifi
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from pymongo.collection import Collection

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger

log = get_logger("rag.store")


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    settings = get_settings()
    if not settings.mongo_uri:
        raise RuntimeError("MONGO_URI is not set. Copy .env.example to .env and fill it in.")

    # Atlas (mongodb+srv://) always uses TLS. A local Atlas deployment
    # (mongodb://localhost...) does not, so TLS follows the connection string.
    use_tls = settings.mongo_uri.startswith("mongodb+srv://") or "tls=true" in settings.mongo_uri.lower()
    tls_options = {"tls": True, "tlsCAFile": certifi.where()} if use_tls else {}

    return MongoClient(
        settings.mongo_uri,
        serverSelectionTimeoutMS=20000,
        appname="placement-rag",
        **tls_options,
    )


def get_collection() -> Collection:
    settings = get_settings()
    return get_client()[settings.mongo_db][settings.mongo_collection]


@lru_cache(maxsize=1)
def get_vector_store() -> MongoDBAtlasVectorSearch:
    from backend.app.rag.embeddings import get_embeddings

    settings = get_settings()
    return MongoDBAtlasVectorSearch(
        collection=get_collection(),
        embedding=get_embeddings(),
        index_name=settings.vector_index_name,
        relevance_score_fn="cosine",
        text_key="text",
        embedding_key="embedding",
        auto_create_index=False,
    )


def ping() -> bool:
    try:
        get_client().admin.command("ping")
        return True
    except Exception as exc:
        log.warning("mongo ping failed: %s", exc)
        return False
