from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger

log = get_logger("rag.embeddings")


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """The one place the embedding model is constructed.

    Ingestion and query time must use identical model *and* identical encode
    settings — a mismatch produces vectors that still have the right shape but
    land in a different space, so retrieval silently returns noise instead of
    failing loudly.
    """
    settings = get_settings()
    log.info("loading embedding model %s", settings.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )
