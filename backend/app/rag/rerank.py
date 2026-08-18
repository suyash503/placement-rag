"""Cross-encoder reranking.

The retriever's bi-encoder embeds the query and each document independently, so
it never actually compares them — it only compares two summaries. A cross-encoder
reads (query, document) together and scores the pair, which is far more accurate
and far too slow to run over the whole collection. The usual arrangement, and the
one used here, is bi-encoder for recall over thousands of documents, cross-encoder
for precision over the ~25 that survive.
"""

from functools import lru_cache

from langchain_core.documents import Document

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger

log = get_logger("rag.rerank")


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    settings = get_settings()
    log.info("loading reranker %s", settings.reranker_model)
    return CrossEncoder(settings.reranker_model, max_length=512, device="cpu")


def rerank(query: str, documents: list[Document], top_k: int) -> list[tuple[Document, float]]:
    if not documents:
        return []

    model = get_reranker()
    scores = model.predict([(query, d.page_content) for d in documents])
    scored = zip(documents, (float(s) for s in scores), strict=True)
    ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]
