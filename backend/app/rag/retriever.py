"""Retrieval funnel: filter -> hybrid search -> rerank -> diversify.

Modes exist so the evaluation harness can measure each stage in isolation rather
than asserting that the whole thing "feels better".
"""

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_mongodb.retrievers.full_text_search import MongoDBAtlasFullTextSearchRetriever
from langchain_mongodb.retrievers.hybrid_search import MongoDBAtlasHybridSearchRetriever

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.rag.query_router import ParsedQuery, parse_query
from backend.app.rag.rerank import rerank
from backend.app.rag.store import get_collection, get_vector_store

log = get_logger("rag.retriever")

RetrievalMode = Literal["vector", "hybrid", "hybrid_rerank"]


@dataclass
class RetrievalResult:
    documents: list[Document]
    parsed: ParsedQuery
    mode: RetrievalMode
    trace: dict[str, Any] = field(default_factory=dict)


def _label(doc: Document) -> str:
    meta = doc.metadata
    if meta.get("doc_type") == "company":
        return f"{meta.get('company')} (profile)"
    return f"{meta.get('company')} @ {meta.get('college')} {meta.get('year')}"


def _diversify(documents: list[Document], max_per_company: int, limit: int) -> list[Document]:
    """Stop one company's many drives from filling the whole context window."""
    seen: dict[str, int] = {}
    kept: list[Document] = []
    for doc in documents:
        company = doc.metadata.get("company", "?")
        if isinstance(company, list):
            company = company[0] if company else "?"
        if seen.get(company, 0) >= max_per_company:
            continue
        seen[company] = seen.get(company, 0) + 1
        kept.append(doc)
        if len(kept) >= limit:
            break
    return kept


def _vector_leg(query: str, k: int, pre_filter: dict | None) -> list[tuple[Document, float]]:
    return get_vector_store().similarity_search_with_score(query, k=k, pre_filter=pre_filter)


def _fulltext_leg(query: str, k: int, pre_filter: dict | None) -> list[Document]:
    settings = get_settings()
    retriever = MongoDBAtlasFullTextSearchRetriever(
        collection=get_collection(),
        search_index_name=settings.fulltext_index_name,
        search_field="text",
        k=k,
        filter=pre_filter,
    )
    return retriever.invoke(query)


def retrieve(
    question: str,
    mode: RetrievalMode = "hybrid_rerank",
    k: int | None = None,
    explain: bool = False,
) -> RetrievalResult:
    settings = get_settings()
    final_k = k or settings.final_k
    parsed = parse_query(question)
    pre_filter = parsed.as_pre_filter()

    trace: dict[str, Any] = {
        "mode": mode,
        "filters": parsed.filters,
        "filter_notes": parsed.notes,
        "candidate_k": settings.candidate_k,
        "final_k": final_k,
        "timings_ms": {},
    }

    started = time.perf_counter()
    if mode == "vector":
        candidates = [doc for doc, _ in _vector_leg(question, settings.candidate_k, pre_filter)]
    else:
        retriever = MongoDBAtlasHybridSearchRetriever(
            vectorstore=get_vector_store(),
            search_index_name=settings.fulltext_index_name,
            k=settings.candidate_k,
            pre_filter=pre_filter,
            vector_penalty=settings.vector_penalty,
            fulltext_penalty=settings.fulltext_penalty,
        )
        candidates = retriever.invoke(question)
    trace["timings_ms"]["search"] = round((time.perf_counter() - started) * 1000)
    trace["candidates_found"] = len(candidates)

    # A filter that matches nothing produces an empty result set. Retrying without
    # it beats answering "I don't know" to a question the data can partly answer.
    if not candidates and pre_filter:
        log.info("filter %s matched nothing — retrying unfiltered", pre_filter)
        trace["filter_relaxed"] = True
        parsed.notes.append("filter matched no documents, retried without it")
        if mode == "vector":
            candidates = [doc for doc, _ in _vector_leg(question, settings.candidate_k, None)]
        else:
            candidates = MongoDBAtlasHybridSearchRetriever(
                vectorstore=get_vector_store(),
                search_index_name=settings.fulltext_index_name,
                k=settings.candidate_k,
            ).invoke(question)
        trace["candidates_found"] = len(candidates)

    if mode == "hybrid_rerank" and settings.rerank_enabled and candidates:
        started = time.perf_counter()
        reranked = rerank(question, candidates, top_k=len(candidates))
        trace["timings_ms"]["rerank"] = round((time.perf_counter() - started) * 1000)
        trace["rerank_top"] = [
            {"doc": _label(d), "score": round(s, 3)} for d, s in reranked[:8]
        ]
        candidates = [d for d, _ in reranked]

    # Applied last: reranking is a relevance signal and would otherwise undo the
    # granularity choice. Stable sort, so the reranked order survives within each group.
    if parsed.prefer_doc_type:
        candidates.sort(key=lambda d: d.metadata.get("doc_type") != parsed.prefer_doc_type)
        trace["preferred_doc_type"] = parsed.prefer_doc_type

    documents = _diversify(candidates, settings.max_per_company, final_k)
    trace["returned"] = [_label(d) for d in documents]

    if explain:
        started = time.perf_counter()
        try:
            vector_hits = _vector_leg(question, 8, pre_filter)
            trace["vector_only"] = [
                {"doc": _label(d), "score": round(s, 3)} for d, s in vector_hits
            ]
        except Exception as exc:
            trace["vector_only_error"] = str(exc)
        try:
            trace["fulltext_only"] = [
                {"doc": _label(d), "score": round(d.metadata.get("score", 0.0), 3)}
                for d in _fulltext_leg(question, 8, pre_filter)
            ]
        except Exception as exc:
            trace["fulltext_only_error"] = str(exc)
        trace["timings_ms"]["explain"] = round((time.perf_counter() - started) * 1000)

    return RetrievalResult(documents=documents, parsed=parsed, mode=mode, trace=trace)
