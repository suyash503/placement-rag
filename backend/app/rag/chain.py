import time
from collections.abc import AsyncIterator
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.rag.retriever import RetrievalMode, retrieve
from backend.app.rag.schemas import ChatMessage, Citation

log = get_logger("rag.chain")

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the campus placement assistant for a college Training & Placement cell. "
            "Answer strictly from the numbered records below.\n\n"
            "Rules:\n"
            "- Cite the record number in square brackets after every fact, like [2].\n"
            "- If the records do not contain the answer, say so plainly and suggest what to ask "
            "instead. Never invent a company, package, or cutoff.\n"
            "- Prefer a short markdown table when listing three or more companies.\n"
            "- Packages are annual CTC in LPA. CGPA cutoffs are the minimum a student needs.\n"
            "- Keep it concise and factual; no filler openings.\n\n"
            "RECORDS:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the follow-up question as a standalone question that carries over any "
            "company, college, year or branch mentioned earlier. Return only the rewritten "
            "question, nothing else.",
        ),
        ("human", "Conversation so far:\n{history}\n\nFollow-up: {question}"),
    ]
)


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.1,
        max_output_tokens=1400,
    )


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] {doc.page_content}" for i, doc in enumerate(documents, start=1)
    )


def build_citations(documents: list[Document]) -> list[Citation]:
    citations = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata
        company = meta.get("company")
        citations.append(
            Citation(
                index=i,
                doc_type=meta.get("doc_type", "record"),
                company=company if isinstance(company, str) else str(company),
                college=meta.get("college"),
                year=meta.get("year"),
                role=meta.get("role"),
                package_lpa=meta.get("package_lpa"),
                cgpa_cutoff=meta.get("cgpa_cutoff"),
                branches=meta.get("branches") or [],
                selection_rounds=meta.get("selection_rounds") or [],
                active_backlogs=meta.get("active_backlogs"),
                job_location=meta.get("job_location"),
                text=doc.page_content,
            )
        )
    return citations


async def condense(question: str, history: list[ChatMessage]) -> str:
    """A bare follow-up like "and its CGPA cutoff?" embeds to nothing useful.

    Folding the prior turns back into the question is what makes multi-turn work.
    """
    if not history:
        return question

    transcript = "\n".join(f"{m.role}: {m.content}" for m in history[-6:])
    chain = CONDENSE_PROMPT | get_llm()
    try:
        result = await chain.ainvoke({"history": transcript, "question": question})
        rewritten = result.content.strip()
        if rewritten and len(rewritten) < 400:
            log.info("condensed %r -> %r", question, rewritten)
            return rewritten
    except Exception as exc:
        log.warning("condense failed, using raw question: %s", exc)
    return question


async def answer_stream(
    question: str,
    history: list[ChatMessage] | None = None,
    mode: RetrievalMode = "hybrid_rerank",
    explain: bool = True,
) -> AsyncIterator[dict]:
    started = time.perf_counter()
    history = history or []

    standalone = await condense(question, history)
    result = retrieve(standalone, mode=mode, explain=explain)

    trace = dict(result.trace)
    if standalone != question:
        trace["condensed_question"] = standalone

    citations = build_citations(result.documents)
    yield {"type": "meta", "citations": [c.model_dump() for c in citations], "trace": trace}

    if not result.documents:
        yield {
            "type": "token",
            "value": "I could not find any placement record matching that. "
            "Try naming a company, a college, a year between 2014 and 2018, or a package range.",
        }
        yield {"type": "done", "latency_ms": round((time.perf_counter() - started) * 1000)}
        return

    chain = ANSWER_PROMPT | get_llm()
    payload = {"context": format_context(result.documents), "question": standalone}

    try:
        async for chunk in chain.astream(payload):
            if chunk.content:
                yield {"type": "token", "value": chunk.content}
    except Exception as exc:
        log.exception("generation failed")
        yield {"type": "error", "message": f"Generation failed: {exc}"}

    yield {"type": "done", "latency_ms": round((time.perf_counter() - started) * 1000)}


async def answer(
    question: str,
    history: list[ChatMessage] | None = None,
    mode: RetrievalMode = "hybrid_rerank",
    explain: bool = False,
) -> dict:
    parts: list[str] = []
    meta: dict = {}
    latency = 0
    async for event in answer_stream(question, history, mode=mode, explain=explain):
        if event["type"] == "token":
            parts.append(event["value"])
        elif event["type"] == "meta":
            meta = event
        elif event["type"] == "done":
            latency = event["latency_ms"]
    return {
        "answer": "".join(parts),
        "citations": meta.get("citations", []),
        "trace": meta.get("trace", {}),
        "latency_ms": latency,
    }
