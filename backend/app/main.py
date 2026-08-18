import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import chat, meta
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger, setup_logging

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()

    # Both models load from disk on first use, which would otherwise show up as a
    # ~10s stall on whoever asks the first question.
    log.info("warming models...")
    started = time.perf_counter()
    try:
        from backend.app.rag.embeddings import get_embeddings
        get_embeddings().embed_query("warmup")
        if settings.rerank_enabled:
            from backend.app.rag.rerank import get_reranker
            get_reranker().predict([("warmup", "warmup")])
        log.info("models ready in %.1fs", time.perf_counter() - started)
    except Exception as exc:
        log.warning("model warm-up failed: %s", exc)

    from backend.app.rag.store import ping
    log.info("mongo reachable: %s", ping())

    yield


app = FastAPI(
    title="Placement RAG API",
    version="1.0.0",
    description="Hybrid retrieval over campus placement records.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - started) * 1000
    response.headers["X-Response-Time-ms"] = f"{elapsed:.0f}"
    if request.url.path.startswith("/api"):
        log.info("%s %s -> %s in %.0fms", request.method, request.url.path, response.status_code, elapsed)
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


app.include_router(meta.router, prefix="/api")
app.include_router(chat.router, prefix="/api")


@app.get("/")
async def root():
    return {"service": "placement-rag", "docs": "/docs", "health": "/api/health"}
