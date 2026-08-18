from functools import lru_cache

from fastapi import APIRouter, Query

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.rag.schemas import HealthResponse, StatsResponse
from backend.app.rag.store import get_collection, ping

log = get_logger("api.meta")

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    mongo_ok = ping()

    documents = None
    if mongo_ok:
        try:
            documents = get_collection().count_documents({})
        except Exception as exc:
            log.warning("count failed: %s", exc)
            mongo_ok = False

    gemini_ok = bool(settings.gemini_api_key)
    return HealthResponse(
        status="ok" if mongo_ok and gemini_ok else "degraded",
        mongo=mongo_ok,
        gemini_key=gemini_ok,
        collection=f"{settings.mongo_db}.{settings.mongo_collection}",
        documents=documents,
    )


@router.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    collection = get_collection()

    pipeline = [
        {
            "$group": {
                "_id": None,
                "companies": {"$addToSet": "$company"},
                "package_min": {"$min": "$package_lpa"},
                "package_max": {"$max": "$package_lpa"},
            }
        }
    ]
    agg = next(iter(collection.aggregate(pipeline)), {})

    records = collection.count_documents({"doc_type": "record"})
    profiles = collection.count_documents({"doc_type": "company"})
    colleges = collection.distinct("college", {"doc_type": "record"})
    years = sorted(y for y in collection.distinct("year", {"doc_type": "record"}) if isinstance(y, int))

    indexes = [
        {"name": idx["name"], "type": idx.get("type", "search"), "status": idx.get("status")}
        for idx in collection.list_search_indexes()
    ]

    return StatsResponse(
        documents=records + profiles,
        records=records,
        company_profiles=profiles,
        companies=len(agg.get("companies", [])),
        colleges=len(colleges),
        years=years,
        package_min=agg.get("package_min"),
        package_max=agg.get("package_max"),
        indexes=indexes,
    )


@lru_cache(maxsize=1)
def _company_names() -> list[str]:
    return sorted(get_collection().distinct("company", {"doc_type": "company"}))


@router.get("/companies")
async def companies(q: str = Query(default="", max_length=80), limit: int = 20) -> list[str]:
    names = _company_names()
    if not q:
        return names[:limit]
    needle = q.lower()
    return [n for n in names if needle in n.lower()][:limit]
