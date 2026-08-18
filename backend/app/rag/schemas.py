from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    mode: Literal["vector", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    explain: bool = True


class Citation(BaseModel):
    index: int
    doc_type: str
    company: str
    college: str | list[str] | None = None
    year: int | list[int] | None = None
    role: str | None = None
    package_lpa: float | None = None
    cgpa_cutoff: float | None = None
    branches: list[str] = Field(default_factory=list)
    selection_rounds: list[str] = Field(default_factory=list)
    active_backlogs: str | None = None
    job_location: str | None = None
    text: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    trace: dict[str, Any]
    latency_ms: int


class StatsResponse(BaseModel):
    documents: int
    records: int
    company_profiles: int
    companies: int
    colleges: int
    years: list[int]
    package_min: float | None = None
    package_max: float | None = None
    indexes: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    mongo: bool
    gemini_key: bool
    collection: str
    documents: int | None = None
