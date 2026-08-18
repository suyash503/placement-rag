from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mongo_uri: str = Field(default="", alias="MONGO_URI")
    mongo_db: str = Field(default="placement_rag", alias="MONGO_DB")
    mongo_collection: str = Field(default="placement_chunks", alias="MONGO_COLLECTION")

    vector_index_name: str = "vector_index"
    fulltext_index_name: str = "text_index"

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    embedding_dimensions: int = 384

    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )
    rerank_enabled: bool = Field(default=True, alias="RERANK_ENABLED")

    # Retrieval is a funnel: pull a wide candidate set, rerank it, keep a few.
    candidate_k: int = 25
    final_k: int = 6
    max_per_company: int = 3
    vector_penalty: float = 60.0
    fulltext_penalty: float = 60.0

    cors_origins: str = Field(default="http://localhost:5173", alias="CORS_ORIGINS")

    @field_validator("mongo_uri")
    @classmethod
    def _reject_placeholder_uri(cls, v: str) -> str:
        if "<password>" in v or "<db_password>" in v:
            raise ValueError(
                "MONGO_URI still contains a placeholder password. Update .env with real credentials."
            )
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def enriched_csv(self) -> Path:
        return DATA_DIR / "placement_enriched.csv"

    @property
    def raw_csv(self) -> Path:
        return DATA_DIR / "sample_placement_doc.csv"


@lru_cache
def get_settings() -> Settings:
    return Settings()
