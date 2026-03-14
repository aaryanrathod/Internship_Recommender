"""
Application configuration module for the Internship Recommendation Engine.

Uses Pydantic BaseSettings to load configuration from environment variables
and a .env file, providing type-safe access to all engine parameters.

All fields have sensible defaults so the app works without a .env file
(e.g. on Streamlit Cloud where .env is not committed).
"""

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator


class Settings(BaseSettings):
    """Central configuration for the Internship Recommendation Engine.

    All fields can be overridden via environment variables or a ``.env`` file
    located in the project root. Environment variables take precedence over
    values defined in the ``.env`` file.

    Note:
        ``SEMANTIC_WEIGHT + KEYWORD_WEIGHT + LOCATION_WEIGHT`` must equal 1.0.
    """

    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model identifier for embedding generation.",
    )
    FAISS_INDEX_PATH: str = Field(
        default="data/faiss.index",
        description="Path to the serialized FAISS index file.",
    )
    INTERNSHIP_DATA_PATH: str = Field(
        default="data/internships_batch1.csv",
        description="Path to the internship listings dataset.",
    )
    TOP_K_RESULTS: int = Field(
        default=10,
        description="Number of top recommendations to return.",
    )
    SEMANTIC_WEIGHT: float = Field(
        default=0.60,
        description="Weight for semantic similarity in hybrid scoring (0-1).",
    )
    KEYWORD_WEIGHT: float = Field(
        default=0.25,
        description="Weight for keyword similarity in hybrid scoring (0-1).",
    )
    LOCATION_WEIGHT: float = Field(
        default=0.15,
        description="Weight for location-based matching in hybrid scoring (0-1).",
    )
    CANDIDATE_LOCATION: str = Field(
        default="",
        description="Default candidate location preference (empty = no preference).",
    )
    REDIS_URL: str = Field(
        default="",
        description="Redis connection URL. Leave empty to disable caching.",
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Application logging level.",
    )
    MAX_TEXT_LENGTH: int = Field(
        default=10000,
        description="Maximum character length for input text fields.",
    )

    @model_validator(mode="after")
    def _check_weights_sum_to_one(self) -> "Settings":
        """Validate that the three scoring weights sum to 1.0 (+-tolerance)."""
        total = self.SEMANTIC_WEIGHT + self.KEYWORD_WEIGHT + self.LOCATION_WEIGHT
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"SEMANTIC_WEIGHT ({self.SEMANTIC_WEIGHT}) + "
                f"KEYWORD_WEIGHT ({self.KEYWORD_WEIGHT}) + "
                f"LOCATION_WEIGHT ({self.LOCATION_WEIGHT}) = {total:.4f}, "
                f"but must sum to 1.0."
            )
        return self

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


# Singleton instance — import this throughout the application.
settings = Settings()
