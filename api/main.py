"""
FastAPI application factory for the Internship Recommendation Engine.

Creates and configures the ASGI application with:

* **Lifespan** — loads the FAISS index and Embedder singleton at startup;
  fails fast if the index file is missing.
* **CORS middleware** — configurable allowed origins.
* **Request logging middleware** — logs method, path, status code, and
  latency for every request via loguru.
* **Health endpoint** — ``GET /health`` returning service status.
* **OpenAPI metadata** — title, description, version.

Run with::

    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.routes.recommend import router as recommend_router


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown logic.

    **Startup**:
        1. Load the ``Embedder`` singleton (downloads / caches the model).
        2. Load the FAISS ``InternshipIndex`` from disk.

    Fails fast with a clear error if the index file is missing, so that
    misconfiguration is caught immediately rather than at first request
    time.

    **Shutdown**:
        Logs a clean shutdown message.

    Args:
        app: The FastAPI application instance.
    """
    # ── Startup ──────────────────────────────────────────────────────────
    logger.info("Starting Internship Recommendation Engine …")

    # Load embedder singleton
    try:
        from engine.embedder import get_embedder

        embedder = get_embedder()
        app.state.embedder_loaded = True
        logger.info("Embedder loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load Embedder: {}", exc)
        app.state.embedder_loaded = False
        raise

    # Load FAISS index
    try:
        from config import settings

        index_path = settings.FAISS_INDEX_PATH
    except Exception:
        index_path = "data/faiss.index"
        logger.warning(
            "Could not read FAISS_INDEX_PATH from Settings; "
            "defaulting to '{}'.",
            index_path,
        )

    try:
        from engine.indexer import InternshipIndex

        index = InternshipIndex()
        index.load(index_path)
        app.state.index = index
        app.state.index_loaded = True
        logger.info(
            "FAISS index loaded: {} listings.",
            len(index.listings),
        )
    except FileNotFoundError as exc:
        logger.error("FAISS index not found — {}", exc)
        app.state.index = None
        app.state.index_loaded = False
        # Fail fast: raise so the server refuses to start with a broken state
        raise RuntimeError(
            f"Cannot start: FAISS index file missing at '{index_path}'. "
            "Run `python scripts/build_index.py` first."
        ) from exc

    app.state.startup_time = datetime.now(timezone.utc)
    logger.info("Startup complete ✓")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("Shutting down Internship Recommendation Engine.")


# ── App factory ──────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A fully configured :class:`FastAPI` instance.
    """
    app = FastAPI(
        title="Internship Recommendation Engine",
        description=(
            "AI-powered API that matches candidate resumes to internship "
            "listings using hybrid semantic + keyword similarity scoring. "
            "Upload a resume (PDF, DOCX, or TXT) and receive ranked "
            "recommendations with match explanations."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS middleware ──────────────────────────────────────────────────
    try:
        from config import settings  # noqa: F811

        # Allow configurable origins via env var (comma-separated)
        origins_raw = getattr(settings, "CORS_ORIGINS", "*")
    except Exception:
        origins_raw = "*"

    if origins_raw == "*":
        allowed_origins = ["*"]
    elif isinstance(origins_raw, str):
        allowed_origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
    else:
        allowed_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request logging middleware ────────────────────────────────────────

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log every HTTP request with method, path, status, and latency."""
        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "{method} {path} → {status} ({latency:.0f}ms)",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency=latency_ms,
        )
        return response

    # ── Health endpoint ──────────────────────────────────────────────────

    @app.get(
        "/health",
        tags=["System"],
        summary="Service health check",
        response_description="Current health status of the API.",
    )
    async def health(request: Request) -> JSONResponse:
        """Return the current health status of the service.

        Fields:
            - ``status``: ``"healthy"`` or ``"degraded"``
            - ``index_loaded``: whether the FAISS index is in memory
            - ``model_loaded``: whether the embedding model is loaded
            - ``timestamp``: current UTC ISO-8601 timestamp
        """
        index_ok = getattr(request.app.state, "index_loaded", False)
        model_ok = getattr(request.app.state, "embedder_loaded", False)

        status = "healthy" if (index_ok and model_ok) else "degraded"

        return JSONResponse(
            content={
                "status": status,
                "index_loaded": index_ok,
                "model_loaded": model_ok,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            status_code=200 if status == "healthy" else 503,
        )

    # ── Register routers ─────────────────────────────────────────────────
    app.include_router(recommend_router, prefix="/api/v1")

    return app


# ── Module-level app instance (for uvicorn api.main:app) ─────────────────────

app = create_app()
