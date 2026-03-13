"""
Recommendation endpoint for the Internship Recommendation Engine API.

Exposes a single ``POST /recommend`` route that accepts either a multipart
file upload (PDF / DOCX / TXT) **or** a JSON body with raw text, then
executes the full pipeline:

    parse → preprocess → extract_profile → embed → FAISS search → rank → respond

Structured HTTP errors are returned for invalid input, oversized files,
empty resumes, and service-unavailability (index not loaded).
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse
from loguru import logger

from api.schemas import (
    CandidateProfile,
    RecommendationResponse,
    RecommendationResult,
)
from engine.embedder import get_embedder
from engine.extractor import extract_profile
from engine.parser import ResumeParseError, parse_resume
from engine.preprocessor import detect_language, preprocess
from engine.scorer import rank_recommendations

# ── Constants ────────────────────────────────────────────────────────────────

_MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 MB
_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".txt"})

router = APIRouter(tags=["Recommendations"])


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_index(request: Request):
    """Retrieve the InternshipIndex from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The loaded :class:`~engine.indexer.InternshipIndex`.

    Raises:
        HTTPException: 503 if the index is not loaded.
    """
    index = getattr(request.app.state, "index", None)
    if index is None or index.index is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "service_unavailable",
                "message": (
                    "FAISS index is not loaded. The service is starting up "
                    "or the index file is missing. Run scripts/build_index.py first."
                ),
            },
        )
    return index


def _log_request_background(
    method: str,
    path: str,
    candidate_name: str,
    num_results: int,
    latency_ms: float,
) -> None:
    """Background task that logs recommendation request metadata.

    Args:
        method: HTTP method.
        path: Request path.
        candidate_name: Name of the candidate.
        num_results: Number of results returned.
        latency_ms: Pipeline latency in milliseconds.
    """
    logger.info(
        "RecommendRequest | {} {} | candidate='{}' | results={} | latency={:.0f}ms",
        method,
        path,
        candidate_name,
        num_results,
        latency_ms,
    )


def _validate_extension(filename: str) -> str:
    """Validate and return the lowercase file extension.

    Args:
        filename: Original filename.

    Returns:
        Lowercase extension string (e.g. ``'.pdf'``).

    Raises:
        HTTPException: 422 if the extension is unsupported.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unsupported_format",
                "message": (
                    f"Unsupported file format '{ext}'. "
                    f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
                ),
            },
        )
    return ext


# ── Endpoint: file upload ────────────────────────────────────────────────────


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Get internship recommendations from a resume",
    description=(
        "Upload a resume file (PDF, DOCX, or TXT) or send raw text to "
        "receive ranked internship recommendations."
    ),
    responses={
        422: {"description": "Invalid input (bad format, too large, empty)"},
        503: {"description": "FAISS index not loaded"},
    },
)
async def recommend(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(default=None, description="Resume file (PDF/DOCX/TXT, max 5 MB)"),
    raw_text: str | None = Form(default=None, description="Raw resume text (alternative to file upload)"),
    candidate_name: str | None = Form(default=None, description="Candidate name (optional)"),
) -> RecommendationResponse:
    """Process a resume and return ranked internship recommendations.

    Accepts **either** a multipart file upload or raw text via form field.
    Executes the full pipeline: parse → preprocess → extract → embed →
    search → rank.

    Args:
        request: The incoming request (used to access app state).
        background_tasks: FastAPI background task runner.
        file: Uploaded resume file (mutually exclusive with ``raw_text``).
        raw_text: Plain-text resume content.
        candidate_name: Optional candidate name override.

    Returns:
        A :class:`~api.schemas.RecommendationResponse` with ranked results.

    Raises:
        HTTPException: 422 for validation errors, 503 if index unavailable.
    """
    t0 = time.perf_counter()

    # ── Ensure index is available ────────────────────────────────────────
    index = _get_index(request)

    # ── Load settings for weights ────────────────────────────────────────
    try:
        from config import settings

        semantic_weight = settings.SEMANTIC_WEIGHT
        keyword_weight = settings.KEYWORD_WEIGHT
        top_k = settings.TOP_K_RESULTS
    except Exception:
        semantic_weight = 0.70
        keyword_weight = 0.30
        top_k = 10

    # ── Extract raw text from file or form ──────────────────────────────
    parsed_text: str

    if file is not None and file.filename:
        # File upload path
        _validate_extension(file.filename)

        # Size check
        content = await file.read()
        if len(content) > _MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "file_too_large",
                    "message": (
                        f"File size ({len(content) / (1024 * 1024):.1f} MB) exceeds "
                        f"the {_MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f} MB limit."
                    ),
                },
            )

        # Write to temp file for parser
        ext = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            parsed_text = parse_resume(tmp_path)
        except ResumeParseError as e:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "parse_failed",
                    "message": str(e),
                },
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    elif raw_text and raw_text.strip():
        parsed_text = raw_text.strip()
    else:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "no_input",
                "message": "Provide either a file upload or raw_text.",
            },
        )

    # ── Preprocess ───────────────────────────────────────────────────────
    cleaned = preprocess(parsed_text)

    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "empty_resume",
                "message": "Resume text is empty after parsing and cleaning.",
            },
        )

    # ── Language check (warn but continue) ───────────────────────────────
    lang = detect_language(cleaned)
    if lang not in ("en", "unknown"):
        logger.warning(
            "Non-English resume detected (lang='{}'). "
            "Continuing, but results may be less accurate.",
            lang,
        )

    # ── Extract profile ──────────────────────────────────────────────────
    profile: CandidateProfile = extract_profile(cleaned)

    # Override candidate name if provided via form
    resolved_name = candidate_name or "Unknown Candidate"
    # Try extracting name from profile extractor
    from engine.extractor import _extract_candidate_name

    if resolved_name == "Unknown Candidate":
        resolved_name = _extract_candidate_name(cleaned)

    # ── Embed → search → rank ────────────────────────────────────────────
    embedder = get_embedder()
    query_vec = embedder.embed_profile(profile)

    faiss_candidates = index.search(query_vec, top_k=20)

    results: list[RecommendationResult] = rank_recommendations(
        profile=profile,
        candidates=faiss_candidates,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
        top_n=top_k,
    )

    # ── Build response ───────────────────────────────────────────────────
    response = RecommendationResponse(
        candidate_name=resolved_name,
        total_results=len(results),
        results=results,
    )

    # ── Background logging ───────────────────────────────────────────────
    latency_ms = (time.perf_counter() - t0) * 1000
    background_tasks.add_task(
        _log_request_background,
        method=request.method,
        path=str(request.url.path),
        candidate_name=resolved_name,
        num_results=len(results),
        latency_ms=latency_ms,
    )

    return response
