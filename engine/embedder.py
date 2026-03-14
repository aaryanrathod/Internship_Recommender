"""
Embedding module for the Internship Recommendation Engine.

Wraps ``sentence-transformers`` to produce dense float32 vectors from text,
with optional Redis caching (hash-keyed, 1-hour TTL) and automatic chunking
for inputs that exceed the model's token window.

Usage::

    from engine.embedder import get_embedder

    embedder = get_embedder()
    vec = embedder.embed_text("Python developer with ML experience")
    profile_vec = embedder.embed_profile(candidate_profile)

Dependencies:
    - ``sentence-transformers``
    - ``numpy``
    - ``redis`` (optional — caching degrades gracefully)
    - ``loguru``
"""

from __future__ import annotations

import hashlib
import time
from typing import List, Optional

import numpy as np
from loguru import logger

from api.schemas import CandidateProfile
from engine.preprocessor import chunk_text

# ── Constants ────────────────────────────────────────────────────────────────

_LONG_TEXT_TOKEN_LIMIT: int = 512
"""Approximate token count above which inputs are chunked before embedding."""

_LONG_TEXT_WORD_LIMIT: int = int(_LONG_TEXT_TOKEN_LIMIT * 0.75)
"""Word-count proxy for the token limit (1 token ≈ 0.75 words)."""

_CACHE_TTL_SECONDS: int = 3600
"""Redis cache time-to-live for embedding vectors (1 hour)."""

_CACHE_KEY_PREFIX: str = "emb:"
"""Redis key prefix to namespace embedding cache entries."""


# ── Embedder class ───────────────────────────────────────────────────────────


class Embedder:
    """Produces dense vector embeddings for text using SentenceTransformers.

    The model is loaded exactly once at initialisation.  Long texts are
    automatically split into overlapping chunks via
    :func:`engine.preprocessor.chunk_text` and the resulting chunk
    embeddings are mean-pooled into a single vector.

    Embedding results can optionally be cached in Redis (keyed by SHA-256
    hash of the input text).  Redis failures are caught silently — the
    engine continues without caching.

    Args:
        model_name: SentenceTransformer model identifier.  When ``None``,
            the value is read from :attr:`config.Settings.EMBEDDING_MODEL`.

    Attributes:
        model_name: Name of the loaded model.
        embedding_dim: Dimensionality of the output vectors.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        # Resolve model name via config if not provided
        if model_name is None:
            try:
                from config import settings
                model_name = settings.EMBEDDING_MODEL
            except Exception:
                model_name = "all-MiniLM-L6-v2"
                logger.warning(
                    "Could not load Settings; defaulting to model '{}'.",
                    model_name,
                )

        self.model_name: str = model_name

        t0 = time.perf_counter()
        self._model = SentenceTransformer(model_name)
        elapsed = time.perf_counter() - t0

        self.embedding_dim: int = self._model.get_sentence_embedding_dimension()

        logger.info(
            "SentenceTransformer '{}' loaded in {:.2f}s  "
            "(embedding_dim={}).",
            self.model_name,
            elapsed,
            self.embedding_dim,
        )

        # ── Optional Redis connection ────────────────────────────────────
        self._redis = self._connect_redis()

    # ── Redis helpers ────────────────────────────────────────────────────

    @staticmethod
    def _connect_redis():
        """Attempt to connect to Redis using the URL from Settings.

        Returns:
            A ``redis.Redis`` client instance, or ``None`` if Redis is
            unavailable or not configured.
        """
        try:
            from config import settings
            if not settings.REDIS_URL:
                logger.debug("REDIS_URL is empty — caching disabled.")
                return None

            import redis

            client = redis.Redis.from_url(
                settings.REDIS_URL,
                decode_responses=False,
                socket_connect_timeout=2,
            )
            client.ping()
            logger.info("Redis cache connected at '{}'.", settings.REDIS_URL)
            return client
        except Exception as exc:
            logger.debug("Redis not available (caching disabled): {}", exc)
            return None

    def _cache_key(self, text: str) -> str:
        """Compute a deterministic Redis key for a given text input.

        Args:
            text: The raw input string.

        Returns:
            Cache key of the form ``emb:<sha256_hex>``.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{_CACHE_KEY_PREFIX}{digest}"

    def _cache_get(self, text: str) -> np.ndarray | None:
        """Look up a cached embedding vector in Redis.

        Args:
            text: Input text whose embedding may be cached.

        Returns:
            The cached ``float32`` numpy vector, or ``None`` on miss or
            error.
        """
        if self._redis is None:
            return None
        try:
            data: bytes | None = self._redis.get(self._cache_key(text))
            if data is not None:
                vec = np.frombuffer(data, dtype=np.float32).copy()
                logger.debug("Cache HIT for text hash '{}'.", self._cache_key(text))
                return vec
        except Exception as exc:
            logger.debug("Redis GET failed: {}", exc)
        return None

    def _cache_set(self, text: str, vector: np.ndarray) -> None:
        """Store an embedding vector in Redis with a 1-hour TTL.

        Args:
            text: Input text (used to derive the cache key).
            vector: The embedding vector to cache.
        """
        if self._redis is None:
            return
        try:
            self._redis.setex(
                self._cache_key(text),
                _CACHE_TTL_SECONDS,
                vector.astype(np.float32).tobytes(),
            )
            logger.debug("Cache SET for key '{}'.", self._cache_key(text))
        except Exception as exc:
            logger.debug("Redis SET failed: {}", exc)

    # ── Core embedding methods ───────────────────────────────────────────

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string into a normalised float32 vector.

        If the text exceeds ~512 tokens (≈384 words) it is split into
        overlapping chunks via :func:`engine.preprocessor.chunk_text` and
        the chunk embeddings are averaged.

        Results are cached in Redis (when available) keyed by the SHA-256
        hash of the input text.

        Args:
            text: The input text to embed.

        Returns:
            A 1-D ``np.ndarray`` of dtype ``float32`` with unit L2 norm.
        """
        # ── Check cache ──────────────────────────────────────────────────
        cached = self._cache_get(text)
        if cached is not None:
            return cached

        # ── Chunk long texts ─────────────────────────────────────────────
        words = text.split()
        if len(words) > _LONG_TEXT_WORD_LIMIT:
            chunks = chunk_text(text, max_tokens=_LONG_TEXT_TOKEN_LIMIT, overlap=64)
            logger.debug(
                "Text exceeds token limit ({} words); split into {} chunks.",
                len(words),
                len(chunks),
            )
            chunk_vecs = self._model.encode(
                chunks,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            # Mean-pool across chunks
            vector = np.mean(chunk_vecs, axis=0).astype(np.float32)
        else:
            vector = self._model.encode(
                text,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)

        # ── L2-normalise ─────────────────────────────────────────────────
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        # ── Store in cache ───────────────────────────────────────────────
        self._cache_set(text, vector)

        return vector

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of text strings.

        Args:
            texts: List of input strings.

        Returns:
            A 2-D ``np.ndarray`` of shape ``(len(texts), embedding_dim)``
            with dtype ``float32``.  Each row is L2-normalised.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        vectors = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=64,
        ).astype(np.float32)

        # Row-wise L2 normalisation
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)  # avoid division by zero
        vectors = vectors / norms

        logger.debug("Batch-embedded {} texts → shape {}.", len(texts), vectors.shape)
        return vectors

    def embed_profile(self, profile: CandidateProfile) -> np.ndarray:
        """Embed a candidate profile using its ``summary_text``.

        This is the recommended way to obtain a candidate's embedding for
        similarity search against the FAISS index.

        Args:
            profile: A populated :class:`~api.schemas.CandidateProfile`.

        Returns:
            A 1-D normalised ``float32`` vector.
        """
        if not profile.summary_text:
            logger.warning(
                "Profile has empty summary_text; embedding raw_text instead."
            )
            return self.embed_text(profile.raw_text[:5000])

        return self.embed_text(profile.summary_text)


# ── Singleton access ─────────────────────────────────────────────────────────

_singleton: Embedder | None = None


def get_embedder(model_name: str | None = None) -> Embedder:
    """Return the module-level singleton :class:`Embedder` instance.

    The model is loaded on the first call; subsequent calls return the
    same object.  Pass ``model_name`` only to override the config on
    first initialisation — it is ignored after the singleton exists.

    Args:
        model_name: Optional model name override (used only on first call).

    Returns:
        The shared :class:`Embedder` instance.
    """
    global _singleton
    if _singleton is None:
        _singleton = Embedder(model_name=model_name)
    return _singleton
