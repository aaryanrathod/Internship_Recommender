"""
FAISS vector index module for the Internship Recommendation Engine.

Manages the lifecycle of a FAISS ``IndexFlatIP`` (inner-product / cosine
similarity on L2-normalised vectors) that maps internship listings to dense
embeddings for nearest-neighbour retrieval.

Usage::

    from engine.indexer import InternshipIndex
    from engine.embedder import get_embedder

    idx = InternshipIndex()
    idx.build(internships, get_embedder())
    idx.save("data/faiss.index")

    # Later …
    idx2 = InternshipIndex()
    idx2.load("data/faiss.index")
    results = idx2.search(query_vec, top_k=10)

Scaling note:
    ``IndexFlatIP`` performs **exact** brute-force search and is ideal for
    datasets up to ~50 000 listings.  For larger corpora, switch to
    ``IndexIVFFlat`` (inverted-file with flat quantiser) — this requires a
    training step but yields sub-linear search time::

        quantiser = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantiser, dim, n_cells)
        index.train(vectors)
        index.add(vectors)
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from loguru import logger

from api.schemas import InternshipListing
from engine.embedder import Embedder

# ── Constants ────────────────────────────────────────────────────────────────

_META_SUFFIX: str = ".meta.pkl"
"""Suffix appended to the FAISS index path to derive the metadata filename."""


# ── Helper ───────────────────────────────────────────────────────────────────


def _listing_to_text(listing: InternshipListing) -> str:
    """Concatenate the key fields of an internship listing into a single
    string suitable for embedding.

    Format: ``"<title> <company> <skills> <description>"``

    Args:
        listing: An :class:`~api.schemas.InternshipListing` instance.

    Returns:
        Combined text string.
    """
    parts = [
        listing.title,
        listing.company,
        " ".join(listing.required_skills),
        listing.description,
    ]
    return " ".join(p for p in parts if p).strip()


# ── InternshipIndex ──────────────────────────────────────────────────────────


class InternshipIndex:
    """FAISS-backed vector index for internship listings.

    Stores a parallel list of :class:`~api.schemas.InternshipListing`
    metadata alongside the FAISS index so that search results can be
    mapped back to full listing objects.

    Attributes:
        index: The underlying ``faiss.IndexFlatIP`` (or ``None`` before
            :meth:`build` / :meth:`load`).
        listings: Parallel metadata list aligned with FAISS row IDs.
    """

    def __init__(self) -> None:
        """Initialise empty index state."""
        self.index: faiss.IndexFlatIP | None = None
        self.listings: List[InternshipListing] = []
        logger.debug("InternshipIndex initialised (empty).")

    # ── Build ────────────────────────────────────────────────────────────

    def build(
        self,
        internships: List[InternshipListing],
        embedder: Embedder,
    ) -> None:
        """Embed all internship listings and build the FAISS index.

        Each listing is converted to text via :func:`_listing_to_text`,
        batch-embedded, and added to a new ``IndexFlatIP``.

        Args:
            internships: List of validated internship listings.
            embedder: An initialised :class:`~engine.embedder.Embedder`.

        Raises:
            ValueError: If ``internships`` is empty.
        """
        if not internships:
            raise ValueError("Cannot build index from an empty listing set.")

        logger.info("Building FAISS index for {} internships …", len(internships))
        t0 = time.perf_counter()

        # ── Prepare texts ────────────────────────────────────────────────
        texts: List[str] = []
        for listing in internships:
            texts.append(_listing_to_text(listing))

        # ── Batch embed ──────────────────────────────────────────────────
        vectors = embedder.embed_batch(texts)  # (n, dim), float32, normalised

        # ── Create FAISS index ───────────────────────────────────────────
        dim = vectors.shape[1]
        # IndexFlatIP = exact inner-product (cosine similarity on unit vecs).
        # For datasets > 50k listings, consider IndexIVFFlat for faster
        # approximate search — see module docstring.
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self.listings = list(internships)

        elapsed = time.perf_counter() - t0
        logger.info(
            "FAISS index built: {} vectors × {} dims in {:.2f}s.",
            self.index.ntotal,
            dim,
            elapsed,
        )

    # ── Persist / restore ────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the FAISS index and metadata to disk.

        Two files are written:
            - ``<path>`` — the FAISS binary index
            - ``<path>.meta.pkl`` — pickled list of
              :class:`~api.schemas.InternshipListing`

        Args:
            path: Destination path for the FAISS index file.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self.index is None or not self.listings:
            raise RuntimeError(
                "Cannot save: index has not been built. "
                "Call build() first."
            )

        index_path = Path(path)
        meta_path = Path(f"{path}{_META_SUFFIX}")

        # Ensure parent directories exist
        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as fh:
            pickle.dump(self.listings, fh, protocol=pickle.HIGHEST_PROTOCOL)

        idx_size = index_path.stat().st_size
        meta_size = meta_path.stat().st_size
        logger.info(
            "Index saved → '{}' ({:.1f} KB) + metadata ({:.1f} KB).",
            index_path,
            idx_size / 1024,
            meta_size / 1024,
        )

    def load(self, path: str) -> None:
        """Load a previously saved FAISS index and metadata from disk.

        Args:
            path: Path to the FAISS index file (the metadata pickle is
                expected at ``<path>.meta.pkl``).

        Raises:
            FileNotFoundError: If either the index file or the metadata
                file is missing.
        """
        index_path = Path(path)
        meta_path = Path(f"{path}{_META_SUFFIX}")

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index file not found: '{index_path}'. "
                "Run scripts/build_index.py to create it."
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Index metadata file not found: '{meta_path}'. "
                "The index may be corrupted — rebuild with scripts/build_index.py."
            )

        t0 = time.perf_counter()
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as fh:
            self.listings = pickle.load(fh)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Index loaded from '{}': {} vectors, {} listings in {:.2f}s.",
            index_path,
            self.index.ntotal,
            len(self.listings),
            elapsed,
        )

    # ── Search ───────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 20,
    ) -> List[Tuple[float, InternshipListing]]:
        """Find the top-k most similar internship listings.

        Args:
            query_vec: A 1-D normalised ``float32`` query embedding
                (e.g. from :meth:`Embedder.embed_profile`).
            top_k: Number of nearest neighbours to return.

        Returns:
            List of ``(score, InternshipListing)`` tuples sorted in
            **descending** order of cosine similarity score.

        Raises:
            RuntimeError: If the index has not been built or loaded.
        """
        if self.index is None:
            raise RuntimeError(
                "Index is empty. Call build() or load() before searching."
            )

        # FAISS expects a 2-D query matrix
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        query_vec = query_vec.astype(np.float32)
        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_vec, k)

        results: List[Tuple[float, InternshipListing]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS sentinel for fewer results than k
            results.append((float(score), self.listings[idx]))

        logger.debug("Search returned {} results (top_k={}).", len(results), top_k)
        return results