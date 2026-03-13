"""
Scoring and ranking module for the Internship Recommendation Engine.

Provides pure, side-effect-free functions that combine semantic similarity
scores (from FAISS) with keyword-based skill overlap to produce a ranked
list of :class:`~api.schemas.RecommendationResult` objects.

No AI models are invoked — explanations are generated via deterministic
template logic.

Public API:
    - :func:`compute_skill_overlap` — Jaccard similarity between skill sets.
    - :func:`generate_explanation` — human-readable match rationale.
    - :func:`rank_recommendations` — hybrid scoring + ranking pipeline.
"""

from __future__ import annotations

from typing import List, Tuple

from api.schemas import (
    CandidateProfile,
    InternshipListing,
    RecommendationResult,
)


# ── Skill overlap ────────────────────────────────────────────────────────────


def compute_skill_overlap(
    candidate_skills: List[str],
    internship_skills: List[str],
) -> float:
    """Compute Jaccard similarity between two skill lists.

    Comparison is **case-insensitive**.  Returns ``0.0`` when both lists
    are empty (avoids division by zero).

    Formula::

        |intersection| / |union|

    Args:
        candidate_skills: Skills extracted from the candidate's resume.
        internship_skills: Skills required by the internship listing.

    Returns:
        A float in ``[0.0, 1.0]`` representing skill overlap.

    Examples:
        >>> compute_skill_overlap(["Python", "SQL"], ["python", "java"])
        0.3333333333333333
        >>> compute_skill_overlap([], [])
        0.0
    """
    cand_set = {s.strip().lower() for s in candidate_skills if s.strip()}
    intern_set = {s.strip().lower() for s in internship_skills if s.strip()}

    if not cand_set and not intern_set:
        return 0.0

    intersection = cand_set & intern_set
    union = cand_set | intern_set

    return len(intersection) / len(union)


# ── Explanation generator ────────────────────────────────────────────────────


def generate_explanation(
    profile: CandidateProfile,
    internship: InternshipListing,
    semantic_score: float,
    skill_overlap: float,
) -> str:
    """Generate a template-based explanation for a recommendation match.

    No AI model is called — the output is fully deterministic.

    Args:
        profile: The candidate's parsed profile.
        internship: The matched internship listing.
        semantic_score: Cosine similarity from FAISS (0–1).
        skill_overlap: Jaccard skill overlap (0–1).

    Returns:
        A 1–2 sentence human-readable explanation string.

    Examples:
        >>> generate_explanation(profile, listing, 0.87, 0.55)
        'Strong match based on 6 overlapping skills including Python and SQL. Semantic similarity score of 0.87 suggests closely aligned experience.'
    """
    cand_lower = {s.strip().lower(): s.strip() for s in profile.skills}
    intern_lower = {s.strip().lower(): s.strip() for s in internship.required_skills}
    overlapping = [
        cand_lower[k] for k in cand_lower if k in intern_lower
    ]

    parts: list[str] = []

    # ── Skill sentence ───────────────────────────────────────────────────
    n_overlap = len(overlapping)
    if n_overlap > 0:
        highlighted = overlapping[:3]  # show up to 3 skill names
        skill_names = ", ".join(highlighted)
        if n_overlap <= 3:
            parts.append(
                f"Match based on {n_overlap} overlapping skill{'s' if n_overlap != 1 else ''}: "
                f"{skill_names}."
            )
        else:
            parts.append(
                f"Strong match based on {n_overlap} overlapping skills including "
                f"{skill_names}."
            )
    elif internship.required_skills:
        parts.append(
            "No direct skill overlap detected; recommendation is driven by "
            "overall profile similarity."
        )
    else:
        parts.append(
            "Listing does not specify required skills; ranking is based on "
            "semantic profile similarity."
        )

    # ── Semantic sentence ────────────────────────────────────────────────
    sem = round(semantic_score, 2)
    if sem >= 0.80:
        parts.append(
            f"Semantic similarity score of {sem} suggests closely aligned experience."
        )
    elif sem >= 0.55:
        parts.append(
            f"Semantic similarity score of {sem} indicates a moderate profile fit."
        )
    else:
        parts.append(
            f"Semantic similarity score of {sem} indicates a partial match — "
            "review listing details for fit."
        )

    return " ".join(parts)


# ── Hybrid ranking ───────────────────────────────────────────────────────────


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a numeric value to the ``[lo, hi]`` interval.

    Args:
        value: The value to clamp.
        lo: Lower bound (default ``0.0``).
        hi: Upper bound (default ``1.0``).

    Returns:
        Clamped value.
    """
    return max(lo, min(hi, value))


def rank_recommendations(
    profile: CandidateProfile,
    candidates: List[Tuple[float, InternshipListing]],
    semantic_weight: float = 0.70,
    keyword_weight: float = 0.30,
    top_n: int = 10,
) -> List[RecommendationResult]:
    """Score, rank, and return the top-N internship recommendations.

    For each ``(semantic_score, InternshipListing)`` pair produced by FAISS
    search:

    1. Compute Jaccard skill overlap.
    2. Hybrid score = ``semantic_weight × semantic_score
       + keyword_weight × skill_overlap``.
    3. Generate a template-based explanation.
    4. Build a :class:`~api.schemas.RecommendationResult`.

    Results are sorted in **descending** order of ``hybrid_score`` and
    truncated to ``top_n``.

    Args:
        profile: The candidate's parsed profile.
        candidates: List of ``(score, InternshipListing)`` tuples from
            :meth:`InternshipIndex.search`.
        semantic_weight: Weight for semantic similarity (default ``0.70``).
        keyword_weight: Weight for skill overlap (default ``0.30``).
        top_n: Maximum number of results to return (default ``10``).

    Returns:
        Sorted list of :class:`~api.schemas.RecommendationResult`, length
        ≤ ``top_n``.  Returns an empty list if ``candidates`` is empty.

    Examples:
        >>> results = rank_recommendations(profile, faiss_results, top_n=5)
        >>> all(0 <= r.match_score <= 1 for r in results)
        True
    """
    if not candidates:
        return []

    scored: list[RecommendationResult] = []

    for raw_semantic, listing in candidates:
        semantic_score = _clamp(raw_semantic)
        skill_overlap = compute_skill_overlap(profile.skills, listing.required_skills)

        hybrid_score = _clamp(
            semantic_weight * semantic_score + keyword_weight * skill_overlap
        )

        explanation = generate_explanation(
            profile, listing, semantic_score, skill_overlap,
        )

        scored.append(
            RecommendationResult(
                internship_id=listing.internship_id,
                title=listing.title,
                company=listing.company,
                location=listing.location,
                required_skills=listing.required_skills,
                match_score=round(hybrid_score, 4),
                skill_overlap_pct=round(skill_overlap, 4),
                explanation=explanation,
            )
        )

    # Sort descending by hybrid score, then by skill overlap as tiebreaker
    scored.sort(key=lambda r: (r.match_score, r.skill_overlap_pct), reverse=True)

    return scored[:top_n]
