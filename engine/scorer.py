"""
Scoring and ranking module for the Internship Recommendation Engine.

Provides pure, side-effect-free functions that combine **three** scoring
dimensions to produce a ranked list of
:class:`~api.schemas.RecommendationResult` objects:

1. **Semantic similarity** — cosine similarity from FAISS.
2. **TF-IDF weighted skill overlap** — replaces simple Jaccard with an
   IDF-aware score that rewards rare, discriminative skills.
3. **Location match** — scored against candidate location preferences.

No AI models are invoked — explanations are generated via deterministic
template logic.

Public API:
    - :class:`SkillWeighter` — corpus-aware TF-IDF skill scorer.
    - :func:`compute_skill_overlap` — legacy Jaccard (kept for tests).
    - :func:`compute_location_score` — location preference scorer.
    - :func:`generate_explanation` — human-readable match rationale.
    - :func:`rank_recommendations` — hybrid scoring + ranking pipeline.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from api.schemas import (
    CandidateLocationPreference,
    CandidateProfile,
    InternshipListing,
    RecommendationResult,
)


# ── TF-IDF Skill Weighter ────────────────────────────────────────────────────


class SkillWeighter:
    """Corpus-aware skill scorer using IDF weights.

    Skills that appear in many internship listings receive a low IDF weight
    (common, less discriminative), while rare skills receive a high weight.

    The score for a candidate–internship pair is the sum of IDF weights of
    overlapping skills normalised by the total IDF weight of the
    internship's required skills.

    Fuzzy matching is applied: if a candidate skill *contains* an internship
    skill (or vice-versa), it counts as a match.  For example,
    ``"ReactJS"`` matches ``"React"``.

    Args:
        all_internships: The full corpus of internship listings used to
            compute IDF values.
    """

    def __init__(self, all_internships: List[InternshipListing]) -> None:
        self._idf: dict[str, float] = {}
        n = len(all_internships)

        # Document frequency: how many listings contain each skill
        df: dict[str, int] = {}
        for listing in all_internships:
            seen: set[str] = set()
            for skill in listing.required_skills:
                key = skill.strip().lower()
                if key and key not in seen:
                    df[key] = df.get(key, 0) + 1
                    seen.add(key)

        # IDF with smoothing: log((1 + N) / (1 + df)) + 1
        for skill, freq in df.items():
            self._idf[skill] = math.log((1 + n) / (1 + freq)) + 1

    def get_idf(self, skill: str) -> float:
        """Return the IDF weight for a skill, defaulting to a high weight
        for unseen skills (since they are maximally discriminative)."""
        return self._idf.get(skill.strip().lower(), 3.0)

    def _fuzzy_match(
        self, cand_key: str, intern_keys: set[str]
    ) -> str | None:
        """Return the best fuzzy match for *cand_key* in *intern_keys*.

        Checks exact match first, then substring containment in both
        directions.  Returns the matched internship key or ``None``.
        """
        if cand_key in intern_keys:
            return cand_key
        for ik in intern_keys:
            if cand_key in ik or ik in cand_key:
                return ik
        return None

    def score(
        self,
        candidate_skills: List[str],
        internship_skills: List[str],
    ) -> float:
        """Compute IDF-weighted skill overlap score.

        Args:
            candidate_skills: Skills extracted from the candidate's resume.
            internship_skills: Skills required by the internship listing.

        Returns:
            A float in ``[0.0, 1.0]``.  Returns ``0.0`` when the
            internship has no required skills.
        """
        if not internship_skills:
            return 0.0

        intern_keys = {s.strip().lower() for s in internship_skills if s.strip()}
        cand_keys = {s.strip().lower() for s in candidate_skills if s.strip()}

        # Total IDF weight of internship skills (denominator)
        total_idf = sum(self.get_idf(k) for k in intern_keys)
        if total_idf == 0.0:
            return 0.0

        # Matched IDF weight (numerator)
        matched_idf = 0.0
        matched_intern_keys: set[str] = set()

        for ck in cand_keys:
            hit = self._fuzzy_match(ck, intern_keys - matched_intern_keys)
            if hit is not None:
                matched_idf += self.get_idf(hit)
                matched_intern_keys.add(hit)

        return min(matched_idf / total_idf, 1.0)


# ── Legacy Jaccard (kept for backwards compatibility / tests) ────────────────


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


# ── Location scoring ─────────────────────────────────────────────────────────


def compute_location_score(
    preference: Optional[CandidateLocationPreference],
    internship: InternshipListing,
) -> float:
    """Score how well an internship's location matches a candidate's preference.

    Returns the **highest applicable** score from the following rules
    (evaluated in order):

    ====  ======================================================  =====
    Prio  Condition                                                Score
    ====  ======================================================  =====
    0     No preference (None or all fields empty)                  0.50
    1     Internship is Remote & candidate is open_to_remote        0.95
    2     Preferred city matches internship city (contains)         1.00
    3     Preferred location_type matches                           0.80
    4     Preferred country matches internship country              0.75
    5     Internship is Remote but candidate not open_to_remote     0.30
    —     No match                                                  0.10
    ====  ======================================================  =====

    Args:
        preference: The candidate's location preferences, or ``None``.
        internship: The internship listing to score.

    Returns:
        A float in ``[0.0, 1.0]``.
    """
    # No preference → neutral score
    if preference is None:
        return 0.5

    has_any = (
        preference.preferred_city is not None
        or preference.preferred_country is not None
        or preference.preferred_location_type is not None
    )
    if not has_any:
        return 0.5

    scores: list[float] = []

    intern_loc = (internship.location or "").strip().lower()
    intern_country = (internship.country or "").strip().lower()
    intern_type = (internship.location_type or "").strip()

    # Remote handling
    is_remote = intern_type == "Remote"
    if is_remote:
        scores.append(0.95 if preference.open_to_remote else 0.30)

    # City match (substring, case-insensitive)
    if preference.preferred_city:
        pref_city = preference.preferred_city.strip().lower()
        if pref_city and pref_city in intern_loc:
            scores.append(1.0)

    # Location type match
    if preference.preferred_location_type:
        if preference.preferred_location_type == intern_type:
            scores.append(0.80)

    # Country match
    if preference.preferred_country:
        pref_country = preference.preferred_country.strip().lower()
        if pref_country and pref_country == intern_country:
            scores.append(0.75)

    return max(scores) if scores else 0.1


# ── Explanation generator ────────────────────────────────────────────────────


def generate_explanation(
    profile: CandidateProfile,
    internship: InternshipListing,
    semantic_score: float,
    skill_overlap: float,
    location_score: float = 0.5,
    location_preference: Optional[CandidateLocationPreference] = None,
) -> str:
    """Generate a template-based explanation for a recommendation match.

    No AI model is called — the output is fully deterministic.

    Args:
        profile: The candidate's parsed profile.
        internship: The matched internship listing.
        semantic_score: Cosine similarity from FAISS (0–1).
        skill_overlap: TF-IDF skill overlap (0–1).
        location_score: Location match score (0–1).
        location_preference: Candidate's location preferences.

    Returns:
        A 1–3 sentence human-readable explanation string.
    """
    cand_lower = {s.strip().lower(): s.strip() for s in profile.skills}
    intern_lower = {s.strip().lower(): s.strip() for s in internship.required_skills}
    overlapping = [cand_lower[k] for k in cand_lower if k in intern_lower]

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

    # ── Location sentence (only when a preference was supplied) ──────────
    if location_preference is not None:
        loc_type = getattr(internship, "location_type", "Remote")
        if location_score >= 0.95:
            parts.append(f"Location is a strong match ({loc_type}).")
        elif location_score >= 0.75:
            parts.append(f"Location aligns with your preference ({internship.location}).")
        elif location_score <= 0.2:
            parts.append("Location does not match your preference.")

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
    location_preference: Optional[CandidateLocationPreference] = None,
    skill_weighter: Optional[SkillWeighter] = None,
    semantic_weight: float = 0.60,
    keyword_weight: float = 0.25,
    location_weight: float = 0.15,
    top_n: int = 10,
) -> List[RecommendationResult]:
    """Score, rank, and return the top-N internship recommendations.

    For each ``(semantic_score, InternshipListing)`` pair produced by FAISS
    search:

    1. Compute TF-IDF weighted skill overlap (or Jaccard fallback).
    2. Compute location match score.
    3. Hybrid score = ``semantic_weight × semantic + keyword_weight × skill
       + location_weight × location``.
    4. Generate a template-based explanation.
    5. Build a :class:`~api.schemas.RecommendationResult`.

    Results are sorted in **descending** order of ``hybrid_score`` and
    truncated to ``top_n``.

    Args:
        profile: The candidate's parsed profile.
        candidates: List of ``(score, InternshipListing)`` tuples from
            :meth:`InternshipIndex.search`.
        location_preference: Optional candidate location preferences.
        skill_weighter: Optional :class:`SkillWeighter` for TF-IDF scoring.
            Falls back to Jaccard if ``None``.
        semantic_weight: Weight for semantic similarity (default ``0.60``).
        keyword_weight: Weight for skill overlap (default ``0.25``).
        location_weight: Weight for location match (default ``0.15``).
        top_n: Maximum number of results to return (default ``10``).

    Returns:
        Sorted list of :class:`~api.schemas.RecommendationResult`, length
        ≤ ``top_n``.  Returns an empty list if ``candidates`` is empty.
    """
    if not candidates:
        return []

    scored: list[RecommendationResult] = []

    for raw_semantic, listing in candidates:
        semantic_score = _clamp(raw_semantic)

        # Skill score: TF-IDF if a weighter is provided, else Jaccard
        if skill_weighter is not None:
            skill_score = skill_weighter.score(profile.skills, listing.required_skills)
        else:
            skill_score = compute_skill_overlap(profile.skills, listing.required_skills)

        # Location score
        loc_score = compute_location_score(location_preference, listing)

        # Hybrid formula
        hybrid_score = _clamp(
            semantic_weight * semantic_score
            + keyword_weight * skill_score
            + location_weight * loc_score
        )

        explanation = generate_explanation(
            profile,
            listing,
            semantic_score,
            skill_score,
            loc_score,
            location_preference,
        )

        scored.append(
            RecommendationResult(
                internship_id=listing.internship_id,
                title=listing.title,
                company=listing.company,
                location=listing.location,
                location_type=getattr(listing, "location_type", "Remote"),
                location_score=round(loc_score, 4),
                required_skills=listing.required_skills,
                match_score=round(hybrid_score, 4),
                skill_overlap_pct=round(skill_score, 4),
                explanation=explanation,
                domain=getattr(listing, "domain", ""),
                stipend_usd=getattr(listing, "stipend_usd", None),
            )
        )

    # Sort descending by hybrid score, then by skill overlap as tiebreaker
    scored.sort(key=lambda r: (r.match_score, r.skill_overlap_pct), reverse=True)

    return scored[:top_n]
