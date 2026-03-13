"""
Integration test for the full recommendation pipeline.

Exercises the complete flow end-to-end using:
    - A synthetic 200-word resume with known skills
    - 10 synthetic InternshipListing objects
    - An in-memory FAISS index (not persisted to disk)

Validates that the output is a valid ``RecommendationResponse`` with
correctly sorted, scored, and explained results.
"""

from __future__ import annotations

from typing import List

import pytest

from api.schemas import (
    CandidateProfile,
    InternshipListing,
    RecommendationResponse,
    RecommendationResult,
)


# ── Synthetic data ───────────────────────────────────────────────────────────

SYNTHETIC_RESUME: str = (
    "Arjun Mehta\n"
    "Email: arjun.mehta@email.com | Phone: +91-9876543210\n"
    "Mumbai, Maharashtra, India\n\n"
    "Summary\n"
    "Motivated Computer Science undergraduate with strong foundations in "
    "Python, Machine Learning, and Data Analysis. Hands-on experience "
    "building end-to-end ML pipelines using TensorFlow, Scikit-learn, and "
    "Pandas. Passionate about applying Natural Language Processing techniques "
    "to solve real-world problems. Quick learner with strong communication "
    "and teamwork skills.\n\n"
    "Education\n"
    "B.Tech in Computer Science and Engineering\n"
    "Indian Institute of Technology, Bombay — 2021 – 2025\n"
    "CGPA: 8.7/10\n\n"
    "Experience\n"
    "Machine Learning Intern — DataWorks Analytics\n"
    "May 2024 – August 2024\n"
    "Developed a text classification pipeline using TensorFlow and spaCy for "
    "customer feedback analysis. Achieved 92% F1-score on production dataset. "
    "Collaborated with a cross-functional team of five engineers.\n\n"
    "Data Science Intern — InsightHub\n"
    "June 2023 – August 2023\n"
    "Performed exploratory data analysis on e-commerce datasets using Pandas "
    "and Matplotlib. Built interactive dashboards with Plotly. Automated "
    "weekly reporting with Python scripts.\n\n"
    "Skills\n"
    "Python, SQL, TensorFlow, Scikit-learn, Pandas, NumPy, Matplotlib, "
    "spaCy, NLP, Machine Learning, Deep Learning, Data Analysis, Git, "
    "Docker, REST API, Communication, Problem Solving\n\n"
    "Certifications\n"
    "AWS Cloud Practitioner\n"
    "DeepLearning.AI TensorFlow Developer Certificate\n"
)

SYNTHETIC_LISTINGS: List[dict] = [
    {
        "internship_id": "INT-001",
        "title": "ML Engineering Intern",
        "company": "AlphaAI",
        "location": "Bangalore",
        "required_skills": ["Python", "TensorFlow", "Machine Learning", "NLP"],
        "description": "Work on NLP models for text classification and entity extraction.",
    },
    {
        "internship_id": "INT-002",
        "title": "Data Science Intern",
        "company": "BetaAnalytics",
        "location": "Mumbai",
        "required_skills": ["Python", "Pandas", "SQL", "Data Analysis"],
        "description": "Analyse business datasets and build predictive models.",
    },
    {
        "internship_id": "INT-003",
        "title": "Backend Engineering Intern",
        "company": "GammaWeb",
        "location": "Hyderabad",
        "required_skills": ["Java", "Spring Boot", "PostgreSQL", "REST API"],
        "description": "Build scalable backend services for a SaaS platform.",
    },
    {
        "internship_id": "INT-004",
        "title": "Deep Learning Research Intern",
        "company": "DeltaLabs",
        "location": "Remote",
        "required_skills": ["Python", "TensorFlow", "Deep Learning", "Computer Vision"],
        "description": "Research novel architectures for image classification tasks.",
    },
    {
        "internship_id": "INT-005",
        "title": "Full Stack Developer Intern",
        "company": "EpsilonTech",
        "location": "Pune",
        "required_skills": ["React", "Node.js", "MongoDB", "JavaScript"],
        "description": "Develop features for a customer-facing web application.",
    },
    {
        "internship_id": "INT-006",
        "title": "NLP Intern",
        "company": "ZetaNLP",
        "location": "Bangalore",
        "required_skills": ["Python", "spaCy", "NLP", "Transformers"],
        "description": "Build and evaluate NLP pipelines for information extraction.",
    },
    {
        "internship_id": "INT-007",
        "title": "DevOps Intern",
        "company": "EtaOps",
        "location": "Chennai",
        "required_skills": ["Docker", "Kubernetes", "AWS", "Linux"],
        "description": "Automate CI/CD pipelines and manage cloud infrastructure.",
    },
    {
        "internship_id": "INT-008",
        "title": "Data Engineering Intern",
        "company": "ThetaData",
        "location": "Delhi",
        "required_skills": ["Python", "SQL", "Apache Spark", "ETL"],
        "description": "Design and maintain ETL pipelines for large-scale datasets.",
    },
    {
        "internship_id": "INT-009",
        "title": "AI Product Intern",
        "company": "IotaProducts",
        "location": "Remote",
        "required_skills": ["Python", "Machine Learning", "Communication", "Agile"],
        "description": "Translate ML research into product features working with PM team.",
    },
    {
        "internship_id": "INT-010",
        "title": "Cybersecurity Intern",
        "company": "KappaSec",
        "location": "Noida",
        "required_skills": ["Network Security", "Penetration Testing", "Linux", "Cryptography"],
        "description": "Conduct security audits and vulnerability assessments.",
    },
]


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def internship_listings() -> List[InternshipListing]:
    """Validated InternshipListing objects from synthetic data."""
    return [InternshipListing(**row) for row in SYNTHETIC_LISTINGS]


@pytest.fixture(scope="module")
def embedder():
    """Module-scoped Embedder singleton (loads model once for all tests)."""
    from engine.embedder import Embedder

    return Embedder(model_name="all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def faiss_index(internship_listings, embedder):
    """In-memory FAISS index built from synthetic listings (not saved to disk)."""
    from engine.indexer import InternshipIndex

    index = InternshipIndex()
    index.build(internship_listings, embedder)
    return index


@pytest.fixture(scope="module")
def candidate_profile() -> CandidateProfile:
    """Profile extracted from the synthetic resume via the preprocessor + extractor."""
    from engine.extractor import extract_profile
    from engine.preprocessor import preprocess

    cleaned = preprocess(SYNTHETIC_RESUME)
    return extract_profile(cleaned)


# ── End-to-end test ──────────────────────────────────────────────────────────


class TestFullPipeline:
    """End-to-end integration test for the recommendation pipeline."""

    def test_pipeline_produces_valid_response(
        self,
        candidate_profile: CandidateProfile,
        embedder,
        faiss_index,
    ) -> None:
        """Full pipeline must return a valid RecommendationResponse with
        sorted, scored, and explained results."""
        from engine.scorer import rank_recommendations

        # ── Embed candidate ──────────────────────────────────────────────
        query_vec = embedder.embed_profile(candidate_profile)
        assert query_vec.ndim == 1
        assert query_vec.shape[0] == embedder.embedding_dim

        # ── FAISS search ─────────────────────────────────────────────────
        top_k_search = 20
        faiss_results = faiss_index.search(query_vec, top_k=top_k_search)
        assert len(faiss_results) > 0
        assert len(faiss_results) <= top_k_search

        # ── Rank ─────────────────────────────────────────────────────────
        top_n = 5
        results: List[RecommendationResult] = rank_recommendations(
            profile=candidate_profile,
            candidates=faiss_results,
            semantic_weight=0.70,
            keyword_weight=0.30,
            top_n=top_n,
        )

        # ── Build response ───────────────────────────────────────────────
        response = RecommendationResponse(
            candidate_name="Arjun Mehta",
            total_results=len(results),
            results=results,
        )

        # ── Assertions ───────────────────────────────────────────────────

        # Type
        assert isinstance(response, RecommendationResponse)

        # Count
        assert response.total_results == len(results)
        assert response.total_results <= top_n
        assert response.total_results > 0

        # Sorted descending by match_score
        scores = [r.match_score for r in response.results]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted descending: {scores}"
        )

        # Top result has a positive score
        assert response.results[0].match_score > 0.0

        # All scores in [0, 1]
        for r in response.results:
            assert 0.0 <= r.match_score <= 1.0
            assert 0.0 <= r.skill_overlap_pct <= 1.0

        # All explanations are non-empty strings
        for r in response.results:
            assert isinstance(r.explanation, str)
            assert len(r.explanation) > 0, (
                f"Empty explanation for {r.internship_id}"
            )

        # All required fields populated
        for r in response.results:
            assert r.internship_id
            assert r.title
            assert r.company

    def test_top_results_are_ml_related(
        self,
        candidate_profile: CandidateProfile,
        embedder,
        faiss_index,
    ) -> None:
        """Given an ML-heavy resume, the top results should lean towards
        ML/Data/NLP roles rather than unrelated ones like Cybersecurity."""
        from engine.scorer import rank_recommendations

        query_vec = embedder.embed_profile(candidate_profile)
        faiss_results = faiss_index.search(query_vec, top_k=10)

        results = rank_recommendations(
            profile=candidate_profile,
            candidates=faiss_results,
            top_n=3,
        )

        top_titles = {r.title for r in results}
        # The cybersecurity and full-stack roles should NOT dominate the top 3
        ml_related_keywords = {"ML", "Data", "NLP", "Deep Learning", "AI"}
        has_relevant = any(
            any(kw.lower() in t.lower() for kw in ml_related_keywords)
            for t in top_titles
        )
        assert has_relevant, (
            f"Expected ML-related roles in top 3, got: {top_titles}"
        )

    def test_candidate_profile_has_extracted_skills(
        self,
        candidate_profile: CandidateProfile,
    ) -> None:
        """The extracted profile should contain known skills from the resume."""
        skills_lower = {s.lower() for s in candidate_profile.skills}

        # These skills are explicitly listed in the synthetic resume
        expected = {"python", "sql", "machine learning"}
        found = expected & skills_lower
        assert len(found) >= 2, (
            f"Expected at least 2 of {expected} in skills, got: {skills_lower}"
        )

    def test_profile_summary_is_nonempty(
        self,
        candidate_profile: CandidateProfile,
    ) -> None:
        """summary_text must be populated for embedder to work correctly."""
        assert isinstance(candidate_profile.summary_text, str)
        assert len(candidate_profile.summary_text) > 10
