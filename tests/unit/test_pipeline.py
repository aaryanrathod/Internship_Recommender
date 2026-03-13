"""
Unit tests for individual engine modules.

Covers:
    - ``engine.parser``       — PDF extraction mock, unsupported extension error
    - ``engine.preprocessor`` — HTML stripping, chunk_text sizing
    - ``engine.extractor``    — skill extraction from hardcoded text, empty input
    - ``engine.scorer``       — Jaccard overlap, ranking order, score clamping
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import pytest

from api.schemas import (
    CandidateProfile,
    Education,
    Experience,
    InternshipListing,
    RecommendationResult,
)


# ============================================================================
#  Parser tests
# ============================================================================


class TestParser:
    """Tests for ``engine.parser``."""

    def test_parse_resume_raises_for_unsupported_extension(self, tmp_path: Path) -> None:
        """ResumeParseError must be raised for an unsupported file format."""
        from engine.parser import ResumeParseError, parse_resume

        unsupported = tmp_path / "resume.xlsx"
        unsupported.write_text("fake data")

        with pytest.raises(ResumeParseError, match="Unsupported file format"):
            parse_resume(str(unsupported))

    def test_parse_resume_raises_for_missing_file(self) -> None:
        """ResumeParseError must be raised when file does not exist."""
        from engine.parser import ResumeParseError, parse_resume

        with pytest.raises(ResumeParseError, match="File not found"):
            parse_resume("/nonexistent/resume.pdf")

    def test_parse_resume_txt_returns_content(self, tmp_path: Path) -> None:
        """A valid .txt file should return its text content."""
        from engine.parser import parse_resume

        txt_file = tmp_path / "resume.txt"
        expected_text = "Jane Doe\nPython Developer with 3 years of experience."
        txt_file.write_text(expected_text, encoding="utf-8")

        result = parse_resume(str(txt_file))
        assert len(result) > 0
        assert "Jane Doe" in result
        assert "Python" in result

    def test_pdf_extraction_returns_nonempty_string(self, tmp_path: Path) -> None:
        """Mocked pdfplumber should produce non-empty text.

        ``pdfplumber`` is imported lazily *inside* ``_extract_pdf``, so it is
        never a module-level attribute of ``engine.parser``.  The correct
        approach is to inject a mock into ``sys.modules["pdfplumber"]`` so
        that the ``import pdfplumber`` call inside the function resolves to
        our mock, then configure ``mock_pdfplumber.open`` as the context
        manager.
        """
        import sys
        from unittest.mock import MagicMock, patch

        from engine.parser import _extract_pdf

        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        # Build the mock page
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Experienced Python developer"

        # Build the mock PDF context-manager object
        mock_pdf_cm = MagicMock()
        mock_pdf_cm.__enter__ = MagicMock(return_value=mock_pdf_cm)
        mock_pdf_cm.__exit__ = MagicMock(return_value=False)
        mock_pdf_cm.pages = [mock_page]

        # Build the mock pdfplumber module
        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.return_value = mock_pdf_cm

        # Inject into sys.modules so the lazy ``import pdfplumber`` inside
        # _extract_pdf retrieves our mock instead of the real package.
        with patch.dict(sys.modules, {"pdfplumber": mock_pdfplumber}):
            result = _extract_pdf(pdf_path)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Python" in result

    def test_parse_resume_raises_for_empty_txt(self, tmp_path: Path) -> None:
        """An empty .txt file must raise ResumeParseError."""
        from engine.parser import ResumeParseError, parse_resume

        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        with pytest.raises(ResumeParseError, match="empty"):
            parse_resume(str(empty_file))


# ============================================================================
#  Preprocessor tests
# ============================================================================


class TestPreprocessor:
    """Tests for ``engine.preprocessor``."""

    def test_html_tags_are_stripped(self) -> None:
        """All HTML tags must be removed from the output."""
        from engine.preprocessor import preprocess

        html_text = "<h1>Jane Doe</h1><p>Python <b>developer</b></p>"
        result = preprocess(html_text)

        assert "<h1>" not in result
        assert "<p>" not in result
        assert "<b>" not in result
        assert "Jane Doe" in result
        assert "Python" in result
        assert "developer" in result

    def test_markdown_artefacts_are_removed(self) -> None:
        """Markdown links, bold, headings should be cleaned."""
        from engine.preprocessor import preprocess

        md_text = "## Skills\n**Python** and [TensorFlow](https://tf.org)"
        result = preprocess(md_text)

        assert "##" not in result
        assert "**" not in result
        assert "](https" not in result
        assert "Python" in result
        assert "TensorFlow" in result

    def test_ligatures_are_fixed(self) -> None:
        """OCR ligature characters must be expanded to ASCII."""
        from engine.preprocessor import preprocess

        ligature_text = "pro\ufb01cient in o\ufb03ce tools"
        result = preprocess(ligature_text)

        assert "proficient" in result
        assert "office" in result

    def test_decorative_lines_removed(self) -> None:
        """Lines like '--------' and '========' must be stripped."""
        from engine.preprocessor import preprocess

        text = "Section A\n--------\nContent here\n========\nSection B"
        result = preprocess(text)

        assert "--------" not in result
        assert "========" not in result
        assert "Section A" in result
        assert "Content here" in result

    @pytest.mark.parametrize(
        "word_count, max_tokens, expected_min_chunks",
        [
            (1000, 256, 4),   # 1000 words, max ~192 words/chunk → ≥4 chunks
            (100, 256, 1),    # fits in one chunk
            (500, 128, 4),    # 500 words, max ~96 words/chunk → ≥4 chunks
        ],
        ids=["1000w_256t", "100w_256t", "500w_128t"],
    )
    def test_chunk_text_returns_correct_chunk_count(
        self,
        word_count: int,
        max_tokens: int,
        expected_min_chunks: int,
    ) -> None:
        """chunk_text must produce at least the expected number of chunks."""
        from engine.preprocessor import chunk_text

        text = " ".join(f"word{i}" for i in range(word_count))
        chunks = chunk_text(text, max_tokens=max_tokens, overlap=32)

        assert len(chunks) >= expected_min_chunks
        # Every chunk's word count must not exceed the token-to-word limit
        max_words = int(max_tokens * 0.75)
        for chunk in chunks:
            assert len(chunk.split()) <= max_words

    def test_chunk_text_empty_input_returns_empty_list(self) -> None:
        """Empty or whitespace-only input should return an empty list."""
        from engine.preprocessor import chunk_text

        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_chunk_text_raises_on_bad_overlap(self) -> None:
        """overlap >= max_tokens should raise ValueError."""
        from engine.preprocessor import chunk_text

        with pytest.raises(ValueError, match="overlap"):
            chunk_text("hello world", max_tokens=10, overlap=10)

    def test_detect_language_english(self) -> None:
        """English text should return 'en'."""
        from engine.preprocessor import detect_language

        result = detect_language(
            "I am a software engineer with five years of experience in Python."
        )
        assert result == "en"

    def test_detect_language_empty_returns_unknown(self) -> None:
        """Empty input should return 'unknown' without raising."""
        from engine.preprocessor import detect_language

        assert detect_language("") == "unknown"


# ============================================================================
#  Extractor tests
# ============================================================================


class TestExtractor:
    """Tests for ``engine.extractor``."""

    SAMPLE_TEXT: str = (
        "Jane Doe is a software developer proficient in Python and SQL. "
        "She has experience with TensorFlow and Machine Learning. "
        "She completed her B.Tech in Computer Science from IIT Delhi in 2024 "
        "with a CGPA of 9.1. She worked as an ML Intern at Google from "
        "May 2023 to August 2023, building recommendation systems."
    )

    def test_known_skills_are_extracted(self, tmp_path: Path) -> None:
        """Skills present in text AND taxonomy should be returned."""
        from engine.extractor import extract_skills

        # Create a mini taxonomy
        import json

        taxonomy_file = tmp_path / "skills.json"
        taxonomy_file.write_text(
            json.dumps(["Python", "SQL", "TensorFlow", "Machine Learning", "Java"]),
            encoding="utf-8",
        )

        skills = extract_skills(self.SAMPLE_TEXT, taxonomy_path=str(taxonomy_file))

        # Python, SQL, TensorFlow, Machine Learning should all be found
        skills_lower = {s.lower() for s in skills}
        assert "python" in skills_lower
        assert "sql" in skills_lower
        # Java is NOT in the text
        assert "java" not in skills_lower

    def test_empty_text_returns_empty_skills(self, tmp_path: Path) -> None:
        """Empty input must return an empty skill list without raising."""
        from engine.extractor import extract_skills

        import json

        taxonomy_file = tmp_path / "skills.json"
        taxonomy_file.write_text(json.dumps(["Python", "SQL"]), encoding="utf-8")

        skills = extract_skills("", taxonomy_path=str(taxonomy_file))
        assert skills == []

    def test_missing_taxonomy_returns_empty_skills(self) -> None:
        """A non-existent taxonomy file should return empty, not raise."""
        from engine.extractor import extract_skills

        skills = extract_skills(
            "Python developer", taxonomy_path="/nonexistent/skills.json"
        )
        assert skills == []

    def test_extract_profile_returns_candidate_profile(self, tmp_path: Path) -> None:
        """extract_profile must return a valid CandidateProfile."""
        from engine.extractor import extract_profile

        import json

        taxonomy_file = tmp_path / "skills.json"
        taxonomy_file.write_text(
            json.dumps(["Python", "SQL", "TensorFlow", "Machine Learning"]),
            encoding="utf-8",
        )

        profile = extract_profile(self.SAMPLE_TEXT, taxonomy_path=str(taxonomy_file))

        assert isinstance(profile, CandidateProfile)
        assert profile.raw_text == self.SAMPLE_TEXT
        assert isinstance(profile.skills, list)
        assert isinstance(profile.education, list)
        assert isinstance(profile.experience, list)
        assert isinstance(profile.summary_text, str)


# ============================================================================
#  Scorer tests
# ============================================================================


class TestScorer:
    """Tests for ``engine.scorer``."""

    # ── Fixtures ─────────────────────────────────────────────────────────

    @pytest.fixture()
    def sample_profile(self) -> CandidateProfile:
        """A minimal CandidateProfile for scoring tests."""
        return CandidateProfile(
            raw_text="Sample resume text",
            skills=["Python", "SQL", "TensorFlow", "Docker"],
            education=[],
            experience=[],
            certifications=[],
            summary_text="Python SQL TensorFlow Docker developer",
        )

    @pytest.fixture()
    def sample_listings(self) -> List[InternshipListing]:
        """Three internship listings with varying skill overlap."""
        return [
            InternshipListing(
                internship_id="INT-001",
                title="ML Intern",
                company="AlphaCorp",
                required_skills=["Python", "TensorFlow", "SQL"],
                description="ML internship",
            ),
            InternshipListing(
                internship_id="INT-002",
                title="Backend Intern",
                company="BetaInc",
                required_skills=["Java", "Spring Boot", "SQL"],
                description="Backend internship",
            ),
            InternshipListing(
                internship_id="INT-003",
                title="DevOps Intern",
                company="GammaTech",
                required_skills=["Docker", "Kubernetes", "AWS"],
                description="DevOps internship",
            ),
        ]

    # ── compute_skill_overlap ────────────────────────────────────────────

    @pytest.mark.parametrize(
        "candidate, internship, expected",
        [
            (["Python", "SQL"], ["Python", "SQL"], 1.0),
            (["Python", "SQL"], ["python", "java"], 1 / 3),
            (["Python"], ["Java"], 0.0),
            ([], [], 0.0),
            (["Python", "SQL", "Docker"], ["python", "sql"], 2 / 3),
        ],
        ids=[
            "perfect_overlap",
            "partial_overlap_case_insensitive",
            "no_overlap",
            "both_empty",
            "superset_candidate",
        ],
    )
    def test_compute_skill_overlap(
        self,
        candidate: List[str],
        internship: List[str],
        expected: float,
    ) -> None:
        """Jaccard similarity must match the expected value."""
        from engine.scorer import compute_skill_overlap

        result = compute_skill_overlap(candidate, internship)
        assert math.isclose(result, expected, rel_tol=1e-9)

    # ── rank_recommendations ─────────────────────────────────────────────

    def test_rank_returns_sorted_descending(
        self,
        sample_profile: CandidateProfile,
        sample_listings: List[InternshipListing],
    ) -> None:
        """Results must be sorted by match_score in descending order."""
        from engine.scorer import rank_recommendations

        # Simulate FAISS results with varied semantic scores
        candidates: List[Tuple[float, InternshipListing]] = [
            (0.50, sample_listings[1]),  # low semantic
            (0.90, sample_listings[0]),  # high semantic
            (0.70, sample_listings[2]),  # medium semantic
        ]

        results = rank_recommendations(
            profile=sample_profile,
            candidates=candidates,
            top_n=3,
        )

        assert len(results) == 3
        scores = [r.match_score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted descending: {scores}"
        )

    def test_rank_returns_empty_for_empty_candidates(
        self,
        sample_profile: CandidateProfile,
    ) -> None:
        """Empty candidates list must return an empty result list."""
        from engine.scorer import rank_recommendations

        results = rank_recommendations(profile=sample_profile, candidates=[])
        assert results == []

    def test_rank_respects_top_n(
        self,
        sample_profile: CandidateProfile,
        sample_listings: List[InternshipListing],
    ) -> None:
        """Result count must not exceed top_n."""
        from engine.scorer import rank_recommendations

        candidates = [(0.8, listing) for listing in sample_listings]
        results = rank_recommendations(
            profile=sample_profile,
            candidates=candidates,
            top_n=2,
        )
        assert len(results) == 2

    def test_scores_clamped_to_unit_interval(
        self,
        sample_profile: CandidateProfile,
        sample_listings: List[InternshipListing],
    ) -> None:
        """All match_score and skill_overlap_pct values must be in [0, 1]."""
        from engine.scorer import rank_recommendations

        # Intentionally pass an out-of-range semantic score
        candidates = [
            (1.5, sample_listings[0]),   # above 1
            (-0.2, sample_listings[1]),  # below 0
        ]

        results = rank_recommendations(
            profile=sample_profile,
            candidates=candidates,
            top_n=2,
        )

        for r in results:
            assert 0.0 <= r.match_score <= 1.0, (
                f"match_score {r.match_score} not in [0, 1]"
            )
            assert 0.0 <= r.skill_overlap_pct <= 1.0, (
                f"skill_overlap_pct {r.skill_overlap_pct} not in [0, 1]"
            )

    def test_explanation_is_nonempty_string(
        self,
        sample_profile: CandidateProfile,
        sample_listings: List[InternshipListing],
    ) -> None:
        """Every recommendation must have a non-empty explanation."""
        from engine.scorer import rank_recommendations

        candidates = [(0.85, sample_listings[0])]
        results = rank_recommendations(
            profile=sample_profile, candidates=candidates, top_n=1,
        )

        assert len(results) == 1
        assert isinstance(results[0].explanation, str)
        assert len(results[0].explanation) > 10

    # ── generate_explanation ─────────────────────────────────────────────

    def test_generate_explanation_mentions_skills(
        self,
        sample_profile: CandidateProfile,
        sample_listings: List[InternshipListing],
    ) -> None:
        """Explanation should reference overlapping skills when present."""
        from engine.scorer import generate_explanation

        explanation = generate_explanation(
            profile=sample_profile,
            internship=sample_listings[0],  # Python, TensorFlow, SQL overlap
            semantic_score=0.87,
            skill_overlap=0.55,
        )

        assert isinstance(explanation, str)
        assert len(explanation) > 0
        # Should mention at least one overlapping skill
        has_skill = any(
            skill.lower() in explanation.lower()
            for skill in ["Python", "SQL", "TensorFlow"]
        )
        assert has_skill, f"Explanation missing skill names: '{explanation}'"

    def test_generate_explanation_no_overlap(
        self,
        sample_profile: CandidateProfile,
    ) -> None:
        """Explanation should gracefully handle zero skill overlap."""
        from engine.scorer import generate_explanation

        listing = InternshipListing(
            internship_id="INT-X",
            title="Intern",
            company="TestCo",
            required_skills=["Rust", "Haskell"],
            description="Functional programming intern",
        )

        explanation = generate_explanation(
            profile=sample_profile,
            internship=listing,
            semantic_score=0.30,
            skill_overlap=0.0,
        )

        assert isinstance(explanation, str)
        assert len(explanation) > 0
