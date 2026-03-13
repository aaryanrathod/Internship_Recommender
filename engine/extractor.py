"""
Profile extraction module for the Internship Recommendation Engine.

Converts cleaned resume text into a structured
:class:`~api.schemas.CandidateProfile` by combining:

* **Skill extraction** — spaCy ``PhraseMatcher`` against a JSON taxonomy.
* **Named Entity Recognition** — spaCy NER for person names, organisations,
  and dates.
* **Regex-based parsing** — degree, job-title, and date-range patterns for
  education and experience sections.

The public entry point is :func:`extract_profile`.

Dependencies:
    - ``spacy`` (``en_core_web_trf`` preferred, ``en_core_web_sm`` fallback)
    - ``loguru``
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

from loguru import logger

from api.schemas import CandidateProfile, Education, Experience

# ── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_TAXONOMY_PATH: str = "data/skills_taxonomy.json"
"""Default path to the skills taxonomy JSON file (relative to project root)."""

_SPACY_MODELS: tuple[str, ...] = ("en_core_web_trf", "en_core_web_sm")
"""spaCy model preference order — transformer model first, small model fallback."""

# ── Degree patterns ──────────────────────────────────────────────────────────

_DEGREE_KEYWORDS: list[str] = [
    # Doctorates
    r"Ph\.?D\.?",
    r"Doctor(?:ate)?\s+of\s+\w+",
    # Masters
    r"M\.?S\.?c?\.?",
    r"M\.?A\.?",
    r"M\.?B\.?A\.?",
    r"M\.?Tech\.?",
    r"M\.?E\.?",
    r"M\.?C\.?A\.?",
    r"M\.?Phil\.?",
    r"Master(?:'?s)?\s+(?:of|in)\s+[\w\s]+",
    # Bachelors
    r"B\.?S\.?c?\.?",
    r"B\.?A\.?",
    r"B\.?Tech\.?",
    r"B\.?E\.?",
    r"B\.?C\.?A\.?",
    r"B\.?B\.?A\.?",
    r"B\.?Com\.?",
    r"Bachelor(?:'?s)?\s+(?:of|in)\s+[\w\s]+",
    # Diplomas / Associates
    r"Associate(?:'?s)?\s+(?:of|in)\s+[\w\s]+",
    r"Diploma\s+in\s+[\w\s]+",
    # High school
    r"High\s+School\s+Diploma",
    r"HSC",
    r"SSC",
    r"(?:10|12)th\s+(?:Grade|Standard|Class)",
]

_DEGREE_RE: re.Pattern[str] = re.compile(
    r"(?:^|[\s,;(])"         # boundary
    r"("
    + "|".join(_DEGREE_KEYWORDS) +
    r")"
    r"(?:\s+(?:in|of)\s+([\w\s&,/()-]+))?"  # optional field-of-study
    r"(?:\s*[\s,;)]|$)",
    re.IGNORECASE | re.MULTILINE,
)

# ── Year pattern ─────────────────────────────────────────────────────────────

_YEAR_RE: re.Pattern[str] = re.compile(
    r"\b(19[89]\d|20[0-3]\d)\b"
)

# ── GPA patterns ─────────────────────────────────────────────────────────────

_GPA_RE: re.Pattern[str] = re.compile(
    r"(?:GPA|CGPA|CPI|Grade|Score|Percentage)"
    r"\s*[:;]?\s*"
    r"(\d{1,2}(?:\.\d{1,2})?)\s*(?:/\s*(?:10|100|4(?:\.0)?))?%?",
    re.IGNORECASE,
)

# ── Date range patterns ─────────────────────────────────────────────────────

_MONTH_NAMES: str = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)"
)

_DATE_RANGE_RE: re.Pattern[str] = re.compile(
    r"("
    + _MONTH_NAMES +
    r"\.?\s*\d{4})"
    r"\s*[-–—to]+\s*"
    r"("
    + _MONTH_NAMES +
    r"\.?\s*\d{4}|[Pp]resent|[Cc]urrent|[Oo]ngoing"
    r")",
    re.IGNORECASE,
)

# ── Job title patterns ───────────────────────────────────────────────────────

_JOB_TITLE_KEYWORDS: list[str] = [
    r"(?:Software|ML|AI|Data|Backend|Frontend|Full[- ]?Stack|DevOps|Cloud|"
    r"QA|Test|Research|Product|Mobile|Web|Java|Python|iOS|Android|System|"
    r"Network|Security|Database|BI|Business|Marketing|Sales|Operations|HR|"
    r"Content|Technical|Solutions|Support|Graduate|Summer|Winter|Spring|Fall)"
    r"\s+"
    r"(?:Engineer|Developer|Intern|Analyst|Scientist|Architect|Designer|"
    r"Manager|Consultant|Associate|Fellow|Trainee|Assistant|Coordinator|"
    r"Specialist|Lead|Administrator|Officer|Executive|Strategist|Writer|"
    r"Researcher)",
    r"(?:Intern(?:ship)?|Trainee|Fellow|Apprentice)"
    r"(?:\s*[-–—]\s*\w[\w\s]*)?",
]

_JOB_TITLE_RE: re.Pattern[str] = re.compile(
    r"(" + "|".join(_JOB_TITLE_KEYWORDS) + r")",
    re.IGNORECASE,
)

# ── Certification patterns ───────────────────────────────────────────────────

_CERT_RE: re.Pattern[str] = re.compile(
    r"(?:^|\n)\s*[-•]?\s*"
    r"("
    r"(?:AWS|Azure|Google|GCP|Oracle|Cisco|CompTIA|IBM|Salesforce|"
    r"Microsoft|Meta|Coursera|Udemy|edX|HackerRank|Kaggle|NPTEL|"
    r"Stanford|DeepLearning\.AI|PMP|ITIL|Scrum|SAFe|TOGAF)"
    r"\s+[\w\s:–—-]{3,80}"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


# ── spaCy loader ─────────────────────────────────────────────────────────────


def _load_spacy_model():
    """Load the best available spaCy language model.

    Tries ``en_core_web_trf`` first (transformer-based, higher accuracy),
    then falls back to ``en_core_web_sm``.  Returns ``None`` if neither
    model is installed, logging a warning.

    Returns:
        A loaded spaCy ``Language`` pipeline, or ``None`` on failure.
    """
    import spacy  # lazy import

    for model_name in _SPACY_MODELS:
        try:
            nlp = spacy.load(model_name, disable=["parser", "lemmatizer"])
            logger.info("Loaded spaCy model: {}", model_name)
            return nlp
        except OSError:
            logger.debug("spaCy model '{}' not found, trying next.", model_name)
            continue

    logger.warning(
        "No spaCy model available (tried {}). "
        "Falling back to regex-only extraction.",
        ", ".join(_SPACY_MODELS),
    )
    return None


# Module-level lazy cache
_nlp = None  # type: ignore[assignment]
_nlp_loaded: bool = False


def _get_nlp():
    """Return the cached spaCy model, loading on first call.

    Returns:
        The spaCy ``Language`` object or ``None``.
    """
    global _nlp, _nlp_loaded
    if not _nlp_loaded:
        _nlp = _load_spacy_model()
        _nlp_loaded = True
    return _nlp


# ── Skills extraction ────────────────────────────────────────────────────────


def _load_skills_taxonomy(taxonomy_path: str | None = None) -> list[str]:
    """Load skill strings from the JSON taxonomy file.

    Args:
        taxonomy_path: Absolute or relative path to the JSON file.  Falls
            back to :data:`_DEFAULT_TAXONOMY_PATH` when ``None``.

    Returns:
        List of skill strings.  Returns an empty list if the file is
        missing or invalid, logging a warning.
    """
    path = Path(taxonomy_path or _DEFAULT_TAXONOMY_PATH)
    if not path.exists():
        logger.warning("Skills taxonomy not found at '{}'. Skill extraction will return empty.", path)
        return []

    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            logger.warning("Skills taxonomy at '{}' is not a JSON list.", path)
            return []
        logger.info("Loaded {} skills from taxonomy '{}'.", len(data), path)
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load skills taxonomy '{}': {}", path, exc)
        return []


def extract_skills(text: str, taxonomy_path: str | None = None) -> list[str]:
    """Extract skills from text using spaCy ``PhraseMatcher``.

    Matches are case-insensitive, deduplicated, and returned in title case.
    If spaCy is not available, falls back to simple substring matching.

    Args:
        text: Cleaned resume text.
        taxonomy_path: Optional override for the taxonomy JSON path.

    Returns:
        Deduplicated, title-cased list of matched skills.  May be empty.
    """
    skills_list = _load_skills_taxonomy(taxonomy_path)
    if not skills_list:
        return []

    nlp = _get_nlp()

    if nlp is not None:
        return _extract_skills_spacy(text, skills_list, nlp)
    else:
        return _extract_skills_fallback(text, skills_list)


def _extract_skills_spacy(text: str, skills_list: list[str], nlp) -> list[str]:
    """Extract skills using spaCy PhraseMatcher (primary strategy).

    Args:
        text: Resume text.
        skills_list: Flat list of canonical skill strings.
        nlp: Loaded spaCy Language pipeline.

    Returns:
        Deduplicated, title-cased skill list.
    """
    from spacy.matcher import PhraseMatcher  # lazy import

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = list(nlp.tokenizer.pipe(skills_list))
    matcher.add("SKILLS", patterns)

    doc = nlp(text)
    matched: set[str] = set()

    for _match_id, start, end in matcher(doc):
        span_text = doc[start:end].text
        matched.add(span_text.strip().title())

    logger.debug("PhraseMatcher found {} unique skills.", len(matched))
    return sorted(matched)


def _extract_skills_fallback(text: str, skills_list: list[str]) -> list[str]:
    """Fallback skill extraction via case-insensitive substring search.

    Used when spaCy is not available.

    Args:
        text: Resume text.
        skills_list: Flat list of canonical skill strings.

    Returns:
        Deduplicated, title-cased skill list.
    """
    text_lower = text.lower()
    matched: set[str] = set()

    for skill in skills_list:
        # Word-boundary aware search
        pattern = re.compile(r"\b" + re.escape(skill.lower()) + r"\b")
        if pattern.search(text_lower):
            matched.add(skill.strip().title())

    logger.debug("Fallback matcher found {} unique skills.", len(matched))
    return sorted(matched)


# ── NER helpers ──────────────────────────────────────────────────────────────


def _extract_candidate_name(text: str) -> str:
    """Extract the candidate's name using spaCy NER.

    If spaCy is unavailable or no PERSON entity is found, returns
    ``'Unknown Candidate'``.

    Heuristic: the first PERSON entity in the document is assumed to be the
    candidate's own name (resumes almost always start with the author's name).

    Args:
        text: Resume text.

    Returns:
        Candidate name string.
    """
    nlp = _get_nlp()

    if nlp is not None:
        doc = nlp(text[:3000])  # only scan the top portion for speed
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                if len(name) > 1 and not name.isdigit():
                    logger.debug("Extracted candidate name via NER: '{}'", name)
                    return name

    # Fallback: try the very first non-empty line (common resume format)
    for line in text.splitlines():
        line = line.strip()
        if line and len(line.split()) <= 5 and not re.search(r"\d{5,}", line):
            # Basic check: looks like a name (short, no long numbers)
            if re.fullmatch(r"[A-Z][a-zA-Z.\-' ]+ [A-Z][a-zA-Z.\-' ]+", line):
                logger.debug("Extracted candidate name via heuristic: '{}'", line)
                return line
            break

    logger.debug("Could not determine candidate name; defaulting to 'Unknown Candidate'.")
    return "Unknown Candidate"


def _extract_organisations(text: str) -> list[str]:
    """Extract ORG entities from text using spaCy NER.

    Args:
        text: Resume text.

    Returns:
        List of unique organisation names.  Empty if spaCy unavailable.
    """
    nlp = _get_nlp()
    if nlp is None:
        return []

    doc = nlp(text)
    orgs: set[str] = set()
    for ent in doc.ents:
        if ent.label_ == "ORG":
            org = ent.text.strip()
            if len(org) > 1:
                orgs.add(org)
    return list(orgs)


# ── Education extraction ─────────────────────────────────────────────────────


def extract_education(text: str) -> list[Education]:
    """Extract educational qualifications from resume text.

    Combines regex-based degree detection with NER organisation entities and
    year patterns to construct :class:`~api.schemas.Education` objects.

    Args:
        text: Cleaned resume text.

    Returns:
        List of ``Education`` entries.  May be empty.
    """
    entries: list[Education] = []
    seen_degrees: set[str] = set()

    orgs = _extract_organisations(text)

    for match in _DEGREE_RE.finditer(text):
        degree_raw = match.group(1).strip()
        field_of_study = (match.group(2) or "").strip()

        degree = degree_raw
        if field_of_study:
            degree = f"{degree_raw} in {field_of_study}"

        degree_key = degree.lower()
        if degree_key in seen_degrees:
            continue
        seen_degrees.add(degree_key)

        # Search the surrounding context (±200 chars) for year and institution
        ctx_start = max(0, match.start() - 200)
        ctx_end = min(len(text), match.end() + 200)
        context = text[ctx_start:ctx_end]

        # Year
        year: str | None = None
        year_matches = _YEAR_RE.findall(context)
        if year_matches:
            year = year_matches[-1]  # prefer the latest year

        # GPA
        gpa: str | None = None
        gpa_match = _GPA_RE.search(context)
        if gpa_match:
            gpa = gpa_match.group(1)

        # Institution: find an ORG entity that appears in the context
        institution = "Unknown Institution"
        for org in orgs:
            if org.lower() in context.lower():
                institution = org
                break

        entries.append(Education(
            degree=degree.strip(),
            institution=institution,
            year=year,
            gpa=gpa,
        ))

    logger.debug("Extracted {} education entries.", len(entries))
    return entries


# ── Experience extraction ────────────────────────────────────────────────────


def extract_experience(text: str) -> list[Experience]:
    """Extract work / internship experience from resume text.

    Detects job-title patterns and nearby date ranges to construct
    :class:`~api.schemas.Experience` objects.

    Args:
        text: Cleaned resume text.

    Returns:
        List of ``Experience`` entries.  May be empty.
    """
    entries: list[Experience] = []
    seen: set[str] = set()

    orgs = _extract_organisations(text)

    for match in _JOB_TITLE_RE.finditer(text):
        title = match.group(1).strip()
        title_key = title.lower()
        if title_key in seen:
            continue
        seen.add(title_key)

        # Context window
        ctx_start = max(0, match.start() - 300)
        ctx_end = min(len(text), match.end() + 300)
        context = text[ctx_start:ctx_end]

        # Duration from date range
        duration: str | None = None
        dr_match = _DATE_RANGE_RE.search(context)
        if dr_match:
            duration = f"{dr_match.group(1)} – {dr_match.group(2)}"

        # Company: find nearest ORG in context
        company = "Unknown Company"
        for org in orgs:
            if org.lower() in context.lower():
                company = org
                break

        # Description: grab the sentence(s) following the title
        description = _extract_description(text, match.end())

        entries.append(Experience(
            title=title,
            company=company,
            duration=duration,
            description=description,
        ))

    logger.debug("Extracted {} experience entries.", len(entries))
    return entries


def _extract_description(text: str, start_pos: int, max_chars: int = 500) -> str | None:
    """Extract a short description following a job title match.

    Grabs text from ``start_pos`` up to the next blank line or
    ``max_chars``, whichever comes first.

    Args:
        text: Full resume text.
        start_pos: Character index right after the matched title.
        max_chars: Cap on description length.

    Returns:
        Trimmed description string, or ``None`` if nothing meaningful found.
    """
    snippet = text[start_pos : start_pos + max_chars]
    # Take everything up to a double newline (next section)
    paragraph = snippet.split("\n\n")[0].strip()
    # Clean residual leading punctuation / bullets
    paragraph = re.sub(r"^[\s,;:\-–—•]+", "", paragraph).strip()
    if len(paragraph) < 10:
        return None
    return paragraph


# ── Certification extraction ────────────────────────────────────────────────


def extract_certifications(text: str) -> list[str]:
    """Extract professional certifications from resume text.

    Uses a regex matching known certification provider prefixes followed by
    a description.

    Args:
        text: Cleaned resume text.

    Returns:
        Deduplicated list of certification strings.
    """
    certs: set[str] = set()
    for match in _CERT_RE.finditer(text):
        cert = match.group(1).strip()
        if cert:
            certs.add(cert)
    logger.debug("Extracted {} certifications.", len(certs))
    return sorted(certs)


# ── Summary generation ───────────────────────────────────────────────────────


def _build_summary(
    skills: list[str],
    education: list[Education],
    experience: list[Experience],
) -> str:
    """Build a compact summary string for embedding / semantic matching.

    Format: ``"[skills] [education summary] [experience summary]"``

    Args:
        skills: Extracted skill list.
        education: Extracted education entries.
        experience: Extracted experience entries.

    Returns:
        Concatenated summary string.
    """
    parts: list[str] = []

    if skills:
        parts.append("Skills: " + ", ".join(skills))

    if education:
        edu_summaries = []
        for edu in education:
            s = edu.degree
            if edu.institution and edu.institution != "Unknown Institution":
                s += f" from {edu.institution}"
            if edu.year:
                s += f" ({edu.year})"
            edu_summaries.append(s)
        parts.append("Education: " + "; ".join(edu_summaries))

    if experience:
        exp_summaries = []
        for exp in experience:
            s = exp.title
            if exp.company and exp.company != "Unknown Company":
                s += f" at {exp.company}"
            if exp.duration:
                s += f" ({exp.duration})"
            exp_summaries.append(s)
        parts.append("Experience: " + "; ".join(exp_summaries))

    return " | ".join(parts)


# ── Public API ───────────────────────────────────────────────────────────────


def extract_profile(
    text: str,
    taxonomy_path: str | None = None,
) -> CandidateProfile:
    """Convert cleaned resume text into a structured ``CandidateProfile``.

    This is the main entry point for the extraction module.  It orchestrates
    skill matching, NER-based name detection, and regex-based education /
    experience parsing, then assembles results into a validated Pydantic
    model.

    Args:
        text: Preprocessed resume text (output of
            :func:`engine.preprocessor.preprocess`).
        taxonomy_path: Optional override for the skills taxonomy JSON path.

    Returns:
        A fully populated :class:`~api.schemas.CandidateProfile`.

    Examples:
        >>> from engine.preprocessor import preprocess
        >>> from engine.extractor import extract_profile
        >>> raw = open("resume.txt").read()
        >>> profile = extract_profile(preprocess(raw))
        >>> profile.skills
        ['Machine Learning', 'Python', 'Sql']
    """
    logger.info("Starting profile extraction ({} chars of input).", len(text))

    # ── Extract individual components ─────────────────────────────────────
    candidate_name = _extract_candidate_name(text)
    skills = extract_skills(text, taxonomy_path=taxonomy_path)
    education = extract_education(text)
    experience = extract_experience(text)
    certifications = extract_certifications(text)
    summary_text = _build_summary(skills, education, experience)

    logger.info(
        "Extraction complete — name='{}', skills={}, education={}, "
        "experience={}, certifications={}.",
        candidate_name,
        len(skills),
        len(education),
        len(experience),
        len(certifications),
    )

    return CandidateProfile(
        raw_text=text,
        skills=skills,
        education=education,
        experience=experience,
        certifications=certifications,
        summary_text=summary_text,
    )
