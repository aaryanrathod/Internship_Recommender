"""
Pydantic v2 request/response schemas for the Internship Recommendation Engine API.

These models serve as the single source of truth for data validation and
serialization across every layer of the application — from resume parsing
through FAISS retrieval to the final JSON response.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── Candidate Sub-Models ─────────────────────────────────────────────────────


class Education(BaseModel):
    """A single educational qualification extracted from a candidate's resume.

    Attributes:
        degree: Name of the degree (e.g. 'B.Tech in Computer Science').
        institution: Name of the university or college.
        year: Graduation year or expected graduation year, if available.
        gpa: Grade point average or percentage, if available.
    """

    degree: str = Field(description="Degree title (e.g. 'B.Tech in Computer Science').")
    institution: str = Field(description="Name of the educational institution.")
    year: str | None = Field(default=None, description="Graduation or expected graduation year.")
    gpa: str | None = Field(default=None, description="GPA or percentage score, if mentioned.")


class Experience(BaseModel):
    """A single work or internship experience entry from a candidate's resume.

    Attributes:
        title: Job or internship title.
        company: Name of the employer or organisation.
        duration: Human-readable duration string (e.g. 'Jun 2024 – Aug 2024').
        description: Summary of responsibilities and achievements.
    """

    title: str = Field(description="Job or internship title.")
    company: str = Field(description="Employer or organisation name.")
    duration: str | None = Field(default=None, description="Employment duration (e.g. 'Jun 2024 – Aug 2024').")
    description: str | None = Field(default=None, description="Summary of duties and accomplishments.")


# ── Candidate Profile ────────────────────────────────────────────────────────


class CandidateProfile(BaseModel):
    """Structured representation of a candidate parsed from their resume.

    This is the primary input to the recommendation engine after the resume
    has been processed by the NLP parsing pipeline.

    Attributes:
        raw_text: The full extracted text of the resume before structuring.
        skills: Deduplicated list of technical and soft skills identified.
        education: List of educational qualifications.
        experience: List of work/internship experiences.
        certifications: List of certifications or professional credentials.
        summary_text: A concise, generated summary used for embedding and
            semantic matching.
    """

    raw_text: str = Field(description="Full raw text extracted from the resume.")
    skills: List[str] = Field(default_factory=list, description="Extracted skill keywords.")
    education: List[Education] = Field(default_factory=list, description="Educational qualifications.")
    experience: List[Experience] = Field(default_factory=list, description="Work and internship experiences.")
    certifications: List[str] = Field(default_factory=list, description="Professional certifications.")
    summary_text: str = Field(default="", description="Generated summary used for semantic matching.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "raw_text": "Jane Doe — Computer Science undergraduate...",
                    "skills": ["Python", "Machine Learning", "SQL"],
                    "education": [
                        {
                            "degree": "B.Tech in Computer Science",
                            "institution": "IIT Delhi",
                            "year": "2025",
                            "gpa": "8.9",
                        }
                    ],
                    "experience": [
                        {
                            "title": "ML Intern",
                            "company": "Acme Corp",
                            "duration": "May 2024 – Jul 2024",
                            "description": "Built an NLP pipeline for resume parsing.",
                        }
                    ],
                    "certifications": ["AWS Cloud Practitioner"],
                    "summary_text": "CS undergrad skilled in Python and ML with internship experience.",
                }
            ]
        }
    }


# ── Location Preference ──────────────────────────────────────────────────────


class CandidateLocationPreference(BaseModel):
    """Location preferences for a candidate, used as a third scoring dimension.

    Attributes:
        preferred_city: Desired city (e.g. 'San Francisco, CA').
        preferred_country: Desired country (e.g. 'United States').
        preferred_location_type: One of Remote / Hybrid / On-site.
        open_to_remote: Whether the candidate accepts remote roles. Defaults
            to ``True``, which gives Remote listings a baseline location
            score even without a city/country match.
    """

    preferred_city: Optional[str] = Field(default=None, description="Preferred city (e.g. 'San Francisco, CA').")
    preferred_country: Optional[str] = Field(default=None, description="Preferred country (e.g. 'United States').")
    preferred_location_type: Optional[Literal["Remote", "Hybrid", "On-site"]] = Field(
        default=None, description="Preferred working arrangement."
    )
    open_to_remote: bool = Field(default=True, description="Accept remote roles regardless of city/country.")


# ── Internship Listing ───────────────────────────────────────────────────────


class InternshipListing(BaseModel):
    """Schema representing a single internship opportunity in the dataset.

    Attributes:
        internship_id: Unique identifier for the listing.
        title: Title of the internship position.
        company: Hiring company or organisation.
        location: City / region (e.g. 'San Francisco, CA').
        country: Full country name (e.g. 'United States').
        location_type: Working arrangement — Remote, Hybrid, or On-site.
        required_skills: Skills explicitly required in the listing.
        preferred_skills: Nice-to-have skills that are not mandatory.
        description: Full text of the internship description.
        domain: Functional domain (e.g. Machine Learning, DevOps).
        duration_months: Expected internship duration in months.
        stipend_usd: Monthly stipend in USD (0 = unpaid).
        experience_level: One of Beginner, Intermediate, or Advanced.
        stipend: Legacy stipend string (kept for backwards compatibility).
        duration: Legacy duration string (kept for backwards compatibility).
    """

    internship_id: str = Field(description="Unique listing identifier.")
    title: str = Field(description="Internship position title.")
    company: str = Field(description="Hiring company name.")
    location: str = Field(default="Remote", description="City / region (e.g. 'San Francisco, CA').")
    country: str = Field(default="", description="Full country name.")
    location_type: Literal["Remote", "Hybrid", "On-site"] = Field(
        default="Remote", description="Working arrangement."
    )
    required_skills: List[str] = Field(default_factory=list, description="Skills required for the role.")
    preferred_skills: List[str] = Field(default_factory=list, description="Nice-to-have skills.")
    description: str = Field(default="", description="Full internship description text.")
    domain: str = Field(default="", description="Functional domain (e.g. 'Machine Learning').")
    duration_months: Optional[int] = Field(default=None, description="Duration in months.")
    stipend_usd: Optional[int] = Field(default=None, description="Monthly stipend in USD (0 = unpaid).")
    experience_level: Literal["Beginner", "Intermediate", "Advanced"] = Field(
        default="Beginner", description="Required experience level."
    )

    # Legacy fields (backwards compatibility with old CSV format)
    stipend: str | None = Field(default=None, description="Stipend or compensation details (legacy).")
    duration: str | None = Field(default=None, description="Expected internship duration (legacy).")


# ── Recommendation Output ────────────────────────────────────────────────────


class RecommendationResult(BaseModel):
    """A single scored internship recommendation for a candidate.

    Attributes:
        internship_id: Identifier of the matched internship listing.
        title: Title of the recommended internship.
        company: Company offering the internship.
        location: Location of the internship.
        location_type: Working arrangement (Remote / Hybrid / On-site).
        location_score: Location match score (0.0 to 1.0).
        required_skills: Skills the listing requires.
        match_score: Hybrid similarity score in the range [0, 1].
        skill_overlap_pct: Fraction of required skills the candidate
            possesses, expressed as a float in [0, 1].
        explanation: Human-readable rationale for the recommendation.
        domain: Functional domain of the listing.
        stipend_usd: Monthly stipend in USD, if available.
    """

    internship_id: str = Field(description="Matched internship listing ID.")
    title: str = Field(description="Title of the recommended internship.")
    company: str = Field(description="Company offering the internship.")
    location: str = Field(default="Remote", description="Internship location.")
    location_type: str = Field(default="Remote", description="Working arrangement.")
    location_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Location match score (0-1).")
    required_skills: List[str] = Field(default_factory=list, description="Skills required by the listing.")
    match_score: float = Field(ge=0.0, le=1.0, description="Hybrid similarity score (0-1).")
    skill_overlap_pct: float = Field(ge=0.0, le=1.0, description="Fraction of required skills the candidate has.")
    explanation: str = Field(default="", description="Human-readable recommendation rationale.")
    domain: str = Field(default="", description="Functional domain of the listing.")
    stipend_usd: Optional[int] = Field(default=None, description="Monthly stipend in USD.")


class RecommendationResponse(BaseModel):
    """Top-level API response wrapping all recommendation results for a candidate.

    Attributes:
        candidate_name: Display name of the candidate.
        total_results: Number of recommendations returned.
        results: Ordered list of internship recommendations, sorted by
            descending match_score.
        location_preference_applied: Whether location-based filtering was active.
        weights_used: Dictionary of scoring weights that were applied.
    """

    candidate_name: str = Field(description="Name of the candidate.")
    total_results: int = Field(ge=0, description="Count of returned recommendations.")
    results: List[RecommendationResult] = Field(
        default_factory=list,
        description="Recommendations sorted by descending match_score.",
    )
    location_preference_applied: bool = Field(
        default=False, description="Whether location preferences were used in scoring."
    )
    weights_used: Dict[str, float] = Field(
        default_factory=lambda: {"semantic": 0.60, "keyword": 0.25, "location": 0.15},
        description="Scoring weights applied for this response.",
    )


# ── Request Models ────────────────────────────────────────────────────────────


class ResumeUploadRequest(BaseModel):
    """Payload for the resume upload endpoint.

    Accepts either raw resume text or a base64-encoded file. At least one of
    the two fields must be provided.

    Attributes:
        candidate_name: Name of the candidate submitting the resume.
        raw_text: Plain-text content of the resume, if already extracted.
        file_base64: Base64-encoded resume file (PDF/DOCX) for server-side
            extraction.
        file_name: Original filename, used to determine the parser
            (e.g. 'resume.pdf').
    """

    candidate_name: str = Field(description="Candidate's full name.")
    raw_text: str | None = Field(default=None, description="Pre-extracted resume text.")
    file_base64: str | None = Field(default=None, description="Base64-encoded resume file.")
    file_name: str | None = Field(default=None, description="Original file name with extension.")
