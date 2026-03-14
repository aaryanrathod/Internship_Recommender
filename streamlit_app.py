"""
Streamlit frontend for the Internship Recommendation Engine.

Directly imports engine modules -- does NOT call the FastAPI server.

Run with::

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so engine.* imports resolve
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Guarded engine imports
# ---------------------------------------------------------------------------

def _import_engine():
    """Import all engine modules with user-friendly error messages."""
    mods = {}
    try:
        from engine.parser import parse_resume, ResumeParseError
        mods["parse_resume"] = parse_resume
        mods["ResumeParseError"] = ResumeParseError
    except ImportError as e:
        st.error(f"Failed to import `engine.parser`: {e}")
        st.stop()

    try:
        from engine.preprocessor import preprocess, detect_language
        mods["preprocess"] = preprocess
        mods["detect_language"] = detect_language
    except ImportError as e:
        st.error(f"Failed to import `engine.preprocessor`: {e}")
        st.stop()

    try:
        from engine.extractor import extract_profile
        mods["extract_profile"] = extract_profile
    except ImportError as e:
        st.error(f"Failed to import `engine.extractor`: {e}")
        st.stop()

    try:
        from engine.embedder import get_embedder
        mods["get_embedder"] = get_embedder
    except ImportError as e:
        st.error(f"Failed to import `engine.embedder`: {e}")
        st.stop()

    try:
        from engine.indexer import InternshipIndex
        mods["InternshipIndex"] = InternshipIndex
    except ImportError as e:
        st.error(f"Failed to import `engine.indexer`: {e}")
        st.stop()

    try:
        from engine.scorer import rank_recommendations
        mods["rank_recommendations"] = rank_recommendations
    except ImportError as e:
        st.error(f"Failed to import `engine.scorer`: {e}")
        st.stop()

    try:
        from api.schemas import CandidateLocationPreference
        mods["CandidateLocationPreference"] = CandidateLocationPreference
    except ImportError:
        mods["CandidateLocationPreference"] = None

    try:
        from config import settings
        mods["settings"] = settings
    except ImportError as e:
        st.error(f"Failed to import `config`: {e}")
        st.stop()

    return mods


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model ...")
def load_embedder():
    """Load the SentenceTransformer model once."""
    from engine.embedder import get_embedder
    return get_embedder()


@st.cache_resource(show_spinner="Loading FAISS index ...")
def load_index(index_path: str):
    """Load the FAISS index from disk once."""
    from engine.indexer import InternshipIndex
    idx = InternshipIndex()
    try:
        idx.load(index_path)
        return idx
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Badge / chip helpers
# ---------------------------------------------------------------------------

def _skill_badge(skill: str) -> str:
    """Return an HTML span styled as a chip/badge for a skill."""
    return (
        f'<span style="display:inline-block;background-color:#EBF2FC;'
        f"color:#1a3c5e;padding:4px 10px;border-radius:12px;"
        f'margin:2px 4px 2px 0;font-size:0.82em;font-weight:500;">'
        f"{skill}</span>"
    )


def _location_chip(location: str, location_type: str) -> str:
    """Return a colored HTML chip for location + work type."""
    colors = {
        "Remote": ("#d4edda", "#155724"),
        "Hybrid": ("#fff3cd", "#856404"),
        "On-site": ("#cce5ff", "#004085"),
    }
    bg, fg = colors.get(location_type, ("#f0f0f0", "#333333"))
    icon = {"Remote": "🌐", "Hybrid": "🏢🏠", "On-site": "🏢"}.get(location_type, "📍")
    label = f"{icon} {location} ({location_type})"
    return (
        f'<span style="display:inline-block;background-color:{bg};'
        f"color:{fg};padding:4px 12px;border-radius:12px;"
        f'margin:2px 4px 2px 0;font-size:0.82em;font-weight:600;">'
        f"{label}</span>"
    )


def _domain_badge(domain: str) -> str:
    """Return a styled HTML chip for a domain."""
    return (
        f'<span style="display:inline-block;background-color:#f3e8ff;'
        f"color:#6b21a8;padding:3px 10px;border-radius:10px;"
        f'margin:2px 4px 2px 0;font-size:0.78em;font-weight:500;">'
        f"{domain}</span>"
    )


# ---------------------------------------------------------------------------
# Profile renderer
# ---------------------------------------------------------------------------

def _render_profile(profile) -> None:
    """Render the extracted candidate profile inside an expander."""
    with st.expander("Your Extracted Profile", expanded=False):
        if profile.skills:
            st.markdown("**Skills**")
            badges = " ".join(_skill_badge(s) for s in profile.skills)
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.info("No skills extracted -- check your resume content.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Education**")
            if profile.education:
                for edu in profile.education:
                    parts = [f"**{edu.degree}**"]
                    if edu.institution:
                        parts.append(f"  {edu.institution}")
                    if edu.year:
                        parts.append(f"  ({edu.year})")
                    if edu.gpa:
                        parts.append(f"  -- GPA: {edu.gpa}")
                    st.markdown(" ".join(parts))
            else:
                st.caption("No education entries extracted.")

        with col2:
            st.markdown("**Experience**")
            if profile.experience:
                for exp in profile.experience:
                    line = f"**{exp.title}**"
                    if exp.company:
                        line += f" at {exp.company}"
                    if exp.duration:
                        line += f" ({exp.duration})"
                    st.markdown(line)
                    if exp.description:
                        st.caption(exp.description[:200])
            else:
                st.caption("No experience entries extracted.")


# ---------------------------------------------------------------------------
# Result card renderer
# ---------------------------------------------------------------------------

def _render_result(rank: int, r) -> None:
    """Render a single recommendation result card."""
    with st.container():
        # Header
        st.markdown(f"### {rank}. {r.title} -- {r.company}")

        # Location chip + domain badge row
        chips = ""
        loc_type = getattr(r, "location_type", "Remote") or "Remote"
        chips += _location_chip(r.location, loc_type)
        domain = getattr(r, "domain", "") or ""
        if domain:
            chips += " " + _domain_badge(domain)
        st.markdown(chips, unsafe_allow_html=True)

        # Score row
        c_bar, c_pct, c_meta = st.columns([3, 1, 2])
        with c_bar:
            st.progress(min(r.match_score, 1.0))
        with c_pct:
            st.markdown(f"**{r.match_score:.0%}**")
        with c_meta:
            stipend = getattr(r, "stipend_usd", None)
            if stipend is not None and stipend > 0:
                st.caption(f"${stipend}/mo")

        # Detail row
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f"**Skill overlap:** {r.skill_overlap_pct:.0%}")
        with mc2:
            loc_score = getattr(r, "location_score", 0.0) or 0.0
            st.markdown(f"**Location match:** {loc_score:.0%}")
        with mc3:
            if hasattr(r, "internship_id"):
                st.caption(f"ID: {r.internship_id}")

        # Skill badges
        if r.required_skills:
            badges = " ".join(_skill_badge(s) for s in r.required_skills)
            st.markdown(badges, unsafe_allow_html=True)

        # Explanation
        st.markdown(f"*{r.explanation}*")

        st.divider()


# =========================================================================
#  MAIN APP
# =========================================================================

def main() -> None:
    """Entry point for the Streamlit app."""

    # -- Page config ------------------------------------------------------
    st.set_page_config(
        page_title="Internship Recommendation Engine",
        page_icon="🎯",
        layout="wide",
    )

    # -- Import engine ----------------------------------------------------
    mods = _import_engine()
    parse_resume = mods["parse_resume"]
    ResumeParseError = mods["ResumeParseError"]
    preprocess = mods["preprocess"]
    detect_language = mods["detect_language"]
    extract_profile = mods["extract_profile"]
    rank_recommendations = mods["rank_recommendations"]
    CandidateLocationPreference = mods["CandidateLocationPreference"]
    settings = mods["settings"]

    # -- Load cached resources --------------------------------------------
    embedder = load_embedder()

    index_path = settings.FAISS_INDEX_PATH
    index = load_index(index_path)

    # -- Header -----------------------------------------------------------
    st.markdown(
        "<h1 style='text-align:center;'>Internship Recommendation Engine</h1>"
        "<p style='text-align:center;color:#666;margin-top:-10px;'>"
        "Upload your resume and find your best-fit internships</p>",
        unsafe_allow_html=True,
    )

    # =====================================================================
    # SIDEBAR
    # =====================================================================
    with st.sidebar:
        st.header("Settings")

        top_n: int = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=20,
            value=10,
        )

        # -- Weights ------------------------------------------------------
        st.subheader("Scoring Weights")

        location_weight: float = st.slider(
            "Location weight",
            min_value=0.0,
            max_value=0.50,
            value=0.15,
            step=0.05,
            help="Weight for location-based matching",
        )

        # Remaining weight distributed proportionally between semantic & keyword
        remaining = round(1.0 - location_weight, 2)
        # Default ratio: semantic 0.60 / (0.60+0.25) = 0.706, keyword = 0.294
        sem_ratio = 0.60 / 0.85
        semantic_weight = round(remaining * sem_ratio, 2)
        keyword_weight = round(remaining - semantic_weight, 2)

        # Show active weights
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            st.metric("Semantic", f"{semantic_weight:.2f}")
        with wc2:
            st.metric("Keyword", f"{keyword_weight:.2f}")
        with wc3:
            st.metric("Location", f"{location_weight:.2f}")

        # -- Location Preferences -----------------------------------------
        st.divider()
        st.subheader("Location Preferences")

        has_loc_pref = st.checkbox("I have location preferences", value=False)

        location_preference = None

        if has_loc_pref and CandidateLocationPreference is not None:
            pref_city = st.text_input(
                "Preferred city",
                placeholder="e.g. San Francisco",
            )
            pref_country = st.text_input(
                "Preferred country",
                placeholder="e.g. United States",
            )
            pref_work_type = st.selectbox(
                "Preferred work type",
                options=["No preference", "Remote", "Hybrid", "On-site"],
                index=0,
            )
            open_to_remote = st.checkbox("Open to remote opportunities", value=True)

            loc_type = pref_work_type if pref_work_type != "No preference" else None
            has_any = bool(pref_city.strip() or pref_country.strip() or loc_type)

            if has_any:
                location_preference = CandidateLocationPreference(
                    preferred_city=pref_city.strip() or None,
                    preferred_country=pref_country.strip() or None,
                    preferred_location_type=loc_type,
                    open_to_remote=open_to_remote,
                )

        # -- About --------------------------------------------------------
        st.divider()
        st.markdown("### About")
        st.caption(
            "This engine embeds your resume using a Sentence Transformer model "
            "and searches a FAISS index of internships via cosine similarity. "
            "Results are ranked by a hybrid of semantic similarity, "
            "TF-IDF skill overlap, and location matching."
        )

        # Index status
        st.divider()
        if index is not None:
            listing_count = len(index.listings) if hasattr(index, "listings") else "?"
            sw_status = "TF-IDF" if getattr(index, "skill_weighter", None) else "Jaccard"
            st.success(f"Index loaded -- **{listing_count}** internships ({sw_status} scoring)")
        else:
            st.error("FAISS index not loaded")

    # -- FAISS index check ------------------------------------------------
    if index is None:
        st.error(
            "FAISS index file not found at "
            f"`{index_path}`.\n\n"
            "**Run this command first:**\n"
            "```\npython scripts/build_index.py --input data/internships_batch1.csv\n```"
        )
        st.stop()

    # -- Input tabs -------------------------------------------------------
    tab_upload, tab_paste = st.tabs(["Upload Resume", "Paste Text"])

    uploaded_file = None
    pasted_text = ""

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF or DOCX)",
            type=["pdf", "docx"],
            help="Max 5 MB",
        )
        if uploaded_file is not None:
            size_kb = uploaded_file.size / 1024
            st.caption(f"**{uploaded_file.name}** -- {size_kb:.1f} KB")

    with tab_paste:
        pasted_text = st.text_area(
            "Paste your resume text",
            height=250,
            placeholder=(
                "Jane Doe\n"
                "Python developer with 3 years of experience in Machine Learning.\n"
                "Skills: Python, TensorFlow, SQL, Pandas, NLP\n"
                "Education: B.Tech CS, IIT Delhi 2024\n"
                "Experience: ML Intern at Google, May 2023 -- Aug 2023\n"
                "..."
            ),
        )
        if pasted_text.strip():
            st.caption(f"Characters: {len(pasted_text):,}")

    # -- Action button ----------------------------------------------------
    st.markdown("")
    find_btn = st.button("Find Internships", type="primary", use_container_width=True)

    if not find_btn:
        st.stop()

    # -- Validate input ---------------------------------------------------
    has_file = uploaded_file is not None
    has_text = bool(pasted_text.strip())

    if not has_file and not has_text:
        st.warning("Please upload a resume file or paste your resume text first.")
        st.stop()

    # -- Process ----------------------------------------------------------
    with st.spinner("Analyzing your resume ..."):

        # Step 1 -- Get raw text
        raw_text: str = ""

        if has_file:
            suffix = os.path.splitext(uploaded_file.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                tmp.write(uploaded_file.getvalue())
                tmp.close()
                raw_text = parse_resume(tmp.name)
            except ResumeParseError as e:
                st.error(f"Resume parsing failed: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error during parsing: {e}")
                st.stop()
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
        else:
            raw_text = pasted_text.strip()

        if not raw_text:
            st.error("No text could be extracted from the resume.")
            st.stop()

        # Step 2 -- Preprocess
        cleaned = preprocess(raw_text)
        if not cleaned:
            st.error("Resume is empty after preprocessing. Please try a different file.")
            st.stop()

        # Step 3 -- Language check
        lang = detect_language(cleaned)
        if lang not in ("en", "unknown"):
            st.warning(
                f"Non-English resume detected (language: `{lang}`). "
                "Results may be less accurate."
            )

        # Step 4 -- Extract profile
        profile = extract_profile(cleaned)

        # Step 5 -- Embed
        query_vec = embedder.embed_profile(profile)

        # Step 6 -- Search
        faiss_results = index.search(query_vec, top_k=30)

        # Step 7 -- Rank with all three dimensions
        skill_weighter = getattr(index, "skill_weighter", None)

        results = rank_recommendations(
            profile=profile,
            candidates=faiss_results,
            location_preference=location_preference,
            skill_weighter=skill_weighter,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            location_weight=location_weight,
            top_n=top_n,
        )

    # -- Results ----------------------------------------------------------
    if not results:
        st.info(
            "No matching internships found. Try adding more skills or "
            "experience to your resume."
        )
        st.stop()

    st.success(f"Found **{len(results)}** matching internships for you!")

    if location_preference is not None:
        parts = []
        if location_preference.preferred_city:
            parts.append(f"city: {location_preference.preferred_city}")
        if location_preference.preferred_country:
            parts.append(f"country: {location_preference.preferred_country}")
        if location_preference.preferred_location_type:
            parts.append(f"type: {location_preference.preferred_location_type}")
        st.info(f"Location preferences applied: {', '.join(parts)}")

    # Extracted profile
    _render_profile(profile)

    # Recommendation cards
    st.markdown("---")
    st.markdown("## Recommended Internships")

    for rank, r in enumerate(results, 1):
        _render_result(rank, r)

    # -- Footer -----------------------------------------------------------
    st.markdown("---")
    st.caption(
        "Built with **Sentence Transformers** + **FAISS** + **spaCy** | "
        "Powered by Streamlit"
    )


if __name__ == "__main__":
    main()
