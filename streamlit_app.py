"""
Streamlit frontend for the Internship Recommendation Engine.

Directly imports engine modules — does NOT call the FastAPI server.

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
    """Import all engine modules with user-friendly error messages.

    Returns:
        Tuple of imported modules/functions, or displays st.error and stops.
    """
    mods = {}
    try:
        from engine.parser import parse_resume, ResumeParseError
        mods["parse_resume"] = parse_resume
        mods["ResumeParseError"] = ResumeParseError
    except ImportError as e:
        st.error(f"❌ Failed to import `engine.parser`: {e}")
        st.stop()

    try:
        from engine.preprocessor import preprocess, detect_language
        mods["preprocess"] = preprocess
        mods["detect_language"] = detect_language
    except ImportError as e:
        st.error(f"❌ Failed to import `engine.preprocessor`: {e}")
        st.stop()

    try:
        from engine.extractor import extract_profile
        mods["extract_profile"] = extract_profile
    except ImportError as e:
        st.error(f"❌ Failed to import `engine.extractor`: {e}")
        st.stop()

    try:
        from engine.embedder import get_embedder
        mods["get_embedder"] = get_embedder
    except ImportError as e:
        st.error(f"❌ Failed to import `engine.embedder`: {e}")
        st.stop()

    try:
        from engine.indexer import InternshipIndex
        mods["InternshipIndex"] = InternshipIndex
    except ImportError as e:
        st.error(f"❌ Failed to import `engine.indexer`: {e}")
        st.stop()

    try:
        from engine.scorer import rank_recommendations
        mods["rank_recommendations"] = rank_recommendations
    except ImportError as e:
        st.error(f"❌ Failed to import `engine.scorer`: {e}")
        st.stop()

    try:
        from config import settings
        mods["settings"] = settings
    except ImportError as e:
        st.error(f"❌ Failed to import `config`: {e}")
        st.stop()

    return mods


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model …")
def load_embedder():
    """Load the SentenceTransformer model once."""
    from engine.embedder import get_embedder
    return get_embedder()


@st.cache_resource(show_spinner="Loading FAISS index …")
def load_index(index_path: str):
    """Load the FAISS index from disk once.

    Args:
        index_path: Path to the FAISS index file.

    Returns:
        InternshipIndex or None if file is missing.
    """
    from engine.indexer import InternshipIndex
    idx = InternshipIndex()
    try:
        idx.load(index_path)
        return idx
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _skill_badge(skill: str) -> str:
    """Return an HTML span styled as a chip/badge for a skill."""
    return (
        f'<span style="display:inline-block;background-color:#EBF2FC;'
        f"color:#1a3c5e;padding:4px 10px;border-radius:12px;"
        f'margin:2px 4px 2px 0;font-size:0.82em;font-weight:500;">'
        f"{skill}</span>"
    )


def _render_profile(profile) -> None:
    """Render the extracted candidate profile inside an expander."""
    with st.expander("📋 Your Extracted Profile", expanded=False):
        # Skills
        if profile.skills:
            st.markdown("**Skills**")
            badges = " ".join(_skill_badge(s) for s in profile.skills)
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.info("No skills extracted — check your resume content.")

        col1, col2 = st.columns(2)

        # Education
        with col1:
            st.markdown("**Education**")
            if profile.education:
                for edu in profile.education:
                    parts = [f"🎓 **{edu.degree}**"]
                    if edu.institution:
                        parts.append(f"  {edu.institution}")
                    if edu.year:
                        parts.append(f"  ({edu.year})")
                    if edu.gpa:
                        parts.append(f"  — GPA: {edu.gpa}")
                    st.markdown(" ".join(parts))
            else:
                st.caption("No education entries extracted.")

        # Experience
        with col2:
            st.markdown("**Experience**")
            if profile.experience:
                for exp in profile.experience:
                    line = f"💼 **{exp.title}**"
                    if exp.company:
                        line += f" at {exp.company}"
                    if exp.duration:
                        line += f" ({exp.duration})"
                    st.markdown(line)
                    if exp.description:
                        st.caption(exp.description[:200])
            else:
                st.caption("No experience entries extracted.")


def _render_result(rank: int, r) -> None:
    """Render a single recommendation result card."""
    with st.container():
        # Header
        st.markdown(
            f"### {rank}. {r.title} — {r.company}"
        )

        # Score row
        c_bar, c_pct, c_loc = st.columns([3, 1, 2])
        with c_bar:
            st.progress(min(r.match_score, 1.0))
        with c_pct:
            st.markdown(f"**{r.match_score:.0%}**")
        with c_loc:
            st.markdown(f"📍 {r.location}")

        # Meta row
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"**Skill overlap:** {r.skill_overlap_pct:.0%}")
        with mc2:
            if hasattr(r, "internship_id"):
                st.caption(f"ID: {r.internship_id}")

        # Skill badges
        if r.required_skills:
            badges = " ".join(_skill_badge(s) for s in r.required_skills)
            st.markdown(badges, unsafe_allow_html=True)

        # Explanation
        st.markdown(f"*{r.explanation}*")

        st.divider()


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  MAIN APP                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    """Entry point for the Streamlit app."""

    # ── Page config ──────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Internship Recommendation Engine",
        page_icon="🎯",
        layout="wide",
    )

    # ── Import engine ────────────────────────────────────────────────────
    mods = _import_engine()
    parse_resume = mods["parse_resume"]
    ResumeParseError = mods["ResumeParseError"]
    preprocess = mods["preprocess"]
    detect_language = mods["detect_language"]
    extract_profile = mods["extract_profile"]
    rank_recommendations = mods["rank_recommendations"]
    settings = mods["settings"]

    # ── Load cached resources ────────────────────────────────────────────
    embedder = load_embedder()

    index_path = settings.FAISS_INDEX_PATH
    index = load_index(index_path)

    # ── Header ───────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center;'>🎯 Internship Recommendation Engine</h1>"
        "<p style='text-align:center;color:#666;margin-top:-10px;'>"
        "Upload your resume and find your best-fit internships</p>",
        unsafe_allow_html=True,
    )

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        top_n: int = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=20,
            value=10,
        )

        semantic_weight: float = st.slider(
            "Semantic weight",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            help="Higher = prioritizes meaning over keywords",
        )

        keyword_weight = round(1.0 - semantic_weight, 2)
        st.metric("Keyword weight (auto)", f"{keyword_weight:.2f}")

        st.divider()

        st.markdown("### ℹ️ About")
        st.caption(
            "This engine embeds your resume using a Sentence Transformer model "
            "and searches a FAISS index of internships via cosine similarity. "
            "Results are ranked by a hybrid of semantic similarity and "
            "keyword skill overlap."
        )

        # Index status
        st.divider()
        if index is not None:
            listing_count = len(index.listings) if hasattr(index, "listings") else "?"
            st.success(f"✅ Index loaded — **{listing_count}** internships")
        else:
            st.error("❌ FAISS index not loaded")

    # ── FAISS index check ────────────────────────────────────────────────
    if index is None:
        st.error(
            "⚠️ FAISS index file not found at "
            f"`{index_path}`.\n\n"
            "**Run this command first:**\n"
            "```\npython scripts/build_index.py --input data/internships.csv\n```"
        )
        st.stop()

    # ── Input tabs ───────────────────────────────────────────────────────
    tab_upload, tab_paste = st.tabs(["📄 Upload Resume", "✏️ Paste Text"])

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
            st.caption(
                f"📎 **{uploaded_file.name}** — {size_kb:.1f} KB"
            )

    with tab_paste:
        pasted_text = st.text_area(
            "Paste your resume text",
            height=250,
            placeholder=(
                "Jane Doe\n"
                "Python developer with 3 years of experience in Machine Learning.\n"
                "Skills: Python, TensorFlow, SQL, Pandas, NLP\n"
                "Education: B.Tech CS, IIT Delhi 2024\n"
                "Experience: ML Intern at Google, May 2023 – Aug 2023\n"
                "..."
            ),
        )
        if pasted_text.strip():
            st.caption(f"Characters: {len(pasted_text):,}")

    # ── Action button ────────────────────────────────────────────────────
    st.markdown("")  # spacing
    find_btn = st.button("🔍 Find Internships", type="primary", use_container_width=True)

    if not find_btn:
        st.stop()

    # ── Validate input ───────────────────────────────────────────────────
    has_file = uploaded_file is not None
    has_text = bool(pasted_text.strip())

    if not has_file and not has_text:
        st.warning("Please upload a resume file or paste your resume text first.")
        st.stop()

    # ── Process ──────────────────────────────────────────────────────────
    with st.spinner("Analyzing your resume …"):

        # Step 1 — Get raw text
        raw_text: str = ""

        if has_file:
            # Save to temp file for the parser
            suffix = os.path.splitext(uploaded_file.name)[1]
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix,
            )
            try:
                tmp.write(uploaded_file.getvalue())
                tmp.close()
                raw_text = parse_resume(tmp.name)
            except ResumeParseError as e:
                st.error(f"❌ Resume parsing failed: {e}")
                st.stop()
            except Exception as e:
                st.error(f"❌ Unexpected error during parsing: {e}")
                st.stop()
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
        else:
            raw_text = pasted_text.strip()

        if not raw_text:
            st.error("❌ No text could be extracted from the resume.")
            st.stop()

        # Step 2 — Preprocess
        cleaned = preprocess(raw_text)
        if not cleaned:
            st.error("❌ Resume is empty after preprocessing. Please try a different file.")
            st.stop()

        # Step 3 — Language check
        lang = detect_language(cleaned)
        if lang not in ("en", "unknown"):
            st.warning(
                f"⚠️ Non-English resume detected (language: `{lang}`). "
                "Results may be less accurate."
            )

        # Step 4 — Extract profile
        profile = extract_profile(cleaned)

        # Step 5 — Embed
        query_vec = embedder.embed_profile(profile)

        # Step 6 — Search
        faiss_results = index.search(query_vec, top_k=30)

        # Step 7 — Rank
        results = rank_recommendations(
            profile=profile,
            candidates=faiss_results,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            top_n=top_n,
        )

    # ── Results ──────────────────────────────────────────────────────────
    if not results:
        st.info(
            "No matching internships found. Try adding more skills or "
            "experience to your resume."
        )
        st.stop()

    st.success(
        f"Found **{len(results)}** matching internships for you!"
    )

    # Extracted profile
    _render_profile(profile)

    # Recommendation cards
    st.markdown("---")
    st.markdown("## 🏆 Recommended Internships")

    for rank, r in enumerate(results, 1):
        _render_result(rank, r)

    # ── Footer ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "Built with **Sentence Transformers** + **FAISS** + **spaCy** · "
        "Powered by Streamlit"
    )


if __name__ == "__main__":
    main()
