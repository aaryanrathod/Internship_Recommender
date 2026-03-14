# 🎯 Internship Recommendation Engine

An AI-powered internship matching system that uses **Sentence Transformers** + **FAISS** + **spaCy** to rank internship listings against a candidate's resume with **three scoring dimensions**: semantic similarity, TF-IDF weighted skill overlap, and location preference matching.

---

## ✨ Features

- **Three-Dimensional Hybrid Scoring** — Semantic similarity (FAISS cosine) + TF-IDF weighted skill matching + location preference scoring
- **TF-IDF Skill Weighter** — Rare, discriminative skills (e.g. QuantLib, LayoutLM) count more than common ones (Python). Includes fuzzy substring matching (`ReactJS` ↔ `React`)
- **Location Preference Matching** — Candidates specify city, country, and preferred work type (Remote / Hybrid / On-site); each internship is scored accordingly
- **Resume Parsing** — Handles PDF, DOCX, and TXT; falls back to OCR (Tesseract) for scanned PDFs
- **Skill Extraction** — spaCy PhraseMatcher against a curated skills taxonomy
- **Streamlit UI** — Interactive frontend with file upload, paste-text mode, 3 scoring weight sliders, location preference controls, and styled result cards with color-coded location chips
- **FastAPI Backend** — REST API with multipart file upload, location preference form fields, OpenAPI docs, and metadata endpoints
- **Redis Caching** — Optional embedding cache (silently disabled if Redis is unavailable)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Input: Resume (PDF / DOCX / TXT / raw text)                │
└───────────────────────┬─────────────────────────────────────┘
                        │
          ┌─────────────▼─────────────┐
          │   engine/parser.py        │  pdfplumber / python-docx / OCR
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │   engine/preprocessor.py  │  Unicode, HTML strip, chunking
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │   engine/extractor.py     │  spaCy NER, skills, edu, exp
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │   engine/embedder.py      │  SentenceTransformer + Redis cache
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │   engine/indexer.py       │  FAISS IndexFlatIP + SkillWeighter
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │   engine/scorer.py        │  Hybrid: semantic + TF-IDF + location
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │  RecommendationResponse   │  Ranked results with explanations
          └───────────────────────────┘
```

---

## 📁 Project Structure

```
Internship Recommendation Engine New/
├── api/
│   ├── main.py               # FastAPI app factory + lifespan
│   ├── schemas.py            # Pydantic v2 data models
│   └── routes/
│       └── recommend.py      # POST /recommend, GET /internships/domains|locations
├── engine/
│   ├── parser.py             # Resume text extraction
│   ├── preprocessor.py       # Text cleaning, chunking, language detection
│   ├── extractor.py          # Profile extraction (skills, edu, exp)
│   ├── embedder.py           # SentenceTransformer + Redis cache
│   ├── indexer.py            # FAISS index build/save/load/search + SkillWeighter
│   └── scorer.py             # SkillWeighter, location scoring, hybrid ranking
├── scripts/
│   └── build_index.py        # CLI to build FAISS index from CSV
├── tests/
│   ├── unit/
│   │   └── test_pipeline.py  # Unit tests
│   └── integration/
│       └── test_full_pipeline.py # End-to-end tests
├── data/
│   ├── internships_batch1.csv  # 200 internship listings (primary dataset)
│   ├── internships.csv         # Legacy 30-row dataset
│   ├── skills_taxonomy.json    # Flat list of skills for extraction
│   └── faiss.index             # Built index (generated — not committed)
├── streamlit_app.py          # Streamlit frontend (direct engine imports)
├── config.py                 # Pydantic Settings from .env
├── requirements.txt
├── pyproject.toml
├── .env.example              # Copy this to .env
└── .gitignore
```

---

## ⚡ Quick Start

### 1. Clone and install

```bash
git clone https://github.com/aaryanrathod/Internship_Recommender.git
cd Internship_Recommender
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure environment

```bash
cp .env.example .env
```

Key `.env` fields:

```ini
EMBEDDING_MODEL=all-MiniLM-L6-v2
FAISS_INDEX_PATH=data/faiss.index
INTERNSHIP_DATA_PATH=data/internships_batch1.csv
TOP_K_RESULTS=10

# Scoring weights — must sum to 1.0
SEMANTIC_WEIGHT=0.60
KEYWORD_WEIGHT=0.25
LOCATION_WEIGHT=0.15

CANDIDATE_LOCATION=          # leave empty for no default preference
REDIS_URL=redis://localhost:6379/0   # optional
LOG_LEVEL=INFO
MAX_TEXT_LENGTH=10000
```

> **Weight validation:** The engine validates that `SEMANTIC_WEIGHT + KEYWORD_WEIGHT + LOCATION_WEIGHT == 1.0` on startup and raises a `ValueError` if not.

### 3. Build the FAISS index

```bash
python scripts/build_index.py --input data/internships_batch1.csv
```

Expected output:
```
============================================================
  FAISS Index Build — Summary
============================================================
  Input file       : data/internships_batch1.csv
  Internships      : 200
  Index dimensions : 384
  Index file       : data/faiss.index
  Index size       : 300.0 KB
  Total time       : 22s
  SkillWeighter    : YES (built & saved)
============================================================

  Domain breakdown:
    Machine Learning           20  ####################
    Research                   20  ####################
    Data Science               18  ##################
    ...

  Location type: Remote 52% | On-site 26% | Hybrid 22%
  Experience:    Intermediate 48% | Beginner 27% | Advanced 25%
```

### 4. Run the app

**Streamlit UI** (recommended):
```bash
streamlit run streamlit_app.py
# → http://localhost:8501
```

**FastAPI server:**
```bash
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs
```

---

## 📊 Internship Data Format

Your CSV must have at minimum `internship_id`, `title`, `company`. All columns:

| Column | Required | Format | Example |
|---|---|---|---|
| `internship_id` | ✅ | `INT-001` | `INT-042` |
| `title` | ✅ | string | `ML Research Intern` |
| `company` | ✅ | string | `Nexora AI Labs` |
| `location` | — | `City, Region` | `San Francisco, CA` |
| `country` | — | full name | `United States` |
| `location_type` | — | `Remote` / `Hybrid` / `On-site` | `Remote` |
| `description` | — | 3–4 sentences | — |
| `required_skills` | — | **pipe-separated** | `Python\|SQL\|TensorFlow` |
| `preferred_skills` | — | **pipe-separated** | `Spark\|Airflow` |
| `domain` | — | see domains below | `Machine Learning` |
| `duration_months` | — | integer | `3` |
| `stipend_usd` | — | integer (0 = unpaid) | `2200` |
| `experience_level` | — | `Beginner` / `Intermediate` / `Advanced` | `Intermediate` |

**Valid domains:** `Machine Learning`, `Data Science`, `Software Engineering`, `Web Development`, `NLP`, `Computer Vision`, `DevOps`, `Cybersecurity`, `Product Management`, `Data Engineering`, `Research`, `Finance Tech`

To rebuild the index after updating your CSV:
```bash
python scripts/build_index.py --input data/internships_batch1.csv --output data/faiss.index
```

---

## 🌐 API Reference

Interactive docs: [`http://localhost:8000/docs`](http://localhost:8000/docs)

### `POST /api/v1/recommend`

Upload a resume and get ranked recommendations.

**Request** (multipart form):

| Field | Type | Description |
|---|---|---|
| `file` | File | PDF / DOCX / TXT (max 5 MB) |
| `raw_text` | string | Alternative: paste resume text |
| `candidate_name` | string | Optional candidate name |
| `preferred_city` | string | e.g. `San Francisco, CA` |
| `preferred_country` | string | e.g. `United States` |
| `preferred_location_type` | string | `Remote` / `Hybrid` / `On-site` |
| `open_to_remote` | bool | Default `true` |

**Response:**

```json
{
  "candidate_name": "Rahul Sharma",
  "total_results": 10,
  "location_preference_applied": true,
  "weights_used": { "semantic": 0.60, "keyword": 0.25, "location": 0.15 },
  "results": [
    {
      "internship_id": "INT-001",
      "title": "Machine Learning Research Intern",
      "company": "Nexora AI Labs",
      "location": "San Francisco, CA",
      "location_type": "Remote",
      "location_score": 0.95,
      "domain": "Machine Learning",
      "required_skills": ["Python", "TensorFlow", "PyTorch"],
      "match_score": 0.61,
      "skill_overlap_pct": 0.42,
      "stipend_usd": 2200,
      "explanation": "Strong match based on 5 overlapping skills including Python, TensorFlow. Semantic similarity of 0.74 indicates a moderate profile fit. Location is a strong match (Remote)."
    }
  ]
}
```

### `GET /api/v1/internships/domains`

Returns unique domains in the loaded index.

```json
{ "domains": ["Computer Vision", "Cybersecurity", "Data Engineering", ...], "count": 12 }
```

### `GET /api/v1/internships/locations`

Returns unique countries and location types in the index.

```json
{
  "countries": ["Australia", "Brazil", "Canada", ...],
  "country_count": 12,
  "location_types": ["Hybrid", "On-site", "Remote"]
}
```

### `GET /health`

```json
{
  "status": "healthy",
  "index_loaded": true,
  "model_loaded": true
}
```

---

## 🖥️ Streamlit UI

```bash
streamlit run streamlit_app.py
```

**Sidebar controls:**
- **Number of recommendations** slider (1–20)
- **Location weight** slider (0.0–0.50, default 0.15) — semantic & keyword auto-redistribute proportionally
- **Active weights** displayed in 3 metric cards
- **Location Preferences** section (toggled by checkbox):
  - Preferred city, country, work type (Remote / Hybrid / On-site)
  - "Open to remote" checkbox

**Input:**
- **Upload Resume** tab — drag-and-drop PDF or DOCX
- **Paste Text** tab — paste raw resume text

**Result cards include:**
- Color-coded location chip: 🌐 **green** (Remote) · 🏢🏠 **orange** (Hybrid) · 🏢 **blue** (On-site)
- Purple domain badge
- Match score progress bar + percentage
- Skill overlap % and **location match %** side by side
- Monthly stipend (when available)
- TF-IDF skill badges
- Template-based explanation mentioning skills, semantic fit, and location

> No FastAPI server needed — the Streamlit app imports the engine directly.

---

## 🔧 Scoring Logic

```
hybrid_score = semantic_weight  × cosine_similarity        (FAISS)
             + keyword_weight   × tfidf_skill_score        (SkillWeighter)
             + location_weight  × location_match_score
```

### TF-IDF Skill Scoring (`SkillWeighter`)

```
idf(skill) = log((1 + N) / (1 + df(skill))) + 1
```

- Skills appearing in many listings get low IDF (common → less discriminative)
- Rare skills get high IDF and contribute more to the score
- **Fuzzy matching**: `ReactJS` matches `React`, `ML` matches `Machine Learning`
- Score is normalized to `[0, 1]` by total IDF of the internship's required skills

### Location Scoring

| Condition | Score |
|---|---|
| No preference specified | 0.50 (neutral) |
| Preferred city matches listing city | 1.00 |
| Internship is Remote + candidate open_to_remote | 0.95 |
| Preferred location_type matches | 0.80 |
| Preferred country matches | 0.75 |
| Internship Remote, candidate not open_to_remote | 0.30 |
| No match at all | 0.10 |

> Returns the **highest applicable** score.

### Default Weights

| Dimension | Weight |
|---|---|
| Semantic similarity | 0.60 |
| TF-IDF skill overlap | 0.25 |
| Location match | 0.15 |

All three must sum to 1.0 (validated on startup).

---

## 🧪 Running Tests

```bash
# Unit tests only (fast — no model loading)
pytest tests/unit -v

# Integration tests (loads SentenceTransformer model)
pytest tests/integration -v

# All tests
pytest -v
```

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | Text embedding model (`all-MiniLM-L6-v2`) |
| `faiss-cpu` | Vector similarity search |
| `spacy` | NER and skill extraction |
| `pdfplumber` | PDF text extraction |
| `python-docx` | DOCX parsing |
| `fastapi` + `uvicorn` | REST API server |
| `streamlit` | Interactive frontend |
| `pydantic` v2 | Data validation |
| `pydantic-settings` | `.env` configuration |
| `loguru` | Structured logging |
| `redis` | Optional embedding cache |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
