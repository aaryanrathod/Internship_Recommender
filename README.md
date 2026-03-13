# 🎯 Internship Recommendation Engine

An AI-powered internship matching system that uses **Sentence Transformers** + **FAISS** + **spaCy** to rank internship listings against a candidate's resume. Features a FastAPI backend and an interactive Streamlit frontend.

---

## ✨ Features

- **Hybrid Scoring** — Combines semantic similarity (FAISS cosine search) with keyword skill overlap (Jaccard) for more relevant rankings
- **Resume Parsing** — Handles PDF, DOCX, and TXT; falls back to OCR (Tesseract) for scanned PDFs
- **Skill Extraction** — spaCy PhraseMatcher against a curated skills taxonomy
- **Streamlit UI** — Interactive frontend with file upload, paste-text mode, sliders for weights, and styled result cards
- **FastAPI Backend** — REST API with multipart file upload, OpenAPI docs, lifespan index loading, and request logging
- **Redis Caching** — Optional embedding cache (silently disabled if Redis is unavailable)
- **Fully Tested** — 22 unit tests + 4 end-to-end integration tests

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
          │   engine/indexer.py       │  FAISS IndexFlatIP search
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │   engine/scorer.py        │  Hybrid score + explanation
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
│       └── recommend.py      # POST /api/v1/recommend endpoint
├── engine/
│   ├── parser.py             # Resume text extraction
│   ├── preprocessor.py       # Text cleaning, chunking, language detection
│   ├── extractor.py          # Profile extraction (skills, edu, exp)
│   ├── embedder.py           # SentenceTransformer + Redis cache
│   ├── indexer.py            # FAISS index build/save/load/search
│   └── scorer.py             # Hybrid scoring and ranking
├── scripts/
│   └── build_index.py        # CLI to build FAISS index from CSV
├── tests/
│   ├── unit/
│   │   └── test_pipeline.py  # 22 unit tests
│   └── integration/
│       └── test_full_pipeline.py # End-to-end tests
├── data/
│   ├── internships.csv       # Internship dataset
│   ├── skills_taxonomy.json  # Flat list of skills for extraction
│   └── faiss.index           # Built index (generated)
├── streamlit_app.py          # Streamlit frontend (direct engine imports)
├── config.py                 # Pydantic Settings from .env
├── requirements.txt
├── pyproject.toml
└── .env                      # Environment variables (not committed)
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

Copy `.env.example` to `.env` and set your paths:

```bash
cp .env.example .env
```

Default `.env` values:

```ini
EMBEDDING_MODEL=all-MiniLM-L6-v2
FAISS_INDEX_PATH=data/faiss.index
INTERNSHIP_DATA_PATH=data/internships.csv
TOP_K_RESULTS=10
SEMANTIC_WEIGHT=0.70
KEYWORD_WEIGHT=0.30
REDIS_URL=redis://localhost:6379/0   # optional
LOG_LEVEL=INFO
MAX_TEXT_LENGTH=10000
```

### 3. Build the FAISS index

```bash
python scripts/build_index.py --input data/internships.csv
```

Expected output:
```
============================================================
  FAISS Index Build — Summary
============================================================
  Input file       : data/internships.csv
  Internships      : 30
  Index dimensions : 384
  Index size       : 45.0 KB
  Total time       : 36s
============================================================
```

### 4. Run the app

**Streamlit UI** (recommended for testing):
```bash
streamlit run streamlit_app.py
# → http://localhost:8501
```

**FastAPI server**:
```bash
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs
```

---

## 📊 Internship Data Format

Your CSV must have these columns:

| Column | Required | Example |
|---|---|---|
| `internship_id` | ✅ | `INT-001` |
| `title` | ✅ | `ML Intern` |
| `company` | ✅ | `Google DeepMind` |
| `location` | — | `Bangalore` |
| `required_skills` | — | `"Python,TensorFlow,NLP"` |
| `description` | — | Full description text |
| `stipend` | — | `₹80000/month` |
| `duration` | — | `3 months` |

To rebuild the index after updating your CSV:

```bash
python scripts/build_index.py --input data/internships.csv --output data/faiss.index
```

---

## 🌐 API Reference

The server starts at `http://localhost:8000`. Interactive docs: [`/docs`](http://localhost:8000/docs)

### `POST /api/v1/recommend`

Upload a resume and get ranked internship recommendations.

**Request** (multipart form):

| Field | Type | Description |
|---|---|---|
| `file` | File | PDF / DOCX / TXT (max 5 MB) |
| `raw_text` | string | Alternative: paste resume text directly |
| `candidate_name` | string | Optional candidate name |

**Response:**

```json
{
  "candidate_name": "Rahul Sharma",
  "total_results": 10,
  "results": [
    {
      "internship_id": "INT-001",
      "title": "Machine Learning Intern",
      "company": "Google DeepMind",
      "location": "Bangalore",
      "required_skills": ["Python", "TensorFlow", "NLP"],
      "match_score": 0.525,
      "skill_overlap_pct": 0.36,
      "explanation": "Strong match based on 4 overlapping skills including Python, NLP, TensorFlow."
    }
  ]
}
```

**Error codes:**

| Code | Reason |
|---|---|
| 422 | File > 5 MB / unsupported format / empty resume / no input provided |
| 503 | FAISS index not loaded — run `build_index.py` first |

### `GET /health`

```json
{
  "status": "healthy",
  "index_loaded": true,
  "model_loaded": true,
  "timestamp": "2026-03-14T00:04:56+00:00"
}
```

---

## 🖥️ Streamlit UI

```bash
streamlit run streamlit_app.py
```

- **Upload Resume tab** — drag-and-drop PDF or DOCX
- **Paste Text tab** — paste raw resume text
- **Sidebar** — adjust number of recommendations, semantic weight, keyword weight
- **Results** — match score progress bars, skill chips, extraction profile, explanations

> No FastAPI server needed — the Streamlit app imports the engine directly.

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

## 🔧 Scoring Logic

```
hybrid_score = semantic_weight × cosine_similarity
             + keyword_weight  × jaccard_skill_overlap
```

- **Semantic similarity** — `all-MiniLM-L6-v2` embedding via FAISS `IndexFlatIP`
- **Skill overlap** — Jaccard: `|candidate ∩ internship| / |candidate ∪ internship|`
- All scores clamped to `[0.0, 1.0]`

> For datasets > 50k listings, upgrade `IndexFlatIP` → `IndexIVFFlat` in `engine/indexer.py`.

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | Text embedding model |
| `faiss-cpu` | Vector similarity search |
| `spacy` | NER and skill extraction |
| `pdfplumber` | PDF text extraction |
| `python-docx` | DOCX parsing |
| `fastapi` + `uvicorn` | REST API server |
| `streamlit` | Interactive frontend |
| `pydantic` v2 | Data validation |
| `loguru` | Structured logging |
| `redis` | Optional embedding cache |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
