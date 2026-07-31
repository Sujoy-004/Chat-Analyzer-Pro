# Technology Stack

**Analysis Date:** 2026-07-31

## Languages

**Primary:**
- Python 3.8+ (root `requirements.txt` header) — all application code; runtime image uses Python 3.10 (`deployment/Dockerfile` line 5: `FROM python:3.10-slim`)

**Secondary:**
- HTML/CSS — embedded styling in `app/streamlit_app.py` (custom CSS + `st.markdown(unsafe_allow_html=True)`)
- TOML — Streamlit config (`.streamlit/config.toml`, `deployment/streamlit_config.toml`)
- JSON — Telegram chat export sample data (`data/sample_chats/telegram_sample.json`)
- Markdown — documentation (`README.md`, `changelog.md`)
- Jupyter Notebook — exploratory development notebooks (`notebooks/01_data_parsing.ipynb` through `notebooks/07_final_integration.ipynb`)

## Runtime

**Environment:**
- Streamlit server (web runtime), port 8501, headless mode — `.streamlit/config.toml`
- Python 3.10 in container image — `deployment/Dockerfile`
- Target: Streamlit Cloud (README links `https://chat-analyzer-pro-sujoy.streamlit.app/`), plus Docker and Heroku configurations

**Package Manager:**
- pip
- Lockfile: **missing** — no `requirements-lock.txt`, `poetry.lock`, or `Pipfile.lock`; root `.gitignore` even comments out lock-file lines (lines 248-250)
- Two requirements files: `requirements.txt` (root, dev-oriented) and `deployment/requirements.txt` (production)

## Frameworks

**Core:**
- Streamlit >=1.28.0 — single-page web UI, session state, caching (`app/streamlit_app.py`)
- pandas >=2.0.0 — all data handling (DataFrames are the core data structure passed between modules)
- numpy >=1.24.0 — numeric computation

**Testing:**
- unittest (Python standard library) — `tests/test_parser.py`, `tests/test_end_to_end.py` use `unittest.TestCase`
- pytest >=7.4.0, pytest-cov >=4.1.0, unittest-xml-reporting >=3.2.0 — declared in `deployment/requirements.txt` (commented out in root `requirements.txt`)

**Build/Dev:**
- Jupyter >=1.0.0, ipython >=8.14.0, notebook >=7.0.0 — `deployment/requirements.txt`
- black >=23.7.0, flake8 >=6.1.0, pylint >=2.17.0, mypy >=1.5.0 — code quality (deployment only)
- sphinx >=7.1.0, sphinx-rtd-theme >=1.3.0 — docs (deployment only)
- Docker — `deployment/Dockerfile` (multi-stage build)

## Key Dependencies

**Critical:**
- streamlit >=1.28.0 — the entire UI (`app/streamlit_app.py`)
- pandas >=2.0.0 — parsers, analysis, reporting all operate on DataFrames
- matplotlib >=3.7.0 + seaborn >=0.12.0 + plotly >=5.15.0 — all visualizations (`app/streamlit_app.py`, `src/utils/visualization.py`)
- reportlab >=4.0.0 — PDF report generation (`src/reporting/pdf_report.py`)
- vaderSentiment >=3.3.2 + nltk >=3.8.0 — sentiment analysis (`src/analysis/sentiment.py`); NLTK data (vader_lexicon, punkt, stopwords) downloaded at build time in `deployment/Dockerfile` line 29

**Infrastructure:**
- networkx >=3.1 — conversation network analysis (`src/analysis/network_graph.py`)
- transformers >=4.30.0 + torch >=2.0.0 — heavy NLP (emotion classification `src/analysis/emotion.py`, summarization `src/analysis/summarizer.py`, optional HF sentiment `src/analysis/sentiment.py`); **included** in `deployment/requirements.txt`, **commented out** in root `requirements.txt` (lines 38-39)
- requests >=2.31.0 — GitHub raw module fetch (`app/streamlit_app.py`), Telegram Bot API (`src/reporting/weekly_digest.py`)
- Pillow >=10.0.0 — image handling (`app/streamlit_app.py`, `src/ingest/ingestion.py`)
- scipy >=1.11.0, python-dateutil >=2.8.2, pytz >=2023.3, regex >=2023.8.8, tqdm >=4.66.0, emoji >=2.8.0, wordcloud >=1.9.0, python-dotenv >=1.0.0 — utilities
- streamlit-option-menu >=0.3.6, pydantic >=2.0.0, gunicorn >=21.2.0, python-multipart >=0.0.6, scikit-learn >=1.3.0 — deployment-only extras (`deployment/requirements.txt`)

**Optional / code-imported (NOT in requirements files — runtime feature detection):**
- pytesseract — OCR (`src/ingest/ingestion.py` lines 49-53); system package `tesseract-ocr` in `apt.txt`
- pdfplumber — PDF text extraction (`src/ingest/ingestion.py` lines 55-59)
- pdf2image — PDF→image for OCR fallback (`src/ingest/ingestion.py` lines 61-65); `poppler-utils` in `apt.txt`
- textblob — secondary sentiment (`src/analysis/sentiment.py` lines 17-22)
- These are probed with try/except ImportError and expose `DEPENDENCIES` status dict in `src/ingest/ingestion.py`

## Configuration

**Environment:**
- No `.env` file present in repo (`.env` is gitignored — `.gitignore` line 91)
- `python-dotenv` is declared but no code calls `load_dotenv()` — credentials are passed as plain dicts (`src/reporting/weekly_digest.py` `create_digest_bot()`)
- Runtime env vars in Docker: `PYTHONPATH=/app`, `STREAMLIT_SERVER_PORT=8501`, `STREAMLIT_SERVER_ADDRESS=0.0.0.0`, `STREAMLIT_SERVER_HEADLESS=true` (`deployment/Dockerfile` lines 45-48)
- Heroku assigns `$PORT` dynamically (`deployment/Procfile` line 5)

**Build:**
- `deployment/Dockerfile` — multi-stage: `python:3.10-slim` base → deps → app; copies `deployment/requirements.txt`, installs NLTK data, copies `src/`, `app/`, `tests/`, `.streamlit/`
- `deployment/Procfile` — Heroku web process runs `streamlit run app/streamlit_app.py`
- `deployment/streamlit_config.toml` — alternate Streamlit config reference copy
- `.streamlit/config.toml` — active Streamlit config (theme, server, upload limits `maxUploadSize = 400`, logging, `dataFrameSerialization = "arrow"`)
- `apt.txt` — Streamlit Cloud system packages: tesseract-ocr (+ eng, script-latn), poppler-utils, libgl1-mesa-glx, libglib2.0-0, libgomp1, libtesseract-dev, libleptonica-dev
- `packages.txt` — additional system packages for matplotlib/image/PDF: libgl1-mesa-glx, libglib2.0-0, libjpeg-dev, libpng-dev, libfreetype6-dev, build-essential

## Platform Requirements

**Development:**
- Python 3.8+ (root `requirements.txt`)
- System libs for matplotlib/OCR (see `apt.txt` / `packages.txt`)
- Jupyter notebooks for exploration

**Production:**
- Docker image `python:3.10-slim` (`deployment/Dockerfile`)
- Streamlit Cloud (`https://chat-analyzer-pro-sujoy.streamlit.app/`) — reads `apt.txt` + `packages.txt`
- Heroku support via `deployment/Procfile`

---

*Stack analysis: 2026-07-31*
