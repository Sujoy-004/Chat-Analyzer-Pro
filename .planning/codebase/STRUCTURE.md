# Codebase Structure

**Analysis Date:** 2026-07-31

## Directory Layout

```
Chat-Analyzer-Pro/
├── app/                    # Streamlit application (deployed entry point)
│   ├── streamlit_app.py    # Single-file UI + logic (1234 lines)
│   └── assets/             # logo.png, style.css
├── src/                    # Library layer (parsers, analysis, reporting, utils)
│   ├── _init_.py           # NOTE: misnamed package marker (see Special Directories)
│   ├── parser/             # whatsapp_parser.py, telegram_parser.py
│   ├── ingest/             # ingestion.py (multi-format ingestion)
│   ├── analysis/           # eda, sentiment, emotion, relationship_health,
│   │                       # network_graph, summarizer
│   ├── reporting/          # pdf_report.py, weekly_digest.py
│   └── utils/              # preprocessing.py, visualization.py
├── tests/                  # unittest suites (4 files, self-contained)
│   ├── test_parser.py
│   ├── test_analysis.py
│   ├── test_reporting.py
│   └── test_end_to_end.py
├── notebooks/              # Development notebooks 01–07 (day-by-day exploration)
├── data/                   # Sample chats + processed outputs
│   ├── sample_chats/       # whatsapp_sample.txt, telegram_sample.json
│   └── processed/          # example_parsed.csv (generated)
├── deployment/             # Docker, Heroku, prod requirements, config reference
│   ├── Dockerfile
│   ├── Procfile
│   ├── requirements.txt
│   └── streamlit_config.toml
├── .streamlit/config.toml  # Active Streamlit config (theme, server, upload limits)
├── requirements.txt        # Root (dev-oriented) dependencies
├── apt.txt                 # Streamlit Cloud system packages (tesseract, poppler, ...)
├── packages.txt            # Additional system packages
├── changelog.md            # Day-by-day build log (Days 1–15)
├── README.md               # Minimal (deployed app URL only)
├── .gitignore
└── .planning/              # GSD planning artifacts
    └── codebase/           # Codebase map docs (STACK.md, ARCHITECTURE.md, STRUCTURE.md)
```

## Directory Purposes

**`app/`:**
- Purpose: The shipped web application. Single-file Streamlit app with UI, runtime module fetching, fallback parsers, health scoring, and all Plotly chart code
- Contains: `streamlit_app.py` (only Python file), `assets/logo.png`, `assets/style.css`
- Key files: `app/streamlit_app.py`
- Entry point for all deployments (`deployment/Dockerfile` CMD, `deployment/Procfile`)

**`src/parser/`:**
- Purpose: Chat export parsers producing message DataFrames
- Contains: `whatsapp_parser.py` (`WhatsAppParser` class + `parse_whatsapp_chat()`), `telegram_parser.py` (`parse_telegram_chat()`)
- Key files: `src/parser/whatsapp_parser.py`, `src/parser/telegram_parser.py`

**`src/ingest/`:**
- Purpose: Multi-format file ingestion — TXT/JSON/ZIP/images/PDF/media → normalized messages
- Contains: `ingestion.py` (dispatch, OCR, PDF extraction, normalization)
- Key files: `src/ingest/ingestion.py`

**`src/analysis/`:**
- Purpose: All chat analytics — EDA, sentiment, emotion, relationship health, network graphs, summarization
- Contains: `eda.py`, `sentiment.py`, `emotion.py`, `relationship_health.py`, `network_graph.py`, `summarizer.py`
- Key files: `src/analysis/relationship_health.py` (largest, 1175 lines; the orchestrator)

**`src/reporting/`:**
- Purpose: Output artifacts — PDF reports and automated weekly digests (email/Telegram)
- Contains: `pdf_report.py`, `weekly_digest.py`
- Key files: `src/reporting/pdf_report.py`

**`src/utils/`:**
- Purpose: Shared helpers — text preprocessing and chart generation
- Contains: `preprocessing.py`, `visualization.py` (`ChatVisualizer`)
- Key files: `src/utils/visualization.py` (715 lines, largest module in `src/`)

**`tests/`:**
- Purpose: Unit + integration tests (unittest framework)
- Contains: `test_parser.py`, `test_analysis.py`, `test_reporting.py`, `test_end_to_end.py`
- Key files: all four; each is runnable standalone (`python tests/test_parser.py`)

**`notebooks/`:**
- Purpose: Day-by-day exploratory development (Days 1–7 per `changelog.md`)
- Contains: `01_data_parsing.ipynb` → `07_final_integration.ipynb` (07 is a 2-byte stub)
- Note: not included in Docker image; they are dev artifacts only

**`data/`:**
- Purpose: Sample chat exports + generated processed output
- Contains: `sample_chats/whatsapp_sample.txt`, `sample_chats/telegram_sample.json`, `processed/example_parsed.csv` (gitignored pattern `data/processed/*.csv`)
- Note: `.gitignore` expects a `data/raw/` directory that does not exist in the repo

**`deployment/`:**
- Purpose: Production packaging — Docker multi-stage build, Heroku Procfile, full prod requirements, config reference
- Contains: `Dockerfile`, `Procfile`, `requirements.txt`, `streamlit_config.toml`
- Key files: `deployment/Dockerfile` (sets `PYTHONPATH=/app`, installs NLTK data, healthcheck)

**`.streamlit/`:**
- Purpose: Active Streamlit server/theme config
- Contains: `config.toml` (maxUploadSize = 400, purple `#667eea` theme, `dataFrameSerialization = "arrow"`)

## Key File Locations

**Entry Points:**
- `app/streamlit_app.py`: Streamlit app — `main()` at line 550, `if __name__ == "__main__"` at line 1233
- `deployment/Dockerfile`: `CMD ["streamlit", "run", "app/streamlit_app.py", ...]`
- `deployment/Procfile`: `web: streamlit run app/streamlit_app.py --server.port=$PORT ...`
- Per-module demo entries (`if __name__ == "__main__":`): `src/parser/whatsapp_parser.py:257`, `src/ingest/ingestion.py:639`, `src/analysis/sentiment.py:436`, `src/analysis/relationship_health.py:1174`, `src/reporting/pdf_report.py:616`

**Configuration:**
- `.streamlit/config.toml`: active Streamlit config
- `deployment/streamlit_config.toml`: reference copy for manual setups
- `requirements.txt` (root) vs `deployment/requirements.txt` (prod) — two divergent manifests
- `apt.txt` + `packages.txt`: Streamlit Cloud system packages (tesseract-ocr, poppler-utils, libgl1-mesa-glx, ...)

**Core Logic:**
- File dispatch + normalization: `src/ingest/ingestion.py` (`process_uploaded_file` line 399, `normalize_message` line 323)
- Parsers: `src/parser/whatsapp_parser.py`, `src/parser/telegram_parser.py`
- Health pipeline: `src/analysis/relationship_health.py` (`analyze_relationship_health` line 1071)
- App fallback logic: `app/streamlit_app.py` (fallback parsers line 165, `process_uploaded_file` line 244, `calculate_basic_health_score` line 379)

**Testing:**
- `tests/test_parser.py`, `tests/test_analysis.py`, `tests/test_reporting.py`, `tests/test_end_to_end.py` — runnable with `python -m unittest tests.test_parser` or directly as scripts

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (`whatsapp_parser.py`, `relationship_health.py`, `preprocessing.py`)
- Test files: `test_<area>.py` (`test_parser.py`, `test_analysis.py`, `test_reporting.py`, `test_end_to_end.py`)
- Notebooks: `NN_<topic>.ipynb` (`01_data_parsing.ipynb`, `02_exploratory_analysis.ipynb`)
- Config: lowercase (`config.toml`, `requirements.txt`, `apt.txt`, `packages.txt`)
- **Exception (known quirk):** package marker files are `_init_.py` instead of `__init__.py` in every `src/` subdirectory

**Directories:**
- Lowercase, singular/plural by role: `parser`, `ingest`, `analysis`, `reporting`, `utils`, `tests`, `notebooks`, `data`, `deployment`, `app`
- Under `src/`: purpose-named packages

**Classes:**
- PascalCase with domain noun: `WhatsAppParser`, `ChatEDA`, `EmotionAnalyzer`, `ConversationSummarizer`, `ChatAnalysisPDFGenerator`, `WeeklyDigestBot`, `ChatVisualizer`, `SentimentConfig`
- Test classes: `Test<Subject>(unittest.TestCase)` (`TestWhatsAppParser`, `TestEDAModule`, `TestCompletePipeline`)

**Functions:**
- snake_case, verb-first: `parse_whatsapp_chat`, `process_uploaded_file`, `normalize_message`, `analyze_relationship_health`, `generate_chat_analysis_pdf`, `create_digest_bot`
- Private helpers prefixed `_`: `_process_zip_file`, `_add_features`, `_format_file_size`, `_ocr_pdf_page`, `_categorize_time`
- Module-level convenience wrappers use `quick_*` prefix: `quick_sentiment_analysis`, `quick_summarize`, `quick_timeline`, `quick_dashboard`

**Variables:**
- snake_case; DataFrames commonly `df`; parsed rows use singular nouns (`msg`, `message`, `sender`); health metrics use `*_points` (app basic mode) or `*_score` (library mode)

## Where to Add New Code

**New Feature (analysis):**
- Primary code: `src/analysis/<feature>.py` — write functions that take a message DataFrame (`datetime`, `sender`, `message` columns) and return `Dict` results, matching the style of `src/analysis/network_graph.py`
- To surface in the deployed app: add the file to `module_urls` in `load_github_modules()` (`app/streamlit_app.py:46-51`) AND add a call site in `main()`; otherwise the app won't use it
- Notebook: `notebooks/08_<feature>.ipynb` for exploration

**New Parser:**
- Implementation: `src/parser/<format>_parser.py` returning a DataFrame with the standard columns (`datetime`, `sender`, `message`, `message_length`, `hour`, ...)
- Ingestion hookup: add a dispatch branch in `process_uploaded_file()` (`src/ingest/ingestion.py:419-446`) and a private `_process_<format>_file()`; add the extension to the app's accepted types list (`app/streamlit_app.py:581`)

**New Report Type:**
- Implementation: `src/reporting/<report>_report.py`, following `src/reporting/pdf_report.py` (class with `generate_report()` + module-level convenience function)

**New Utility:**
- Shared helpers: `src/utils/` (extend `preprocessing.py` with functions, or add a new module; re-export in `src/utils/_init_.py` if the package marker naming is ever fixed)

**New Tests:**
- Tests: `tests/test_<area>.py` using `unittest.TestCase`; add a `run_<area>_tests()` runner. Prefer importing the real `src/` modules over the mock-parser pattern currently used in `tests/test_parser.py` and `tests/test_end_to_end.py`

**UI/Chart changes:**
- `app/streamlit_app.py` — all charts are built inline in `main()` with Plotly; the library counterpart `src/utils/visualization.py` uses Matplotlib/Seaborn and is not consumed by the app

## Special Directories

**`src/` package markers (`_init_.py`):**
- Purpose: intended as package initializers with metadata + re-exports (`src/_init_.py`, `src/analysis/_init_.py`, `src/parser/_init_.py`, `src/reporting/_init_.py`, `src/utils/_init_.py`)
- Generated: No
- Committed: Yes
- **Critical caveat:** they are named `_init_.py`, not `__init__.py`, so Python does not treat `src/` subdirectories as regular packages. Imports like `from src.analysis.relationship_health import ...` only work via PEP 420 namespace-package behavior when the repo root is on `sys.path` (Docker sets `PYTHONPATH=/app`, `deployment/Dockerfile:48`). Renaming them to `__init__.py` would make the relative re-export imports in those files start executing — verify them before doing so (e.g., `src/analysis/_init_.py:16` imports `plot_relationship_health_dashboard` which is not defined in `src/analysis/relationship_health.py` — the actual name is `plot_relationship_health_dashboard_enhanced`)

**`data/processed/`:**
- Purpose: CSV output of the parser demo (`data/processed/example_parsed.csv`)
- Generated: Yes (by running `src/parser/whatsapp_parser.py` as a script)
- Committed: Yes, but gitignored (`data/processed/*.csv` in `.gitignore`) — currently tracked as an artifact

**`notebooks/`:**
- Purpose: Day-by-day development history (Days 1–7 per `changelog.md`); `07_final_integration.ipynb` is a 2-byte stub
- Generated: No
- Committed: Yes

**`.planning/`:**
- Purpose: GSD planning artifacts (codebase maps, plans, phase docs)
- Generated: Yes (by GSD workflows)
- Committed: Yes

**`deployment/`:**
- Purpose: Production packaging; the Docker image copies only `src/`, `app/`, `tests/`, `.streamlit/`, and `deployment/requirements.txt` — `notebooks/`, `data/`, and `deployment/streamlit_config.toml` are not copied
- Generated: No
- Committed: Yes

---

*Structure analysis: 2026-07-31*
