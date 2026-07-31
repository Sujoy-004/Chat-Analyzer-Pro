<!-- refreshed: 2026-07-31 -->
# Architecture

**Analysis Date:** 2026-07-31

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────┐
│                     PRESENTATION — Streamlit App                      │
│                    `app/streamlit_app.py` (1234 lines)               │
│  UI rendering · session state · fallback parsers · chart building    │
└────────────────────────────────────┬─────────────────────────────────┘
                                     │ exec() of fetched source
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│              RUNTIME MODULE FETCHER (GitHub raw + requests)          │
│            `load_github_modules()` in `app/streamlit_app.py`         │
│   downloads 4 modules: ingestion, whatsapp_parser, telegram_parser,  │
│   relationship_health — stored in `st.session_state` as namespaces   │
└────────────────────────────────────┬─────────────────────────────────┘
                                     │ "advanced" path only
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│   LIBRARY LAYER — `src/` (standalone modules, not imported by app)   │
│                                                                      │
│  PARSER          INGEST                    ANALYSIS                  │
│  `src/parser/`   `src/ingest/ingestion.py`  `src/analysis/`          │
│  whatsapp_parser  dispatch by ext +           eda, sentiment,        │
│  telegram_parser  normalize_message()         emotion,               │
│                                                relationship_health,  │
│                                                network_graph,        │
│                                                summarizer            │
│  REPORTING            UTILS                                          │
│  `src/reporting/`     `src/utils/`                                   │
│  pdf_report,          preprocessing,                                 │
│  weekly_digest        visualization                                  │
└────────────────────────────────────┬─────────────────────────────────┘
                                     │ pandas DataFrame (message schema)
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│  DATA CONTRACT — normalized message dict / DataFrame                  │
│  uid · date · time · author · text · source · media · meta            │
│  produced by `normalize_message()` in `src/ingest/ingestion.py`      │
│  consumed by health scoring, charts, and (in theory) reports         │
└──────────────────────────────────────────────────────────────────────┘
```

**Key architectural fact:** the deployed Streamlit app (`app/streamlit_app.py`) does **not** `import` from `src/`. It downloads four `src/` files from `https://raw.githubusercontent.com/Sujoy-004/Chat-Analyzer-Pro/...` at runtime (`load_github_modules()`, `app/streamlit_app.py:42-70`) and executes them via `exec()` into session-state namespaces (`load_and_execute_modules()`, `app/streamlit_app.py:73-111`). Every other `src/` module (eda, sentiment, emotion, network_graph, summarizer, pdf_report, weekly_digest, visualization, preprocessing) is **not wired into the app** — they are library code exercised via notebooks and tests.

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Streamlit UI | Page config, upload widget, module loading status, all chart rendering, welcome/feature screens | `app/streamlit_app.py` |
| Module fetcher | Download 4 src modules from GitHub raw; cache via `@st.cache_data`; record availability flags | `app/streamlit_app.py:42-70` |
| App-side ingestion | Dispatch by extension: txt/json/zip/images/pdf/media; produce DataFrame | `app/streamlit_app.py` `process_uploaded_file()` (line 244) |
| App fallback parsers | Inline WhatsApp regex parser + Telegram JSON parser (used when downloaded modules fail) | `app/streamlit_app.py:165-241` |
| App health scoring | Basic 5-component health score (balance/initiation/response/consistency/engagement) used when advanced module unavailable | `app/streamlit_app.py` `calculate_basic_health_score()` (line 379) |
| Ingestion library | Multi-format file dispatch, ZIP recursion, OCR/PDF/media metadata, normalization to canonical schema | `src/ingest/ingestion.py` |
| WhatsApp parser | Class-based line/date-format parsing, feature engineering (media flags, emoji counts, time periods), CSV save | `src/parser/whatsapp_parser.py` |
| Telegram parser | Function-based JSON export parsing (file path or URL) | `src/parser/telegram_parser.py` |
| EDA | Time-based feature engineering + volume/dynamics/content summary | `src/analysis/eda.py` |
| Sentiment | Multi-engine (VADER/TextBlob/HF) with availability flags + consensus voting | `src/analysis/sentiment.py` |
| Emotion | HF transformer emotion classification (6 classes) with cached global analyzer | `src/analysis/emotion.py` |
| Relationship health | Conversation starters, initiator ratio, response patterns, dominance, weighted health score, rolling score, gamification (friendship index, streaks, emoji personality, milestones) | `src/analysis/relationship_health.py` |
| Network graph | NetworkX directed interaction graph, centrality metrics, community detection | `src/analysis/network_graph.py` |
| Summarizer | T5-small abstractive summarization, group dynamics, periodic summaries | `src/analysis/summarizer.py` |
| PDF reports | ReportLab document with gauge/radar/pie charts and recommendations | `src/reporting/pdf_report.py` |
| Weekly digest | Email (SMTP) + Telegram bot delivery of scheduled summaries | `src/reporting/weekly_digest.py` |
| Preprocessing | Text cleaning, emoji/URL extraction, tokenization helpers | `src/utils/preprocessing.py` |
| Visualization | Matplotlib/Seaborn chart library (timeline, heatmap, wordcloud, sentiment, gauges, dashboards) | `src/utils/visualization.py` |

## Pattern Overview

**Overall:** Two parallel worlds: a **library layer** (`src/`) built module-by-module (days 1–15, tracked in `changelog.md`) and a **self-contained app layer** (`app/`) that dynamically pulls a subset of that library from GitHub at runtime, with built-in fallbacks so it degrades gracefully to "basic mode" when the network fetch fails.

**Key Characteristics:**
- Function-style modules: most of `src/` exports standalone functions taking/returning `pd.DataFrame` or `Dict` — only a few class abstractions (`WhatsAppParser`, `ChatEDA`, `EmotionAnalyzer`, `ConversationSummarizer`, `ChatAnalysisPDFGenerator`, `WeeklyDigestBot`, `ChatVisualizer`)
- DataFrame-centric: every parser normalizes to a message DataFrame; every analysis function consumes one; columns `datetime`, `sender`, `message` are the de-facto interface
- Dependency-drifting: heavy/optional dependencies (transformers, torch, pytesseract, pdfplumber, textblob) are probed with `try/except ImportError` and features degrade at runtime (`DEPENDENCIES` dict in `src/ingest/ingestion.py:34-65`, `*_AVAILABLE` flags in `src/analysis/sentiment.py:10-29`)
- Fallback-first UX: the app always tries "advanced" (downloaded module) then "basic" (inline fallback) and surfaces which mode ran via `processing_method`
- `if __name__ == "__main__":` demo/example blocks in most modules (e.g., `src/parser/whatsapp_parser.py:257`, `src/ingest/ingestion.py:639`, `src/analysis/sentiment.py:436`, `src/analysis/relationship_health.py:1174`, `src/reporting/pdf_report.py:616`)

## Layers

**Presentation (app):**
- Purpose: Streamlit UI — upload, process, score, chart
- Location: `app/streamlit_app.py`
- Contains: `main()` + ~20 helper functions; embedded CSS in `load_css()`; all Plotly chart construction inline
- Depends on: network (GitHub raw fetch), streamlit, pandas, plotly, matplotlib/seaborn (imported but charts use Plotly)
- Used by: Streamlit runtime (`streamlit run app/streamlit_app.py`; also `deployment/Dockerfile`, `deployment/Procfile`)

**Ingestion (library):**
- Purpose: Any supported file → normalized message list + media analysis
- Location: `src/ingest/ingestion.py` (+ app-side twin `process_uploaded_file()` in `app/streamlit_app.py:244`)
- Contains: extension dispatch (`process_uploaded_file`), per-type private processors (`_process_zip_file`, `_process_text_file`, `_process_json_file`, `_process_image_file`, `_process_pdf_file`, `_process_media_file`, `_process_unknown_file`), normalization (`normalize_message`)
- Depends on: stdlib (zipfile, json, uuid, logging), optional PIL/pytesseract/pdfplumber/pdf2image
- Used by: only itself (notebooks, `__main__` demo) and the app via runtime `exec()`

**Parsers (library):**
- Purpose: Raw chat exports → message DataFrame
- Location: `src/parser/whatsapp_parser.py`, `src/parser/telegram_parser.py`
- Contains: `WhatsAppParser` class (regex line parsing, multiline continuation, feature columns), `parse_telegram_chat()` function (JSON from file or URL)
- Depends on: re, pandas, datetime (+ requests for URL input)
- Used by: notebooks, tests; whatsapp_parser + telegram_parser fetched by the app at runtime (though the app's `process_uploaded_file` calls only `ingestion`, whose `_process_text_file`/`_process_json_file` replicate parsing internally)

**Analysis (library):**
- Purpose: Message DataFrame → insight dicts / enriched DataFrames / figures
- Location: `src/analysis/` (eda.py, sentiment.py, emotion.py, relationship_health.py, network_graph.py, summarizer.py)
- Contains: metric functions and small classes; `analyze_relationship_health()` (`src/analysis/relationship_health.py:1071`) is the orchestrator that chains starters → initiator → response → dominance → weighted score → gamification
- Depends on: pandas, numpy, matplotlib, seaborn, networkx, optional vaderSentiment/textblob/transformers
- Used by: notebooks, tests; only relationship_health.py fetched by the app at runtime

**Reporting (library):**
- Purpose: Analysis dict → PDF file / email / Telegram message
- Location: `src/reporting/pdf_report.py`, `src/reporting/weekly_digest.py`
- Contains: `ChatAnalysisPDFGenerator` (ReportLab story builder), `WeeklyDigestBot` (SMTP + Telegram Bot API, scheduling via `schedule_weekly_digest()` at line 510)
- Depends on: reportlab, matplotlib, smtplib, requests
- Used by: notebooks, tests only (not wired into the app)

**Utils (library):**
- Purpose: Shared text + chart helpers
- Location: `src/utils/preprocessing.py`, `src/utils/visualization.py`
- Contains: `preprocess_text`, `clean_messages`, `extract_emojis`, ... ; `ChatVisualizer` (12 plot methods + `create_summary_dashboard`)
- Depends on: re, pandas, matplotlib, seaborn, wordcloud
- Used by: `src/analysis/relationship_health.py:800` (`from src.utils.visualization import ChatVisualizer`), notebooks

**Tests:**
- Purpose: Unit + integration coverage of the library layer
- Location: `tests/test_parser.py`, `tests/test_analysis.py`, `tests/test_reporting.py`, `tests/test_end_to_end.py`
- Contains: `unittest.TestCase` classes; each file has a `run_*_tests()` runner + `unittest.main()`
- Depends on: unittest, pandas, numpy (self-contained — **tests re-implement mock parsers instead of importing `src/`**)

## Data Flow

### Primary Request Path (deployed app)

1. User uploads file via `st.sidebar.file_uploader` (`app/streamlit_app.py:587`)
2. On first run, `load_and_execute_modules()` fetches and `exec()`s the 4 src modules from GitHub raw into `st.session_state` (`app/streamlit_app.py:73-111`)
3. `process_uploaded_file()` (`app/streamlit_app.py:244`) tries `ingestion["process_uploaded_file"](uploaded_file)` → normalized message dicts → `convert_normalized_messages_to_df()` (`app/streamlit_app.py:313`) → DataFrame with columns `datetime, sender, message, date, time, hour, message_length, source, uid`
4. On advanced failure, falls back to inline `fallback_whatsapp_parser()` / `fallback_telegram_parser()` (`app/streamlit_app.py:165-241`) → same DataFrame shape, `source` = `whatsapp_fallback` / `telegram_fallback`
5. `calculate_relationship_health()` (`app/streamlit_app.py:348`) calls downloaded `relationship_health["analyze_relationship_health"](df)` → `health_score` dict; on failure, `calculate_basic_health_score()` (`app/streamlit_app.py:379`) computes a 5-point weighted score
6. Chart-building blocks render Plotly figures (pie, bar, line, heatmap, weekday, participant timeline, source analysis) directly in `main()`; `method` flag switches between advanced vs basic insights sections (`app/streamlit_app.py:948-987`)
7. `display_media_results()` (`app/streamlit_app.py:493`) shows OCR/media notes for non-message files

### Library Pipeline (notebooks / tests)

1. `parse_whatsapp_chat()` / `parse_telegram_chat()` → DataFrame (`src/parser/*`)
2. `ChatEDA(df).generate_comprehensive_summary()` or `add_sentiment_analysis(df)` or `EmotionAnalyzer().analyze_emotions(df)` (`src/analysis/*`)
3. `analyze_relationship_health(df)` → nested dict (`src/analysis/relationship_health.py:1071`)
4. `generate_chat_analysis_pdf(results)` → PDF (`src/reporting/pdf_report.py:534`) or `WeeklyDigestBot` → email/Telegram (`src/reporting/weekly_digest.py:557` `create_digest_bot()`)

**State Management:**
- Streamlit session state for module namespaces and availability flags: `modules_loaded`, `executed_modules`, `module_{name}` namespaces, `ingestion_available`, etc. (`app/streamlit_app.py:33-40`, `92-105`)
- `@st.cache_data` on `load_github_modules()` so the network fetch runs once per session (`app/streamlit_app.py:42`)
- Module-level singleton analyzers to avoid reloading ML models: `_vader_analyzer` / `_hf_analyzer` (`src/analysis/sentiment.py:39-40`), `_emotion_analyzer` / `_emotion_model_loaded` (`src/analysis/emotion.py:29-30`)
- No persistence layer — data lives only in the in-memory DataFrame for the request/session

## Key Abstractions

**Normalized Message Contract:**
- Purpose: canonical schema every parser/ingester produces; defined in `normalize_message()` (`src/ingest/ingestion.py:323-380`)
- Fields: `uid`, `date` ("YYYY-MM-DD"), `time` ("HH:MM"), `author`, `text`, `source`, `media`, `meta`
- DataFrame projection used by the app: `datetime, sender, message, date, time, hour, message_length, source, uid` (`convert_normalized_messages_to_df()`, `app/streamlit_app.py:313`)
- Pattern: adapter/normalizer — heterogeneous inputs (WhatsApp lines, Telegram JSON, OCR text) → uniform rows

**Analysis Results Dict:**
- Purpose: the output contract of the health pipeline; consumed by PDF generator and app
- Shape: `{'conversation_stats', 'initiator_analysis', 'response_analysis', 'dominance_analysis', 'health_score': {'overall_health_score', 'grade', 'component_scores', 'strengths', 'areas_for_improvement'}, 'friendship_index', 'streaks', 'milestones', 'emoji_personality', 'rolling_health', 'prepared_data'}` (`src/analysis/relationship_health.py:1101-1128`)
- Note: `deployment` app's basic mode produces a **different, flatter** dict (`method`, `total_score`, `grade`, `*_points`, `message_counts`, `response_df`, `initiators_df` — `app/streamlit_app.py:479-491`), so report generators written for one shape don't consume the other without adaptation

**Optional-Dependency Gates:**
- Purpose: graceful degradation when heavy libs absent
- Examples: `DEPENDENCIES` dict (`src/ingest/ingestion.py:34-65`), `VADER_AVAILABLE`/`TEXTBLOB_AVAILABLE`/`TRANSFORMERS_AVAILABLE` (`src/analysis/sentiment.py:10-29`), `try: from src.utils.visualization import ChatVisualizer / except ImportError` fallback to `_plot_original_dashboard` (`src/analysis/relationship_health.py:798-863`)

## Entry Points

**Streamlit App:**
- Location: `app/streamlit_app.py`
- Triggers: `streamlit run app/streamlit_app.py` (per `deployment/Dockerfile` CMD and `deployment/Procfile`)
- Responsibilities: `main()` (line 550) orchestrates everything; `if __name__ == "__main__": main()` at line 1233

**Standalone module demos (each has `if __name__ == "__main__":`):**
- `src/parser/whatsapp_parser.py:257` — parses `data/sample_chats/whatsapp_sample.txt`
- `src/ingest/ingestion.py:639` — prints dependency status; accepts a file path argv
- `src/analysis/sentiment.py:436` — prints usage banner
- `src/analysis/relationship_health.py:1174` — runs `example_usage()` (synthetic Alice/Bob data)
- `src/reporting/pdf_report.py:616` — generates `example_chat_report.pdf` from sample dict
- `src/reporting/weekly_digest.py` — `create_digest_bot()` / `send_quick_digest()` (line 557/599)

**Test runners:**
- `tests/test_parser.py:314` `run_parser_tests()`, `tests/test_analysis.py:381` `run_analysis_tests()`, `tests/test_reporting.py:406` `run_reporting_tests()`, `tests/test_end_to_end.py:628` `run_end_to_end_tests()`; each file also runs via `unittest.main()`

**Scheduled automation:**
- `WeeklyDigestBot.schedule_weekly_digest()` (`src/reporting/weekly_digest.py:510`) — in-module scheduler for recurring digests

## Architectural Constraints

- **Threading:** Single-threaded Streamlit server process (one Python event loop per session). No threading, async, or multiprocessing used anywhere. Model loads (`transformers` pipelines) block the request thread (`src/analysis/emotion.py`, `src/analysis/summarizer.py`).
- **Global state:** Module-level singletons — `_vader_analyzer`/`_hf_analyzer` (`src/analysis/sentiment.py:39-40`), `_emotion_analyzer`/`_emotion_model_loaded` (`src/analysis/emotion.py:29-30`); Streamlit session-state module namespaces (`app/streamlit_app.py:92-105`); `DEPENDENCIES` module dict (`src/ingest/ingestion.py:34`). `logging.basicConfig(level=logging.INFO)` called at import time in `src/analysis/relationship_health.py:24`, `src/reporting/weekly_digest.py:23`, `src/utils/visualization.py:19` (reconfigured per-module when imported).
- **Package integrity:** `src/` uses `_init_.py` (underscore) files — **not** `__init__.py` — so the directories only work as PEP 420 namespace packages, and the relative re-export imports inside those `_init_.py` files (e.g., `from .relationship_health import ...` in `src/analysis/_init_.py:16`) never execute. Code that imports `src.*` works only when the repo root is on `sys.path` (Docker sets `PYTHONPATH=/app` in `deployment/Dockerfile:48`; ad-hoc scripts rely on CWD).
- **App/library decoupling:** `app/streamlit_app.py` must never import `src/` directly (it fetches raw source instead); any new module to be used by the deployed app must be added to `module_urls` in `load_github_modules()` (`app/streamlit_app.py:46-51`).
- **Runtime network dependency:** advanced mode requires HTTPS access to `raw.githubusercontent.com`; on failure the app silently degrades to inline fallbacks and reports "Basic mode only".

## Anti-Patterns

### Remote code download + `exec()`

**What happens:** `load_github_modules()` fetches Python source over HTTP and runs it with `exec(code, namespace)` (`app/streamlit_app.py:42-111`).
**Why it's wrong:** Executing code from a remote URL at runtime is a supply-chain risk (if the repo/URL is compromised, every app session executes attacker code), is untransparent, and defeats static analysis/import tooling. The downloaded modules are the same files already in the repo.
**Do this instead:** Import `src/ingest/ingestion.py`, `src/parser/*`, `src/analysis/relationship_health.py` normally (or vendor them under `app/`), and drop the fetch/exec machinery. All needed source ships in the deployed artifact (`deployment/Dockerfile` copies `src/` already).

### Duplicated logic between app and library

**What happens:** The app re-implements what `src/` already provides: inline `fallback_whatsapp_parser`/`fallback_telegram_parser` (`app/streamlit_app.py:165-241`) vs `src/parser/*`; `calculate_basic_health_score` (`app/streamlit_app.py:379`) vs `src/analysis/relationship_health.py`; all charts inline vs `src/utils/visualization.py`; file dispatch in `process_uploaded_file` (`app/streamlit_app.py:244`) vs `src/ingest/ingestion.py:399`.
**Why it's wrong:** Two implementations of the same logic drift (the app's basic health dict shape differs from the library's `health_score` dict), doubling maintenance and test surface.
**Do this instead:** Single implementation in `src/` (or `app/`), imported by both the interactive app and the notebooks; keep only genuine degradation fallbacks, not full parallel implementations.

### Self-contained mock tests that never exercise library code

**What happens:** `tests/test_parser.py` and `tests/test_end_to_end.py` define private `_parse_whatsapp_file`/`_parse_telegram_json` mock parsers (e.g., `tests/test_parser.py:135-168`, `237-261`) and assert against them instead of importing `src/parser/*`.
**Why it's wrong:** Green test suites can pass while the real parsers are broken; the "88+ tests" in `changelog.md` do not actually validate the shipped modules.
**Do this instead:** Import and exercise `WhatsAppParser.parse_file`, `parse_telegram_chat`, `process_uploaded_file`, etc. directly, keeping mocks only for external IO (network, filesystem).

### Divergent requirements files

**What happens:** Root `requirements.txt` (comment-out heavy, `transformers`/`torch` disabled, lines 38-39) vs `deployment/requirements.txt` (everything enabled) vs app code expecting more (pytesseract/pdfplumber probed but absent from both).
**Why it's wrong:** Dev, test, and prod environments resolve different dependency sets; optional features silently turn on/off depending on which file was used.
**Do this instead:** One canonical requirements file (or a `requirements-{dev,prod}.txt` split with a shared base) that matches what the code actually imports.

## Error Handling

**Strategy:** Defensive per-call `try/except` with graceful degradation at every layer; the app warns (`st.warning`) then falls back to basic implementations instead of failing.

**Patterns:**
- Per-module fallback: advanced module call wrapped in try/except → inline fallback (`app/streamlit_app.py:250-258`, `354-374`)
- Optional import gating: `try: import pytesseract / except ImportError: DEPENDENCIES['pytesseract'] = False` (`src/ingest/ingestion.py:49-53`); same pattern for sentiment engines
- Per-row resilience: normalization failures produce fallback rows with `_normalization_error: True` instead of aborting (`src/ingest/ingestion.py:454-470`); parse failures return empty DataFrames/None rather than raising
- User-facing errors: `st.error` + `st.exception(e)` on the file-processing path (`app/streamlit_app.py:1078-1080`)
- Logging: mixed — `logger = logging.getLogger(__name__)` + `NullHandler` in `src/ingest/ingestion.py:30-31`; `logging.basicConfig(level=logging.INFO)` + bare `print()`/`print(f"✅ ...")` in analysis/reporting modules

## Cross-Cutting Concerns

**Logging:** No shared logger configuration. `src/ingest/ingestion.py` uses a NullHandler logger; `relationship_health.py`, `weekly_digest.py`, `visualization.py` call `logging.basicConfig(level=logging.INFO)` at import; sentiment/emotion/summarizer and the app rely on `print()`. `.streamlit/config.toml` sets `[logger] level = "info"`.

**Validation:** Minimal. Message fields are coerced/backfilled with `or` fallbacks and `try/except` in `normalize_message()` (`src/ingest/ingestion.py:323-380`); no schema validation library in the root requirements (pydantic only in `deployment/requirements.txt`); `tests/test_end_to_end.py` `TestDataValidation` (line 579) covers validation scenarios against mock data only.

**Authentication:** None in the app. `WeeklyDigestBot` takes `smtp_server`/`sender_password`/`bot_token`/`chat_id` as plain dict values passed to `create_digest_bot()` (`src/reporting/weekly_digest.py:557`); `python-dotenv` is declared but no `load_dotenv()` call exists.

---

*Architecture analysis: 2026-07-31*
