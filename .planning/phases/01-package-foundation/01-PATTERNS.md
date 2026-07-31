# Phase 1: Package Foundation - Pattern Map

**Mapped:** 2026-07-31
**Files analyzed:** 22 actions (5 new, 12 modified/moved, 5 deleted)
**Analogs found:** 16 / 22 (pyproject.toml + 5 deletions have no code analog)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `pyproject.toml` (new) | config | build | none (research-backed; deps from `requirements.txt`) | no-analog |
| `src/chat_analyzer/__init__.py` (rename from `src/_init_.py`) | config (marker) | — | `src/_init_.py` itself (content already valid, keep) | exact (self) |
| `src/chat_analyzer/analysis/__init__.py` (rename) | config (marker) | — | `src/analysis/_init_.py` itself (must strip broken re-exports) | exact (self) |
| `src/chat_analyzer/parser/__init__.py` (rename) | config (marker) | — | `src/parser/_init_.py` itself (fix `parse_telegram_json` → `parse_telegram_chat`) | exact (self) |
| `src/chat_analyzer/utils/__init__.py` (rename) | config (marker) | — | `src/utils/_init_.py` itself (re-exports all valid — keep) | exact (self) |
| `src/chat_analyzer/reporting/__init__.py` (rename) | config (marker) | — | `src/reporting/_init_.py` itself (re-exports all valid — keep) | exact (self) |
| `src/chat_analyzer/ingest/__init__.py` (new) | config (marker) | — | `src/utils/_init_.py` (cleanest existing marker shape) | role-match |
| `src/chat_analyzer/cli/__init__.py` (new) | config (marker) | — | `src/utils/_init_.py` (marker shape) | role-match |
| `src/chat_analyzer/cli/main.py` (new) | controller | request-response | `src/ingest/ingestion.py:639-665` demo block (path → `process_uploaded_file`) | role-match |
| `src/chat_analyzer/cli/__main__.py` (new) | entry point | request-response | `src/ingest/ingestion.py:639-665` demo block (`sys.argv[1]` path handling) | role-match |
| `src/chat_analyzer/analysis/relationship_health.py` (moved) | service | batch | itself (fix `from src.*` at line 800) | exact (self) |
| `src/chat_analyzer/analysis/emotion.py` (moved) | service | batch | itself (fix docstring ref at line 15) | exact (self) |
| `src/chat_analyzer/utils/visualization.py` (moved) | service | transform | itself (fix docstring ref at line 685) | exact (self) |
| `src/chat_analyzer/{analysis,parser,ingest,utils,reporting}/*.py` (9 modules, moved unchanged) | service/utility | batch/transform | themselves — pure `git mv`, zero content change | exact (self) |
| `app/`, `deployment/`, `.streamlit/`, `apt.txt`, `packages.txt` (deleted) | — | — | no analog (git rm; history preserves) | n/a |
| `.planning/PROJECT.md` (edited) | config | — | itself (stale `>=3.8` → `>=3.11`) | exact (self) |

## Pattern Assignments

### `pyproject.toml` (config, build — no code analog)

**Analog:** none in repo. Use `.planning/research/ARCHITECTURE.md:165-181` (PEP 621 template) + `.planning/research/STACK.md` (verified versions) + `requirements.txt` (existing dep list, **minus** `streamlit`, `plotly`, `python-dotenv`, `tqdm`, `requests`, `python-dateutil`, `pytz` unless a module actually imports them).

**Reference layout** (from research, ARCHITECTURE.md:165-181):
```toml
[project]
requires-python = ">=3.11"          # D-09: floor, NOT research's >=3.9
dependencies = [ "pandas>=2.0", ... ]

[project.optional-dependencies]
nlp = ["torch>=2.0", "transformers>=4.30"]   # [nlp] extra (CLI-01/QUAL-01)

[project.scripts]
chat-analyzer = "chat_analyzer.cli:app"       # D-01: console script
```

**Source-of-truth for dependency names** — `requirements.txt` groups (all currently installed deps the moved modules import):
- Data: `pandas>=2.0.0`, `numpy>=1.24.0`
- Viz: `matplotlib>=3.7.0`, `seaborn>=0.12.0`, `wordcloud>=1.9.0` (no plotly — D-06)
- Sentiment: `vaderSentiment>=3.3.2`, `nltk>=3.8.0`
- Reporting: `reportlab>=4.0.0`, `Pillow>=10.0.0`
- Network: `networkx>=3.1`, `emoji>=2.8.0`
- `[nlp]`: `transformers>=4.30.0`, `torch>=2.0.0` (from requirements.txt commented block; research pins `transformers<6`)
- CLI (new, from STACK.md): `typer>=0.12`, `rich>=13`, `plotext>=5.2`

---

### `src/chat_analyzer/cli/main.py` (controller, request-response — interactive prompt)

**Analog:** `src/ingest/ingestion.py:639-665` — the only existing "take a file path, process it, print results" flow. The CLI prompt is this demo block generalized to a Typer app with an interactive prompt instead of `sys.argv`.

**Core pattern** (ingestion.py:648-655 — path → process → report counts):
```python
if len(sys.argv) > 1:
    filepath = sys.argv[1]
    try:
        messages, media = process_uploaded_file(filepath)
        print(f"\nProcessed {filepath}:")
        print(f"Messages: {len(messages)}")
        print(f"Media items: {len(media)}")
        ...
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
```

**File-input handoff** — `process_uploaded_file` already accepts a plain path string. The CLI needs **zero** new file-reading code; it prompts for a path (D-03) and passes `str(path)`:
- `src/ingest/ingestion.py:391-394` — `elif isinstance(uploaded_file, str) and os.path.exists(uploaded_file): with open(uploaded_file, "rb") as f: return os.path.basename(uploaded_file), f.read()`
- `src/ingest/ingestion.py:399` — signature `def process_uploaded_file(uploaded_file: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]`
- Returns `(normalized_messages, media_results)` — the CLI's downstream input contract.

**Heavy-import guard (CRITICAL for instant `--help`):** `cli/main.py` must keep stdlib/light imports at top (per research Anti-Pattern 2, ARCHITECTURE.md:264-268). Analysis modules (`ChatEDA`, `EmotionAnalyzer`, `ConversationSummarizer`, `ChatVisualizer`) are imported inside the command handler, never at module top.

---

### `src/chat_analyzer/cli/__main__.py` (entry point, request-response)

**Analog:** `src/ingest/ingestion.py:639-665` (`if __name__ == "__main__":` demo block) and `src/parser/whatsapp_parser.py:257-268` (path-exists guard + friendly message).

**Core pattern** (whatsapp_parser.py:257-268 — existence check before processing):
```python
if __name__ == "__main__":
    sample_file = "data/sample_chats/whatsapp_sample.txt"
    if os.path.exists(sample_file):
        df = parse_whatsapp_chat(sample_file, output_file)
        ...
    else:
        print(f"Sample file not found: {sample_file}")
```

`__main__.py` shape (D-02): thin shim that defers to `cli/main.py`'s app — `from chat_analyzer.cli.main import app` then `app()` (or `raise SystemExit(app())`) so `python -m chat_analyzer` runs the identical Typer app.

---

### `src/chat_analyzer/*/__init__.py` (package markers — 6 renamed + 2 new)

**Analog:** current `_init_.py` files themselves. The rename activates currently-dead re-exports (Python 3 namespace-package behavior makes them silent today), so each marker must be verified against **actual** symbols before the rename lands (research Anti-Pattern 1, ARCHITECTURE.md:258-262).

**Verification results — which re-exports are broken vs. valid:**

| Marker | Current re-export | Actual symbol | Verdict |
|--------|-------------------|---------------|---------|
| `src/analysis/_init_.py:19` | `plot_relationship_health_dashboard` | `plot_relationship_health_dashboard_enhanced` (relationship_health.py:784) | **BROKEN — strip/rename** |
| `src/analysis/_init_.py:24-26` | `analyze_sentiment`, `perform_eda`, `classify_emotions` | none exist (grep: no such defs) | **BROKEN — strip** (kept behind `try/except ImportError` today) |
| `src/parser/_init_.py:14` | `parse_telegram_json` | `parse_telegram_chat` (telegram_parser.py:6) | **BROKEN — rename to `parse_telegram_chat`** |
| `src/parser/_init_.py:13` | `parse_whatsapp_chat` | `parse_whatsapp_chat` (whatsapp_parser.py:231) | valid |
| `src/utils/_init_.py:6-7` | `ChatVisualizer`, `preprocess_text`, `clean_messages`, `extract_emojis` | all exist (visualization.py:30, preprocessing.py:11/46/88) | valid |
| `src/reporting/_init_.py:12-14` | `generate_chat_analysis_pdf`, `ChatAnalysisPDFGenerator` | both exist (pdf_report.py:534, :34) | valid |
| `src/_init_.py:19-22` | `from . import parser, analysis, reporting, utils` | all subpackages exist post-move | valid |

**Target marker shape** — keep the existing style (docstring + `__all__` + `__version__` + `__author__`), do NOT strip the docstring headers. Model: `src/utils/_init_.py` (14 lines, the cleanest):
```python
"""
Utility Functions Package
Contains visualization, preprocessing, and helper functions.
"""

from .visualization import ChatVisualizer
from .preprocessing import preprocess_text, clean_messages, extract_emojis

__all__ = [
    'ChatVisualizer',
    'preprocess_text',
    'clean_messages',
    'extract_emojis'
]
```

`src/chat_analyzer/__init__.py` keeps its current content as-is (it's valid): docstring + `__version__ = "1.0.0"` + `from . import parser, analysis, reporting, utils` (add `ingest` and `cli` if desired — note `ingest` gains a marker for the first time; it has none today).

**New markers** (`cli/`, `ingest/`): mirror `src/utils/_init_.py` — docstring + valid `from .x import y` lines + `__all__`. `cli/__init__.py` should be minimal (docstring + `__version__`) so importing `chat_analyzer.cli` does not trigger heavy analysis imports.

---

### `src/chat_analyzer/analysis/relationship_health.py` (service, batch — import-site fix)

**Analog:** itself. One-line fix at line 800 — intra-core lazy import inside function body:
```python
# relationship_health.py:798-801 (current)
if use_viz_module:
    try:
        from src.utils.visualization import ChatVisualizer
        viz = ChatVisualizer(figsize=(12, 6))
```
→ `from chat_analyzer.utils.visualization import ChatVisualizer`. Keep the try/except-lazy structure; only the import path changes.

### `src/chat_analyzer/analysis/emotion.py` (service, batch — docstring fix)

**Analog:** itself. Line 15 is inside the module docstring (`from src.analysis.emotion import EmotionAnalyzer`) — update to `from chat_analyzer.analysis.emotion import EmotionAnalyzer`. No code change.

### `src/chat_analyzer/utils/visualization.py` (service, transform — docstring fix)

**Analog:** itself. Line 685 is inside the `if __name__ == "__main__":` demo docstring (`from src.utils.visualization import ChatVisualizer, quick_dashboard`) — update to `from chat_analyzer.utils.visualization import ...`. No code change.

---

### Moved core modules (9 files, unchanged)

`src/analysis/{eda,sentiment,emotion,relationship_health,network_graph,summarizer}.py`, `src/parser/{whatsapp_parser,telegram_parser}.py`, `src/ingest/ingestion.py`, `src/utils/{preprocessing,visualization}.py`, `src/reporting/{pdf_report,weekly_digest}.py` → `git mv` into `src/chat_analyzer/...`, zero content change (per D-10/D-11 and "reuse, not rewrite"). The three `from src.*` sites above are the *only* content edits.

**Note on `src/analysis/summarizer.py:12`** — `from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration` at module top is a known base-install breaker (research Anti-Pattern 2). It is **not** in the 3 fixed import sites, so per CONTEXT.md's surgical-scope it stays as-is this phase unless the plan explicitly opts into the lazy-import change (it only breaks when `[nlp]` features actually load — acceptable for Phase 1 scope; flag to planner).

---

## Shared Patterns

### Optional-dependency gate (try/except ImportError + availability flags)
**Source:** `src/ingest/ingestion.py:33-65` (DEPENDENCIES dict) and `src/analysis/sentiment.py:9-29` (`*_AVAILABLE` flags)
**Apply to:** `cli/main.py` (guard `[nlp]`-gated features), future `cli/pipeline.py` lazy imports
```python
# ingestion.py:34-47 — dict-flag pattern
DEPENDENCIES = {'PIL': False, 'pytesseract': False, 'pdfplumber': False, 'pdf2image': False}
try:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    DEPENDENCIES['PIL'] = True
except ImportError:
    logger.info("PIL not available - image processing disabled")
```
```python
# sentiment.py:10-15 — module-flag pattern (degrade-not-crash with actionable hint)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️ VADER not available. Install with: pip install vaderSentiment")
```
The CLI's `[nlp]` gate follows this convention: `EmotionAnalyzer`/`ConversationSummarizer` imports happen inside the command handler, wrapped in try/except, with the "install `chat-analyzer-pro[nlp]`" hint (research Pattern 3, ARCHITECTURE.md:159-191).

### Logging pattern (module logger + NullHandler)
**Source:** `src/ingest/ingestion.py:29-31`
**Apply to:** `cli/main.py` (and any new cli module)
```python
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
```
CLI configures logging once in `main.py` (research Data Flow note, ARCHITECTURE.md:230).

### Degrade-not-crash error handling
**Source:** `src/ingest/ingestion.py:409-472` (outer try/except, per-message fallback dicts)
**Apply to:** `cli/main.py` file-prompt error UX — never raise to a traceback for user input errors; log + print friendly message (mirrors the `_process_unknown_file` / fallback-normalization pattern).
```python
# ingestion.py:409-413 — error containment shape
try:
    filename, content = _read_file_content(uploaded_file)
except Exception as e:
    logger.error(f"Failed to read file: {e}")
    return [], [{"file": "unknown", "note": f"File reading error: {e}"}]
```

### Path-exists guard before processing
**Source:** `src/parser/whatsapp_parser.py:262-268`
**Apply to:** `cli/main.py` prompt loop — validate the entered path exists before calling `process_uploaded_file`; on failure, re-prompt (non-technical user flow, D-03) rather than crash.

### Demo-block convention (`if __name__ == "__main__":`)
**Source:** 9 existing demo blocks (`ingestion.py:639`, `preprocessing.py:249`, `sentiment.py:436`, `whatsapp_parser.py:257`, `network_graph.py:538`, `visualization.py:680`, `pdf_report.py:616`, `weekly_digest.py:632`, `relationship_health.py:1174`)
**Apply to:** post-move modules keep their demo blocks untouched; `cli/__main__.py` is the *formalized* version of this pattern for the package. No cleanup needed this phase beyond the docstring import fixes.

### Test fixtures for smoke tests
**Source:** `data/sample_chats/whatsapp_sample.txt`, `data/sample_chats/telegram_sample.json` (both exist)
**Apply to:** `--help` smoke test and CLI prompt test — `process_uploaded_file("data/sample_chats/whatsapp_sample.txt")` is the end-to-end exercise. Existing tests (`tests/test_parser.py` etc.) use `unittest`; they must be rewired to import `chat_analyzer.*` — that is a **separate plan/phase concern** (research build-order step 7), only note the import-path breakage now.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `pyproject.toml` | config | build | No existing pyproject in repo; use research ARCHITECTURE.md:165-181 template + requirements.txt dep names |
| `app/` (deleted) | — | — | Deletion only; git history preserves (D-05) |
| `deployment/` (deleted) | — | — | Deletion only |
| `.streamlit/` (deleted) | — | — | Deletion only |
| `apt.txt`, `packages.txt` (deleted) | — | — | Deletion only |

## Metadata

**Analog search scope:** `src/**` (all 17 .py files), `tests/`, repo root, `.planning/research/`
**Files scanned:** 17 source files + 4 test files + requirements.txt
**Pattern extraction date:** 2026-07-31
