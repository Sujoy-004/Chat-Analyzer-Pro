# Coding Conventions

**Analysis Date:** 2026-07-31

## Naming Patterns

**Files:**
- `snake_case.py` for all Python modules: `src/parser/whatsapp_parser.py`, `src/analysis/relationship_health.py`, `src/reporting/weekly_digest.py`, `app/streamlit_app.py`
- `snake_case.py` for tests, prefixed `test_`: `tests/test_parser.py`, `tests/test_analysis.py`, `tests/test_reporting.py`, `tests/test_end_to_end.py`
- Package `__init__.py` files present in every package (`src/_init_.py`, `src/analysis/_init_.py`, `src/parser/_init_.py`, `src/utils/_init_.py`, `src/reporting/_init_.py`)

**Classes:**
- PascalCase, typically one primary class per module: `WhatsAppParser` (`src/parser/whatsapp_parser.py`), `ChatEDA` (`src/analysis/eda.py`), `ChatVisualizer` (`src/utils/visualization.py`), `WeeklyDigestBot` (`src/reporting/weekly_digest.py`), `EmotionAnalyzer` (`src/analysis/emotion.py`), `ConversationSummarizer` (`src/analysis/summarizer.py`), `ChatAnalysisPDFGenerator` (`src/reporting/pdf_report.py`)
- Config classes use PascalCase with UPPER_SNAKE attributes: `SentimentConfig.HF_MODEL`, `SentimentConfig.POSITIVE_THRESHOLD` (`src/analysis/sentiment.py:31-36`)

**Functions:**
- `snake_case`, verb-first where possible: `parse_whatsapp_chat` (`src/parser/whatsapp_parser.py:231`), `analyze_relationship_health` (`src/analysis/relationship_health.py:1071`), `process_uploaded_file` (`src/ingest/ingestion.py:399`), `generate_weekly_summary` (`src/reporting/weekly_digest.py:48`)
- Private helpers prefixed with single underscore `_`: `_add_features`, `_categorize_time_period` (`src/parser/whatsapp_parser.py`), `_process_zip_file`, `_ocr_pdf_page`, `_read_file_content`, `_format_file_size` (`src/ingest/ingestion.py`), `_plot_original_dashboard` (`src/analysis/relationship_health.py:970`)
- Module-level "quick" wrapper functions exposed at bottom of modules: `quick_sentiment_analysis` (`src/analysis/sentiment.py:412`), `quick_timeline`/`quick_heatmap`/`quick_wordcloud`/`quick_sentiment`/`quick_dashboard` (`src/utils/visualization.py:649-673`), `create_digest_bot`/`send_quick_digest` (`src/reporting/weekly_digest.py:557,599`)

**Variables:**
- `snake_case`; DataFrame variables named `df`, config dicts named `config`, summaries named `summary` (consistent across all modules)
- Module-level flags for optional dependencies use UPPER_SNAKE: `VADER_AVAILABLE`, `TEXTBLOB_AVAILABLE`, `TRANSFORMERS_AVAILABLE` (`src/analysis/sentiment.py:12,18,26`), `DEPENDENCIES` dict (`src/ingest/ingestion.py:34`)
- Module-level lazy-initialized singletons use leading underscore: `_vader_analyzer`, `_hf_analyzer` (`src/analysis/sentiment.py:39-40`), `_emotion_analyzer`, `_emotion_model_loaded` (`src/analysis/emotion.py:29-30`)

**Types:**
- Type hints are used in newer/most modules but are NOT consistent across the codebase
- Modules WITH full type hints (use these as the reference pattern): `src/ingest/ingestion.py`, `src/reporting/weekly_digest.py`, `src/reporting/pdf_report.py`, `src/analysis/relationship_health.py`, `src/analysis/network_graph.py`, `src/utils/preprocessing.py`, `src/utils/visualization.py`
- Modules WITHOUT type hints (older style): `src/analysis/eda.py` (bare `def __init__(self, df)`), `src/parser/telegram_parser.py` (`def parse_telegram_chat(source)`), `src/analysis/sentiment.py`
- Standard type vocabulary: `Dict`, `List`, `Optional`, `Tuple`, `Any`, `Union`, `pd.DataFrame`, `plt.Figure`
- Example to follow: `def process_uploaded_file(uploaded_file: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:` (`src/ingest/ingestion.py:399`)

## Code Style

**Formatting:**
- 4-space indentation, PEP 8 line lengths (~120 chars tolerated — long lines exist in `src/analysis/relationship_health.py` and `src/reporting/pdf_report.py`)
- No formatter configured. `black>=23.7.0` and `flake8>=6.1.0` are listed but COMMENTED OUT in `requirements.txt:89-91`
- Quote style is inconsistent: single quotes in older files (`src/parser/whatsapp_parser.py`, `src/analysis/sentiment.py`, `src/analysis/eda.py`), double quotes in newer files (`src/ingest/ingestion.py`, `src/reporting/weekly_digest.py`, `src/analysis/relationship_health.py`, `src/utils/preprocessing.py`). **Use double quotes in new code** to match the current majority.
- Two blank lines between top-level functions/classes; docstring conventions per Google style

**Linting:**
- No linting configuration exists (no `.flake8`, `setup.cfg`, `pyproject.toml`, `ruff.toml`)
- No pre-commit hooks, no CI config (no `.github/` directory)

## Import Organization

**Order:**
1. Standard library (`os`, `re`, `json`, `logging`, `datetime`, `typing`)
2. Third-party (`pandas as pd`, `numpy as np`, `matplotlib.pyplot as plt`, `seaborn as sns`, `streamlit as st`)
3. Local/relative imports (`from .whatsapp_parser import ...` in `src/parser/_init_.py`)

**Pattern observed:**
- `import` statements are NOT strictly grouped with blank lines between stdlib/third-party; a single continuous block is the norm (e.g., `src/parser/whatsapp_parser.py:1-5`, `src/analysis/relationship_health.py:15-21`)
- Standard aliases everywhere: `import pandas as pd`, `import numpy as np`, `import matplotlib.pyplot as plt`, `import seaborn as sns`
- `from datetime import datetime, timedelta` and `from typing import ...` are imported directly rather than as modules
- Optional heavy dependencies (transformers, torch) are imported lazily inside `try/except ImportError` blocks at module top, NOT at the top import block (see `src/analysis/emotion.py:60-61`, `src/analysis/summarizer.py:12`)

**Path Aliases:**
- No aliases/`sys.path` manipulation in `src/`; relative imports used in `__init__.py` files (`.whatsapp_parser`, `.relationship_health`)
- `app/streamlit_app.py` dynamically loads modules from GitHub raw URLs and executes them via `exec(code, namespace)` (`app/streamlit_app.py:42-111`) — do NOT copy this pattern into `src/`

## Error Handling

**Patterns:**
- **Graceful degradation via optional-dependency guards** — the dominant pattern. Imports wrapped in `try/except ImportError` set a module-level `_AVAILABLE` flag, and functions return safe defaults when the flag is False:
  ```python
  # src/analysis/sentiment.py:10-15
  try:
      from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
      VADER_AVAILABLE = True
  except ImportError:
      VADER_AVAILABLE = False
      print("⚠️ VADER not available. Install with: pip install vaderSentiment")
  ```
- **Return-safe-defaults instead of raising**: functions return zero-value dicts/DataFrames on missing data or missing dependencies:
  ```python
  # src/analysis/sentiment.py:80-85
  if not VADER_AVAILABLE or _vader_analyzer is None:
      return {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}
  if pd.isna(text) or str(text).strip() == "":
      return {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}
  ```
- **`{'error': ...}` result dicts** instead of exceptions for analysis functions operating on empty/invalid data:
  ```python
  # src/parser/whatsapp_parser.py:209-210
  if df.empty:
      return {"error": "No data to analyze"}
  ```
  Also `src/analysis/relationship_health.py:78-79` (`return {'error': 'No conversation starters found'}`)
- **Broad `except Exception as e:` with logging** — pervasive; log the exception and return a fallback:
  ```python
  # src/ingest/ingestion.py:448-450
  except Exception as e:
      logger.error(f"Processing failed for {filename}: {e}")
      media_results.append({"file": filename, "note": f"Processing error: {e}"})
  ```
- **Known pitfall — bare `except:`** exists in `src/parser/whatsapp_parser.py:62,78` and `tests/test_parser.py:158`. **Never add new bare excepts**; catch `(ValueError, TypeError)` or `Exception as e` explicitly.
- Per-item try/except with skip-and-continue in loops (e.g., `calculate_rolling_health_score` in `src/analysis/relationship_health.py:412-430`, ZIP member processing in `src/ingest/ingestion.py:485-518`)

## Logging

**Framework:** Two coexisting styles — do not mix within one module:

1. **`logging` module** — use for production `src/` code. Pattern:
   ```python
   # src/ingest/ingestion.py:29-31
   logger = logging.getLogger(__name__)
   logger.addHandler(logging.NullHandler())
   ```
   Used in `src/ingest/ingestion.py`, `src/reporting/weekly_digest.py:22-24`, `src/analysis/relationship_health.py:24-25`, `src/utils/visualization.py:18-20`. Some modules call `logging.basicConfig(level=logging.INFO)` at import (this is noisy — prefer just `getLogger(__name__)`).
   - Levels: `logger.info` for optional-dep disabled, `logger.warning` for recoverable failures, `logger.error` for processing failures.

2. **`print()` with emoji status** — used in older/notebook-derived analysis modules (`src/analysis/sentiment.py`, `src/analysis/emotion.py`, `src/analysis/summarizer.py`, `src/parser/whatsapp_parser.py`). Emojis: 🚀 init, ✅ success, ⚠️ warning, ❌ error.
   - Acceptable in `if __name__ == "__main__":` demo blocks and Streamlit app (`st.warning`/`st.error`), but **prefer `logging` for functions imported by other modules**.

## Comments

**When to Comment:**
- Inline comments explain non-obvious regex, fallback logic, or business thresholds (e.g., `# 120min = very slow` in `src/analysis/relationship_health.py:161`, `# Common email limit` in `tests/test_reporting.py:321`)
- Section banner comments `# ===...===` separate logical phases in `src/analysis/relationship_health.py:375-377,435-437` and `requirements.txt`
- `# Mock ...` comments in tests annotate that test data is fabricated (`tests/test_analysis.py:93,107,174`)

**JSDoc/TSDoc (Python docstrings):**
- **Module docstrings** — triple-quoted at top of every module describing purpose, dependencies, and usage example (see `src/analysis/emotion.py:1-20`, `src/ingest/ingestion.py:1-17`)
- **Google-style docstrings with `Args:` / `Returns:`** — the universal convention:
  ```python
  # src/utils/preprocessing.py:11-24
  def preprocess_text(text: str, lowercase: bool = True, remove_urls: bool = True,
                     remove_mentions: bool = False) -> str:
      """
      Preprocess text message for analysis.
      
      Args:
          text: Input text string
          lowercase: Convert to lowercase
          remove_urls: Remove URLs
          remove_mentions: Remove @mentions
          
      Returns:
          Preprocessed text string
      """
  ```
- Older modules add `(str)` type annotations inside Args (`src/parser/whatsapp_parser.py:28-32`); newer modules omit the parenthesized types. Follow the newer style (bare parameter name) in new code.
- One-line docstrings acceptable for private helpers: `def _format_file_size(size_bytes: int) -> str: """Convert bytes to human-readable format."""` (`src/ingest/ingestion.py:314-315`)

## Function Design

**Size:** Functions are long (100-200+ lines common, e.g., `process_uploaded_file` at `src/ingest/ingestion.py:399-472`, `create_summary_dashboard` at `src/utils/visualization.py:558`). Large operations are decomposed into private `_process_*` helpers (`src/ingest/ingestion.py:475-623`) — follow this decomposition pattern when adding functionality.

**Parameters:**
- Keyword-friendly signatures with defaults for all tunable knobs: `def identify_conversation_starters(df: pd.DataFrame, gap_threshold_minutes: int = 60) -> pd.DataFrame` (`src/analysis/relationship_health.py:28`)
- Config dicts passed for multi-value config: `email_config: Optional[Dict[str, str]] = None` (`src/reporting/weekly_digest.py:35`), weights dict with defaults (`src/analysis/relationship_health.py:282`)

**Return Values:**
- Analysis functions return **dicts** with nested structure (flat strings as keys, e.g., `snake_case` keys in `summary` dicts)
- DataFrame-mutating functions return a **new DataFrame** after `df = df.copy()` defensive copy (see `src/analysis/relationship_health.py:39`, `src/utils/preprocessing.py:59`, `src/analysis/sentiment.py:186`)
- Convenience wrappers return the primary artifact: `parse_whatsapp_chat` returns `pd.DataFrame`, `quick_sentiment_analysis` returns `(df_analyzed, summary)`

## Module Design

**Exports:**
- Modules expose a small set of public functions (or one class) and rely on `__init__.py` re-exports for package-level API:
  - `src/parser/_init_.py` re-exports `parse_whatsapp_chat`
  - `src/utils/_init_.py` re-exports `ChatVisualizer`, `preprocess_text`, `clean_messages`, `extract_emojis`
  - `src/analysis/_init_.py` re-exports `analyze_relationship_health`, `calculate_relationship_health_score`, `plot_relationship_health_dashboard` — note: this file's `try/except ImportError` block references functions that do not exist (`analyze_sentiment`, `classify_emotions`, `perform_eda`); the except swallows it. Do not extend this pattern; keep `__init__.py` re-exports in sync with actual function names.
- No barrel files beyond these `__init__.py` re-exports.

**File structure (canonical order within each module):**
1. Module docstring
2. Imports
3. Optional-dependency detection (`try/except ImportError` + flags)
4. Constants / config classes
5. Class definition(s) or function definitions (private helpers interspersed or at bottom)
6. Public wrapper functions
7. `if __name__ == "__main__":` demo/usage block (present in most modules: `src/parser/whatsapp_parser.py:257`, `src/analysis/relationship_health.py:1174`, `src/ingest/ingestion.py:639`)

**Conventions for class-based modules:**
- `__init__` stores config on `self`, initializes heavy resources in a separate `_initialize_*` method guarded by a module-level singleton flag (see `EmotionAnalyzer._initialize_model` in `src/analysis/emotion.py:51-79`)
- Plot methods take `ax` or `figsize` parameters and use matplotlib directly; plot helpers prefixed `_plot_` (`src/analysis/relationship_health.py:867-967`)
- `ChatVisualizer` (`src/utils/visualization.py`) centralizes color schemes in `self.colors` and applies `plt.style.use(style)` in `__init__`

**Conventions for function-based modules:**
- Chain small pure functions; each returns a dict summary (e.g., `src/analysis/relationship_health.py` composes `identify_conversation_starters` → `calculate_initiator_ratio` → `analyze_response_patterns` → `calculate_dominance_scores` → `calculate_relationship_health_score` in `analyze_relationship_health` at line 1071)
- Module-level `logger` and `warnings.filterwarnings('ignore')` at top for analysis modules

## Testing-Specific Conventions

- Test classes named `Test<Module>` (`TestWhatsAppParser`, `TestEDAModule`, `TestWeeklyDigest`, `TestCompletePipeline`), see `tests/`
- Test methods named `test_<behavior_description>` with a docstring on every method
- `setUp()`/`tearDown()` for fixture construction and temp-file cleanup — see `tests/test_parser.py:16-31`
- Each test file includes a `run_*_tests()` aggregator + `if __name__ == '__main__': unittest.main()` — see `tests/test_analysis.py:381-401`
- See `TESTING.md` for the full testing conventions.

---

*Convention analysis: 2026-07-31*
