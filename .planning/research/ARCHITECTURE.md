# Architecture Research

**Domain:** pip-installable CLI tool wrapping an existing analysis library (Chat-Analyzer-Pro pivot)
**Researched:** 2026-07-31
**Confidence:** HIGH (structure verified against actual `src/` code + PyPA official docs + 2026 ecosystem sources)

## Executive Decision

The existing `src/` is **85% correct** — a src-layout container with DataFrame-centric, UI-independent modules. Two things block pip distribution today: the `_init_.py` misnamed package markers and the generic top-level package names (`parser`, `analysis`, `utils`). Fix those by nesting everything under one real import package `chat_analyzer`, add a `cli/` subpackage for all CLI-specific code, and wire a Typer-based entry point. The analysis core is reused as-is; the CLI is a thin orchestration + rendering layer.

---

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 ENTRY — `analyze` console script                         │
│             chat_analyzer.cli.main:app (Typer app)                       │
│   argument parsing · file validation · exit codes · --help              │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ calls
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                ORCHESTRATION — chat_analyzer/cli/pipeline.py             │
│   run_pipeline(path, options) → AnalysisResults                         │
└───────┬──────────────┬──────────────┬───────────────────┬───────────────┘
        │ ingest       │ analyze      │ adapt             │ render
        ▼              ▼              ▼                   ▼
┌───────────────┐ ┌────────────────────────────┐ ┌──────────────────────┐
│ CORE (library)│ │  analysis/ (all reused)    │ │ PRESENTATION (cli/)  │
│ ingest/       │ │  eda · sentiment · emotion │ │ render.py            │
│  ingestion.py │ │  relationship_health ·     │ │  rich + plotext      │
│  (+moved df   │ │  network_graph · summarizer│ │ report_html.py       │
│   converter)  │ │  parser/ whatsapp, telegram│ │  single-file HTML    │
│  utils/       │ │  reporting/ (deferred)     │ │  + base64 PNG charts │
│  preprocessing│ │  utils/visualization.py    │ │                      │
└───────┬───────┘ │  (matplotlib → PNG for    │ └──────────────────────┘
        │         │   HTML embedding)          │
        │         └──────────────┬─────────────┘
        ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DATA CONTRACT — message DataFrame + AnalysisResults                     │
│  df: datetime · sender · message · message_length · hour · source · uid  │
│  AnalysisResults: TypedDict normalized from module dicts (adapters.py)   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| `cli/main.py` | Command parsing, file/option validation, exit codes, error UX, wiring console script | Typer app; `@app.command()` per subcommand |
| `cli/pipeline.py` | The full run: ingest → df → analyze → results; owns ordering and progress reporting | Single `run_pipeline(path, opts)` function |
| `cli/adapters.py` | Map each analysis module's dict into the canonical `AnalysisResults` contract | Small pure functions per module |
| `cli/render.py` | Terminal output: rich tables/panels + plotext ASCII charts | Rich `Console`, plotext figures |
| `cli/report_html.py` | Self-contained HTML report from `AnalysisResults` + base64 PNGs | stdlib template (no jinja) |
| `cli/errors.py` | User-facing errors, "how to export chats" instructions, upgrade hints for missing extras | Custom exception → exit code mapping |
| `ingest/ingestion.py` | Any file → normalized message dicts (already handles paths; `_read_file_content` line 391) | Existing, unchanged |
| `ingest/ingestion.py` `messages_to_dataframe()` | Normalized dicts → message DataFrame (**moved from app/streamlit_app.py:313**) | New function in core |
| `parser/*` | Raw export → DataFrame (`datetime, sender, message, ...`) | Existing `WhatsAppParser`, `parse_telegram_chat` |
| `analysis/*` | DataFrame → insight dicts / enriched DataFrames | Existing modules, unchanged except summarizer import fix |
| `utils/visualization.py` | Matplotlib charts → figures for HTML embedding | Existing `ChatVisualizer` (12 plot methods) |

---

## Recommended Project Structure

```text
Chat-Analyzer-Pro/
├── pyproject.toml                  # NEW: setuptools src-layout, deps, extras, entry point
├── README.md                       # quickstart "how a friend uses it"
├── data/sample_chats/              # unchanged
├── tests/                          # rewired to import chat_analyzer.* (currently self-contained)
└── src/
    └── chat_analyzer/              # SINGLE top-level import package (replaces src/ as import root)
        ├── __init__.py             # renamed from _init_.py, content cleaned (see pitfalls)
        ├── parser/                 # git mv src/parser → src/chat_analyzer/parser
        │   ├── __init__.py         #   renamed + cleaned
        │   ├── whatsapp_parser.py  #   unchanged
        │   └── telegram_parser.py  #   unchanged
        ├── ingest/
        │   ├── __init__.py
        │   └── ingestion.py        # + messages_to_dataframe() moved from app/streamlit_app.py:313
        ├── analysis/
        │   ├── __init__.py
        │   ├── eda.py · sentiment.py · emotion.py
        │   ├── relationship_health.py · network_graph.py
        │   └── summarizer.py       # transformers import made lazy (top-level today — breaks base install)
        ├── reporting/
        │   ├── __init__.py
        │   ├── pdf_report.py       # deferred, kept importable
        │   └── weekly_digest.py    # deferred, kept importable
        ├── utils/
        │   ├── __init__.py
        │   └── preprocessing.py · visualization.py
        └── cli/                    # NEW: ALL CLI-specific code lives here
            ├── __init__.py
            ├── __main__.py         # python -m chat_analyzer
            ├── main.py             # Typer app + console script target
            ├── pipeline.py         # orchestration
            ├── contracts.py        # AnalysisResults TypedDict
            ├── adapters.py         # module dicts → AnalysisResults
            ├── render.py           # rich + plotext
            ├── report_html.py      # single-file HTML
            └── errors.py           # exit codes + friendly errors
```

### Structure Rationale

- **`src/chat_analyzer/` — one top-level import package.** PyPA and setuptools conventions: one and only one top-level import package per project, and `src/` is the container directory. Today the repo exposes `src.parser`, `src.analysis`, `src.utils`, `src.reporting` as top-level names — `utils` and `parser` are guaranteed to collide with unrelated packages on PyPI. Nesting under `chat_analyzer` (matching distribution `chat-analyzer-pro`) is the standard fix. (`packaging.python.org` src-layout discussion — HIGH)
- **`cli/` inside the package, not a sibling.** The wheel ships the library *and* the CLI; `chat_analyzer.cli` keeps one artifact, and the `[project.scripts] analyze` entry point points at `chat_analyzer.cli.main:app`. Nothing in `cli/` is importable or needed by library consumers.
- **Core modules stay untouched (reuse, not rewrite).** They are already UI-independent functions on `DataFrame → Dict`. The CLI adapts their outputs; it does not re-implement analysis. Only four surgical core changes (see Build Order): rename+clean package markers, fix 3 `from src.*` import sites, add `messages_to_dataframe()`, make `summarizer.py`'s transformers import lazy.
- **`app/` is dropped.** Everything the app did that the CLI needs (DataFrame conversion, analysis orchestration) moves into core or `cli/`; the `exec()` module-fetcher, fallback parsers, and basic health scoring are deleted with it (per PROJECT.md and CONCERNS.md).

---

## Architectural Patterns

### Pattern 1: Thin CLI over a library core (Facade + Pipeline)

**What:** The CLI is a facade: argument parsing and rendering on one side, the library's public functions on the other, connected by a linear pipeline function. No business logic lives in the CLI.
**When to use:** Always for this project — it is the core architectural decision of the pivot.
**Trade-offs:** Requires an adapter layer to normalize heterogeneous module dicts (small cost); prevents the duplicated-logic drift the old app suffered (big win).

```python
# cli/pipeline.py — the only orchestration in the system
def run_pipeline(path: Path, opts: Options) -> AnalysisResults:
    messages, media = process_uploaded_file(str(path))          # 1. ingest (path supported today)
    df = messages_to_dataframe(messages)                        # 2. moved from app/streamlit_app.py:313
    eda = ChatEDA(df).generate_comprehensive_summary()
    sentiment = add_sentiment_analysis(df)                      # VADER path — always available
    health = analyze_relationship_health(df)                    # core orchestrator, reused
    if opts.with_nlp:                                           # heavy features opt-in (see Pattern 3)
        emotions = EmotionAnalyzer().get_emotion_summary(df)
        summary = ConversationSummarizer().generate_full_report(df)
    return adapt(eda=eda, sentiment=sentiment, health=health,  # 3. normalize (cli/adapters.py)
                 emotions=emotions, summary=summary)
```

### Pattern 2: Canonical results contract via adapters

**What:** Each analysis module returns its own dict shape (e.g. `relationship_health` returns a 12-key nested dict at line 1101; sentiment returns its own). `cli/adapters.py` maps each into one `AnalysisResults` TypedDict that both `render.py` and `report_html.py` consume.
**When to use:** When two renderers (terminal + HTML) must show the same data and the core modules are not to be modified.
**Trade-offs:** One mapping function per module (~5 small functions); keeps core clean and renderers stable.

```python
# cli/contracts.py
class AnalysisResults(TypedDict):
    stats: dict            # volume, participants, date range
    sentiment: dict        # summary + per-message series
    health: dict           # normalized health_score + components
    emotions: dict | None  # None when [nlp] extra missing
    summary: str | None
    charts: dict[str, str] # name → base64 PNG for HTML embedding
```

### Pattern 3: Optional heavy deps — extras + lazy imports + degrade-not-crash

**What:** `torch`/`transformers` ship in an optional extra; heavy modules are imported inside functions, never at module top level; when the extra is missing, features degrade with an actionable hint instead of failing.
**When to use:** Any distributed tool with ML dependencies. Verified current practice: PEP 621 `[project.optional-dependencies]` extras are the idiomatic mechanism; transformers itself lazily imports backends and degrades capabilities ("import never fails, capabilities degrade gracefully"); lazy in-function imports are the standard CLI pattern (instant `--help`). (HIGH)
**Trade-offs:** Users must run `pip install chat-analyzer-pro[nlp]` for emotion+summary; the trade is a fast, light base install vs. "everything always works". Matches the codebase's existing `DEPENDENCIES`/`*_AVAILABLE` degradation machinery.

```toml
# pyproject.toml
[project]
requires-python = ">=3.9"          # see pitfall: 3.8 is EOL; Typer 0.12 needs 3.9+
dependencies = [
  "pandas>=2.0", "numpy>=1.24", "matplotlib>=3.7", "seaborn>=0.12",
  "networkx>=3.1", "vaderSentiment>=3.3.2", "nltk>=3.8", "emoji>=2.8",
  "rich>=13", "plotext>=5.2", "typer>=0.12",
]

[project.optional-dependencies]
nlp = ["torch>=2.0", "transformers>=4.30"]   # emotion + summarization
dev = ["pytest>=7.4", "pytest-cov>=4.1"]     # replace self-contained unittest drift

[project.scripts]
analyze = "chat_analyzer.cli.main:app"
```

```python
# chat_analyzer/analysis/summarizer.py — required change: import inside method
class ConversationSummarizer:
    def _ensure_model(self):
        try:
            from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration  # lazy
        except ImportError:
            raise FeatureUnavailable("summarization needs: pip install chat-analyzer-pro[nlp]")
```

### Pattern 4: Single-file HTML report via data URIs

**What:** `report_html.py` renders `AnalysisResults` into one self-contained HTML file; matplotlib charts (from `ChatVisualizer`) are saved to PNG bytes and embedded as `data:image/png;base64,...`. No external assets, no CDN, no server.
**When to use:** The "shareable report card" requirement — a file a friend can open by double-clicking.
**Trade-offs:** Larger file size vs. external assets; zero runtime dependencies. Use a stdlib `string.Template` — no jinja2 dep needed for one template.

---

## Data Flow

### Request Flow (`analyze chat.txt`)

```
User runs: analyze chat.txt --output report.html
    ↓
cli/main.py (Typer) — validates file exists, builds Options
    ↓
cli/pipeline.py run_pipeline()
    ├─ process_uploaded_file(path)            → (messages, media)   [ingest/ingestion.py — path already handled, line 391]
    ├─ messages_to_dataframe(messages)        → df                 [MOVED from app/streamlit_app.py:313]
    ├─ ChatEDA(df) → summary                  ─┐
    ├─ add_sentiment_analysis(df)             ├─ analysis core      [all reused unchanged]
    ├─ analyze_relationship_health(df)        ─┘
    ├─ [--with-nlp] EmotionAnalyzer / ConversationSummarizer  (lazy imports)
    └─ adapt() → AnalysisResults
    ↓
cli/render.py (rich panels/tables + plotext ASCII charts)  → terminal
cli/report_html.py (template + base64 PNGs from ChatVisualizer) → report.html
    ↓
Exit code 0 (or 2 for usage errors, 1 for analysis failures — via cli/errors.py)
```

### State Management

- **Process-local only.** No persistence. The DataFrame and `AnalysisResults` live for the duration of one command run.
- **Module-level singletons are correct for a CLI** — `_emotion_analyzer`, `_vader_analyzer` (already in emotion.py/sentiment.py) load ML models exactly once per run. Keep them.
- **No shared mutable state across components.** Each pipeline step takes data in, returns data out. `cli/adapters.py` is pure.
- **Logging:** CLI configures logging once in `main.py`. Core modules that call `logging.basicConfig` at import time (relationship_health.py:24, visualization.py:19) must be neutralized — see pitfalls.

### Key Data Flows

1. **File → DataFrame:** path string → `process_uploaded_file` (already accepts `str` paths) → normalized dicts → `messages_to_dataframe` → df with `datetime, sender, message, message_length, hour, source, uid`. This is the de-facto interface every analysis module consumes (verified: `analyze_relationship_health` docstring requires `datetime, sender, message`).
2. **DataFrame → insights:** each analysis module independently maps df → dict; `adapters.py` normalizes into `AnalysisResults`.
3. **AnalysisResults → two outputs:** `render.py` (terminal) and `report_html.py` (HTML) consume the same contract — one analysis run, two presentations.
4. **tz-awareness caveat (from CONCERNS.md):** Telegram parsing produces tz-aware datetimes; WhatsApp naive. `messages_to_dataframe` must normalize to naive (as the old `convert_normalized_messages_to_df` did) or downstream `strptime` paths break.

---

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 1–10k messages (typical chat export) | Current design is fine; single process, in-memory DataFrame |
| 10k–100k messages (long group histories) | First bottleneck: row-wise VADER sentiment + regex parsing get slow. Fix: rich `Progress` bars; batch sentiment; sample messages for summarization (summarizer already caps at `max_messages=100`) |
| 100k+ messages | Model memory: transformer pipelines over full df can OOM. Fix: process emotion in batches (emotion.py already batch-oriented); keep matplotlib charts to the same aggregates, not per-message |

### Scaling Priorities

1. **First bottleneck: startup import cost.** If `cli/main.py` eagerly imports analysis modules (which import matplotlib/pandas), `--help` is slow. Keep heavy imports inside `pipeline.py` command handlers. Also the summarizer top-level transformers import *breaks base installs entirely* — fix first.
2. **Second bottleneck: first-run model download.** `EmotionAnalyzer` downloads `j-hartmann/emotion-english-distilroberta-base` from HuggingFace Hub on first use — needs network + disk, then caches. Print a one-line "downloading model…" notice via rich so the CLI doesn't look hung.

---

## Anti-Patterns

### Anti-Pattern 1: Shipping the `_init_.py` markers or the generic top-level names

**What people do:** Add a `pyproject.toml` to the repo as-is. The misnamed `_init_.py` markers make `src.*` a PEP 420 namespace package; setuptools would either not discover it (regular `find`) or ship implicit namespaces (`find` defaults) with the collision-prone names `parser`/`analysis`/`utils`/`reporting`. Even if it installed, `import utils` would shadow/collide with unrelated PyPI packages.
**Why it's wrong:** Wheels silently missing modules, or broken imports for users who have any package named `parser`/`utils` installed.
**Do this instead:** `git mv src/* src/chat_analyzer/`, rename `_init_.py` → `__init__.py`, and rewrite the `__init__.py` contents as minimal markers. **Do not** reuse the existing contents: `src/analysis/_init_.py:16` imports `plot_relationship_health_dashboard` (actual: `plot_relationship_health_dashboard_enhanced`), `src/parser/_init_.py:14` imports `parse_telegram_json` (actual: `parse_telegram_chat`), and `sentiment`/`eda`/`emotion` re-exports (`analyze_sentiment`, `perform_eda`, `classify_emotions`) don't exist. Renaming triggers these broken imports. Minimal docstring+`__version__` markers de-risk the rename.

### Anti-Pattern 2: Eager heavy imports at CLI startup (and the summarizer landmine)

**What people do:** `import` analysis modules at the top of `cli/main.py`; leave `from transformers import ...` at `summarizer.py:12`.
**Why it's wrong:** The summarizer's top-level import means the entire module is unimportable without torch/transformers — a base install breaks when the CLI merely imports the pipeline. Eager imports make `analyze --help` take seconds.
**Do this instead:** Move transformers import into `ConversationSummarizer._ensure_model()` (or `__init__`); import heavy modules inside `run_pipeline()`; keep stdlib/light imports at top. (2026 lazy-import practice — HIGH)

### Anti-Pattern 3: Re-implementing analysis in the CLI (the old app's sin)

**What people do:** Copy `calculate_basic_health_score` / fallback parsers / inline charts into `cli/` because "it's easier than wiring the module".
**Why it's wrong:** The codebase map already documents this drift — the app's basic-mode health dict has a *different shape* than the library's, doubling maintenance. It burned the Streamlit version.
**Do this instead:** Call `analyze_relationship_health(df)`, `ChatEDA`, `add_sentiment_analysis` directly. Degradation happens in the *features* (skip emotion if extra missing), never in parallel implementations.

### Anti-Pattern 4: `logging.basicConfig()` and `print()` leaking from core into CLI output

**What people do:** Ignore that `relationship_health.py:24` and `visualization.py:19` call `logging.basicConfig` at import and that sentiment/emotion/summarizer `print()` status lines.
**Why it's wrong:** The CLI loses control of logging config (the import order of modules decides the log format) and bare `print()`s pollute structured terminal output.
**Do this instead:** In the CLI phase, remove/replace the two `logging.basicConfig` calls with `getLogger(__name__)` + `NullHandler` (matching `ingestion.py:30-31`); decide whether core `print("✅ …")` lines are acceptable flavor or route them through a module-level logger. The `warnings.filterwarnings('ignore')` calls in emotion/summarizer should also be scoped, not global.

### Anti-Pattern 5: Keeping `convert_normalized_messages_to_df` in app code

**What people do:** Write a new DataFrame builder inside `cli/pipeline.py` because the function "belongs to the web app".
**Why it's wrong:** Third copy of the same schema logic; the tz-naive normalization bug in CONCERNS.md propagates differently per copy.
**Do this instead:** Move it into `ingest/ingestion.py` as `messages_to_dataframe(messages)` — the single source for dicts → df. It's the one piece of app glue that genuinely belongs in core.

### Anti-Pattern 6: Python 3.8 runtime floor

**What people do:** Keep `requires-python = ">=3.8"` from the old requirements.txt.
**Why it's wrong:** 3.8 is EOL (Oct 2024); current tooling and Typer 0.12 (3.9+) have moved on; 2026 distributions shouldn't promise support for an unsupported interpreter.
**Do this instead:** `requires-python = ">=3.9"` (or `>=3.10` matching dev); flag this as a PROJECT.md decision update in the roadmap.

---

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| HuggingFace Hub | Direct download at first use (transformers `from_pretrained`) | `j-hartmann/emotion-english-distilroberta-base`, `t5-small`; cache in HF cache dir; needs network once. Not used on base install (extra-gated) |
| (None else) | — | No accounts, no hosting, no SMTP/Telegram in v1 (weekly_digest deferred) |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `cli/*` → core (`analysis/`, `parser/`, `ingest/`, `utils/`) | Direct function calls | One direction only. Core modules must **never** import `cli.*` |
| `cli/pipeline.py` → `cli/render.py`, `cli/report_html.py` | `AnalysisResults` TypedDict | Both renderers consume the same contract; adding an output means adding a renderer, not changing the pipeline |
| `adapters.py` → module result dicts | Direct reads of returned dicts | Adapters are the only place that knows each module's internal dict shape |
| core analysis ↔ core utils (`visualization.py`) | `try/except` import at `relationship_health.py:800` | The one intra-core import site to fix in the package rename (`from src.utils...` → `from chat_analyzer.utils...`) |

---

## Build Order (for roadmap)

Dependency-ordered — each step unblocks the next:

1. **Package surgery (prereq for everything).** `git mv src/* src/chat_analyzer/`; rename+clean all `_init_.py` → `__init__.py`; fix 3 `from src.*` import sites (relationship_health.py:800, emotion.py:15, visualization.py:685); add `pyproject.toml` (setuptools, `packages.find where=["src"]`, `requires-python>=3.9`, scripts entry, `[nlp]`/`[dev]` extras); verify `pip install -e .` and `analyze --help`. **Risk gate:** do NOT rename `_init_.py` without cleaning contents (stale re-exports fail at import).
2. **CLI skeleton.** `cli/main.py` (Typer app, `analyze <file>`, `--output`, `--with-nlp`, `--format`), `cli/__main__.py`, `cli/errors.py` (exit codes, WhatsApp/Telegram export instructions). Instant `--help` verified.
3. **Pipeline core.** Move `messages_to_dataframe` into `ingest/ingestion.py`; make summarizer's transformers import lazy; build `cli/pipeline.py` + `contracts.py` + `adapters.py` wired to EDA + sentiment + relationship_health (the always-available insights).
4. **Terminal rendering.** `cli/render.py` with rich panels/tables + plotext charts; neutralize `logging.basicConfig` in core; decide on `print()` leakage.
5. **HTML report.** `cli/report_html.py` single-file template; `ChatVisualizer` → PNG → base64 embedding; `--output report.html`.
6. **NLP extras gate.** Wire `--with-nlp`/`[nlp]` extra, model-download notice, upgrade hints, degrade paths for emotion/summarization.
7. **Polish.** Rewire `tests/` to import `chat_analyzer.*` (replacing the self-contained mock-parser tests — CONCERNS.md anti-pattern); README quickstart; delete `app/` and `deployment/`.

---

## Sources

- PyPA "src layout vs flat layout" — https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/ (HIGH)
- setuptools package discovery (src-layout, one import package per project) — https://setuptools.pypa.io/en/latest/userguide/package_discovery.html (HIGH)
- pyOpenSci Python Package Structure (src-layout strongly suggested) — https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-structure.html (HIGH)
- Typer vs Click vs Argparse, 2026 comparisons (Typer for greenfield type-hinted CLIs; built on Click; rich integration) — https://johal.in/comparison-click-vs-typer-vs-argparse-2026-python , https://www.guvi.in/blog/typer-vs-click-vs-argparse/ (MEDIUM — blog sources, but consistent; Typer official docs: https://typer.tiangolo.com/)
- CLI lazy imports for fast `--help` — https://codepointers.com/2026/02/05/optimize-python-cli-performance-with-lazy-imports/ (MEDIUM)
- Optional extras pattern (PEP 621 `[project.optional-dependencies]`; transformers `[torch]` extra endorsement) — https://github.com/huggingface/tokenizers/issues/1973 , https://discuss.python.org/t/help-packaging-optional-application-features-using-extras/14074 (HIGH)
- transformers lazy-loading architecture ("import never fails, capabilities degrade") — https://readoss.com/en/huggingface/transformers/how-import-transformers-works-lazy-loading-architecture (MEDIUM)
- PEP 771 default extras (noted; explicit `[nlp]` chosen instead) — https://peps.python.org/pep-0771/ (HIGH)
- Repo-internal verification: `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`, `src/analysis/summarizer.py:12`, `src/ingest/ingestion.py:383-396,399`, `app/streamlit_app.py:313-345`, `src/analysis/_init_.py`, `src/parser/_init_.py`, `src/analysis/relationship_health.py:800,1071-1128`

---

*Architecture research for: Chat-Analyzer-Pro CLI pivot (project research, architecture dimension)*
*Researched: 2026-07-31*
