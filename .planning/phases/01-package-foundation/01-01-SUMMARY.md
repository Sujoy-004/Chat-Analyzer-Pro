---
phase: 01-package-foundation
plan: 01
subsystem: packaging
tags: [hatchling, pyproject, src-layout, package-markers, typer, rich, plotext, lazy-imports]

# Dependency graph
requires:
  - phase: 00-init
    provides: project context, verified research stack (STACK.md), codebase conventions, phase-1 decisions D-01..D-11
provides:
  - Importable `chat_analyzer.*` src-layout package (valid markers, zero stale `src.*` imports, lazy transformers gate)
  - pyproject.toml (hatchling, requires-python >=3.11, lean base deps, [nlp]/[dev] extras, D-01 console script target)
  - Web-app artifacts deleted (app/, deployment/, .streamlit/, apt.txt, packages.txt, requirements.txt)
  - Human-approved package-legitimacy gate for typer/rich/plotext/hatchling (T-01-SC mitigated)
affects: [01-02 (CLI + pip install), Phase 2 (parser hardening + pipeline), Phase 3 (HTML report), Phase 4 (NLP quality)]

# Tech tracking
tech-stack:
  added: [hatchling>=1.31.0 (build backend), typer>=0.27, rich>=13, plotext>=5.3 (CLI stack, declared not yet installed)]
  patterns:
    - "src-layout packaging (PEP 517/621) with hatchling wheel target"
    - "Light package marker — top-level __init__ exposes version/metadata only; no eager subpackage imports"
    - "Lazy heavy-dep gating: transformers import moved into method try-block; sys.modules assertion proves import-time lightness"
    - "degrade-not-crash for optional deps (existing convention preserved: textblob/wordcloud/pytesseract/pdfplumber try/except)"

key-files:
  created: [pyproject.toml, src/chat_analyzer/__init__.py, src/chat_analyzer/ingest/__init__.py]
  modified: [src/chat_analyzer/analysis/__init__.py, src/chat_analyzer/parser/__init__.py, src/chat_analyzer/reporting/__init__.py, src/chat_analyzer/analysis/relationship_health.py, src/chat_analyzer/analysis/emotion.py, src/chat_analyzer/utils/visualization.py, src/chat_analyzer/analysis/summarizer.py, .planning/PROJECT.md, .planning/phases/01-package-foundation/01-01-PLAN.md]

key-decisions:
  - "Base deps = verified-import list only (grep over src/, not requirements.txt): pandas, numpy, matplotlib, seaborn, vaderSentiment, wordcloud, networkx, requests, reportlab, Pillow, typer, rich, plotext"
  - "transformers<6 pin in [nlp] extra — 5.x breaks the 4.x-era core code; torch/transformers excluded from base by design (PKG-03)"
  - "DELIBERATELY EXCLUDED from pyproject (verified zero imports): streamlit, plotly, python-dotenv, nltk, scipy, tqdm, emoji, regex, python-dateutil, pytz, textblob (last degrades to VADER-only when absent)"
  - "requirements.txt deleted — superseded by pyproject.toml (avoids duplicated-manifests drift, CONCERNS.md:42-45); recoverable from git"
  - "Package-legitimacy gate for typer/rich/plotext/hatchling human-approved (all four real on PyPI, versions in range) — plan 02 may pip install -e ."

patterns-established:
  - "Light top-level marker: docstring + __version__ (single source of truth, matches pyproject) + __author__/__project__ + __all__; NO `from . import ...` eager imports"
  - "Lazy transformers import pattern: import inside __init__ try-block after self.model_name assignment (verified only 3 names referenced, all in __init__)"
  - "In-function lazy imports preserved for heavy/optional deps (visualization ChatVisualizer, wordcloud, PIL)"

requirements-completed: [PKG-01, PKG-02, PKG-03, PKG-04]

# Metrics
duration: 20min
completed: 2026-07-31
---

# Phase 1 Plan 1: Package Foundation — Restructure into a Shipable src-Layout Package Summary

**Repackaged `src/` into the importable `chat_analyzer.*` package (valid `__init__.py` markers, zero stale `src.*` imports, lazy transformers gate), created a hatchling `pyproject.toml` with `>=3.11` floor and `[nlp]` extra, deleted the entire web-app surface, and human-approved the new install-time dependency gate (typer/rich/plotext/hatchling) for plan 02's `pip install -e .`**

## Performance

- **Duration:** ~20 min (wave-1 launch 2026-07-31T12:39Z → completion 2026-07-31T12:59Z; two executor sessions: Tasks 1-2, then Task 3 gate + verification)
- **Started:** 2026-07-31T12:39:00Z
- **Completed:** 2026-07-31T12:59:13Z
- **Tasks:** 3 (2 auto + 1 blocking-human checkpoint, all closed)
- **Files modified:** 34 (20 moved/rewritten in Task 1, 13 in Task 2 incl. deletions, 1 plan annotation)

## Accomplishments

- `src/` restructured via `git mv` into `src/chat_analyzer/` with 5 renamed markers (`_init_.py` → `__init__.py`) plus a new `ingest/__init__.py`; all 4 content-edit modules otherwise byte-identical to pre-move (reuse, not rewrite)
- Light top-level marker — `import chat_analyzer` loads zero heavy modules (verified via sys.modules assertion); eager `from . import parser/analysis/reporting/utils` lines removed
- All 6 `from src.*` sites fixed (3 real import/docstring sites + 3 marker docstrings, reporting/`__init__.py:9` included); zero-match grep gate holds
- Lazy transformers import in `summarizer.py` — module imports cleanly WITHOUT transformers installed (proven in smoke test)
- `pyproject.toml`: hatchling src-layout, `requires-python = ">=3.11"` (D-09), lean verified-import base deps, `[nlp]` extra (`torch>=2.0`, `transformers>=4.30,<6`), `[dev]` extra, D-01 console script `chat-analyzer = "chat_analyzer.cli:app"`
- Web-app artifacts deleted (`app/` incl. the `exec()` fetcher + `unsafe_allow_html`, `deployment/`, `.streamlit/`, `apt.txt`, `packages.txt`) and `requirements.txt` removed (duplicated-manifests anti-pattern); PROJECT.md floor corrected 3.8+ → 3.11+
- Package-legitimacy gate human-approved — threat T-01-SC mitigated, plan 02 unblocked for `pip install -e .`

## Task Commits

Each task was committed atomically:

1. **Task 1: Restructure src/ into src/chat_analyzer/ with valid package markers and fix all import sites** - `1c279aa` (feat)
2. **Task 2: Create pyproject.toml, delete web-app artifacts, fix PROJECT.md floor** - `7823759` (feat)
3. **Task 3: Package legitimacy gate (blocking-human checkpoint)** - `85df6d0` (docs: approve gate)

**Plan metadata:** pending final phase-1 commit (this SUMMARY + STATE/ROADMAP)

## Files Created/Modified

- `pyproject.toml` - hatchling PEP 517/621 src-layout build config; requires-python >=3.11; verified-import base deps; [nlp]/[dev] extras; D-01 console script target
- `src/chat_analyzer/__init__.py` - light marker: docstring + `__version__ = "0.1.0"` (matches pyproject) + `__all__`; eager subpackage imports removed
- `src/chat_analyzer/analysis/__init__.py` - cleaned re-exports (`plot_relationship_health_dashboard_enhanced`), broken try/except symbols stripped, docstring aligned to real symbols
- `src/chat_analyzer/parser/__init__.py` - re-export fix `parse_telegram_json` → `parse_telegram_chat`
- `src/chat_analyzer/reporting/__init__.py` - docstring `src.reporting` → `chat_analyzer.reporting` (line 9, required for zero-match grep gate)
- `src/chat_analyzer/ingest/__init__.py` - NEW marker: `process_uploaded_file`, `get_dependency_status`, `get_supported_formats`
- `src/chat_analyzer/analysis/relationship_health.py` - lazy import path fix `src.utils.visualization` → `chat_analyzer.utils.visualization` (line 800)
- `src/chat_analyzer/analysis/emotion.py` - docstring src.* fix + emoji-print guard under `__main__` (deviation, see below)
- `src/chat_analyzer/utils/visualization.py` - docstring src.* fix (line 685)
- `src/chat_analyzer/analysis/summarizer.py` - transformers import moved into `__init__` try-block (line 51)
- `.planning/PROJECT.md` - Python floor 3.8+ → 3.11+ (D-09); `analyze` → `chat-analyzer` command name (D-01)
- `.planning/phases/01-package-foundation/01-01-PLAN.md` - Task 3 gate approval annotation
- Deleted: `app/` (streamlit_app.py, assets), `deployment/` (Dockerfile, Procfile, configs), `.streamlit/`, `apt.txt`, `packages.txt`, `requirements.txt`

## Decisions Made

- Base dependencies = verified-import list only (grep over `src/`, not blind-copied requirements.txt) — excludes streamlit/plotly/dotenv (web-app deps, D-06), nltk/scipy/tqdm/emoji/regex/pytz (never imported), textblob (degrades to VADER-only), python-dateutil (transitive via pandas)
- `transformers>=4.30,<6` pin — 5.x breaks the 4.x-era core code; torch/transformers gated behind `[nlp]` extra (PKG-02/03)
- `requirements.txt` deleted — keeping both manifests re-creates the duplicated-manifests drift (CONCERNS.md:42-45); recoverable from git
- `chat-analyzer` console script declared now (entry-point resolution is lazy) — actual `chat_analyzer.cli` created and verified in plan 02

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Guarded emotion.py module-info emoji prints under `__main__`**
- **Found during:** Task 1 (marker/import-site edits)
- **Issue:** `src/analysis/emotion.py` ended with three top-level `print("🎭 ...")` module-info lines containing emoji. They executed at import time and raise `UnicodeEncodeError` on Windows cp1252 consoles (the dev machine's default), and pollute import output even when encoding succeeds. Would have broken `import chat_analyzer.analysis.emotion` in the smoke test.
- **Fix:** Moved the module-info block under `if __name__ == "__main__":` so it only runs when the module is executed directly.
- **Files modified:** src/chat_analyzer/analysis/emotion.py
- **Verification:** Import smoke test passes (`IMPORTS-OK`); no emoji print on import.
- **Committed in:** 1c279aa (Task 1 commit)

**2. [Rule 3 - Blocking] Installed 4 dev-deps from requirements.txt for the smoke test**
- **Found during:** Task 1 verify (import smoke test)
- **Issue:** The smoke test's full import surface required packages not present in the dev environment. Verified via dist-info timestamps (2026-07-31 18:09, immediately before the Task 1 commit): **seaborn, wordcloud, vaderSentiment, reportlab** were installed from requirements.txt (which still existed at that point; deleted later in Task 2). pandas/numpy/matplotlib/requests were already present.
- **Fix:** `pip install` of the 4 missing packages into user site-packages (environment-only change — no repo files touched, no pyproject drift).
- **Files modified:** none (environment only)
- **Verification:** Import smoke test passes (`IMPORTS-OK`).
- **Committed in:** n/a (environment setup for 1c279aa verification)

**3. [Rule 2 - Missing Critical] Aligned analysis/__init__.py docstring usage symbols with real exports**
- **Found during:** Task 1 (marker rewrite)
- **Issue:** The original marker docstring showed `from src.analysis.sentiment import analyze_sentiment` and `from src.analysis.emotion import classify_emotions` — both symbols do not exist (the old try/except swallowed the ImportError). Keeping them in the docstring after the `src.analysis` → `chat_analyzer.analysis` rewrite would leave a documentation example that cannot run.
- **Fix:** While rewriting docstring paths, also aligned the symbols to real exports: `quick_sentiment_analysis` (sentiment.py:412) and `EmotionAnalyzer` (emotion.py:33) — both verified to exist.
- **Files modified:** src/chat_analyzer/analysis/__init__.py
- **Verification:** Symbols confirmed present in module source; docstring matches actual API.
- **Committed in:** 1c279aa (Task 1 commit)

**4. [Process] Soft-reset re-commit of Task 1**
- **Found during:** Task 1 commit step
- **Issue:** The initial Task 1 commit (`8a5a623`, 18:12:52) was soft-reset to HEAD~1 and re-committed as `1c279aa` (18:13:26) with an identical tree (`git diff 8a5a623 1c279aa` is empty) and identical message — a commit-metadata/ordering correction, not a content change.
- **Fix:** Re-committed with corrected metadata; final commit `1c279aa` is the canonical Task 1 commit.
- **Files modified:** none (commit history only)
- **Verification:** `git diff 8a5a623 1c279aa` empty; reflog confirms `reset: moving to HEAD~1`.
- **Committed in:** 1c279aa

---

**Total deviations:** 4 (2 bug/blocking auto-fixes, 1 doc-correctness fix, 1 process correction)
**Impact on plan:** All fixes necessary for correctness or local execution. No scope creep; no plan-mandated file was left unfixed.

## Issues Encountered

- Smoke test emitted expected degrade-not-crash INFO logs: `pytesseract not available - OCR disabled`, `pdfplumber not available`, `pdf2image not available` from `ingest/ingestion.py`. These are the existing optional-dependency degrade paths (PIL-only ingestion), not failures — note for plan 02/03 whether these optional OCR deps should be declared.
- No auth gates encountered (Task 3 was a human-verify legitimacy gate, not an auth gate).

## User Setup Required

None - no external service configuration required. Task 3's human verification (checking typer/rich/plotext/hatchling on pypi.org) is complete and recorded in the plan.

## Verification Results

All plan gates pass (re-run in full on continuation):

| Gate | Result |
|------|--------|
| `python -m compileall -q src/chat_analyzer` | exit 0 |
| `*_init_.py` files remaining | 0 |
| `from src\.` / `import src\.` matches (all .py) | 0 |
| Import smoke (transformers NOT installed) | IMPORTS-OK |
| Light marker sys.modules assertion (pandas/matplotlib/reportlab/seaborn/networkx) | LIGHT-OK |
| tomllib pyproject assertions | PYPROJECT-OK |
| Deletion Test-Path (app, deployment, .streamlit, apt.txt, packages.txt, requirements.txt) | all False |
| `3.8` matches in PROJECT.md | 0 |
| analysis/__init__.py: `plot_relationship_health_dashboard_enhanced` present; analyze_sentiment/perform_eda/classify_emotions absent | PASS |
| parser/__init__.py: `parse_telegram_chat` present; `parse_telegram_json` absent | PASS |
| reporting/__init__.py: `chat_analyzer.reporting` present; `src.reporting` absent | PASS |
| ingest/__init__.py: `process_uploaded_file` present | PASS |
| summarizer.py: `from transformers import` only at line 51 (inside `__init__` try-block) | PASS |
| `__version__` (0.1.0) matches pyproject `version` | PASS |
| `gsd-sdk query verify.key-links` | {"verified":true} |
| Task 3 package-legitimacy gate | human-approved (commit 85df6d0) |

## Known Stubs

- **`chat-analyzer` console script → `chat_analyzer.cli:app`** (pyproject.toml `[project.scripts]`): the target module does not exist yet — INTENTIONAL, explicitly planned as key_link D-01 ("target created in plan 02"). Entry-point resolution is lazy so `pip install -e .` succeeds now; invocation verified in plan 02. Not a blocker for this plan's goal (structural package readiness).

## Next Phase Readiness

- **Plan 01-02 can `pip install -e .` immediately** — gate approved, pyproject valid, hatchling backend in place.
- Environment already satisfies most base deps (pandas 3.0.2, numpy 1.26.4, matplotlib 3.10.8, seaborn 0.13.2, networkx 3.6.1, reportlab 5.0.0, requests 2.32.5, vaderSentiment 3.3.2, wordcloud 1.9.6, Pillow 12.1.0; dev: pytest 9.0.2, pytest-cov 7.1.0, ruff 0.14.11). Plan 02 install will add: **plotext (not installed)**, upgrade **typer 0.24.1 → >=0.27**, and hatchling.
- `transformers`/`torch` deliberately NOT installed — lazy gate proven; `[nlp]` extra install is a plan-02+ decision (or Phase 4).
- The four package-legitimacy findings (typer 0.27.x, rich 15.x, plotext 5.3.x, hatchling 1.31.x) recorded in the plan file's gate annotation — plan 02 can reference them without re-verification.
- OCR optional deps (pytesseract/pdfplumber/pdf2image) undeclared — decide in plan 02/03 whether to add to an extra.
- Phase 2 must NOT re-introduce `from src.*` imports; the zero-match grep gate is the standing invariant.

---
*Phase: 01-package-foundation*
*Completed: 2026-07-31*
