# Phase 1: Package Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-31
**Phase:** 1-Package Foundation
**Areas discussed:** Command name, CLI interaction style, Web app fate, Distribution, Package scope

---

## Command Name

| Option | Description | Selected |
|--------|-------------|----------|
| chat-analyzer | Descriptive, no PyPI collision, memorable | ✓ |
| cpro | Short but cryptic | |
| analyze (collision) | Keeps PROJECT.md name, collides on PyPI | |
| Both (alias) | Main + short alias | |

**User's choice:** chat-analyzer
**Notes:** Also confirmed keeping the `python -m chat_analyzer` fallback entry point.

---

## CLI Interaction Style

| Option | Description | Selected |
|--------|-------------|----------|
| Interactive prompt | Tool asks "Enter path to chat export:" — zero CLI knowledge needed | ✓ |
| Prompt + arg support | Prompt if no path given, arg also accepted | |
| Argument only | Strictly positional, friend must know syntax | |

**User's choice:** Interactive prompt only
**Notes:** The friend's flow is: clone repo → open terminal → `python -m chat_analyzer` → terminal asks for the exported file → calculations run. This emerged from the Distribution discussion when the user described the exact workflow.

---

## Web App Fate

| Option | Description | Selected |
|--------|-------------|----------|
| Delete entirely | Remove app/, deployment/, .streamlit/, apt.txt, packages.txt | ✓ |
| Archive in repo | Move to web-app/ folder, keep in git history | |
| Leave in place | Excluded from package but untouched | |

**User's choice:** Delete entirely
**Notes:** The Streamlit app is fully replaced by the CLI. Git history preserves the old app.

---

## Distribution

| Option | Description | Selected |
|--------|-------------|----------|
| Publish to PyPI | Friend runs `pip install chat-analyzer-pro` | |
| Install from GitHub | `pip install git+https://...` | |
| Local install only | `pip install .` locally | |

**User's choice:** Clone-and-run (freeform, none of the options exactly matched)
**Notes:** User specified: "my friend will clone this repo. open it. and from the terminal run the 'python -m chat_analyzer'. terminal will ask for the exported file, then the calculations will be done accordingly." This drove the CLI Interaction Style decision. Package must still be pip-installable (PKG-01..05) but no PyPI publication needed in v1.

---

## Package Scope

| Option | Description | Selected |
|--------|-------------|----------|
| All modules | All of src/analysis, src/parser, src/ingest, src/utils | ✓ |
| Core path only | Parsers, ingest, EDA, sentiment only | |
| All + reporting | Everything including pdf_report, weekly_digest | |

**User's choice:** All modules
**Notes:** Reporting modules move in the restructure but their CLI exposure is deferred to v2 (PDF/digest out of scope for v1 CLI).

---

## the agent's Discretion

- `pyproject.toml` build backend and metadata (hatchling recommended by research)
- How to clean stale re-exports in the `_init_.py` → `__init__.py` rename
- Fixing the 3 `from src.*` import sites after the move
- Whether `src/reporting` stays in the moved tree (recommended: yes, moved)

## Deferred Ideas

- PyPI publication and `pip install git+...` instructions — beyond v1
- PDF report and Telegram digest CLI exposure — v2 (OUT-06, OUT-07)
- Instagram/Messenger/Discord import — v2 (FMT-01..03)
