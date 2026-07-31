# Phase 1: Package Foundation - Context

**Gathered:** 2026-07-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Repackage the existing `src/` library into `src/chat_analyzer/` as a real importable package (valid `__init__.py` markers), add a `pyproject.toml` with an `[nlp]` optional extra, expose a working `chat-analyzer` command (plus `python -m chat_analyzer` fallback) with an interactive file-path prompt, and ensure the existing analysis core survives the restructure intact. The web app is deleted entirely — no Streamlit code ships.

</domain>

<decisions>
## Implementation Decisions

### Command Name
- **D-01:** The main command is `chat-analyzer` (console script `chat-analyzer = chat_analyzer.cli:app`). Avoids the PyPI `analyze` collision.
- **D-02:** Keep `python -m chat_analyzer` working as a fallback entry point for environments where the console script isn't on PATH.

### CLI Interaction Style
- **D-03:** The primary UX is interactive: running `chat-analyzer` (or `python -m chat_analyzer`) with no arguments prompts the user for the chat file path (e.g., "Enter path to chat export:"). No CLI-argument knowledge required — target user is a non-technical friend who clones the repo and runs one command.
- **D-04:** No positional-argument support required in v1 — the interactive prompt is the flow. (Argument-style invocation is not needed; do not over-engineer flag support beyond `--help`.)

### Web App Fate
- **D-05:** Delete `app/`, `deployment/`, `.streamlit/`, `apt.txt`, and `packages.txt` entirely from the repo. The Streamlit app is fully replaced by the CLI. Git history preserves the old app if ever needed.
- **D-06:** The pip package must contain no web-app-only code — no `exec()` module fetcher, no `unsafe_allow_html` patterns, no Streamlit/plotly dependencies.

### Distribution
- **D-07:** v1 distribution is clone-and-run: friend clones the repo, opens a terminal in it, and runs `python -m chat_analyzer`. The console script `chat-analyzer` still exists for PATH-installed users.
- **D-08:** The project must still be `pip install`-able (pyproject + build backend correct, `pip install .` works) since PKG-01..05 require a valid installable package — but publishing to PyPI or providing `pip install git+...` instructions is NOT required in v1.
- **D-09:** Python floor `>=3.11` is enforced; PROJECT.md's stale "3.8+" constraint must be updated during this phase.

### Package Scope
- **D-10:** Ship ALL existing `src/` modules in the package: `src/analysis` (EDA, sentiment, emotion, relationship_health, network_graph, summarizer), `src/parser` (whatsapp, telegram), `src/ingest`, `src/utils` (preprocessing, visualization). Nothing is excluded from the package in Phase 1.
- **D-11:** Reporting modules (`src/reporting`) — PDF and weekly digest — are included in the package only insofar as repackaging moves them; their CLI exposure is deferred to v2 (PDF/digest output explicitly out of scope for v1). Do not wire them into the CLI in this phase.

### the agent's Discretion
- Exact `pyproject.toml` build backend choice (hatchling recommended by research) and package metadata
- How to clean the stale re-exports in the `_init_.py` → `__init__.py` rename (verified approach: clean the markers of broken re-export imports; they currently import functions that don't exist)
- Which of the 3 `from src.*` import sites need fixing after the move
- Whether `src/reporting` stays in the tree (moved) or is handled differently during restructure

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project context
- `.planning/PROJECT.md` — project context, core value, constraints, key decisions (contains the stale `>=3.8` constraint that must be corrected in this phase)
- `.planning/REQUIREMENTS.md` — 28 v1 requirements; this phase owns PKG-01..05, CLI-01, CLI-05, QUAL-01, QUAL-04
- `.planning/ROADMAP.md` §Phase 1 — phase goal, success criteria (5), requirement mapping

### Research (from new-project)
- `.planning/research/STACK.md` — verified stack: Python >=3.11 floor, typer, rich, plotext, jinja2, hatchling/uv; `transformers<6` pin; repackaging findings
- `.planning/research/ARCHITECTURE.md` — package layout (`src/chat_analyzer/`), CLI subpackage design, `[nlp]` extra + lazy imports, 4 surgical core changes
- `.planning/research/SUMMARY.md` — consolidated recommendations including the `_init_.py` rename and 9 import-site fix

### Codebase map
- `.planning/codebase/STACK.md` — current dependencies, runtime, configuration (docker/heroku references become obsolete this phase)
- `.planning/codebase/ARCHITECTURE.md` — component responsibilities, the `_init_.py` package-marker problem (line 202), divergent requirements files (line 228), anti-patterns
- `.planning/codebase/STRUCTURE.md` — directory layout, the `_init_.py` quirk (§Special Directories, lines 175-179), naming conventions
- `.planning/codebase/CONVENTIONS.md` — code style to preserve when cleaning re-exports
- `.planning/codebase/CONCERNS.md` — the `exec()`/`unsafe_allow_html` web-app concerns that must NOT carry into the CLI

### Project instruction file
- `AGENTS.md` — project conventions enforced during planning/execution (Python >=3.11, no heavy deps in base install, reuse analysis modules)

No external specs — requirements fully captured in decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Entire `src/` library (`src/parser/`, `src/analysis/`, `src/ingest/`, `src/utils/`): the analysis core being repackaged as-is; all modules take/return DataFrames or dicts
- `src/ingest/ingestion.py` `process_uploaded_file` (line 399) already accepts file paths — reusable for the CLI's file input
- `normalize_message()` (`src/ingest/ingestion.py:323`) — canonical message contract the whole pipeline relies on
- Sample chats in `data/sample_chats/` (whatsapp_sample.txt, telegram_sample.json) — usable as fixtures for `--help` smoke tests

### Established Patterns
- Function-style modules with a few classes (WhatsAppParser, ChatEDA, EmotionAnalyzer, etc.)
- Optional-dependency gates via try/except ImportError (DEPENDENCIES dict, `*_AVAILABLE` flags) — the `[nlp]` lazy-import pattern should follow this existing convention
- `if __name__ == "__main__":` demo blocks in most modules — candidates for cleanup or preservation during restructure

### Integration Points
- The `src/` → `src/chat_analyzer/` move requires fixing 3 known `from src.*` import sites (`src/analysis/relationship_health.py:800`, `src/analysis/emotion.py:15`, `src/utils/visualization.py:685` per research)
- New CLI subpackage `chat_analyzer/cli/` is the only new code in this phase; it must not import heavy deps at module load
- `_init_.py` → `__init__.py` rename activates currently-silent re-exports — they must be cleaned (they import functions that don't exist, e.g. `plot_relationship_health_dashboard` vs actual `plot_relationship_health_dashboard_enhanced`)

</code_context>

<specifics>
## Specific Ideas

- The target user is a real friend cloning the repo — "Enter path to chat export:" prompt must be dead simple; no flags to learn
- The CLI should work offline and locally; no accounts, no hosting, no telemetry
- `python -m chat_analyzer` is the documented primary flow for v1 (matches D-07 distribution decision)

</specifics>

<deferred>
## Deferred Ideas

- PyPI publication and `pip install git+...` instructions — deferred beyond v1 (D-08 keeps installability, but no publishing)
- PDF report and Telegram digest CLI exposure — v2 (OUT-06, OUT-07)
- Instagram/Messenger/Discord import — v2 (FMT-01..03)

</deferred>

---

*Phase: 1-Package Foundation*
*Context gathered: 2026-07-31*
