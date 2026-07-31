# AGENTS.md

Project instruction file for Chat-Analyzer-Pro.

## Project

Chat-Analyzer-Pro is being pivoted from a Streamlit web app into a pip-installable CLI tool. One command — `analyze <chat_file>` — turns a WhatsApp `.txt` or Telegram `.json` export into terminal insights (with inline charts) plus a self-contained HTML report. Heavy NLP (emotion, summarization, relationship health, network graph) is gated behind an optional `[nlp]` extra.

The analysis core in `src/` is being repackaged into `src/chat_analyzer/` (valid package markers required for PyPI shipping). Existing modules: parsers, sentiment (VADER), emotion, summarizer, network graph, relationship health, EDA, visualization, preprocessing, ingestion, reporting.

## GSD Workflow

This repo uses the GSD planning system. Planning docs live in `.planning/`:
- `PROJECT.md` — project context and requirements
- `ROADMAP.md` — phases (4 phases, MVP mode)
- `REQUIREMENTS.md` — 28 v1 requirements (REQ-IDs: CLI-*, ANAL-*, OUT-*, PKG-*, QUAL-*)
- `STATE.md` — project memory
- `codebase/` — codebase map (STACK, ARCHITECTURE, STRUCTURE, CONVENTIONS, TESTING, INTEGRATIONS, CONCERNS)
- `research/` — domain research (STACK, FEATURES, ARCHITECTURE, PITFALLS, SUMMARY)

Workflows: `/gsd-plan-phase`, `/gsd-execute-phase`, `/gsd-discuss-phase`, `/gsd-verify-work`, `/gsd-progress`.

## Conventions

- Python >= 3.11 floor (enforced)
- Base install must NOT pull torch/transformers/streamlit/plotly — heavy deps gated behind `[nlp]` extra with lazy imports
- Reuse existing analysis modules; do not rewrite analysis logic
- Terminal output via typer/rich/plotext; HTML report is single-file (base64-embedded assets), chat content escaped
- No web-app-only code: do not reintroduce `exec()` of remote code or `unsafe_allow_html` patterns
- Tests exercise real `chat_analyzer.*` modules

## Verification

- Tests: pytest (see `.planning/codebase/TESTING.md` for current patterns)
- Run lint/typecheck per phase plan before committing
