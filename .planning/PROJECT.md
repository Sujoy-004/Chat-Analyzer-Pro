# Chat-Analyzer-Pro

## What This Is

A pip-installable command-line tool that analyzes WhatsApp and Telegram chat exports. A user points it at an exported chat file (`analyze chat.txt`), and it parses, analyzes, and visualizes the conversation — printing insights and inline charts to the terminal and generating a shareable HTML report. It keeps the heavy NLP features (emotion classification, summarization) running locally.

This is a pivot from the existing Streamlit web app: same analysis core (`src/`), new CLI interface, no hosting.

## Core Value

One command turns a raw chat export into real insights about the conversation — locally, fast, no accounts, no hosting.

## Requirements

### Validated

<!-- Shipped and confirmed valuable (from existing codebase). -->

- ✓ Parse WhatsApp `.txt` exports — existing `src/parser/whatsapp_parser.py`
- ✓ Parse Telegram `.json` exports — existing `src/parser/telegram_parser.py`
- ✓ Sentiment analysis (VADER + optional HuggingFace) — existing `src/analysis/sentiment.py`
- ✓ Emotion classification — existing `src/analysis/emotion.py`
- ✓ Conversation summarization — existing `src/analysis/summarizer.py`
- ✓ Network graph analysis — existing `src/analysis/network_graph.py`
- ✓ Relationship health analysis — existing `src/analysis/relationship_health.py`
- ✓ EDA / descriptive statistics — existing `src/analysis/eda.py`
- ✓ Visualization generation — existing `src/utils/visualization.py`
- ✓ Text preprocessing — existing `src/utils/preprocessing.py`
- ✓ File ingestion (txt/json/pdf with optional OCR) — existing `src/ingest/ingestion.py`

### Active

<!-- Current scope. Building toward these. -->

- [ ] CLI entry point: single command like `analyze <chat_file>` that runs the full pipeline
- [ ] Terminal output: insights with inline charts (plotext/rich)
- [ ] HTML report generation: shareable report of the analysis
- [ ] Installable via pip (`pip install chat-analyzer-pro` + `analyze` command)
- [ ] User-friendly error handling and export instructions for WhatsApp/Telegram
- [ ] README updated with "how a friend uses it" quickstart

### Out of Scope

- Streamlit web deployment — replacing the web app, not hosting it; no Vercel (Streamlit can't run there)
- Instagram / Messenger / Discord / Signal import — clunky or no native export (v2 candidate)
- PDF report — existing `src/reporting/pdf_report.py` deferred unless asked for
- Telegram bot digest (`src/reporting/weekly_digest.py`) — deferred unless asked for
- GUI of any kind — this is a pure CLI tool

## Context

- Brownfield project: existing Streamlit app with a complete analysis core in `src/`, already mapped in `.planning/codebase/`.
- Motivations for the pivot: Streamlit takes too long to load; the README's Streamlit Cloud link is dead; recruiters prefer Vercel-deployed links but Vercel cannot run Streamlit.
- Terminal can render graphs via `plotext` (bar/line/scatter in-ASCII) and `rich` (tables/panels) — full insights stay in-terminal; HTML is the shareable "report card."
- Heavy NLP (torch, transformers) is fine in a CLI because it runs locally — install is slower, runtime is not constrained by server limits.
- Known codebase issues from mapping (`.planning/codebase/CONCERNS.md`) include an `exec()` of runtime-downloaded code and `unsafe_allow_html` — these are web-app concerns and should be dropped with the Streamlit app, not carried into the CLI.

## Constraints

- **Tech stack**: Python 3.8+ (runtime) / 3.10 (dev), pip packaging. Reuse existing `src/` analysis modules rather than rewriting analysis logic.
- **Dependencies**: Heavy NLP (torch, transformers) retained per decision — local runtime accepts the install cost.
- **Output**: Must produce both terminal output and an HTML report for v1.
- **Distribution**: pip-installable (PyPI or GitHub) — no hosting, no account, no web server.
- **Formats**: WhatsApp `.txt` + Telegram `.json` only for v1.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Pivot from Streamlit web app to CLI tool | Streamlit slow to load, dead README link, Vercel can't run Streamlit | — Pending |
| Deploy via pip (PyPI/GitHub), not Vercel | Vercel is incompatible with Streamlit and can't fit torch/transformers | — Pending |
| Reuse existing `src/` analysis modules | Parsing/analysis is already UI-independent — only the wrapper changes | — Pending |
| Keep heavy NLP (emotion, summarization) | User chose to keep features; CLI runs locally so 2GB deps are acceptable | — Pending |
| Terminal + HTML output | Terminal = instant insights with inline charts; HTML = shareable report card | — Pending |
| WhatsApp + Telegram formats | Both have native one-tap exports and parsers already exist in `src/` | — Pending |
| Drop web-app-only code (streamlit_app.py, unsafe_allow_html, exec-of-remote-code) | Not needed for CLI; removes the security concerns mapped in CONCERNS.md | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-07-31 after initialization*
