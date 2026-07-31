# Roadmap: Chat-Analyzer-Pro

## Overview

This is a brownfield pivot: the existing Streamlit app's analysis core (`src/`) is repackaged into a pip-installable CLI tool — one command, `analyze <chat_file>`, turns a raw WhatsApp `.txt` or Telegram `.json` export into terminal insights with inline charts plus a self-contained, shareable HTML report. The analysis engine already exists and is reused as-is; the work is exposure, packaging, and robustness.

Four coarse-grained phases (compressing the research plan's 7 dependency-ordered steps) walk from "it installs" → "one command prints correct insights" → "shareable report" → "heavy NLP + quality gate". Every phase ships a complete, user-observable capability; no phase delivers a horizontal technical layer.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Package Foundation** - Repackage `src/` → `src/chat_analyzer/`, pyproject with `[nlp]` extra, working `analyze` command; existing analysis core survives intact
- [ ] **Phase 2: One-Command Terminal Insights** - Parser hardening + pipeline + rich/plotext terminal output: correct insights in one command
- [ ] **Phase 3: Shareable HTML Report** - Self-contained single-file report with `--output` path and `--no-report` opt-out
- [ ] **Phase 4: NLP Extras & Quality Gate** - `[nlp]`-gated features (emotion, health, summary, network), friendly errors with export instructions, tests, README quickstart

## Phase Details

### Phase 1: Package Foundation
**Goal**: The tool is pip-installable and exposes a working `analyze` command; the existing analysis core survives the restructure intact.
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: PKG-01, PKG-02, PKG-03, PKG-04, PKG-05, CLI-01, CLI-05, QUAL-01, QUAL-04
**Success Criteria** (what must be TRUE):
  1. User can `pip install` the project and gets an `analyze` command (plus a working `python -m chat_analyzer` fallback) that responds with `--help` instantly
  2. User's base install downloads no torch/transformers/streamlit/plotly — `pip install .[nlp]` is the documented way to get heavy deps, and `pip freeze` shows the lean base set
  3. Installing on Python older than 3.11 fails with a clear error message (floor enforced)
  4. All existing analysis modules still import and run after the move to `chat_analyzer.*` (parsers, sentiment, EDA, relationship health produce results)
  5. The installed package contains no web-app-only code (no `app/`, no `deployment/`, no `exec()` module fetcher)
**Plans**: 2 plans
Plans:
- [ ] 01-01-PLAN.md — Package surgery: src/ → src/chat_analyzer/ restructure, pyproject (>=3.11, [nlp] extra), web-app deletion, dep legitimacy gate
- [ ] 01-02-PLAN.md — CLI interactive slice: installable command, prompt flow + smoke tests, QUAL-01/04 quality gates

### Phase 2: One-Command Terminal Insights
**Goal**: `analyze <chat_file>` parses real WhatsApp/Telegram exports correctly and prints trusted insights with inline charts to the terminal.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: CLI-02, CLI-03, ANAL-01, ANAL-02, ANAL-03, ANAL-04, ANAL-05, OUT-01, OUT-02
**Success Criteria** (what must be TRUE):
  1. User runs `analyze chat.txt` or `analyze telegram.json` and the full pipeline runs automatically end-to-end, printing results in one command
  2. User sees summary statistics (total messages, participants, date range), per-participant statistics (messages, average length, response behavior), timeline/activity trends (messages per day/week/hour, busiest times), top words and emojis, and VADER sentiment breakdown
  3. User sees results in the terminal as rich tables/panels with color, plus inline ASCII charts for trends (bar/line via plotext)
  4. User sees a progress indicator while the pipeline runs, with the parsed-message count surfaced early
  5. User's reported timestamps and counts match the export — no fabricated dates for unfamiliar date formats, and skipped/malformed lines are counted and surfaced rather than silently dropped
**Plans**: TBD

### Phase 3: Shareable HTML Report
**Goal**: User gets a self-contained single-file HTML report they can share by double-clicking.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: OUT-03, OUT-04, OUT-05
**Success Criteria** (what must be TRUE):
  1. User can generate a self-contained single-file HTML report (charts/images base64-embedded, no external assets, opens offline by double-click)
  2. User can specify the output path with `--output` and the report is written there, with the absolute path printed
  3. User can skip the report (`--no-report`) and still get full terminal output
  4. All chat-derived content in the report is escaped — a message containing markup cannot inject script into a shared report
**Plans**: TBD

### Phase 4: NLP Extras & Quality Gate
**Goal**: The full v1 feature set — heavy NLP insights gated behind the `[nlp]` extra, friendly errors with export instructions, tests that exercise the real code, and a README a friend can follow.
**Mode:** mvp
**Depends on**: Phase 2, Phase 3
**Requirements**: ANAL-06, ANAL-07, ANAL-08, ANAL-09, CLI-04, QUAL-02, QUAL-03
**Success Criteria** (what must be TRUE):
  1. User who installed the `[nlp]` extra gets 6-class emotion classification, relationship health score, conversation summarization, and network graph analysis
  2. User without the `[nlp]` extra gets an actionable hint (e.g., `pip install chat-analyzer-pro[nlp]`) instead of a traceback, and model downloads are announced with name and size before they start
  3. User who runs the tool on a missing, wrong-format, empty, or unparseable file gets a friendly, actionable error with WhatsApp/Telegram export instructions and a correct exit code
  4. Tests exercise the real `chat_analyzer.*` modules (parser fixtures, parse → analyze → render pipeline, HTML/encoding) and pass in a clean environment
  5. A friend can follow the README quickstart (export → pip install → one command) and get results
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Package Foundation | 0 / 2 | Not started | - |
| 2. One-Command Terminal Insights | TBD | Not started | - |
| 3. Shareable HTML Report | TBD | Not started | - |
| 4. NLP Extras & Quality Gate | TBD | Not started | - |
