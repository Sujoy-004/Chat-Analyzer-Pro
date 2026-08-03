# Roadmap: Chat-Analyzer-Pro

## Overview

This is a brownfield pivot: the existing Streamlit app's analysis core (`src/`) is repackaged into a pip-installable CLI tool — one command, `analyze <chat_file>`, turns a raw WhatsApp `.txt` or Telegram `.json` export into terminal insights with inline charts plus a self-contained, shareable HTML report. The analysis engine already exists and is reused as-is; the work is exposure, packaging, and robustness.

Four coarse-grained phases (compressing the research plan's 7 dependency-ordered steps) walk from "it installs" → "one command prints correct insights" → "shareable report" → "heavy NLP + quality gate". Every phase ships a complete, user-observable capability; no phase delivers a horizontal technical layer.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Package Foundation** - Repackage `src/` → `src/chat_analyzer/`, pyproject with `[nlp]` extra, working `analyze` command; existing analysis core survives intact
- [x] **Phase 2: One-Command Terminal Insights** - Parser hardening + pipeline + self-contained HTML report card: correct insights in one command
- [ ] **Phase 3: Shareable HTML Report** - Re-scope pending (OUT-03/04/05 pulled forward into Phase 2)
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

**Wave 1** — 01-01-PLAN (package surgery: restructure + pyproject + deletions) — DONE

**Wave 2** — 01-02-PLAN (CLI interactive slice) — DONE

**Cross-cutting constraints:** lean base install (no torch/transformers/streamlit/plotly in base — gated behind `[nlp]` extra with lazy imports); Python >=3.11 floor enforced; reuse existing analysis modules (no rewrite); reporting modules move but are NOT wired into the CLI; console script `chat-analyzer` (D-01) with `python -m chat_analyzer` fallback (D-02)

### Phase 2: One-Command Terminal Insights
**Goal**: One command (`chat-analyzer <file>` or `python -m chat_analyzer`) parses a real WhatsApp `.txt` or Telegram `.json` export correctly and produces a self-contained, decorated HTML report card with inferred insights. Terminal shows stage narration, a compact summary panel, skip counts, and the absolute report path. The report auto-opens.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: CLI-02, CLI-03, ANAL-01, ANAL-02, ANAL-03, ANAL-04, ANAL-05, OUT-01, OUT-03, OUT-04, OUT-05, CLI-08
**Success Criteria** (what must be TRUE):
  1. User runs `chat-analyzer chat.txt` or `chat-analyzer telegram.json` and the full pipeline runs automatically end-to-end, producing the report in one command
  2. User sees summary statistics (total messages, participants, date range), per-participant statistics (messages, average length, response behavior), timeline/activity trends (messages per day/week/hour, busiest times), top words and emojis, and VADER sentiment breakdown
  3. User sees the decorated HTML report card with inferred insights (replaces the plotext inline-chart wording)
  4. User sees a progress indicator while the pipeline runs, with the parsed-message count surfaced early
  5. User's reported timestamps and counts match the export — no fabricated dates for unfamiliar date formats, and skipped/malformed lines are counted and surfaced rather than silently dropped
**Plans**: 1 plan (02-PLAN.md)

### Phase 3: Shareable HTML Report
**Goal**: Re-scope pending — OUT-03/04/05 were pulled forward into Phase 2; the single-file report card already ships with default-path behavior (no `--output`/`--no-report` flags in v1). This phase is flagged for re-scope or absorption into Phase 4 during the next planning pass.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: OUT-03, OUT-04, OUT-05 (moved to Phase 2 — see traceability)
**Success Criteria** (what must be TRUE):
  1. User can generate a self-contained single-file HTML report (charts/images base64-embedded, no external assets, opens offline by double-click) — DELIVERED in Phase 2
  2. User can specify the output path with `--output` and the report is written there, with the absolute path printed — flag deferred (D-03); default-path behavior ships
  3. User can skip the report (`--no-report`) and still get full terminal output — not applicable in Phase 2; revisit in Phase 4
  4. All chat-derived content in the report is escaped — a message containing markup cannot inject script into a shared report — DELIVERED in Phase 2
**Plans**: TBD (re-scope)

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
**Plans**: TBD (add note: OUT-05 `--no-report` semantics revisit lands here)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Package Foundation | 2/2 | Complete | 2026-08-01 |
| 2. One-Command Terminal Insights | 1/1 | In Progress | - |
| 3. Shareable HTML Report | TBD | Not started | - |
| 4. NLP Extras & Quality Gate | TBD | Not started | - |
