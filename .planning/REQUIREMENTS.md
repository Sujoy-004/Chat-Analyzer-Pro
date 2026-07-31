# Requirements: Chat-Analyzer-Pro

**Defined:** 2026-07-31
**Core Value:** One command turns a raw chat export into real insights about the conversation — locally, fast, no accounts, no hosting.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Core CLI

- [ ] **CLI-01**: User installs the tool with `pip install chat-analyzer-pro` and gets an `analyze` command
- [ ] **CLI-02**: User runs `analyze <chat_file>` on a WhatsApp `.txt` or Telegram `.json` export and the full pipeline runs automatically
- [ ] **CLI-03**: User sees a progress indicator while the pipeline runs
- [ ] **CLI-04**: User gets a friendly, actionable error with export instructions when a file can't be parsed
- [ ] **CLI-05**: User can see CLI help (`analyze --help`) with clear usage

### Analysis

- [ ] **ANAL-01**: User gets summary statistics (total messages, participants, date range, message counts)
- [ ] **ANAL-02**: User gets per-participant statistics (messages sent, average length, response behavior)
- [ ] **ANAL-03**: User gets timeline/activity trends (messages per day/week/hour, busiest times)
- [ ] **ANAL-04**: User gets top words and emojis with frequency
- [ ] **ANAL-05**: User gets sentiment analysis breakdown (VADER-based, per-message and per-participant)
- [ ] **ANAL-06**: User gets emotion classification (6-class) — requires `[nlp]` extra
- [ ] **ANAL-07**: User gets relationship health score — requires `[nlp]` extra
- [ ] **ANAL-08**: User gets conversation summarization — requires `[nlp]` extra
- [ ] **ANAL-09**: User gets network graph analysis — requires `[nlp]` extra

### Output

- [ ] **OUT-01**: User sees analysis results in the terminal with tables, panels, and color
- [ ] **OUT-02**: User sees inline charts in the terminal (bar/line via plotext)
- [ ] **OUT-03**: User gets a self-contained single-file HTML report (charts/images base64-embedded)
- [ ] **OUT-04**: User can specify an output path for the HTML report
- [ ] **OUT-05**: User can skip the HTML report (`--no-report`) or keep it minimal

### Packaging

- [ ] **PKG-01**: Code is restructured into a single importable package (`src/chat_analyzer/`) with valid package markers
- [ ] **PKG-02**: Heavy NLP deps (torch, transformers) are gated behind an optional `[nlp]` extra and lazy imports
- [ ] **PKG-03**: Base install avoids heavy deps and installs quickly
- [ ] **PKG-04**: Python >= 3.11 floor is documented and enforced
- [ ] **PKG-05**: Package is installable from the repo (and PyPI-ready)

### Quality

- [ ] **QUAL-01**: Existing analysis modules still work (parsers, sentiment, analysis functions)
- [ ] **QUAL-02**: Tests pass for the new CLI (parse → analyze → render pipeline)
- [ ] **QUAL-03**: README documents the "how a friend uses it" quickstart (export → pip install → one command)
- [ ] **QUAL-04**: Web-app-only code (streamlit_app.py, unsafe_allow_html, exec-of-remote-code) is removed or excluded from the package

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Formats

- **FMT-01**: User can analyze Instagram DMs (JSON via Meta portal)
- **FMT-02**: User can analyze Facebook Messenger (JSON via Meta portal)
- **FMT-03**: User can analyze Discord exports

### Output

- **OUT-06**: User can generate a PDF report
- **OUT-07**: User can generate a Telegram bot weekly digest

### CLI

- **CLI-06**: User can run a `--light` fast path (skip heavy analysis)
- **CLI-07**: User can filter analysis by date range or participant
- **CLI-08**: User can auto-open the HTML report after generation

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Streamlit/web deployment | Replacing the web app entirely; Vercel cannot run Streamlit |
| GUI of any kind | Pure CLI tool by decision |
| TUI/interactive mode | Research identifies as anti-feature; scope creep |
| Cloud processing / upload / telemetry | Kills the privacy story (local-only is the selling point) |
| PDF export (v1) | HTML report covers v1 sharing needs |
| Telegram bot digest (v1) | Deferred to v2; not core to one-command analysis |
| Instagram/Messenger/Discord import (v1) | Clunky/non-native exports; v2 candidate |
| OCR/PDF chat ingestion | Existing capability but not needed for WhatsApp/Telegram CLI v1 |
| Carrying over web-app `exec()`/`unsafe_allow_html` patterns | Security concern from CONCERNS.md; excluded from CLI |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLI-01 | Phase 1 | Pending |
| CLI-02 | Phase 1 | Pending |
| CLI-03 | Phase 1 | Pending |
| CLI-04 | Phase 1 | Pending |
| CLI-05 | Phase 1 | Pending |
| ANAL-01 | Phase 1 | Pending |
| ANAL-02 | Phase 1 | Pending |
| ANAL-03 | Phase 1 | Pending |
| ANAL-04 | Phase 1 | Pending |
| ANAL-05 | Phase 1 | Pending |
| ANAL-06 | Phase 1 | Pending |
| ANAL-07 | Phase 1 | Pending |
| ANAL-08 | Phase 1 | Pending |
| ANAL-09 | Phase 1 | Pending |
| OUT-01 | Phase 1 | Pending |
| OUT-02 | Phase 1 | Pending |
| OUT-03 | Phase 1 | Pending |
| OUT-04 | Phase 1 | Pending |
| OUT-05 | Phase 1 | Pending |
| PKG-01 | Phase 1 | Pending |
| PKG-02 | Phase 1 | Pending |
| PKG-03 | Phase 1 | Pending |
| PKG-04 | Phase 1 | Pending |
| PKG-05 | Phase 1 | Pending |
| QUAL-01 | Phase 1 | Pending |
| QUAL-02 | Phase 1 | Pending |
| QUAL-03 | Phase 1 | Pending |
| QUAL-04 | Phase 1 | Pending |

**Coverage:**
- v1 requirements: 28 total
- Mapped to phases: 0
- Unmapped: 28 (awaiting roadmap)

---
*Requirements defined: 2026-07-31*
*Last updated: 2026-07-31 after initial definition*
