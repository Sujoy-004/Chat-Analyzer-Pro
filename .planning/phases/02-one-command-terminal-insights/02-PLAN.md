---
phase: 02-one-command-terminal-insights
plan: 02
type: execute
wave: 1
depends_on: ["01-01", "01-02"]
files_modified:
  - "src/chat_analyzer/cli/contracts.py (NEW)"
  - "src/chat_analyzer/cli/pipeline.py (NEW)"
  - "src/chat_analyzer/cli/adapters.py (NEW)"
  - "src/chat_analyzer/cli/render.py (NEW)"
  - "src/chat_analyzer/cli/report_html.py (NEW)"
  - "src/chat_analyzer/cli/main.py (MODIFIED)"
  - "src/chat_analyzer/parser/whatsapp_parser.py (MODIFIED)"
  - "src/chat_analyzer/parser/telegram_parser.py (MODIFIED)"
  - "src/chat_analyzer/ingest/ingestion.py (MODIFIED)"
  - "src/chat_analyzer/utils/visualization.py (MODIFIED, 1 line)"
  - "pyproject.toml (MODIFIED)"
  - "tests/fixtures/whatsapp_system_skip.txt (NEW)"
  - "tests/fixtures/telegram_full_export.json (NEW)"
  - "tests/fixtures/telegram_bare_entity.json (NEW)"
  - "tests/test_phase2_whatsapp.py (NEW)"
  - "tests/test_phase2_telegram.py (NEW)"
  - "tests/test_phase2_builder.py (NEW)"
  - "tests/test_phase2_pipeline.py (NEW)"
  - "tests/test_phase2_report.py (NEW)"
  - "tests/test_phase2_cli.py (NEW)"
autonomous: true
requirements: [CLI-02, CLI-03, ANAL-01, ANAL-02, ANAL-03, ANAL-04, ANAL-05, OUT-01, OUT-03, OUT-04, OUT-05, CLI-08]

must_haves:
  truths:
    - "User runs one command (`chat-analyzer <file>` or `python -m chat_analyzer` + interactive prompt) and the full pipeline runs end-to-end automatically"
    - "Terminal narrates each stage (Parsing/Computing/Writing) with ASCII spinners and surfaces the parsed-message count immediately after parsing (D-05, CLI-03)"
    - "No fabricated timestamps anywhere: unparseable lines are counted in skipped_lines and surfaced on a single line; a fake timestamp is corrupt data (D-15/D-16)"
    - "System messages (encryption notice, X added Y, header-without-sender) are classified as type=system, counted, and never appended to the previous message (D-18)"
    - "Telegram exports parse in both shapes (bare Chat + chats.list[]), join entity-array text recursively, filter service messages, and normalize tz-aware dates to naive UTC (D-19/D-20)"
    - "User gets a self-contained, decorated HTML report card: 5 tabbed sections, each opening with a narrative insight lead-in, matplotlib charts base64-embedded, top words/emojis, VADER sentiment (D-10..D-13)"
    - "Report is written next to the input as `<sanitized_name>_report.html`, auto-opens in the default browser (degrading to a printed path on failure), and every chat-derived byte is HTML-escaped (D-08/D-09/D-10/D-14)"
  artifacts:
    - path: "src/chat_analyzer/cli/contracts.py"
      provides: "ParseReport dataclass + AnalysisResults TypedDict — the single contract consumed by pipeline/adapters/render/report_html"
      contains: "class AnalysisResults"
    - path: "src/chat_analyzer/cli/pipeline.py"
      provides: "run_pipeline(path, console) -> AnalysisResults — the ONLY orchestration (Agg bootstrap, parse threading, analysis, charts→data URI)"
      exports: ["run_pipeline", "fig_to_data_uri"]
    - path: "src/chat_analyzer/cli/adapters.py"
      provides: "Module dicts (ChatEDA/sentiment/parse counts) → AnalysisResults + narrative insight lead-in builders (D-11)"
      exports: ["adapt"]
    - path: "src/chat_analyzer/cli/render.py"
      provides: "rich terminal narration: parsed count, skip line (D-16), ASCII summary panel (D-07), absolute path (D-08)"
      exports: ["show_summary"]
    - path: "src/chat_analyzer/cli/report_html.py"
      provides: "jinja2 autoescape single-file report (5 tabs, lead-ins, base64 charts), filename sanitize (D-14), utf-8 write, auto-open (D-09)"
      exports: ["write_report", "open_report", "sanitize_filename"]
    - path: "src/chat_analyzer/parser/whatsapp_parser.py"
      provides: "Strict date parsing (no datetime.now() at lines 61/63/77/79), system classification (D-18), skip counter (D-15), parse_file_with_report"
      contains: "parse_file_with_report"
    - path: "src/chat_analyzer/parser/telegram_parser.py"
      provides: "Both JSON shapes (D-19), recursive text join, service filter, tz→naive UTC (D-20), parse_telegram_chat_with_report"
      contains: "parse_telegram_chat_with_report"
    - path: "src/chat_analyzer/ingest/ingestion.py"
      provides: "messages_to_dataframe() — the single canonical dict→df builder (tz-naive, timestamp alias); normalize_message tz-safe ISO handling"
      contains: "def messages_to_dataframe"
    - path: "pyproject.toml"
      provides: "Manifest: + jinja2>=3.1, − plotext>=5.3 (OUT-02 dropped, verified imported nowhere)"
      contains: "jinja2"
  key_links:
    - from: "src/chat_analyzer/cli/pipeline.py"
      to: "src/chat_analyzer/parser/whatsapp_parser.py"
      via: "parse_file_with_report call in run_pipeline"
      pattern: "parse_file_with_report"
    - from: "src/chat_analyzer/cli/pipeline.py"
      to: "src/chat_analyzer/parser/telegram_parser.py"
      via: "parse_telegram_chat_with_report call in run_pipeline"
      pattern: "parse_telegram_chat_with_report"
    - from: "src/chat_analyzer/cli/pipeline.py"
      to: "src/chat_analyzer/ingest/ingestion.py"
      via: "messages_to_dataframe(rows) — single canonical builder (Anti-Pattern 5)"
      pattern: "messages_to_dataframe"
    - from: "src/chat_analyzer/cli/pipeline.py"
      to: "matplotlib.use('Agg')"
      via: "first line of run_pipeline — before any matplotlib import (D-12, Pitfall 7)"
      pattern: "matplotlib.use"
    - from: "src/chat_analyzer/cli/adapters.py"
      to: "src/chat_analyzer/cli/contracts.py"
      via: "adapt() returns AnalysisResults consumed by render + report_html"
      pattern: "AnalysisResults"
    - from: "src/chat_analyzer/cli/main.py"
      to: "src/chat_analyzer/cli/pipeline.py"
      via: "positional chat_file / interactive prompt routes into run_pipeline"
      pattern: "run_pipeline"
    - from: "src/chat_analyzer/cli/report_html.py"
      to: "webbrowser.open"
      via: "open_report() — file:// URL in try/except per D-09"
      pattern: "webbrowser.open"
---

# Phase 2: One-Command Terminal Insights — Plan

**One command → trusted parse → decorated HTML report card.** `chat-analyzer <chat_file>`
(or `python -m chat_analyzer` with the interactive prompt) strictly parses a real
WhatsApp `.txt` / Telegram `.json` export, computes insights with the existing
analysis core (ChatEDA + VADER — **reused, never rewritten**), and produces a
self-contained, tabbed HTML report card with narrative insight lead-ins and
base64-embedded matplotlib charts. The terminal stays thin: stage narration,
an immediate parsed-message count, a skip-count line, a compact summary panel,
and the absolute report path. The report auto-opens in the default browser.

> **CONTEXT reshaped this phase (02-CONTEXT.md):** the HTML report card is the
> primary deliverable. OUT-03/04/05 and CLI-08 are **pulled forward** into this
> phase. OUT-02 (plotext inline terminal charts) is **dropped — plotext never
> ships**. Relationship health (ANAL-07), emotion/summary/network (ANAL-06/08/09)
> and friendly export-instruction errors (CLI-04) stay Phase 4.

## User Story

**As a** user who wants to understand what a conversation is really about,
**I want to** run one command on a WhatsApp or Telegram chat export and get a
decorated, self-contained HTML report card (auto-opened in my browser) that
shows where the conversation is going — backed by charts and narrative insights —
**so that** I immediately see the insights without any setup, flags, or
interpretation.

*(Story derived from 02-CONTEXT.md `<domain>` and `<specifics>`; the ROADMAP Phase 2
goal line is stale and is re-written post-planning — see "Post-Planning Doc Updates".)*

## Phase Requirements Mapping

Reconciled per 02-CONTEXT.md decisions. **REQUIREMENTS.md/ROADMAP.md are stale and
must be updated post-planning** (see section below); this table is the new truth.

| Req ID | Description (reconciled) | Status in Phase 2 | Delivered By |
|--------|--------------------------|-------------------|--------------|
| CLI-02 | `chat-analyzer <chat_file>` runs the full pipeline automatically | **IN SCOPE** (D-02 positional arg; no-arg still prompts) | Task 8, Task 5, Task 9 |
| CLI-03 | Progress indicator + parsed-message count surfaced early | **IN SCOPE** (D-05 rich Status, ASCII `line` spinner) | Task 6, Task 5 |
| ANAL-01 | Summary stats (volume, participants, date range, counts) | **IN SCOPE** (D-13 full depth via ChatEDA) | Task 5 |
| ANAL-02 | Per-participant stats (messages, avg length, response behavior) | **IN SCOPE** | Task 5 |
| ANAL-03 | Timeline/activity trends (per day/week/hour, busiest times) | **IN SCOPE** | Task 5 |
| ANAL-04 | Top words and emojis with frequency | **IN SCOPE** | Task 5 |
| ANAL-05 | VADER sentiment breakdown (per-message + per-participant) | **IN SCOPE** (VADER always available in base install) | Task 5 |
| OUT-01 | Terminal tables/panels — **REVERSED** by D-04/D-07 | Terminal shows a **compact summary panel** (volume + participants + date range) + absolute path only; insights live in the HTML report | Task 6 |
| OUT-02 | Inline plotext terminal charts | **DROPPED — plotext never ships**; charts exist only in the HTML report. Remove `plotext>=5.3` from pyproject (verified imported nowhere in `src/`) | Task 4 (removal) |
| OUT-03 | Self-contained single-file HTML report | **PULLED FORWARD from Phase 3** (D-04/D-10) | Task 7 |
| OUT-04 | Output path — **default-path behavior only** | **PULLED FORWARD**: `<sanitized_input_stem>_report.html` next to the input (D-08/D-14); the `--output` flag is deferred (D-03) | Task 7 |
| OUT-05 | `--no-report` opt-out | **PULLED FORWARD — resolved**: report IS the deliverable (D-04), no flags ship (D-03) → report is always generated; NOT applicable in Phase 2 | Task 7 (always-on) |
| CLI-08 | Auto-open the HTML report | **PULLED FORWARD from v2** (D-09) with graceful degradation to path print | Task 7 |
| ANAL-07 | Relationship health | **NOT in Phase 2** — stays Phase 4 (`[nlp]`-labeled) | — |
| CLI-04, ANAL-06/08/09 | Friendly export-instruction errors; emotion/summary/network | **NOT in Phase 2** — Phase 4 | — |

## Reconciliation Summary (CONTEXT → plan)

- **Phases 2+3 merged:** the report card is the deliverable; the terminal is
  entry point + progress + pointer (D-04).
- **Parser hardening is correctness-critical:** both live bugs were demonstrated
  at research time — an unparseable WhatsApp date was silently stamped
  `datetime.now()` (now at `whatsapp_parser.py:61,63,77,79`), and the encryption
  notice was appended to the previous message body. Both end here.
- **Pipeline data path (research Open Question 1):** `run_pipeline` calls the
  **hardened parser modules directly** for `.txt`/`.json` (rows + counts dict);
  `process_uploaded_file` is untouched for other formats/back-compat.
- **OUT-05 (Open Question 2):** resolved as "not applicable in Phase 2 — report
  always generated"; REQUIREMENTS.md gets a note for Phase 4.
- **Parser API (Open Question 3):** parsers return `(rows, counts_dict)`; the
  `ParseReport` dataclass lives in `cli/contracts.py` (core never imports `cli.*`,
  so parsers return plain dicts the pipeline wraps). Old signatures preserved.
- **Sentiment print capture (Open Question 4):** `contextlib.redirect_stdout`
  around the analysis stage in `pipeline.py` (Pitfall 5).
- **`--version` (Open Question 5):** `chat-analyzer 0.1.0` via
  `importlib.metadata.version("chat-analyzer-pro")` — typer 0.27 has **no**
  `version` param [VERIFIED], so a manual eager callback.

## Dependency Graph

```
Task 1 (contracts + fixtures)  ──►  Task 2 (WhatsApp)   ─┐
                              ──►  Task 3 (Telegram)   ─┼─► Task 4 (df builder + pyproject)
                                                          └─► Task 5 (pipeline + adapters) ─► Task 6 (render)
                                                                                                  │
Task 7 (report_html) ─► Task 8 (main.py wiring) ◄───────────┘
                                   │
Task 9 (CLI e2e tests + full verification)
```

| Wave | Tasks | Files touched (no intra-wave overlap) | Parallel |
|------|-------|--------------------------------------|----------|
| 0 | Task 1 — contracts + fixtures | `cli/contracts.py`, `tests/fixtures/*` | — (foundation) |
| 1 | Task 2 — WhatsApp hardening; Task 3 — Telegram hardening | Task 2: `parser/whatsapp_parser.py`, `tests/test_phase2_whatsapp.py`; Task 3: `parser/telegram_parser.py`, `tests/test_phase2_telegram.py` | **Yes** (disjoint files) |
| 2 | Task 4 — `messages_to_dataframe` + `normalize_message` + pyproject | `ingest/ingestion.py`, `pyproject.toml`, `tests/test_phase2_builder.py` | after Wave 1 |
| 3 | Task 5 — pipeline + adapters + viz logging fix; Task 6 — render | Task 5: `cli/pipeline.py`, `cli/adapters.py`, `utils/visualization.py`, `tests/test_phase2_pipeline.py`; Task 6: `cli/render.py` | **Yes** (Task 6 does not touch Task 5 files) |
| 4 | Task 7 — report_html; Task 8 — main.py wiring | Task 7: `cli/report_html.py`, `tests/test_phase2_report.py`; Task 8: `cli/main.py` | **Yes** |
| 5 | Task 9 — CLI e2e tests + full verification | `tests/test_phase2_cli.py` | after Wave 4 |

## Task Breakdown

### Wave 0 — Contracts & fixtures (everything downstream consumes these)

- [ ] **Task 1: Write interface contracts (`cli/contracts.py`) and parser test fixtures**

<task type="auto">
  <name>Task 1: Write interface contracts (cli/contracts.py) and parser test fixtures</name>
  <files>
    src/chat_analyzer/cli/contracts.py (NEW), tests/fixtures/whatsapp_system_skip.txt (NEW),
    tests/fixtures/telegram_full_export.json (NEW), tests/fixtures/telegram_bare_entity.json (NEW)
  </files>
  <action>
    Create `src/chat_analyzer/cli/contracts.py` — the single contract for the phase. Content:

    - `@dataclass class ParseReport`: fields `source: str` ("whatsapp"|"telegram"),
      `total_lines: int = 0`, `parsed_messages: int = 0`, `skipped_lines: int = 0`,
      `system_messages: int = 0`. Constructed by `pipeline.py` from the parsers'
      plain counts dicts (core never imports `cli.*` — the parsers return dicts).
    - `class AnalysisResults(TypedDict)`: keys `source: str`; `parse: Dict[str, int]`
      ({"parsed_messages", "skipped_lines", "system_messages"}); `stats: Dict[str, Any]`
      (total_messages, participants, date_range{"start","end"}, duration_days,
      busiest_day, peak_hour, avg_response_time, media_messages);
      `participants: Dict[str, Any]` ({sender: {"messages", "avg_length", "share_pct"}});
      `content: Dict[str, Any]` (top_words: list[tuple[str,int]], top_emojis: list[tuple[str,int]],
      total_words, unique_words); `sentiment: Dict[str, Any]` (distribution,
      avg_compound, by_sender, daily_avg); `charts: Dict[str, str]`
      (timeline|activity|participants|sentiment → base64 PNG data URI);
      `insights: List[str]` (narrative lead-ins, D-11); `report_path: str` (filled by
      `main.py` after `report_html.write_report`).
    - Use double quotes, `from typing import Any, Dict, List, TypedDict`, `from dataclasses import dataclass`.
      Do NOT put implementation logic here — contracts only (TypedDicts are runtime-checkable no-ops).

    Create `tests/fixtures/` directory and three fixture files (exact content, byte-for-byte —
    tests assert exact counts):

    1. `tests/fixtures/whatsapp_system_skip.txt` (exactly 7 lines, LF line endings):
       ```
       12/25/23, 9:30 AM - Alice: First message
       Messages and calls are end-to-end encrypted.
       25/13/26, 9:30 AM - Alice: bad date line
       12/25/23, 9:31 AM - Alice: Second message
       this is a continuation line
       12/25/23, 9:32 AM - Bob: Third message
       Alice added Bob
       ```
       Expected counts (locked in the plan): parsed=3, skipped=1 (line 3, month 13),
       system=2 (line 2 encryption notice + line 7 bare "Alice added Bob"), total_lines=7.
       Line 5 must append to "Second message" body; no row may contain "end-to-end encrypted".

    2. `tests/fixtures/telegram_full_export.json` (chats.list[] shape):
       ```json
       {"chats": [{"name": "Team", "type": "private_supergroup", "messages": [
         {"id": 1, "type": "message", "date": "2025-09-15T09:45:00", "from": "Sujoy",
          "text": [{"type": "bold", "text": "hello "}, "world"]},
         {"id": 2, "type": "service", "date": "2025-09-15T09:50:00", "actor": "Ravi",
          "action": "edit_group_title", "title": "Team"},
         {"id": 3, "type": "message", "date": "2025-09-15T10:00:00Z", "from": "Ravi", "text": "zoned time"},
         {"id": 4, "type": "message", "date": "2025-09-15T10:01:00", "from": "Ananya", "text": "ok"},
         {"id": 5, "type": "message", "date": "not-a-date", "from": "X", "text": "bad"},
         {"id": 6, "type": "forwarded", "from": "Y", "date": "2025-09-15T10:02:00", "text": "fwd"}
       ]}]}
       ```
       Expected: parsed=3 (ids 1,3,4), system=1 (id 2), skipped=2 (id 5 bad date, id 6 non-message type).

    3. `tests/fixtures/telegram_bare_entity.json` (bare Chat + entity-array text):
       ```json
       {"name": "Chat", "type": "personal_chat", "messages": [
         {"id": 1, "type": "message", "date": "2025-09-15T09:45:00", "from": "Sujoy",
          "text": [{"type": "mention", "text": "@team"}, " check this"]},
         {"id": 2, "type": "message", "date": "2025-09-15T09:46:00+05:30", "from": "Ravi", "text": "offset time"}
       ]}
       ```
       Expected: parsed=2, system=0, skipped=0; msg 1 text joined to "@team check this";
       msg 2's `+05:30` offset normalizes to naive UTC `04:16:00`.

    Do NOT modify the existing `data/sample_chats/*` (Phase 1 smoke tests assert
    whatsapp_sample.txt parses to 27 messages).
  </action>
  <verify>
    <automated>
      $env:PYTHONPATH="src"; python -c "from chat_analyzer.cli.contracts import ParseReport, AnalysisResults; r=ParseReport(source='whatsapp'); assert r.parsed_messages==0; print('CONTRACTS-OK')"
      # fixture files exist and line counts match
      (Get-Content tests/fixtures/whatsapp_system_skip.txt).Count -eq 7
      Test-Path tests/fixtures/telegram_full_export.json; Test-Path tests/fixtures/telegram_bare_entity.json
    </automated>
  </verify>
  <done>
    contracts.py imports with both symbols; the three fixture files exist with the exact
    content above; existing sample_chats untouched.
  </done>
</task>

### Wave 1 — Parser hardening (Tasks 2 and 3 run in parallel; disjoint files)

- [ ] **Task 2: Harden `whatsapp_parser.py` — strict dates (no `datetime.now()`), system classification, skip counting (D-15/D-16/D-17/D-18)**
- [ ] **Task 3: Harden `telegram_parser.py` — both JSON shapes, recursive text join, service filter, tz→naive UTC, skip counting (D-19/D-20)**

<task type="tdd" tdd="true">
  <name>Task 2: Harden WhatsApp parser (strict dates, system classification, counters)</name>
  <files>
    src/chat_analyzer/parser/whatsapp_parser.py, tests/test_phase2_whatsapp.py (NEW)
  </files>
  <behavior>
    RED first — create tests/test_phase2_whatsapp.py with these expectations before touching the parser:
    - Test 1 (no fabrication): a line matching the header regex with an unparseable date
      (month 13, e.g. "25/13/26, 9:30 AM - Alice: hello") produces NO row and increments
      skipped_lines; assert no parsed timestamp equals today's date (datetime.now().date()).
    - Test 2 (system messages): "Messages and calls are end-to-end encrypted." and a bare
      "Alice added Bob" line are counted in system_messages and NEVER appended to the
      previous message body (assert no row body contains "end-to-end encrypted" or
      "added Bob").
    - Test 3 (header-without-sender): a line with a timestamp header but no "sender: "
      part (e.g. "12/26/23, 10:00 AM - Alice joined the group" WITHOUT colon-sender is not
      this case — use a genuinely senderless header line like "12/26/23, 10:00 AM - Group
      renamed") is counted as system, never a continuation.
    - Test 4 (common formats): US 12h ("12/25/23, 9:30 AM"), EU 24h ("25/12/2023, 21:07"),
      iOS bracket ("[14/06/2024, 2:30:45 PM] Maria: msg"), 4-digit year — each parses to the
      correct datetime (D-17: no M/D-vs-D/M heuristics — %m/%d is tried first; document only).
    - Test 5 (multiline): a continuation line joins the previous message with "\n".
    - Test 6 (exact counts): parse_file_with_report(tests/fixtures/whatsapp_system_skip.txt)
      returns parsed=3, skipped=1, system=2, total_lines=7, and exactly 3 rows.
    - Test 7 (QUAL-01): parse_file(path) still returns a pandas DataFrame (hardened internals).
  </behavior>
  <action>
    Implement in `src/chat_analyzer/parser/whatsapp_parser.py` (modify, do not rewrite):

    1. **Delete the four `datetime.now()` fallbacks** at lines 61, 63, 77, 79 entirely
       (D-15). Add a class-level `DATE_FORMATS` tuple (module-level is fine) listing the
       common formats only (D-17 — no disambiguation heuristics):
       `%m/%d/%y %I:%M %p`, `%d/%m/%y %I:%M %p`, `%m/%d/%Y %I:%M %p`, `%d/%m/%Y %I:%M %p`,
       `%m/%d/%y %I:%M:%S %p`, `%d/%m/%y %I:%M:%S %p`, `%m/%d/%y %H:%M`, `%d/%m/%y %H:%M`,
       `%m/%d/%Y %H:%M`, `%d/%m/%Y %H:%M`, `%m/%d/%y %H:%M:%S`, `%d/%m/%y %H:%M:%S`.
    2. Add a private `_parse_datetime_strict(self, datetime_str) -> Optional[datetime]`
       that tries each format and returns None on total failure (research Pitfall 1 code
       example) — NEVER `datetime.now()`.
    3. **Counter state:** in `__init__` add `self.skipped_lines = 0`,
       `self.system_messages = 0`, `self.total_lines = 0` (reset per parse call — see step 6).
    4. **System classification (D-18):** add a header-only regex
       `self.system_header_pattern = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}),?\s(\d{1,2}:\d{2}(?::\d{2})?)\s?([AaPp][Mm])?\s?-\s(.+)$")`
       and a bare-system-phrase regex for the Pitfall 3 no-header cases:
       `self.system_phrase_pattern = re.compile(r"^(.+?)\s+(added|removed|left|joined|created (the )?group|changed (the )?(group|subject|name)|named)\b", re.IGNORECASE)`
       gated to lines <= 120 chars to avoid swallowing prose continuations. Also match the
       encryption notice explicitly:
       `self.encryption_notice = re.compile(r"^Messages and calls are end-to-end encrypted\.?$", re.IGNORECASE)`.
    5. **Per-line classification order in a new `parse_line_strict(line) -> Optional[Dict]`**
       (keep `parse_line` as a thin delegate for QUAL-01, or have parse_line call it):
       a. strip line; if empty → None.
       b. `message_pattern`/`alt_pattern` match WITH sender → parse date strictly; if
          `_parse_datetime_strict` returns None → `self.skipped_lines += 1`, return None
          (never a continuation — D-15); else return the message dict with key `datetime`
          (not `timestamp`), plus `sender`, `message`, `message_length`, `type: "message"`.
       c. encryption-notice match → `self.system_messages += 1`; return None.
       d. system-header match (timestamp, no sender) → `self.system_messages += 1`; return None.
       e. system-phrase match → `self.system_messages += 1`; return None.
       f. otherwise → None (caller decides continuation vs skip).
    6. **`parse_file_with_report(self, file_path: str) -> Tuple[List[Dict], Dict]`** — NEW
       entry point: reset counters, open the file with `encoding="utf-8-sig", errors="replace"`
       (BOM-safe, Pitfall Integration table), loop lines with `self.total_lines += 1` per
       non-empty line, classify per step 5; a bare non-message line when a current message
       exists → append as continuation (`"\n" + stripped`); a bare non-message line with NO
       current message → `self.skipped_lines += 1` (honest count, no orphan fabrication).
       Returns `(rows, {"total_lines":..., "parsed_messages": len(rows),
       "skipped_lines":..., "system_messages":...})`. System rows never enter `rows`.
    7. **`parse_file(self, file_path) -> pd.DataFrame`** — KEPT for QUAL-01: delegate to the
       same hardened loop (call `parse_file_with_report`), build the df from rows, then
       `df = df.rename(columns={"datetime": "timestamp"})` before the existing
       `_add_features` call so its `sort_values('timestamp')` keeps working. Behavior change
       is the point: failed-date lines are dropped (counted), never stamped "now".
  </action>
  <verify>
    <automated>
      # RED→GREEN: tests first, then implementation
      $env:PYTHONPATH="src"; python -m pytest tests/test_phase2_whatsapp.py -q
      # grep gate: zero datetime.now() remaining in the parser (grep hygiene: count code tokens, not comments)
      (Select-String -Path src/chat_analyzer/parser/whatsapp_parser.py -Pattern "datetime\.now\(" | Measure-Object).Count
      # QUAL-01 regression: whole phase-1 import matrix still green
      $env:PYTHONPATH="src"; python -m pytest tests/test_phase1_smoke.py::test_import_matrix -q
    </automated>
  </verify>
  <done>
    All 7 behavior tests pass; exactly 0 matches for `datetime.now()` in whatsapp_parser.py;
    parse_file_with_report returns exact counts on the fixture; test_import_matrix still passes.
  </done>
</task>

<task type="tdd" tdd="true">
  <name>Task 3: Harden Telegram parser (both shapes, recursive text join, service filter, tz→naive UTC)</name>
  <files>
    src/chat_analyzer/parser/telegram_parser.py, tests/test_phase2_telegram.py (NEW)
  </files>
  <behavior>
    RED first — create tests/test_phase2_telegram.py before implementation:
    - Test 1 (bare Chat): parse_telegram_chat_with_report("data/sample_chats/telegram_sample.json")
      → parsed=5, system=0, skipped=0 (existing fixture, 5 messages).
    - Test 2 (chats.list[]): tests/fixtures/telegram_full_export.json → parsed=3, system=1,
      skipped=2, total_lines=6; msg 1 text == "hello world" (recursive join of str + entity dict).
    - Test 3 (entity array, bare Chat): tests/fixtures/telegram_bare_entity.json → parsed=2;
      msg 1 text == "@team check this".
    - Test 4 (service excluded): no row has sender "Ravi" for the service message (id 2 in
      full-export); system count increments instead.
    - Test 5 (tz→naive UTC, D-20): every returned row datetime has `.tzinfo is None`; the
      Z-suffix row ("2025-09-15T10:00:00Z") is 10:00:00 naive UTC; the "+05:30" row is
      04:16:00 naive UTC.
    - Test 6 (malformed dropped honestly): bad-date message → skipped_lines += 1 (never a
      silent bare `except: continue`); non-"message" type (e.g. "forwarded") → skipped += 1.
    - Test 7 (not a chat export): `{"chats": []}` or a dict with neither "messages" nor
      "chats" → raises ValueError containing "Not a Telegram chat export".
    - Test 8 (QUAL-01): parse_telegram_chat(source) still returns a pandas DataFrame
      (hardened internals; system/service rows excluded; datetime naive UTC).
  </behavior>
  <action>
    Implement in `src/chat_analyzer/parser/telegram_parser.py` (modify, do not rewrite):

    1. Add module-level helpers per research Pitfall 4 code example:
       - `_load_messages(data: dict) -> list`: if `data.get("messages")` is a list → return
         it (bare Chat, single-chat export); elif `data.get("chats")` is a list → flatten
         `chat["messages"]` for every dict chat that has a messages list (full export);
         else raise `ValueError("Not a Telegram chat export (no 'messages' or 'chats' key)")`
         — replaces the `data.get('messages', [])` at line 25 and the silent empty result.
       - `_join_text(parts) -> str`: str → as-is; list → join str parts AND entity-dict parts'
         `"text"` values (this fixes the dropped-dict-parts bug at lines 44-46); dict parts
         without a text key are skipped; empty result + `photo`/`video`/`document`/`audio`
         present → `"<Media omitted>"`.
       - `_to_naive_utc(date_str: str) -> datetime`: `datetime.fromisoformat(date_str.replace("Z", "+00:00"))`,
         then `.astimezone(timezone.utc).replace(tzinfo=None)` when tz-aware, pass-through
         when naive (D-20; Telegram exports are UTC per official schema).
    2. **`parse_telegram_chat_with_report(source: str) -> Tuple[List[Dict], Dict]`** — NEW:
       load JSON (keep the existing URL-vs-file handling with `requests`); `messages =
       _load_messages(data)`; loop each message (counters: `total_lines = len(messages)`):
       - `type == "service"` → `system_messages += 1`; continue (D-18/19).
       - `type != "message"` → `skipped_lines += 1`; continue (D-19 — honest, not silent).
       - `_to_naive_utc(msg["date"])` wrapped in try/except (ValueError, TypeError) → on
         failure `skipped_lines += 1`; continue — NEVER a bare `except: continue` that
         drops without counting (fixes lines 35-36).
       - `text = _join_text(msg.get("text"))`; `sender = msg.get("from") or msg.get("actor")
         or "Unknown"` (channels/service-adjacent posts; D-19).
       - row = `{"datetime": dt, "sender": sender, "message": text, "message_length": len(text)}`
         (optionally keep `id`/`message_id` when present — the builder tolerates extra keys).
       Returns `(rows, {"total_lines", "parsed_messages": len(rows), "skipped_lines", "system_messages"})`.
    3. **`parse_telegram_chat(source)`** — KEPT for QUAL-01: delegate to the hardened logic
       and return a DataFrame with the same core columns as today (datetime, sender, message,
       date, time, hour, message_length, message_id, type) — system/service rows excluded,
       datetime naive UTC.
  </action>
  <verify>
    <automated>
      $env:PYTHONPATH="src"; python -m pytest tests/test_phase2_telegram.py -q
      # no bare except:continue regression — assert no line matches "except:\s*$" followed by continue on next line
      Select-String -Path src/chat_analyzer/parser/telegram_parser.py -Pattern "except:\s*$" | Measure-Object
      $env:PYTHONPATH="src"; python -m pytest tests/test_phase1_smoke.py::test_import_matrix -q
    </automated>
  </verify>
  <done>
    All 8 behavior tests pass; parse_telegram_chat_with_report returns exact counts on both
    new fixtures and 5/5 on the sample; every row datetime is tz-naive; no bare `except: continue`.
  </done>
</task>

### Wave 2 — Canonical DataFrame + manifest

- [ ] **Task 4: Add `messages_to_dataframe()` to `ingest/ingestion.py`, fix `normalize_message` tz-safe ISO handling, update pyproject (+jinja2, −plotext)**

<task type="tdd" tdd="true">
  <name>Task 4: Canonical dict→df builder (messages_to_dataframe), normalize_message ISO fix, pyproject deps</name>
  <files>
    src/chat_analyzer/ingest/ingestion.py, pyproject.toml, tests/test_phase2_builder.py (NEW)
  </files>
  <behavior>
    RED first — create tests/test_phase2_builder.py before implementation:
    - Test 1 (schema): messages_to_dataframe([{"datetime": <naive dt>, "sender": "A",
      "message": "hi"}]) yields columns datetime, timestamp, date, hour, sender, message,
      message_length, source, uid; `timestamp` values equal `datetime`; `date` is a date
      object; `hour` is int; `source` defaults "unknown"; `uid` is a str.
    - Test 2 (tz normalization, D-20): input datetime tz-aware (UTC+2) → output
      `df['datetime'].dt.tz is None` and the hour is the UTC hour (2h earlier).
    - Test 3 (ingestion-path telegram bug): dict with `date="2025-09-15T09:45:00", time=""`
      (full ISO in date field — verified live bug) → datetime 2025-09-15 09:45 naive.
    - Test 4 (ingestion-path whatsapp): dict with `date="2025-09-15", time="09:45"` →
      datetime 2025-09-15 09:45.
    - Test 5 (unparseable dropped): dict with no datetime/date/time → dropped from df
      (caller's skip accounting), no crash.
  </behavior>
  <action>
    Modify `src/chat_analyzer/ingest/ingestion.py` (additive):

    1. Add `from datetime import timezone` to the existing datetime import (line 25).
    2. Add a module-level `_to_naive_utc(value) -> datetime` helper (same logic as the
       telegram parser's — single normalization contract, D-20): accepts datetime objects
       and ISO strings (`str.replace("Z", "+00:00")` → `fromisoformat` → aware → naive UTC;
       naive → unchanged). Raise nothing — return the best-effort parse and let the caller
       decide.
    3. Add `def messages_to_dataframe(messages: List[Dict]) -> pd.DataFrame` (research
       Anti-Pattern 5 — the SINGLE source for dicts→df; never a second copy in `cli/`):
       per message: `dt = m.get("datetime") or m.get("timestamp")`; elif `"T" in str(m.get("date",""))`
       → `_to_naive_utc(m["date"])` (ingestion-path telegram full-ISO bug); elif
       `m.get("date") and m.get("time")` → `pd.to_datetime(f"{m['date']} {m['time']}")`;
       else → skip (continue — caller's skip accounting, never fabricate). Build rows with
       `datetime=dt`, `timestamp=dt` (alias — ChatVisualizer REQUIRES 'timestamp', verified
       at visualization.py:83/135/367), `date=dt.date()`, `hour=dt.hour`,
       `sender=m.get("author") or m.get("sender") or m.get("from") or "unknown"`,
       `message=m.get("text") or m.get("message") or ""`,
       `message_length=len(m.get("text") or m.get("message") or "")`,
       `source=m.get("source") or m.get("source_hint") or "unknown"`,
       `uid=m.get("uid") or m.get("id") or str(uuid.uuid4())`. Build `pd.DataFrame(rows)`,
       then `df["datetime"] = pd.to_datetime(df["datetime"])` (pandas 3.x-safe — no uppercase
       freq aliases anywhere in the new code; use lowercase/ISO only, per Pitfall 8). Return df.
    4. `normalize_message` (line 323): in the `elif raw_msg.get("datetime")` branch, replace
       `datetime.fromisoformat(str(raw_msg["datetime"]).replace('Z', '+00:00'))` with a call
       to `_to_naive_utc(str(raw_msg["datetime"]))` so tz-aware telegram values normalize to
       naive UTC at the ingestion boundary too. No other normalize_message changes.
    5. Do NOT modify `process_uploaded_file` (line 399) — it stays the back-compat path for
       other formats; the pipeline calls the hardened parsers directly (research Open Question 1).

    Modify `pyproject.toml`:
    - Add `"jinja2>=3.1",` to `[project] dependencies` (jinja2 3.1.6 is IN the env but MISSING
      from the manifest — verified; autoescape must be set explicitly in Task 7 because
      plain jinja2 `Environment` defaults to `autoescape=False`).
    - Remove `"plotext>=5.3",` from `[project] dependencies` (OUT-02 dropped; verified
      imported nowhere in `src/`). This does NOT break `tests/test_phase1_smoke.py` —
      `test_lean_base_structural` asserts typer/rich presence via pyproject but does NOT
      assert plotext (verified by reading the test).
  </action>
  <verify>
    <automated>
      $env:PYTHONPATH="src"; python -m pytest tests/test_phase2_builder.py -q
      python -c "import tomllib,pathlib; d=tomllib.loads(pathlib.Path('pyproject.toml').read_text()); deps=' '.join(d['project']['dependencies']); assert 'jinja2' in deps, 'jinja2 missing'; assert 'plotext' not in deps, 'plotext still present'; print('DEPS-OK')"
      # Phase 1 structural smoke still green (does not assert plotext)
      $env:PYTHONPATH="src"; python -m pytest tests/test_phase1_smoke.py::test_lean_base_structural -q
    </automated>
  </verify>
  <done>
    All 5 builder tests pass; DEPS-OK prints; test_lean_base_structural still passes.
  </done>
</task>

### Wave 3 — Pipeline, adapters, terminal render (Tasks 5 and 6 run in parallel; disjoint files)

- [ ] **Task 5: Build `cli/pipeline.py` + `cli/adapters.py` (Agg bootstrap, parse threading, analysis, charts→data URI, narrative insights); neutralize `visualization.py` logging (D-05/D-12/D-13/D-11)**
- [ ] **Task 6: Build `cli/render.py` — rich narration, ASCII summary panel, skip line, absolute path (D-05/D-07/D-16)**

<task type="tdd" tdd="true">
  <name>Task 5: Pipeline orchestration (run_pipeline) + adapters (AnalysisResults + narrative insights)</name>
  <files>
    src/chat_analyzer/cli/pipeline.py (NEW), src/chat_analyzer/cli/adapters.py (NEW),
    src/chat_analyzer/utils/visualization.py (MODIFIED, 1 line), tests/test_phase2_pipeline.py (NEW)
  </files>
  <behavior>
    RED first — create tests/test_phase2_pipeline.py before implementation:
    - Test 1 (whatsapp e2e): run_pipeline(data/sample_chats/whatsapp_sample.txt, Console(file=io.StringIO()))
      → AnalysisResults with source="whatsapp", parse.parsed_messages==27, parse.skipped_lines==0,
      stats.total_messages==27, stats.participants==2, date_range start "2023-12-25" end "2023-12-27",
      participants has 2 entries, content.top_words non-empty, sentiment.distribution non-empty,
      charts has exactly {"timeline","activity","participants","sentiment"} all starting with
      "data:image/png;base64,", insights is a non-empty list of non-empty strings.
    - Test 2 (telegram e2e): run_pipeline(data/sample_chats/telegram_sample.json, ...)
      → source="telegram", parse.parsed_messages==5.
    - Test 3 (all-skipped): run_pipeline on tests/fixtures/whatsapp_system_skip.txt is fine,
      but a fixture where EVERY line is unparseable → ValueError whose message contains
      "No messages could be parsed" (friendly, no traceback downstream).
    - Test 4 (no emoji-print pollution): during run_pipeline, analysis prints from
      sentiment.py ("🚀", "✅", "🔍" emoji lines) are captured and do NOT appear in the
      captured stdout buffer (redirect_stdout in the analysis stage, Pitfall 5).
    - Test 5 (unsupported format): run_pipeline(Path("chat.pdf"), ...) → ValueError
      containing "Unsupported" (only .txt/.json supported in this phase).
    - Test 6 (Agg headless): importing cli.pipeline and calling run_pipeline succeeds with
      no display/TclError; matplotlib backend is "Agg" during the run.
  </behavior>
  <action>
    Neutralize core logging first (CONTEXT-mandated, research Anti-Pattern 4):
    - `src/chat_analyzer/utils/visualization.py` line 19: replace
      `logging.basicConfig(level=logging.INFO)` with
      `logging.getLogger(__name__).addHandler(logging.NullHandler())` (keep line 20
      `logger = logging.getLogger(__name__)` and all 12 plot methods byte-identical).

    Create `src/chat_analyzer/cli/pipeline.py` (the ONLY orchestration — research Pattern 1):
    - `def run_pipeline(path: Path, console) -> AnalysisResults`:
      1. **First line**: `import matplotlib; matplotlib.use("Agg")` — BEFORE importing
         eda/sentiment/visualization (all three `import matplotlib.pyplot` at module top;
         Pitfall 7). Import everything else lazily inside the function (Anti-Pattern 2).
      2. Dispatch by suffix: `.txt` → `WhatsAppParser().parse_file_with_report(str(path))`;
         `.json` → `parse_telegram_chat_with_report(str(path))`; else raise
         `ValueError(f"Unsupported file type: {path.suffix} — expected .txt (WhatsApp) or .json (Telegram)")`.
         Wrap each in `with console.status("Parsing chat...", spinner="line"):` (ASCII
         spinner 'line' = `-\|/`, verified — D-05).
      3. **Immediately after parsing** (before analysis — D-05/CLI-03, Pitfall 9):
         `console.print(f"[OK] Parsed {counts['parsed_messages']} messages from {participants} participants")`
         (participant count computed from rows' senders) and, when
         `counts["skipped_lines"] > 0`, `console.print(f"[WARN] Skipped {counts['skipped_lines']} lines that couldn't be parsed")`
         (D-16 — single line, no per-line examples). Build `ParseReport(source=...,
         **counts)` from `cli/contracts.py`.
      4. `df = messages_to_dataframe(rows)` (ingest/ingestion.py — the single builder).
         If `df.empty`: raise `ValueError("No messages could be parsed from this file")`.
      5. `with console.status("Computing insights...", spinner="line"):` — wrap the whole
         analysis block in `with contextlib.redirect_stdout(io.StringIO()) as captured:`
         (captures sentiment.py's emoji `print()`s at module import AND per-call — Pitfall 5;
         log captured text via a module logger if desired, never to console):
         - `from chat_analyzer.analysis.eda import ChatEDA`; `eda = ChatEDA(df)`;
           `summary = eda.generate_comprehensive_summary()`;
           `volume = eda.analyze_message_volume()`;
           `dynamics = eda.analyze_conversation_dynamics()`;
           `content = eda.analyze_content()` (all REUSED, zero rewrites).
         - `from chat_analyzer.analysis.sentiment import add_sentiment_analysis, get_sentiment_summary`;
           `df_sent = add_sentiment_analysis(df)`; `sent_summary = get_sentiment_summary(df_sent)`
           (VADER path — always available in base install; `consensus_sentiment` degrades to
           VADER, verified A6).
         - Charts (D-12): `from chat_analyzer.utils.visualization import ChatVisualizer`;
           `viz = ChatVisualizer()`; build the 4 figures with
           `viz.plot_message_timeline(df, resample_freq="D")` (needs 'timestamp' — provided),
           `viz.plot_activity_heatmap(df)`, `viz.plot_user_activity(df, top_n=10)`,
           `viz.plot_sentiment_timeline(df_sent, sentiment_score_col="vader_compound")`;
           encode each via `fig_to_data_uri(fig)` (module-level helper: savefig to
           io.BytesIO png dpi=120 bbox_inches="tight", plt.close(fig) to avoid figure leak,
           base64 encode → "data:image/png;base64," + b64 — verified 25 KB data URI in env).
           Each chart wrapped in try/except that substitutes an empty string on failure —
           a chart crash must never kill the report (Pitfall 6 degrade spirit).
      6. Return `adapt(source, parse_report, df, summary, volume, dynamics, content, sent_summary, charts)`
         from `cli/adapters.py`.

    Create `src/chat_analyzer/cli/adapters.py` (research Pattern 2 — the ONLY place that
    knows each module's internal dict shape):
    - `def adapt(source, parse: ParseReport, df, summary, volume, dynamics, content,
      sentiment, charts) -> AnalysisResults`: build the contract keys from the module dicts —
      stats (total_messages from len(df); participants count + list; date_range from
      df['datetime'].min()/max() as %Y-%m-%d; duration_days; busiest day from
      volume['hourly_activity'].sum(axis=1).idxmax(); peak_hour from summary
      ['activity_patterns']['peak_hour']; avg_response_time from dynamics
      ['avg_response_time']; media_messages from df['message'].str.contains('<Media omitted>',
      case=False, na=False).sum()); participants dict (per-sender messages count, avg
      message_length, share_pct = messages/total*100 rounded to 1dp, sorted desc);
      content (top_words = content['word_frequency'].most_common(15); top_emojis =
      content['emoji_frequency'].most_common(15); total_words; unique_words); sentiment
      (distribution from sentiment['sentiment_distribution']; avg_compound from
      sentiment['average_scores'].get('vader_compound', {}).get('mean'); by_sender from
      sentiment['by_sender']; daily_avg from sentiment['temporal_analysis'].get('daily_avg_sentiment', {})).
      Defensive `.get()` everywhere — an empty edge-case dict must never KeyError.
    - `def build_insights(stats, participants, content, sentiment) -> List[str]` (D-11 —
      narrative lead-ins, one per tab, natural-language): e.g. (i) f"Most messages land on
      {busiest_day} — {busiest_share}% of the week's activity."; (ii) f"{top_sender} is the
      most active participant, sending {share}% of all messages."; (iii) f"Replies take on
      average {avg_response_time} minutes when they come at all."; (iv) f"The most-used word
      is '{top_word}'."; (v) f"The overall tone leans {dominant_sentiment} ({pct}% of messages).";
      (vi) f"This conversation spans {duration_days} days and {total_messages} messages.";
      (vii) f"The busiest hour is {peak_hour}:00." Return 5-7 short sentences; every value
      comes from the stats (no hardcoded text beyond sentence scaffolding).
    - `adapt()` returns a dict including `insights=build_insights(...)` and
      `parse={"parsed_messages":..., "skipped_lines":..., "system_messages":...}`.
  </action>
  <verify>
    <automated>
      $env:PYTHONPATH="src"; python -m pytest tests/test_phase2_pipeline.py -q
      # Agg-first proof: importing cli.pipeline must not import matplotlib.pyplot at module import time
      $env:PYTHONPATH="src"; python -c "import sys, chat_analyzer.cli.pipeline; assert 'matplotlib' not in sys.modules, 'eager matplotlib import in pipeline module top'; print('LAZY-OK')"
      # visualization logging neutralized — no basicConfig at import
      Select-String -Path src/chat_analyzer/utils/visualization.py -Pattern "logging\.basicConfig" | Measure-Object
    </automated>
  </verify>
  <done>
    All 6 pipeline tests pass; LAZY-OK prints; zero `logging.basicConfig` in visualization.py;
    run_pipeline produces a complete AnalysisResults (all 9 keys) for both sample exports.
  </done>
</task>

<task type="auto">
  <name>Task 6: Terminal narration (cli/render.py) — stages, parsed count, skip line, ASCII summary panel, path</name>
  <files>
    src/chat_analyzer/cli/render.py (NEW)
  </files>
  <action>
    Create `src/chat_analyzer/cli/render.py` — thin, ASCII-first (D-05/D-07/D-16; Pitfall 5):
    - `def show_summary(results: AnalysisResults, console) -> None`: prints, in order:
      1. Parsed-count line — already printed by pipeline right after parsing (D-05); this
         function does NOT repeat it (single-source narration: pipeline owns stage lines,
         render owns the end summary).
      2. Skip line when `results["parse"]["skipped_lines"] > 0`:
         `console.print(f"[WARN] Skipped {n} lines that couldn't be parsed")` (D-16 — one
         line, no examples).
      3. System line when `results["parse"]["system_messages"] > 0`:
         `console.print(f"[INFO] Excluded {n} system messages from stats")` (D-18 — counted
         and surfaced, not hidden).
      4. Compact ASCII summary panel (D-07 — the ONLY terminal insights; NO sentiment, NO
         tables, NO charts in the terminal):
         `console.print(Panel(f"Total messages: {stats['total_messages']}\nParticipants: {stats['participants']}\nDate range: {start} to {end}", title="Summary", box=box.ASCII))`
         — `from rich.panel import Panel; from rich import box`; `box.ASCII` is pure `+-|`
         (verified on cp1252-hostile consoles).
      5. Absolute report path: `console.print(f"Report: {results['report_path']}")` (D-08).
    - ASCII-safe symbols only (`[OK]`/`[WARN]`/`[INFO]`, `+`, `-`, `|`); no emoji, no
      box-drawing glyphs (Pitfall 5 — the CLI's utf-8 reconfigure is a safety net, not a
      license to ship non-ASCII).
    - No business logic: render reads AnalysisResults only.
  </action>
  <verify>
    <automated>
      $env:PYTHONPATH="src"; python -c "
import io
from rich.console import Console
from chat_analyzer.cli.render import show_summary
buf = io.StringIO(); c = Console(file=buf, force_terminal=False)
res = {'parse': {'parsed_messages': 3, 'skipped_lines': 2, 'system_messages': 1},
       'stats': {'total_messages': 3, 'participants': 2, 'date_range': {'start': '2023-12-25', 'end': '2023-12-27'}},
       'report_path': r'C:\x\chat_report.html'}
show_summary(res, c)
out = buf.getvalue()
for token in ('Skipped 2 lines', 'Excluded 1 system', 'Total messages: 3', 'Date range: 2023-12-25 to 2023-12-27', 'Report: C:\\x\\chat_report.html'):
    assert token in out, token
print('RENDER-OK')
"
    </automated>
  </verify>
  <done>
    RENDER-OK prints; show_summary emits exactly the D-07/D-16/D-18 lines in ASCII; no logic
    beyond reading AnalysisResults.
  </done>
</task>

### Wave 4 — HTML report card + CLI wiring (Tasks 7 and 8 run in parallel; disjoint files)

- [ ] **Task 7: Build `cli/report_html.py` — jinja2 autoescape single-file report card (5 tabs, lead-ins, base64 charts, sanitize, utf-8, auto-open) (D-08/D-09/D-10/D-11/D-12/D-14)**
- [ ] **Task 8: Wire `cli/main.py` — positional arg, `--version` eager callback, pipeline routing, re-prompt loop (D-01/D-02/D-03/D-06)**

<task type="tdd" tdd="true">
  <name>Task 7: HTML report card (report_html.py) — autoescape, 5 tabs, narrative lead-ins, base64 charts, sanitize, auto-open</name>
  <files>
    src/chat_analyzer/cli/report_html.py (NEW), tests/test_phase2_report.py (NEW)
  </files>
  <behavior>
    RED first — create tests/test_phase2_report.py before implementation. Generate a report
    from a crafted AnalysisResults (message containing `<script>alert(1)</script>` and `<3`
    in content + sender name "Alice <3 Bob") and assert:
    - Test 1 (single-file): output contains no `http://`/`https://`/`src=` external refs,
      no `<script src`, and contains at least one `data:image/png;base64,` chart URI.
    - Test 2 (charset/encoding): file starts with `<!DOCTYPE html>`, contains
      `<meta charset="utf-8">`; file reads back as valid UTF-8 with the emoji content intact.
    - Test 3 (escaping, D-10/V5): output does NOT contain the raw substring `<script>` from
      the message, does contain `&lt;script&gt;`; the `<3` in a sender name is escaped.
    - Test 4 (filename sanitize, D-14): sanitize_filename("..\\..\\chat<name>:1.txt") and
      sanitize_filename("con") / leading-dot names produce safe strings (no path separators,
      no control chars, no leading dots, no `< > : " / \\ | ? *`); empty-after-strip falls
      back to "chat_analysis".
    - Test 5 (location, D-08): write_report returns a Path equal to
      `input.parent / f"{sanitize(input.stem)}_report.html"`; the file exists.
    - Test 6 (auto-open degrade, D-09): with webbrowser.open monkeypatched to raise, calling
      open_report(path) does NOT raise and returns False; with it returning True, returns True
      and was called with a `file://` URL containing the resolved absolute path.
    - Test 7 (skip note): when parse.skipped_lines > 0 the HTML body contains a note with
      the skip count; when 0, no skip note.
    - Test 8 (tabs + insights): output contains all 5 tab ids (overview, participants, flow,
      words, sentiment) and the first insight sentence text.
  </behavior>
  <action>
    Create `src/chat_analyzer/cli/report_html.py` (research Pattern 4; Pitfall 11):
    - `TEMPLATE` as a module-level inline string constant (NOT a separate `.j2` file —
      sidesteps hatchling wheel package-data risk, research A1). Structure:
      `<!DOCTYPE html>`, `<html lang="en">`, `<head>` with `<meta charset="utf-8">`,
      `<title>{{ title }}</title>`, inline `<style>` (embedded CSS — card layout, tab
      buttons, tables, `img { max-width: 100% }`), and inline `<script>` with a tiny
      `showTab(id)` function toggling `.panel` visibility (no external libs, no CDN — D-10).
      Body: `<h1>{{ title }}</h1>`, `<p class="subtitle">{{ subtitle }}</p>`, tab button
      row, then 5 panels: `overview` (insight[0] + timeline chart + stats table),
      `participants` (insight + participant bar chart + per-participant table),
      `flow` (insight + activity heatmap + busiest day/hour/avg response table),
      `words` (insight + top-words table + top-emojis list),
      `sentiment` (insight + sentiment timeline chart + distribution table).
      Every value interpolated via jinja2 `{{ }}` (autoescape — no `|safe` anywhere except
      the base64 `charts` values, which are internally generated PNG data URIs — still
      validate with `startswith("data:image/png;base64,")` before passing to the template).
      Skip note rendered only when `parse.skipped_lines > 0`.
    - `def sanitize_filename(name: str) -> str` (D-14): strip path separators and invalid
      Windows chars via `re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', '', name)`; strip leading dots
      and whitespace; `name.strip(" .")`; fall back to `"chat_analysis"` if empty.
    - `def write_report(results: AnalysisResults, input_path: Path) -> Path`:
      `env = Environment(autoescape=select_autoescape(["html", "xml"]))` — **explicit**
      (plain jinja2 defaults to autoescape=False; VERIFIED env 3.1.6);
      `html_out = env.from_string(TEMPLATE).render(title=sanitize(input_path.stem).replace("_"," ").title()
      ... )`; report_path = `input_path.parent / f"{sanitize_filename(input_path.stem)}_report.html"`;
      `open(report_path, "w", encoding="utf-8")` (Pitfall 11 — NEVER platform-default);
      return report_path. `html.escape()` additionally applied to any raw string interpolated
      outside the template (defense in depth, V5).
    - `def open_report(path: Path) -> bool` (D-09): `webbrowser.open("file://" + str(path.resolve()))`
      inside try/except → returns success bool; caller (`main.py`) prints the path regardless.
  </action>
  <verify>
    <automated>
      $env:PYTHONPATH="src"; python -m pytest tests/test_phase2_report.py -q
      # no unsafe patterns shipped (QUAL-04 regression scan)
      Select-String -Path src/chat_analyzer/cli/report_html.py -Pattern "unsafe_allow_html|exec\(|markupsafe\.escape" | Measure-Object
    </automated>
  </verify>
  <done>
    All 8 report tests pass; report file is single-file, escaped, utf-8, next to input,
    sanitized; auto-open degrades without crashing; no unsafe tokens in report_html.py.
  </done>
</task>

<task type="auto">
  <name>Task 8: Wire cli/main.py — positional chat_file, --version eager callback, pipeline routing, re-prompt loop</name>
  <files>
    src/chat_analyzer/cli/main.py
  </files>
  <action>
    Modify `src/chat_analyzer/cli/main.py` (extend, don't rewrite — Phase 1 conventions hold:
    encoding bootstrap stays, BLE001 degrade-not-crash stays, docstring updated):

    1. **`--version`** (D-03 — typer 0.27.0 has NO `version` param, VERIFIED): add
       ```
       def _version_callback(value: bool) -> None:
           if value:
               from importlib.metadata import version
               typer.echo(f"chat-analyzer {version('chat-analyzer-pro')}")
               raise typer.Exit()
       ```
    2. **Command signature** (D-02/D-03 — positional argument, no other flags):
       `def main(chat_file: Optional[Path] = typer.Argument(None, help="Path to WhatsApp .txt or Telegram .json export"), version: Optional[bool] = typer.Option(None, "--version", is_eager=True, callback=_version_callback, help="Show version and exit")) -> None`
       Keep the existing utf-8 stdout/stderr reconfigure loop as the FIRST statements.
    3. **Positional path** (D-02): if `chat_file` is not None:
       - `if not chat_file.is_file(): typer.echo(f"File not found: {chat_file}", err=True); raise typer.Exit(1)`.
       - if suffix not in {".txt", ".json"}: friendly
         `typer.echo("Unsupported file type: expected a WhatsApp .txt or Telegram .json export", err=True)`; exit 1.
       - else run the shared `_analyze_path(path)` helper (below) and exit 0/1.
    4. **Interactive no-arg** (D-01/D-06): keep the Phase 1 prompt loop ("Enter path to chat
       export", strip quotes, `File not found: ...` re-prompt), but each successful
       path goes through `_analyze_path`; on `ValueError` (UnsupportedFormat /
       "No messages could be parsed" / "Not a Telegram chat export") print the friendly
       message and `continue` (loop back to re-prompt — D-06); on success `break` + exit 0.
       EOF/Abort behavior unchanged (Phase 1 accepted: exits 1 "Aborted.").
    5. **`_analyze_path(path: Path) -> None`** helper (module-level function):
       - `console = Console()` (rich; create once).
       - `results = run_pipeline(path, console)` — stage narration + immediate parsed count
         happen inside (Task 5).
       - `with console.status("Writing report...", spinner="line"): results["report_path"] = str(write_report(results, path).resolve())`.
       - `show_summary(results, console)` (Task 6 — summary panel + path).
       - `open_report(Path(results["report_path"]))` (D-09 — returns bool; the path is
         already printed by show_summary, so failure degrades to a printed path).
       - Heavy imports (`run_pipeline`, `write_report`, `show_summary`) stay inside the
         helper (Anti-Pattern 2 — `--help`/`--version` stay instant).
    6. Do NOT reference `reporting/*`, `reportlab`, `plotly`, `plotext` anywhere in main.py
       (test_phase1_smoke `test_reporting_importable_but_not_wired` asserts the first three
       are absent; plotext never ships).
  </action>
  <verify>
    <automated>
      # Phase 1 smoke regression — CLI contract unchanged where it must be (10 tests)
      $env:PYTHONPATH="src"; python -m pytest tests/test_phase1_smoke.py -q
      # --version works (typer 0.27 has no built-in — the eager callback closes the gap)
      $env:PYTHONPATH="src"; python -m chat_analyzer --version
      # help still instant
      $env:PYTHONPATH="src"; python -m chat_analyzer --help
    </automated>
  </verify>
  <done>
    All 10 Phase 1 smoke tests still pass; `python -m chat_analyzer --version` prints
    "chat-analyzer 0.1.0" and exits 0; --help exits 0; positional + interactive paths route
    into run_pipeline; no reporting/plotly/plotext tokens in main.py.
  </done>
</task>

### Wave 5 — End-to-end CLI tests + full verification

- [ ] **Task 9: CLI end-to-end tests (`tests/test_phase2_cli.py`) + full verification run (positional run, interactive run, --version, unsupported ext, no tracebacks, legacy baseline untouched)**

<task type="auto">
  <name>Task 9: CLI end-to-end tests and full phase verification</name>
  <files>
    tests/test_phase2_cli.py (NEW)
  </files>
  <action>
    Create `tests/test_phase2_cli.py` — follow `test_phase1_smoke.py`'s plain-pytest style
    (subprocess against the real installed `chat-analyzer` console script and
    `python -m chat_analyzer`; assert on stdout/stderr; the 5 ROADMAP criteria):
    - Test 1 (ROADMAP crit 1 — one command end-to-end): subprocess
      `chat-analyzer data/sample_chats/whatsapp_sample.txt` → returncode 0; stdout contains
      "Parsed 27 messages" (CLI-03 count surfaced) and "Report:".
    - Test 2 (ROADMAP crit 4 — stage narration): stdout contains all three stage lines
      ("Parsing chat", "Computing insights", "Writing report" — narrate via rich Status,
      which writes to stderr when not a tty; assert across stdout+stderr) and the parsed
      count appears before the summary panel (index ordering assertion).
    - Test 3 (ROADMAP crit 5 — counts match export): report file exists at
      `data/sample_chats/whatsapp_sample_report.html` (D-08 location); delete it after.
    - Test 4 (ROADMAP crit 2+3 — report card exists and is well-formed): generated HTML
      contains all 5 tab ids and >= 4 `data:image/png;base64,` URIs and `<meta charset="utf-8">`.
    - Test 5 (interactive path, D-01): `chat-analyzer` with stdin
      "data/sample_chats/whatsapp_sample.txt\n" → exit 0, count surfaced.
    - Test 6 (--version, D-03): `chat-analyzer --version` → exit 0, output matches
      `chat-analyzer \d+\.\d+\.\d+`.
    - Test 7 (unsupported extension re-prompt, D-06): stdin
      "chat.pdf\nnonexistent.txt\n" → does NOT crash, no "Traceback", stays in the loop
      (assert the friendly message "expected a WhatsApp .txt or Telegram .json" appears);
      positional `chat-analyzer chat.pdf` → exit 1, no traceback.
    - Test 8 (telegram, D-19/D-20): `chat-analyzer data/sample_chats/telegram_sample.json`
      → exit 0, stdout contains "Parsed 5 messages".
    - Test 9 (skip surfacing, D-15/D-16): a small tmp fixture with one valid + one
      bad-date line → stdout contains "Skipped 1 lines that couldn't be parsed" and the
      report contains a skip note; no timestamp equals today in the report's stats.
    - Test 10 (no pollution, Pitfall 5): full stdout+stderr of the whatsapp run does NOT
      contain "🚀" or "Initializing Sentiment" (analysis prints captured).
    Use `tmp_path` for generated reports (write a copy of the sample into tmp_path and run
    there) so the repo stays clean; clean up `data/sample_chats/whatsapp_sample_report.html`
    in Test 3.

    Then run the FULL verification battery (below) and confirm the 39 pre-existing legacy
    failures are unchanged (Phase 4 QUAL-02 scope — do NOT fix, do NOT commit the
    uncommitted `tests/test_analysis.py` `freq='6H'`→`'6h'` change, do NOT touch legacy
    test files).
  </action>
  <verify>
    <automated>
      $env:PYTHONPATH="src"; python -m pytest tests/test_phase2_cli.py -q
    </automated>
  </verify>
  <done>
    All 10 CLI e2e tests pass; full battery below green; legacy suite state confirmed
    unchanged (39 failed / 72 passed baseline; `git status` shows only the pre-existing
    `tests/test_analysis.py` modification).
  </done>
</task>

## Verification (Phase-Level Battery)

Run in order at the end of the phase (Windows PowerShell):

```powershell
# 1. New Phase 2 suites (all must pass)
$env:PYTHONPATH="src"; python -m pytest tests/test_phase2_whatsapp.py tests/test_phase2_telegram.py tests/test_phase2_builder.py tests/test_phase2_pipeline.py tests/test_phase2_report.py tests/test_phase2_cli.py -q

# 2. Phase 1 regression (must stay 10 passed)
$env:PYTHONPATH="src"; python -m pytest tests/test_phase1_smoke.py -q

# 3. No datetime.now() anywhere in the parsers (grep hygiene: code tokens, not comments)
(Select-String -Path src/chat_analyzer/parser/*.py -Pattern "datetime\.now\(" | Measure-Object).Count   # expect 0

# 4. No plotext in manifest, jinja2 present
python -c "import tomllib,pathlib; d=tomllib.loads(pathlib.Path('pyproject.toml').read_text()); deps=' '.join(d['project']['dependencies']); assert 'jinja2' in deps and 'plotext' not in deps; print('DEPS-OK')"

# 5. Manual smoke — one command end-to-end on both samples (ROADMAP crit 1)
$env:PYTHONPATH="src"; python -m chat_analyzer data/sample_chats/whatsapp_sample.txt
$env:PYTHONPATH="src"; python -m chat_analyzer data/sample_chats/telegram_sample.json
# Assert: exit 0, stage narration, parsed counts (27 / 5), ASCII summary panel, absolute
# report path, report file exists next to input, auto-open attempted (or degraded).

# 6. Legacy baseline unchanged — count, do not fix
python -m pytest tests -q   # expect the SAME 39 failed / 72 passed baseline as research time
git status --short           # expect only the pre-existing tests/test_analysis.py modification
```

## Test Plan (new tests, all exercise real `chat_analyzer.*` modules per AGENTS.md)

| Test file | Covers | Requirement |
|-----------|--------|-------------|
| `tests/test_phase2_whatsapp.py` | strict date parsing (no `datetime.now()`), skip counting, system classification (encryption notice + "X added Y"), header-without-sender, common formats (US 12h/EU 24h/iOS bracket/4-digit year), multiline continuation, exact fixture counts, `parse_file` QUAL-01 | D-15, D-16, D-17, D-18, ANAL-01 |
| `tests/test_phase2_telegram.py` | both JSON shapes (bare Chat + chats.list[]), recursive entity-array text join, service-message filter, malformed drop counting (no silent `except: continue`), tz-aware → naive UTC (Z and +offset), "not a chat export" error, `parse_telegram_chat` QUAL-01 | D-19, D-20 |
| `tests/test_phase2_builder.py` | canonical df schema (datetime/timestamp alias/date/hour/sender/message/message_length/source/uid), tz normalization, ingestion-path telegram full-ISO date bug, unparseable-row drop | D-20, Anti-Pattern 5 |
| `tests/test_phase2_pipeline.py` | run_pipeline e2e (whatsapp 27, telegram 5), AnalysisResults shape + 4 base64 charts + insights, all-skipped → friendly error, unsupported format error, emoji-print capture, Agg headless | CLI-02, CLI-03, ANAL-01..05, D-05, D-12 |
| `tests/test_phase2_report.py` | single-file report (no external refs), `<meta charset="utf-8">` + valid UTF-8, HTML escaping (`<script>`/`<3` inert), filename sanitize + fallback, report next to input, auto-open degrade, skip note, 5 tabs + insights | OUT-03, OUT-04, OUT-05, D-08, D-09, D-10, D-11, D-14, V5 |
| `tests/test_phase2_cli.py` | one-command e2e (positional + interactive), stage narration order, counts match export, report card well-formed, `--version`, unsupported-extension re-prompt, skip surfacing, no console pollution | CLI-02, CLI-03, D-01, D-02, D-03, D-06, D-15, D-16, ROADMAP crits 1-5 |

**Scope guards:** legacy `tests/` files (test_parser.py, test_end_to_end.py,
test_reporting.py, test_analysis.py — 39 pre-existing failures) are NOT touched
(Phase 4 QUAL-02). The uncommitted `tests/test_analysis.py` change
(`freq='6H'` → `'6h'`, pandas 3.x compat) is NOT committed, reverted, or expanded.

## Success Criteria Mapping (ROADMAP Phase 2, reconciled)

| ROADMAP criterion | Where delivered | Verified by |
|-------------------|-----------------|-------------|
| 1. One command runs the full pipeline end-to-end | Task 8 (routing) + Task 5 (pipeline) + Task 4 (df builder) | test_phase2_cli.py tests 1/5/8; verification battery step 5 |
| 2. Summary stats, per-participant, trends, top words/emojis, VADER sentiment — **in the report**; terminal shows compact summary per D-07 | Task 5 (ChatEDA + VADER reuse via adapters) + Task 7 (report tabs) + Task 6 (terminal panel) | test_phase2_pipeline.py tests 1/2; test_phase2_report.py test 8; RENDER-OK |
| 3. (OUT-02 superseded) user sees the **decorated HTML report card with inferred insights** | Task 7 + Task 5 (`build_insights`) | test_phase2_report.py tests 1/8; test_phase2_cli.py test 4 |
| 4. Progress indicator + parsed-message count surfaced early | Task 6 + Task 5 (status stages + immediate count) | test_phase2_cli.py test 2; RENDER-OK; pipeline tests |
| 5. Timestamps/counts match export — no fabricated dates, skipped lines counted and surfaced | Task 2 (WhatsApp), Task 3 (Telegram), Task 4 (tz-naive df) | test_phase2_whatsapp.py, test_phase2_telegram.py, test_phase2_builder.py; verification battery step 3 (0 × `datetime.now()`) |

## Threat Model

Security enforcement is enabled (`.planning/config.json` has no
`security_enforcement` key → treated as enabled).

### Trust Boundaries

| Boundary | Description |
|----------|-------------|
| chat export file → parser | Untrusted external input (any WhatsApp `.txt` / Telegram `.json` the user points at the tool); malformed lines, system lines, entity-array texts, tz-aware dates cross here |
| chat content → HTML report | Untrusted message/sender content rendered into a *shared, double-clickable* HTML file — the old app's `unsafe_allow_html` bug must not survive the pivot (CONCERNS.md) |
| input filename → report filename | User-controlled input path is the seed for the report file path (D-08/D-14) |
| env → manifest | One new dependency (`jinja2`) enters the install graph this phase |

### STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-02-01 | Tampering | HTML report (`report_html.py` template, chat content) | mitigate | jinja2 `Environment(autoescape=select_autoescape(["html","xml"]))` set EXPLICITLY (plain jinja2 defaults to autoescape=False — VERIFIED 3.1.6); `html.escape()` as defense-in-depth; no `|safe` except base64 chart URIs validated with `startswith("data:image/png;base64,")`; test_phase2_report.py Test 3 asserts `<script>`/`<3` render inert |
| T-02-02 | Tampering | report filename derivation (D-14) | mitigate | `sanitize_filename()` strips `<>:"/\\\|?*` + control chars + leading dots, falls back to `"chat_analysis"`; report path derived from input stem, never user-supplied in Phase 2 (no `--output` flag, D-03); test_phase2_report.py Test 4 |
| T-02-03 | (availability) | report file encoding | mitigate | `open(report_path, "w", encoding="utf-8")` (never platform-default cp1252) + `<!DOCTYPE html>` + `<meta charset="utf-8">`; test_phase2_report.py Test 2 |
| T-02-04 | (integrity) | WhatsApp/Telegram parsers — timestamp fabrication | mitigate | `datetime.now()` deleted at whatsapp_parser.py:61/63/77/79; strict `_parse_datetime_strict` → None + `skipped_lines += 1`; Telegram date parse failures counted, never bare `except: continue`; verification battery step 3 asserts 0 matches |
| T-02-05 | (availability) | terminal output on cp1252 consoles (emoji/box-drawing crash — Pitfall 5) | mitigate | `sys.stdout/stderr.reconfigure(encoding="utf-8", errors="replace")` preserved at main.py entry (Phase 1); rich `box.ASCII` + `spinner="line"` (ASCII frames); analysis emoji `print()`s captured via `contextlib.redirect_stdout` in run_pipeline |
| T-02-06 | (integrity) | tz-naive/aware datetime mixing (Pitfall 9) | mitigate | single `_to_naive_utc` normalization contract at both parser boundaries + defensive re-check in `messages_to_dataframe`; schema tests assert `df['datetime'].dt.tz is None` (test_phase2_telegram.py Test 5, test_phase2_builder.py Test 2) |
| T-02-07 | (denial of service) | unbounded local file read | accept | Local CLI reading the user's own files by design (Phase 1 QUAL-04 scope); WhatsApp exports cap ~40k messages per export; no size cap in CONTEXT scope — revisit if remote ingestion ever ships |
| T-02-SC | Tampering | pip installs (pyproject manifest change) | mitigate (transfer) | Only one manifest addition: `jinja2` — verified in RESEARCH.md `## Package Legitimacy Audit`: slopcheck `[OK]`, ~18 yrs old, universal templating lib, version 3.1.6 matches manifest `>=3.1`; no [ASSUMED]/[SUS] entries → no blocking-human checkpoint required; `plotext` removed (verified imported nowhere in `src/`) |

## Post-Planning Doc Updates (perform AFTER this plan is approved — do not edit now)

The CONTEXT reshaped the phase; REQUIREMENTS.md / ROADMAP.md / STATE.md are stale and must
be reconciled:

**`REQUIREMENTS.md`:**
1. **OUT-02** — mark DROPPED (move to Out of Scope table or strike): "Inline charts in the
   terminal (bar/line via plotext)" — plotext never ships; charts live only in the HTML report.
2. **OUT-01** — reword: terminal shows a compact summary panel (volume, participants, date
   range) + absolute report path; full insights live in the HTML report (D-04/D-07).
3. **OUT-03** — phase mapping Phase 3 → Phase 2 (pulled forward); Traceability table update.
4. **OUT-04** — phase mapping Phase 3 → Phase 2 with note: **default-path behavior ships
   now**; the `--output` flag itself is deferred (D-03).
5. **OUT-05** — phase mapping Phase 3 → Phase 2 with resolution note: report is the
   deliverable (D-04); `--no-report` does not exist in Phase 2; revisit for Phase 4.
6. **CLI-08** — move from v2 section to v1 Phase 2 (auto-open, pulled forward D-09).
7. **Traceability table** — update phase/status for OUT-01/02/03/04/05 and CLI-08.

**`ROADMAP.md`:**
1. **Phase 2 Goal** — rewrite to the CONTEXT `<domain>` wording: "...parses a real
   WhatsApp .txt or Telegram .json export correctly and produces a self-contained,
   decorated HTML report card... Terminal shows stage narration, a compact summary panel,
   skip counts, and the absolute report path. The report auto-opens."
2. **Phase 2 Requirements line** → `CLI-02, CLI-03, ANAL-01..05, OUT-01, OUT-03, OUT-04,
   OUT-05, CLI-08` (OUT-02 removed).
3. **Phase 2 Success Criteria** — criterion 3 replaced: "User sees the decorated HTML
   report card with inferred insights" (plotext wording removed).
4. **Phase 2 Plans** → `02-PLAN.md` (this plan); mark plan count.
5. **Phase 3 (Shareable HTML Report)** — OUT-03/04/05 are now empty; flag for re-scope or
   absorption into Phase 4 during the next planning pass.
6. **Phase 4** — unchanged mapping (ANAL-06/07/08/09, CLI-04, QUAL-02, QUAL-03); add note
   that OUT-05 `--no-report` semantics revisit lands there.

**`STATE.md`:** log the re-mapping (OUT-02 dropped; OUT-03/04/05 + CLI-08 pulled forward;
ANAL-07 stays Phase 4) and this plan's creation.

## Output

Create `.planning/phases/02-one-command-terminal-insights/02-SUMMARY.md` when done.
