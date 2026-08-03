---
phase: 02-one-command-terminal-insights
verified: 2026-08-03T00:00:00Z
status: human_needed
score: 7/7 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Run `python -m chat_analyzer data/sample_chats/whatsapp_sample.txt` (BROWSER neutralized not needed — real run) and observe the generated `whatsapp_sample_report.html` auto-opens in the default browser with 5 working tabs (overview/participants/flow/words/sentiment), 4 charts rendering, narrative lead-ins, and correctly escaped content."
    expected: "Report opens automatically via file:// URL with 5 selectable tabs, charts visible, no unescaped markup injection, terminal shows a clean ASCII summary panel + absolute report path."
    why_human: "webbrowser.open fires an external default browser — a machine-specific behavior that only a human can confirm fires; tests monkeypatch the browser so auto-open cannot be observed programmatically. Visual layout of the tabs/charts also cannot be asserted by grep."
  - test: "Run the tool on a real terminal (tty) and confirm the rich Status ASCII spinners ('line' frames) for the Parsing / Computing / Writing stages are visible, and stage narration reaches stdout."
    why_human: "rich Status spinners render only when stdout is a tty; verification runs used a non-tty Console (force_terminal=False) so spinner animation is unobservable programmatically."
---

# Phase 2: One-Command Terminal Insights — Verification Report

**Phase Goal:** One command (`chat-analyzer <file>` or `python -m chat_analyzer`) parses a real WhatsApp `.txt` or Telegram `.json` export correctly and produces a self-contained, decorated HTML report card with inferred insights. Terminal shows stage narration, a compact summary panel, skip counts, and the absolute report path. The report auto-opens.
**Verified:** 2026-08-03
**Status:** human_needed (all 7/7 machine-verifiable must-haves pass; browser auto-open + visual/spinner appearance deferred to a human on a live machine)
**Re-verification:** No — initial verification (no prior VERIFICATION.md existed)

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | User runs one command (`chat-analyzer <file>` or `python -m chat_analyzer` + interactive prompt) and the full pipeline runs end-to-end automatically | ✓ VERIFIED | `cli/main.py` positional `chat_file` routes into `run_pipeline` (line 45) + interactive re-prompt loop (lines 101-118). `test_phase2_cli.py::test_positional_run` asserts `Parsed 27 messages` + `Messages: 27`; exit 0. I ran `run_pipeline` directly on both samples → complete `AnalysisResults`. Console script `chat-analyzer = "chat_analyzer.cli:app"` in pyproject. |
| 2 | Terminal narrates each stage (Parsing/Computing/Writing) with ASCII spinners and surfaces the parsed-message count immediately as BOTH `[OK] Parsed N messages...` stage line AND a `Messages: N` token (case-sensitive `Messages:\s*(\d+)` smoke shape) | ✓ VERIFIED | `pipeline.py` lines 74-101: `[OK] Parsed N messages from P participants` + `[WARN] Skipped...`; `main.py` line 50 prints the single `Messages: {N}` token; `stage_status()` ASCII `spinner="line"` (non-tty degrade). Phase 1 smoke `test_phase1_smoke.py:87` regex `Messages:\s*(\d+)` kept green (10/10 ran). CLI e2e test 2 asserts token order. |
| 3 | No fabricated timestamps anywhere; unparseable lines counted in skipped_lines and surfaced on a single line | ✓ VERIFIED | 0 matches `datetime.now(` in `parser/*.py`. `whatsapp_parser._parse_datetime_strict` returns None on failure → `skipped_lines += 1` (never `datetime.now()`). Telegram `_load_messages`/date failures → skipped. Behavioral: fixture `whatsapp_system_skip.txt` → parsed=3, skipped=1, system=2 exactly; bad-date month-13 line dropped. CLI test asserts `Skipped 1 lines that couldn't be parsed`. |
| 4 | System messages (encryption notice, X added Y, header-without-sender) classified as type=system, counted, never appended to previous message | ✓ VERIFIED | `whatsapp_parser.parse_line_strict` classification order: encryption-notice → system_header_pattern (timestamp, no sender) → system_phrase_pattern (`added/removed/left/joined/...`); system rows never enter `rows`. Behavioral check: no row contains `end-to-end encrypted` or `added Bob`. Test 2/3/6 green. |
| 5 | Telegram exports parse in both shapes (bare Chat + chats.list[]), join entity-array recursively, filter service messages, normalize tz→naive UTC | ✓ VERIFIED | `telegram_parser._load_messages` (both shapes), `_join_text` (recursive entity dict text), service filter, `_to_naive_utc` (fromisoformat accepts `Z` +05:30 aware→naive UTC). Ran both fixtures: full_export parsed=3/system=1/skipped=2 with `hello world` (joined entity+str); bare parsed=2, text `@team check this`, row2 `2025-09-15 04:16:00` naive UTC. `{"chats":[]}`/`{"messages":[]}`/missing-key → `ValueError("Not a Telegram chat export")`. |
| 6 | Self-contained, decorated HTML report card: 5 tabbed sections, each opening with a narrative insight lead-in, matplotlib charts base64-embedded, top words/emojis, VADER sentiment | ✓ VERIFIED | `report_html.py` TEMPLATE: 5 tab ids (`tab-overview/participants/flow/words/sentiment` each with `insights[i]` lead-in + chart + table). `test_phase2_report.py` asserts 5 tabs + insights. I generated a real report: exactly 5 tabs, ≥4 `data:image/png;base64,` chart URIs, `<meta charset="utf-8">`, `<!DOCTYPE html>`. Pipeline test: top_words non-empty, sentiment distribution non-empty, 4 base64 charts on both samples. |
| 7 | Report written next to input as `<sanitized_name>_report.html`, auto-opens in default browser (degrades to printed path), every chat-derived byte HTML-escaped | ✓ VERIFIED | `write_report` returns `input.parent/f"{sanitize_filename(stem)}_report.html"` written utf-8. `open_report` uses `webbrowser.open("file://"+str(path.resolve()))` in try/except → returns bool, caller prints path regardless. Escaping verified: injected `<script>alert(1)</script>`, `<3`, `<img onerror>` all escaped to `&lt;...&gt;` (autoescape explicit `select_autoescape(["html","xml"])`); template's own `showTab` JS intact. sanitize_filename strips separators/invalid chars → `chatname1.txt`; fallback `chat_analysis`. |

**Score:** 7/7 truths verified (machine-verified)

### Deferred Items (Step 9b — accounted for by later phases, not actionable gaps)

| # | Item | Addressed In | Evidences |
|---|------|--------------|-----------|
| 1 | OUT-04 `--output` flag (output-path selection) | Phase 3/4 | ROADMAP Phase 3 criterion 2: "flag deferred (D-03); default-path behavior ships". Default path ships and is verified | `</` in this phase — not a gap. |
| 2 | OUT-05 `--no-report` semantics | Phase 4 | ROADMAP Phase 4 note: "OUT-05 `--no-report` semantics revisit lands here"; report is always generated in Phase 2 by design (not applicable). |

### Required Artifacts (Level 1-4)

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/chat_analyzer/cli/contracts.py` | ParseReport + AnalysisResults TypedDict (9 keys) | ✓ VERIFIED | 39 lines; `class AnalysisResults(TypedDict)`. Wired: imported by pipeline/adapters/render/report_html. |
| `src/chat_analyzer/cli/pipeline.py` | `run_pipeline` + `fig_to_data_uri` (Agg-first, lazy, redirect_stdout) | ✓ VERIFIED | 157 lines; `matplotlib.use("Agg")` first line of `run_pipeline`. Data-flow FLOWING: real_rows → messages_to_dataframe → ChatEDA/VADER → 4 base64 charts → adapt. |
| `src/chat_analyzer/cli/adapters.py` | `adapt` + `build_insights` | ✓ VERIFIED | 165 lines; defensive `.get()`; avg_response_time None handling. Wired (returned by pipeline; imported in test). |
| `src/chat_analyzer/cli/render.py` | `show_summary` ASCII panel + skip/system lines + absolute path | ✓ VERIFIED | 50 lines; `box.ASCII`; prints `Skipped N lines`, `Excluded N system`, `Total messages`, `Report:`. RENDER-OK check. |
| `src/chat_analyzer/cli/report_html.py` | jinja2 autoescape template (`write_report`, `open_report`, `sanitize_filename`) | ✓ VERIFIED | 202 lines; 5-tab TEMPLATE; base64 validation; utf-8 write; `webbrowser.open` in try/except. Generated+verified live. |
| `src/chat_analyzer/parser/whatsapp_parser.py` | `parse_file_with_report` strict dates + system counts | ✓ VERIFIED | 351 lines; 16 DATE_FORMATS, no `datetime.now()`, counters + system classification. Behavioral count check passed. |
| `src/chat_analyzer/parser/telegram_parser.py` | `parse_telegram_chat_with_report` both shapes + tz→naive UTC | ✓ VERIFIED | 159 lines; `_load_messages`/`_join_text`/`_to_naive_utc`; service filter. Behavioral count + tz check passed. |
| `src/chat_analyzer/ingest/ingestion.py` | `messages_to_dataframe` (single canonical builder) | ✓ VERIFIED | 727 lines; schemas datetime/timestamp/date/hour/sender/message/message_length/source/uid; tz-normalized; rows dropped on unparseable. `_to_naive_utc` + `normalize_message` wired. |
| `pyproject.toml` | `jinja2>=3.1`, no `plotext` | ✓ VERIFIED | `jinja2>=3.1` present (line 23); 0 `plotext` across src + pyproject. |
| `tests/test_phase2_*.py` (6 files) | 48 tests total, real `chat_analyzer.*` modules | ✓ VERIFIED | 6 suites; pytest run = **48 passed** across whatsapp/telegram/builder/pipeline/report/cli. All green. |
| fixtures (3) | byte-exact per plan | ✓ VERIFIED | `whatsapp_system_skip.txt` (7 lines), `telegram_full_export.json` (chats.shape), `telegram_bare_entity.json` — match plan. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| pipeline.py | whatsapp_parser.py | `parse_file_with_report` | WIRED | `WhatsAppParser().parse_file_with_report(str(path))` at pipeline.py:78 |
| pipeline.py | telegram_parser.py | `parse_telegram_chat_with_report` | WIRED | pipeline.py:85 |
| pipeline.py | ingest/ingestion.py | `messages_to_dataframe(rows)` | WIRED | pipeline.py:105; defensive empty-df raise |
| pipeline.py | matplotlib | `matplotlib.use("Agg")` first line before any pyplot import | WIRED | pipeline.py:68 (LAZY-OK: importing cli.pipeline pulls no matplotlib) |
| adapters.py | contracts.py | `AnalysisResults` | WIRED | `AnalysisResults(...)` returned + `from ...contracts import AnalysisResults` |
| main.py | pipeline.py | `run_pipeline` | WIRED | main.py:40/45 |
| report_html.py | webbrowser | `webbrowser.open` (file://, try/except) | WIRED | report_html.py:196-202 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| pipeline.run_pipeline | rows/counts | `parse_file_with_report` / `parse_telegram_chat_with_report` | real file parse (27 / 5 messages) | ✓ FLOWING |
| run_pipeline df | rows | `messages_to_dataframe` | real rows, tz-naive | ✓ FLOWING |
| adapt.adapt stats/participants/content/sentiment | ChatEDA + VADER dicts | `generate_comprehensive_summary`/`analyze_message_volume`/`analyze_content`/`get_sentiment_summary` | non-empty on real samples | ✓ FLOWING |
| charts dict | 4 base64 PNG URIs | `ChatVisualizer` + `fig_to_data_uri` | 4 charts on both samples (26-89 KB each) | ✓ FLOWING |
| write_report | template render | `AnalysisResults` → jinja2 autoescape | report.html written + readable | ✓ FLOWING |
| show_summary | parse/stats/report_path | AnalysisResults | skip/system/panel/path emitted | ✓ FLOWING |

### Behavioral Spot-Checks (browser-neutralized)

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| LAZY-OK (no eager matplotlib) | `python -c "import ...pipeline; assert 'matplotlib' not in sys.modules"` | `LAZY-OK` | ✓ PASS |
| WhatsApp exact counts | `parse_file_with_report(fixture)` | parsed=3, skipped=1, system=2, total=7; no encryption/added-Bob leak | ✓ PASS |
| Telegram both shapes + tz | `parse_telegram_chat_with_report` (full, bare) | 3/1/2 & 2/0/0; `04:16:00` naive UTC; `@team check this`; ValueError on empty/missing keys | ✓ PASS |
| Pipelines end-to-end (whatsapp + telegram) | `run_pipeline(...)` | 27 & 5 messages; 4 base64 charts each; top_words/sentiment non-empty; 7 insights each | ✓ PASS |
| Report write (temp dir, no repo write) | `write_report` on analyze of telegram | tab-{5 ids}, ≥4 charts, `<meta charset>` `<!DOCTYPE html>`, next to input | ✓ PASS |
| HTML escaping (defense) | inject `<script>`, `<3`, `<img onerror>` | escaped to `&lt;...&gt;`; template JS intact | ✓ PASS |
| render narration | `show_summary(... Console(file=StringIO))` | `Total messages: 5`, `Report:`, `Date range:` emitted | ✓ PASS |
| sanitize_filename | `..\..\chat<name>:1.txt`, `con`, `..` | `chatname1.txt`, `con`, `chat_analysis` | ✓ PASS |
| `--version` | `python -m chat_analyzer --version` | `chat-analyzer 0.1.0`, exit 0 | ✓ PASS |

### Probe Execution

Phase 2 declares no explicit probe-scripts (`scripts/*/tests/probe-*.sh`). The auto-open gate (the one external action) was handled by neutralizing the browser (`$env:BROWSER="echo"`) during all behavioral runs — `webbrowser.open` never fired a real browser. Probe section: N/A (no probes declared).

### Requirements Coverage

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|----------|
| CLI-02 | `analyze <chat_file>` runs full pipeline automatically | SATISFIED | positional/in-prompt routing; e2e test exit 0; live run_pipeline both samples |
| CLI-03 | progress indicator + parsed-message count early | SATISFIED | `stage_status` + `[OK] Parsed N` + `Messages: N` token; order asserted |
| ANAL-01 | summary stats (total, participants, date range) | SATISFIED | stats block + panel; pipeline.test asserts 27/2/2023-12-25..27 |
| ANAL-02 | per-participant stats (messages, avg length, share) | SATISFIED | participants dict (messages/avg_message_length/share_pct) |
| ANAL-03 | timeline/activity trends | SATISFIED | busiest_day, peak_hour, daily trend + activity heatmap chart |
| ANAL-04 | top words and emojis with frequency | SATISFIED | content.top_words / top_emojis in stats + words tab |
| ANAL-05 | VADER sentiment systematically (per-message + per-participant) | SATISFIED | by_sender, distribution, avg_compound, daily_avg; sentiment tab |
| OUT-01 | terminal tables/panels — REVERSED to compact ASCII summary | SATISFIED | render panel (volume/participants/date-range) + absolute path |
| OUT-03 | self-contained single-file HTML report (base64, no external assets) | SATISFIED | report_html.py single-file; no http refs, no CDN; charts base64 |
| OUT-04 | output path — **default-path only** | SATISFIED | `<stem>_report.html` next to input; `--output` flag deferred (D-03) |
| OUT-05 | `--no-report` opt-out — **Not applicable** (report always generated) | SATISFIED (by design) | no `--no-report` flag; report is the deliverable; revisit Phase 4 |
| CLI-08 | auto-open the HTML report | SATISFIED | `open_report` `webbrowser.open` file://, degrades to path print |

All 12 required IDs accounted for. REQUIREMENTS.md traceability table already marks them Phase 2 Complete / Dropped (OUT-02) / Not applicable (OUT-05) — consistent with the plan reconciliation.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Resolution |
|------|------|---------|----------|------------|
| (none in phase-2 files) | — | — | — | No `TBD/FIXME/XXX`, no placeholder/“not implemented”, no `return null` stubs, no `unsafe_allow_html`, no `exec()`, no `|safe` (except validated base64 chart URIs), no raw `<script>` injection (defense verified). |
| src/chat_analyzer/analysis/eda.py | 4/7/135 | pre-existing ruff I001/F401/RUF059 | Info | Documented in `deferred-items.md`; `eda.py` is a reuse-only module outside the ruff gates (AGENTS.md) — not a phase-2 regression. |

### Human Verification Required

1. **Report card auto-open + visual confirmation** — Run the real CLI against `data/sample_chats/whatsapp_sample.txt` and confirm the HTML report opens in the default browser with the 5 tabs rendering (Overview/participants/flow/words/sentiment), 4 charts visible, narrative lead-ins, and no-markup escaping (message with markup stays inert).
   - Why: `webbrowser.open` fires the machine's real default browser — an external, machine-specific action unverifiable via automated tests (they monkeypatch the browser). Visual layout of the tab buttons/panels cannot be asserted by grep.
2. **TTY spinner/ASCII narration on a real terminal** — Observe the Parsing / Computing / Writing spinners (`line` frames) are visually distinct on a tty; run as stdin `python -m chat_analyzer` and confirm interactive completion + exit 0.
   - Why: rich `Status` spinner animation only renders on a tty; the verification environment is non-interactive.

### Gaps Summary

No machine-verifiable gaps. All 7 must-haves, 9 required artifacts, all key links, all 12 requirement IDs, and the full 48/48 phase-2 suite + 10/10 phase-1 smoke pass. The remaining items are exclusively browser-visual and real-terminal observations that only a human on a live machine can confirm, which drives the `human_needed` status rather than `passed`.

---

_Verified: 2026-08-03_
_Verifier: the agent (gsd-verifier)_