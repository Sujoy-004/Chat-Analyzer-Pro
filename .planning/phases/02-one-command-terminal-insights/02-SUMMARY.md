---
phase: 02-one-command-terminal-insights
plan: 02
subsystem: cli
tags: [cli, typer, rich, jinja2, parsers, ingestion, html-report, pytest, ruff, tdd]

# Dependency graph
requires:
  - phase: 01-package-foundation
    provides: "pyproject.toml with chat-analyzer console script target, src/ restructured into src/chat_analyzer/, Phase 1 smoke suite (10 tests incl. the Messages: N contract)"
provides:
  - "chat-analyzer <chat_file>" positional run (D-02) plus the Phase 1 interactive prompt (D-01), both routing through one run_pipeline
  - "python -m chat_analyzer --version" printing chat-analyzer 0.1.0 via an eager callback (typer 0.27 has no built-in)
  - Strict WhatsApp parser hardening: datetime.now() fabrication deleted, 16 DATE_FORMATS, _parse_datetime_strict → None + skipped_lines, system classification (encryption notice, header-without-sender, X added/removed/left patterns), parse_file_with_report returning (rows, counts)
  - Strict Telegram parser hardening: both JSON shapes (bare Chat + chats.list[]), recursive entity-array text join, service-message filter, tz→naive UTC via direct fromisoformat, "Not a Telegram chat export" ValueError on empty/missing-key exports, parse_telegram_chat_with_report
  - messages_to_dataframe() canonical dict→df builder (tz-safe, timestamp alias for ChatVisualizer) + normalize_message _to_naive_utc fix
  - run_pipeline orchestration + adapt() building AnalysisResults (9 keys) + narrative insight lead-ins (build_insights); visualization.py logging neutralized
  - render.py terminal narration: [OK]/[WARN]/[INFO] lines, ASCII summary panel, absolute report path
  - report_html.py: jinja2 autoescape single-file report (5 tabs), sanitize_filename, utf-8 write, auto-open degrade
  - 48 new phase-2 tests across 6 suites
affects: [03-shareable-html-report, 04-nlp-quality-gate]

# Tech tracking
tech-stack:
  added:
    - jinja2 3.1.6 (runtime, HTML report template; autoescape set explicitly)
    - removed: plotext>=5.3 from manifest (OUT-02 dropped, verified imported nowhere in src/)
  patterns:
    - matplotlib.use("Agg") as the FIRST line of run_pipeline (before any matplotlib import)
    - contextlib.redirect_stdout around the analysis stage (captures sentiment.py emoji prints)
    - Lazy heavy imports inside CLI handlers and pipeline (instant --help/--version)
    - Single canonical dict→df builder (messages_to_dataframe) — no second copy in cli/
    - jinja2 autoescape + html.escape defense-in-depth; no |safe except validated base64 chart URIs
    - tmp_path discipline in CLI e2e tests (reports land next to the tmp copy, never the repo)

key-files:
  created:
    - src/chat_analyzer/cli/contracts.py — ParseReport dataclass + AnalysisResults TypedDict
    - src/chat_analyzer/cli/pipeline.py — run_pipeline orchestration + fig_to_data_uri
    - src/chat_analyzer/cli/adapters.py — adapt() + build_insights()
    - src/chat_analyzer/cli/render.py — show_summary() ASCII narration
    - src/chat_analyzer/cli/report_html.py — write_report/open_report/sanitize_filename
    - tests/fixtures/whatsapp_system_skip.txt, telegram_full_export.json, telegram_bare_entity.json
    - tests/test_phase2_whatsapp.py, test_phase2_telegram.py, test_phase2_builder.py, test_phase2_pipeline.py, test_phase2_report.py, test_phase2_cli.py
  modified:
    - src/chat_analyzer/parser/whatsapp_parser.py — hardened (no datetime.now, strict dates, system classification, counters)
    - src/chat_analyzer/parser/telegram_parser.py — hardened (both shapes, entity join, service filter, tz→naive UTC)
    - src/chat_analyzer/ingest/ingestion.py — messages_to_dataframe + _to_naive_utc + normalize_message fix
    - src/chat_analyzer/utils/visualization.py — logging.basicConfig → NullHandler (1 line)
    - src/chat_analyzer/cli/main.py — positional arg, --version, pipeline routing, re-prompt loop
    - pyproject.toml — +jinja2, −plotext

key-decisions:
  - "The Messages: N smoke-contract token is printed by main.py's _analyze_path ONCE; pipeline and render never repeat it — keeps Phase 1 test_phase1_smoke message_count() (CRITICAL #1) green in both positional and interactive modes"
  - "Pipeline calls the hardened parser *_with_report entry points directly (rows + counts dict); process_uploaded_file stays the back-compat path for other formats"
  - "OUT-02 (plotext inline terminal charts) is dropped — charts live only in the HTML report; terminal shows a compact ASCII summary panel (D-07)"
  - "OUT-03/04/05 and CLI-08 pulled forward from Phase 3 into Phase 2 (report is the deliverable; default-path behavior ships, no --output/--no-report flags)"
  - "jinja2 Environment(autoescape=select_autoescape(['html','xml'])) set explicitly — plain jinja2 defaults to autoescape=False"
  - "datetime.fromisoformat called directly (Python >= 3.11 accepts trailing Z natively) — removes FURB162 .replace('Z', '+00:00') at both parser/ingestion boundaries"
  - "Single _to_naive_utc normalization contract at parser boundaries + defensive re-check in messages_to_dataframe (D-20, Pitfall 9)"

patterns-established:
  - "matplotlib Agg-first + lazy imports: importing cli.pipeline must not import matplotlib.pyplot at module top (LAZY-OK gate)"
  - "Stage narration ownership split: pipeline owns [OK] Parsed N lines, main owns the Messages: token, render owns the end summary"
  - "HTML report is single-file with inline CSS/JS + base64 chart data URIs — no external assets, no CDN, chat-derived bytes always escaped"

requirements-completed: [CLI-02, CLI-03, ANAL-01, ANAL-02, ANAL-03, ANAL-04, ANAL-05, OUT-01, OUT-03, OUT-04, OUT-05, CLI-08]

# Metrics
duration: 270min
completed: 2026-08-03
---

# Phase 2 Plan 2: One-Command Terminal Insights Summary

**One command — `chat-analyzer <chat_file>` (or `python -m chat_analyzer` interactive) — strictly parses a real WhatsApp `.txt` / Telegram `.json` export, computes insights with the reused analysis core (ChatEDA + VADER), and produces a self-contained tabbed HTML report card with narrative insight lead-ins and base64-embedded matplotlib charts, auto-opened in the default browser. The terminal narrates stages with ASCII spinners, surfaces the parsed-message count twice (the `[OK] Parsed N messages...` stage line and the `Messages: N` smoke-contract token), prints skip/system counts on a single line each, and shows a compact ASCII summary panel plus the absolute report path.**

## Performance

- **Duration:** ~4.5 h (6 task waves, 15 commits)
- **Completed:** 2026-08-03
- **Tasks:** 9
- **Files:** 20 (5 new cli/ modules, 3 new fixtures, 6 new test suites, 5 modified)

## Accomplishments

- **One command end-to-end (CLI-02):** `chat-analyzer data/sample_chats/whatsapp_sample.txt` exits 0, narrates Parsing→Computing→Writing, prints `[OK] Parsed 27 messages` + the `Messages: 27` smoke-contract token, and writes `whatsapp_sample_report.html` next to the input, auto-opened in the browser (verified live — report opens via `file://` URL). Telegram sample: `Parsed 5 messages` / `Messages: 5`.
- **Parser correctness (D-15/D-16/D-17/D-18/D-19/D-20):** zero `datetime.now()` matches in `parser/*.py`; WhatsApp strict dates (16 formats, 2/4-digit year, optional seconds, US/EU/iOS-bracket), system classification (encryption notice, header-without-sender, "X added Bob"), skip counting; Telegram both JSON shapes, recursive entity-array text join, service filter, tz-aware→naive UTC (Z and +05:30 both verified), empty/missing-key exports raise "Not a Telegram chat export".
- **Canonical data path (Anti-Pattern 5):** `messages_to_dataframe()` is the single dict→df builder with the `timestamp` alias ChatVisualizer requires; `_to_naive_utc` normalizes at parser and ingestion boundaries; `normalize_message` tz-safe ISO handling fixed (FURB162 removed).
- **HTML report card (OUT-03/04/05, CLI-08):** jinja2 autoescape single-file report with 5 tabs (overview/participants/flow/words/sentiment), each opening with a narrative insight lead-in, 4 base64 PNG chart data URIs, `sanitize_filename` (D-14), utf-8 write, and auto-open that degrades to a printed path on failure. Chat-derived bytes verified escaped (`<script>`/`<3` render inert).
- **Terminal narration (D-05/D-07/D-16/D-18):** rich `Status` spinners with ASCII `line` frames, single-line skip/system counts, `box.ASCII` summary panel, absolute report path; analysis emoji prints captured via `redirect_stdout` (no console pollution).
- **Phase 1 regression preserved:** `test_phase1_smoke.py` still 10/10 — the `Messages: N` token in `_analyze_path` keeps `message_count()` tests 3 & 4 green (CRITICAL #1).
- **Manifest hygiene:** pyproject now has `jinja2>=3.1` and no `plotext>=5.3` (DEPS-OK gate); heavy deps stay out of the base install.
- **Lint gates:** `ruff check src/chat_analyzer/cli tests/test_phase2_*.py` → all clean (hard gate); `parser/ + ingestion.py` → 63 findings ≤ 84 baseline (non-growth gate, tasks 2-4 removed ~10 legacy findings and added zero).

## Task Commits

Each task was committed atomically (RED→GREEN where tdd):

1. **Task 1: CLI contracts + parser fixtures** - `b56dace` (feat)
2. **Task 2 RED: WhatsApp hardening tests** - `5579e92` (test)
3. **Task 2 GREEN: WhatsApp parser hardening** - `c143e75` (feat)
4. **Task 3 RED: Telegram hardening tests** - `524b91c` (test)
5. **Task 3 GREEN: Telegram parser hardening** - `c862da8` (feat)
6. **Task 4 RED: builder tests** - `97b9141` (test)
7. **Task 4 GREEN: messages_to_dataframe + normalize_message + pyproject** - `77031ee` (feat)
8. **Task 5 RED: pipeline + adapters tests** - `4d9897f` (test)
9. **Task 5 GREEN: run_pipeline + adapters + viz logging fix** - `cc8b5ac` (feat)
10. **Task 6: render.py narration** - `d8c1c92` (feat)
11. **Task 7 RED: HTML report tests** - `12201e1` (test)
12. **Task 7 GREEN: report_html.py** - `424bd23` (feat)
13. **Task 8: main.py wiring (positional, --version, routing)** - `9c43eee` (feat)
14. **Task 8 fix: stage narration when stdout is not a tty** - `d93d692` (fix)
15. **Task 9: CLI e2e tests** - `262b6a5` (test)

## Files Created/Modified

- `src/chat_analyzer/cli/contracts.py` - ParseReport dataclass + AnalysisResults TypedDict (single contract for pipeline/adapters/render/report_html)
- `src/chat_analyzer/cli/pipeline.py` - run_pipeline (Agg-first, parse threading, redirect_stdout analysis, 4 charts→data URI) + fig_to_data_uri
- `src/chat_analyzer/cli/adapters.py` - adapt() assembling AnalysisResults from module dicts + build_insights() narrative lead-ins (LOW #9 None-avg defensive)
- `src/chat_analyzer/cli/render.py` - show_summary(): skip/system lines, ASCII panel, absolute path
- `src/chat_analyzer/cli/report_html.py` - inline jinja2 TEMPLATE, sanitize_filename, write_report (utf-8), open_report (degrade)
- `src/chat_analyzer/parser/whatsapp_parser.py` - hardened strict parsing (see accomplishments)
- `src/chat_analyzer/parser/telegram_parser.py` - hardened strict parsing (see accomplishments)
- `src/chat_analyzer/ingest/ingestion.py` - messages_to_dataframe + _to_naive_utc + normalize_message fix
- `src/chat_analyzer/utils/visualization.py` - logging.basicConfig → NullHandler (1 line, 12 plot methods byte-identical)
- `src/chat_analyzer/cli/main.py` - positional chat_file, --version eager callback, _analyze_path routing, re-prompt loop
- `pyproject.toml` - +jinja2, −plotext
- `tests/fixtures/whatsapp_system_skip.txt`, `telegram_full_export.json`, `telegram_bare_entity.json`
- `tests/test_phase2_whatsapp.py`, `test_phase2_telegram.py`, `test_phase2_builder.py`, `test_phase2_pipeline.py`, `test_phase2_report.py`, `test_phase2_cli.py`

## Decisions Made

- `Messages: N` token ownership: single source in main.py `_analyze_path` (positional + interactive) — see key-decisions (CRITICAL #1)
- Pipeline calls hardened parsers directly; `process_uploaded_file` untouched (back-compat)
- OUT-02 dropped / OUT-03/04/05 + CLI-08 pulled forward (report is the deliverable)
- jinja2 autoescape explicit; no `|safe` except validated base64 chart URIs
- Direct `datetime.fromisoformat` (3.11 floor) — removes FURB162 across parser/ingestion

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Stage narration degrades when stdout is not a tty**
- **Found during:** Task 8 verification / CLI e2e tests
- **Issue:** rich `Status` spinners write to stderr when stdout is not a tty, so the stage narration (Parsing/Computing/Writing) did not appear in captured stdout for subprocess tests — breaking the stage-narration assertion.
- **Fix:** assert stage lines across stdout+stderr in `test_phase2_cli.py` (Test 2 reads both streams); added `d93d692` fix commit documenting the degradation behavior.
- **Files modified:** tests/test_phase2_cli.py
- **Verification:** `python -m pytest tests/test_phase2_cli.py -q` — 10/10 pass.
- **Committed in:** d93d692 (Task 8 fix)

**2. [Rule 1 - Bug] CLI e2e Test 3 (report next to input) could false-fail when a report already exists in the repo sample dir**
- **Found during:** Task 9 execution
- **Issue:** the original assertion `assert not list(REPO_ROOT.rglob("*_report.html"))` failed if a prior manual smoke run left `data/sample_chats/whatsapp_sample_report.html` on disk — an artifact of the Phase-1-style interactive smoke, not a Phase 2 repo write.
- **Fix:** snapshot `*_report.html` files before the run and assert no NEW repo writes after (`after == before`), preserving the D-08/LOW #8 intent.
- **Files modified:** tests/test_phase2_cli.py (uncommitted refinement, committed with this close-out)
- **Verification:** `python -m pytest tests/test_phase2_cli.py -q` — 10/10 pass.

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both were test-harness robustness fixes; no scope creep, no legacy-module content changes.

## Issues Encountered

- **Pre-existing `tests/test_analysis.py` change** (`freq='6H'` → `'6h'`, pandas 3.x compat) left uncommitted exactly as the plan mandates — not committed, not reverted, not expanded.
- **Legacy suite (39 pre-existing failures)** across test_parser.py/test_end_to_end.py/test_reporting.py/test_analysis.py remains — Phase 4 QUAL-02 scope; intentionally untouched this phase. Full-suite passed count is not pinned (LOW #7) — it grew with the 48 new phase-2 tests.
- **torch pre-exists in the local base env:** `test_lean_base_structural` degrades to the structural-only half with a warning (pre-existing, expected).
- **gsd-sdk state handlers:** the installed gsd-sdk lacks phase-level state/roadmap/requirements write verbs for some updates in this environment, so tracking files were updated by direct edit per the workflow's intent where the SDK returned no subcommand.

## TDD Gate Compliance

| Gate | Commit | Evidence |
|------|--------|----------|
| RED | `5579e92` test(02-02) | failing WhatsApp tests before parser change |
| GREEN | `c143e75` feat(02-02) | WhatsApp hardening makes tests pass |
| RED | `524b91c` test(02-02) | failing Telegram tests before parser change |
| GREEN | `c862da8` feat(02-02) | Telegram hardening makes tests pass |
| RED | `97b9141` test(02-02) | failing builder tests before implementation |
| GREEN | `77031ee` feat(02-02) | builder + normalize_message makes tests pass |
| RED | `4d9897f` test(02-02) | failing pipeline + adapters tests |
| GREEN | `cc8b5ac` feat(02-02) | run_pipeline + adapters makes tests pass |
| RED | `12201e1` test(02-02) | failing HTML report tests |
| GREEN | `424bd23` feat(02-02) | report_html makes tests pass |

Gate sequence validated: each `test(02-02)` commit precedes its `feat(02-02)` commit in git log. PASS.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- **Phase 2 is complete** (1/1 plan). Every Phase 2 success criterion is machine-verified: one command end-to-end on both samples, report card well-formed (5 tabs, 4 base64 charts, utf-8, escaped), stage narration + `Messages: N` token early, counts match the export (27/5), skip/system lines surfaced, `--version`, friendly exit-1-no-traceback on malformed files.
- Ready for **Phase 3: Shareable HTML Report** — note OUT-03/04/05 were pulled forward into Phase 2, so Phase 3 requires re-scope (per plan's Post-Planning Doc Updates §ROADMAP item 5).
- Phase 4 absorbs ANAL-06/07/08/09, CLI-04, QUAL-02 (legacy suite rewiring), QUAL-03, and the OUT-05 `--no-report` semantics revisit.

---

*Phase: 02-one-command-terminal-insights*
*Completed: 2026-08-03*

## Self-Check: PASSED

- Created files verified on disk: `src/chat_analyzer/cli/{contracts,pipeline,adapters,render,report_html}.py`, `tests/test_phase2_{whatsapp,telegram,builder,pipeline,report,cli}.py`, 3 fixtures, `02-SUMMARY.md`
- Commits verified in git log: 15 commits across tasks 1-9 under `feat/test/fix(02-02)` scope
- Phase 2 suite: `python -m pytest tests/test_phase2_*.py -q` → 48 passed
- Phase 1 regression: `python -m pytest tests/test_phase1_smoke.py -q` → 10 passed
- Grep gate: 0 × `datetime.now()` in `parser/*.py`
- Manifest: DEPS-OK (jinja2 present, plotext absent)
- Lint hard gate: `python -m ruff check src/chat_analyzer/cli tests/test_phase2_*.py` → All checks passed
- Lint non-growth gate: `ruff check parser/ + ingestion.py` → 63 ≤ 84 baseline
