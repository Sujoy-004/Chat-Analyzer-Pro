---
phase: 01-package-foundation
plan: 02
subsystem: cli
tags: [cli, typer, packaging, pytest, ruff, smoke-tests, tdd]

# Dependency graph
requires:
  - phase: 01-package-foundation
    provides: "pyproject.toml with chat-analyzer console script target, src/ restructured into src/chat_analyzer/, web-app code deleted"
provides:
  - "chat-analyzer" console script (D-01) answering --help instantly
  - "python -m chat_analyzer" fallback entry point (D-02)
  - Interactive file-path prompt loop (D-03) with re-prompt on invalid paths and graceful exit 1 on unprocessable input
  - tests/test_phase1_smoke.py — 10 subprocess/in-process smoke tests (help, prompt flow, import matrix, web-app scan, lean-base proof, reporting non-wiring)
affects: [02-one-command-terminal-insights, 04-nlp-quality-gate]

# Tech tracking
tech-stack:
  added:
    - typer 0.27.0 (runtime, click-free CLI framework with bundled rich)
    - plotext 5.3.2 (runtime, declared but unused this plan)
    - ruff 0.16.1 (dev, lint gate)
    - pytest-cov (dev extra, installed with [dev])
  patterns:
    - Lazy heavy imports inside CLI command handlers (instant --help)
    - Windows console encoding bootstrap (sys.stdout/stderr.reconfigure utf-8 errors=replace) as first statement of main() (T-01-07)
    - Re-prompt validation loop with path.is_file() guard before any file-open (T-01-04)
    - Degrade-not-crash error containment (except Exception -> friendly message + exit code, never a traceback)
    - Subprocess smoke tests using -X utf8 + explicit encoding (cp1252 Windows pitfall)

key-files:
  created:
    - src/chat_analyzer/cli/main.py — Typer app with main command, prompt loop, error exits
    - src/chat_analyzer/cli/__init__.py — D-01 console-script target (chat_analyzer.cli:app)
    - src/chat_analyzer/__main__.py — D-02 python -m fallback (SystemExit(app()))
    - tests/test_phase1_smoke.py — 10 smoke tests (pytest style)
  modified:
    - src/chat_analyzer/cli/main.py — hardened with while-loop validation + error containment (Task 2 GREEN)

key-decisions:
  - "Import-matrix subprocess runs with -X utf8 and explicit utf-8 decode: legacy sentiment.py prints an emoji fallback warning at module load (textblob absent) which crashes a bare cp1252 python -c subprocess; test-harness mitigation keeps legacy modules byte-identical (reuse-not-rewrite)"
  - "typer.prompt on EOF raises typer Abort, which the app converts to exit 1 'Aborted.' with no traceback — accepted as the re-prompt-loop EOF behavior (Test 5 relies on it)"
  - "BLE001 # noqa on main.py's except Exception: the plan mandates broad Exception containment (degrade-not-crash convention, no bare except); ruff's blanket-ban rule conflicts and is overridden with an inline justification"
  - "plotly 6.7.0 pre-exists in the local base env (old Streamlit era) — pip install -e . did not pull it; QUAL-04 proof is pyproject structural confinement + package-tree scan (Test 7), both clean"

patterns-established:
  - "CLI command files import only stdlib + typer at module top; analysis/ingest modules load lazily inside handlers"
  - "Windows encoding bootstrap + ASCII-first CLI output prevents cp1252 crashes (T-01-07)"
  - "Smoke tests that exercise the real chat_analyzer.* modules via subprocess for entry-point behavior and in-process for the analysis core"

requirements-completed: [CLI-01, CLI-05, PKG-02, PKG-03, PKG-05, QUAL-01, QUAL-04]

# Metrics
duration: 25min
completed: 2026-08-01
---

# Phase 1 Plan 2: CLI Interactive Slice Summary

**Installable `chat-analyzer` command (plus `python -m chat_analyzer`) that prompts for a chat export path, processes it through the reused ingestion core, reports message counts, and is proven by a 10-test smoke suite — with the analysis core importable, the base install structurally lean, and zero web-app patterns shipping**

## Performance

- **Duration:** 25 min
- **Started:** 2026-08-01T06:02:00Z
- **Completed:** 2026-08-01T06:27:00Z
- **Tasks:** 3
- **Files modified:** 5 (4 new, 1 hardened)

## Accomplishments

- `chat-analyzer` (D-01 console script) and `python -m chat_analyzer` (D-02) both answer `--help` instantly — verified no heavy module loads (`CLI-LIGHT-OK`: no pandas/matplotlib/reportlab/seaborn/networkx in `sys.modules`)
- Interactive prompt (D-03): piped valid export → `Messages: 27` on the real WhatsApp sample; invalid paths re-prompt instead of crashing; unprocessable input (directory) exits 1 with a friendly message and no traceback
- TDD RED→GREEN: failing smoke tests (re-prompt + error-exit) committed before the `main()` hardening that made them pass
- QUAL-01: all 20 `chat_analyzer.*` modules import in a clean subprocess; Test 8 proves the analysis core produces results (VADER columns, EDA sender counts, relationship-health `total_messages == 4`)
- QUAL-04: installed package tree scan finds zero `exec(code` / `unsafe_allow_html` / streamlit / plotly tokens
- PKG-02/03: tomllib structural proof — torch/transformers/streamlit/plotly absent from base deps, `[nlp]` extra contains exactly torch + transformers, `requires-python == ">=3.11"`
- D-10/D-11: `pdf_report` and `weekly_digest` import after install (`REPORTING-OK`) while `cli/main.py` references neither nor reportlab
- Lint gate: `python -m ruff check` exits 0 on all new phase code

## Task Commits

Each task was committed atomically:

1. **Task 1: CLI entry slice (cli/ + __main__.py, pip install -e ., instant --help)** - `60f7bb1` (feat)
2. **Task 2 RED: smoke tests (tests 4-5 failing)** - `6d58f37` (test)
3. **Task 2 GREEN: prompt hardening (validation loop, error exits)** - `7f1ff59` (feat)
4. **Task 3: QUAL gates (lean-base proof, reporting non-wiring, ruff clean)** - `95d525e` (feat)

**Plan metadata:** final docs commit below.

## Files Created/Modified

- `src/chat_analyzer/cli/main.py` - Typer app; `main` command with encoding bootstrap, re-prompt validation loop, lazy `process_uploaded_file` import, exit codes
- `src/chat_analyzer/cli/__init__.py` - D-01 console-script target; re-exports `app` only
- `src/chat_analyzer/__main__.py` - D-02 `python -m chat_analyzer` shim; `raise SystemExit(app())`
- `tests/test_phase1_smoke.py` - 10 pytest-style smoke tests (help, prompt happy path, re-prompt, error exit, import matrix, web-app scan, core-produces-results, lean-base structural, reporting non-wiring)

## Decisions Made

- Import-matrix subprocess uses `-X utf8` + explicit `encoding="utf-8"` — see key-decisions (legacy sentiment.py emoji module-load print crashes bare cp1252 subprocesses; test-harness mitigation preserves byte-identical legacy modules)
- `typer.prompt` EOF behavior (Abort → exit 1 "Aborted.", no traceback) accepted as the loop's EOF path
- `# noqa: BLE001` on the plan-mandated broad `except Exception` (degrade-not-crash convention)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Import-matrix subprocess crashed on cp1252 console encoding**
- **Found during:** Task 2 RED phase (test_import_matrix)
- **Issue:** `src/chat_analyzer/analysis/sentiment.py` prints an emoji fallback warning (`⚠️ TextBlob not available...`) at module load when textblob is absent. A bare `python -c "import chat_analyzer..."` subprocess has no encoding bootstrap (research Pitfall 5), so cp1252 stdout raised UnicodeEncodeError and the import matrix failed for the wrong reason (encoding, not importability).
- **Fix:** Run the import check with `-X utf8` and decode the child's output with explicit `encoding="utf-8", errors="replace"` (same mitigation family as the CLI's T-01-07 stdout reconfigure). Kept legacy modules byte-identical per reuse-not-rewrite.
- **Files modified:** tests/test_phase1_smoke.py
- **Verification:** `python -m pytest tests/test_phase1_smoke.py` — test_import_matrix passes; RED failures reduced to exactly tests 4-5.
- **Committed in:** 6d58f37 (Task 2 RED commit)

**2. [Rule 1 - Bug] `text=True` subprocess decode crashed pytest's reader thread**
- **Found during:** Task 2 RED phase (same test)
- **Issue:** With `text=True` and no `encoding`, subprocess decodes child output with the locale (cp1252); the child's UTF-8 emoji bytes from sentiment.py produced a UnicodeDecodeError in the `_readerthread`, surfaced by pytest 9 as `PytestUnhandledThreadExceptionWarning` with `result.stdout = None`.
- **Fix:** Pass `encoding="utf-8"` explicitly (replacing `text=True`) so the parent decodes the `-X utf8` child output correctly.
- **Files modified:** tests/test_phase1_smoke.py
- **Verification:** test_import_matrix passes standalone and under pytest.
- **Committed in:** 6d58f37 (Task 2 RED commit)

**3. [Rule 3 - Blocking lint gate] ruff BLE001 flagged the plan-mandated `except Exception`**
- **Found during:** Task 3 lint gate (`python -m ruff check`)
- **Issue:** The plan explicitly requires `try/except Exception as exc` around `process_uploaded_file` (degrade-not-crash convention, "no bare except clauses"). ruff's default `BLE001` ("Do not catch blind exception") flagged it.
- **Fix:** Added inline `# noqa: BLE001 - degrade-not-crash convention` with justification rather than narrowing the catch (narrowing would let unexpected parser errors leak tracebacks, violating the plan's T-01-04 requirement).
- **Files modified:** src/chat_analyzer/cli/main.py
- **Verification:** `python -m ruff check src/chat_analyzer/cli src/chat_analyzer/__main__.py tests/test_phase1_smoke.py` exits 0.
- **Committed in:** 95d525e (Task 3 commit)

---

**Total deviations:** 3 auto-fixed (1 bug, 2 blocking)
**Impact on plan:** All three were required for the plan's own acceptance criteria to pass (import matrix must exit 0; ruff gate must exit 0). No scope creep; no legacy-module content changes.

## Issues Encountered

- **Pre-existing env plotly:** `plotly==6.7.0` remains in the local base env from the old Streamlit era. `pip install -e .` did not pull it (pyproject base deps exclude it) and the QUAL-04 package-tree scan is clean — documented in `deferred-items.md`, not a packaging defect.
- **Pre-existing env torch:** torch is importable in the local base env; `test_lean_base_structural` degraded to the structural-only half exactly as the plan prescribed (warns, doesn't fail).
- **typer.prompt EOF discovery:** probing confirmed EOF raises typer `Abort`, which the app converts to exit 1 with "Aborted." — verified acceptable for Test 5's "no traceback" assertion before writing the GREEN loop.
- **gsd-sdk state handlers unavailable:** the installed `gsd-sdk` lacks `state.advance-plan` / `state.record-metric` / `requirements.mark-complete` subcommands (no local node_modules copy either), so STATE.md, ROADMAP.md, and REQUIREMENTS.md were updated by direct edit per the workflow's intent.
- **Old test suites (`tests/test_parser.py` etc.)** still import the pre-restructure `src.*` paths — known, Phase 4 QUAL-02 rewires them; intentionally not touched this plan.

## TDD Gate Compliance

| Gate | Commit | Evidence |
|------|--------|----------|
| RED | `6d58f37` test(01-02) | tests 4-5 failed (re-prompt, exit 1) before any main.py change; 1/2/3/6/7/8 passed |
| GREEN | `7f1ff59` feat(01-02) | all 8 tests pass after prompt-loop hardening |
| REFACTOR | — | not needed; GREEN implementation was minimal, subsequent changes were Task 3 scope |

Gate sequence validated: `test(01-02)` commit precedes `feat(01-02)` commit in git log. PASS.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- **Phase 1 is complete** (2/2 plans). Every Phase 1 success criterion is machine-verified: install works, both entry points answer `--help` instantly, base install is lean, all analysis modules import and produce results, no web-app code ships, reporting ships importable-but-unwired.
- Ready for **Phase 2: One-Command Terminal Insights** (parser hardening with strict parse + skip counts, pipeline, rich/plotext terminal output).
- The CLI's lazy-import pattern and `process_uploaded_file` handoff are the extension points Phase 2 builds on; the `[nlp]`-gated modules (emotion, summarizer) import cleanly in the base env thanks to their existing lazy guards.

---

*Phase: 01-package-foundation*
*Completed: 2026-08-01*

## Self-Check: PASSED

- Created files verified on disk: `src/chat_analyzer/cli/main.py`, `src/chat_analyzer/cli/__init__.py`, `src/chat_analyzer/__main__.py`, `tests/test_phase1_smoke.py`, `01-02-SUMMARY.md`
- Commits verified in git log: `60f7bb1` (feat Task 1), `6d58f37` (test RED), `7f1ff59` (feat GREEN), `95d525e` (feat Task 3)
- Final suite run: `python -m pytest tests/test_phase1_smoke.py -q` → 10 passed
- `python -m ruff check src/chat_analyzer/cli src/chat_analyzer/__main__.py tests/test_phase1_smoke.py` → All checks passed
