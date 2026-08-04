---
phase: 04-nlp-extras-quality-gate
plan: 03
subsystem: cli
tags: [typer, rich, subprocess-pip, tty-menu, friendly-errors, cli-04]

# Dependency graph
requires:
  - phase: 04-nlp-extras-quality-gate
    provides: 04-02 nlp_gate silent availability probe + locked model constants + CHAT_ANALYZER_FORCE_NLP override; gated emotion/summary pipeline stages
provides:
  - D-04 3-option NLP download menu (tty-only) dispatching to a guarded runtime install_nlp with pre-download name+size announcement (D-05)
  - D-06 single hint line for positional and piped/no-menu runs (never a prompt off-tty)
  - CLI-04 friendly-error taxonomy: 4 failure classes, each a distinct message + inline WhatsApp/Telegram export steps, exit 1 positional / re-prompt interactive, zero tracebacks
affects: [04-04-legacy-tests, 04-05-quality-gate (README D-18/D-19 neutral option presentation)]

# Tech tracking
tech-stack:
  added: []  # NO new packages — runtime install re-declares the audited [nlp] extras via subprocess pip
  patterns:
    - "Guarded runtime installer: subprocess.run([sys.executable, -m pip install ...], capture_output=True, check=False) + returncode check, never shell=True (T-04-10)"
    - "rich output safety: soft_wrap=True keeps the hint one line on narrow non-tty consoles; \[nlp] escaped so rich markup cannot corrupt the package name"
    - "_friendly_error(chat_file, exc) classifier: exception type/text -> distinct composed message + inline export steps (D-13/D-14, T-04-11)"

key-files:
  created: [tests/test_phase4_cli.py]
  modified: [src/chat_analyzer/cli/main.py, src/chat_analyzer/cli/nlp_gate.py]

key-decisions:
  - "Menu rendered with rich console.print option lines + typer.prompt(default='2'); option 2 (CPU-only ~0.6GB) is the default (D-04/T-04-13), option 3 always available"
  - "Any message containing 'chat-analyzer-pro[nlp]' goes through typer.echo (no markup) or escapes \[nlp] in console.print — rich markup silently drops unknown-bracket tags"
  - "install_nlp uses explicit check=False to satisfy ruff PLW1510 while still inspecting returncode to raise the friendly RuntimeError"

requirements-completed: [CLI-04, QUAL-02]

# Metrics
duration: 20min
completed: 2026-08-04
---

# Phase 4 Plan 3: Interactive NLP Download Menu + Friendly Errors Summary

**The "always-integrated" user contract: silent startup availability check, a tty-only 3-option NLP download menu dispatched to a guarded subprocess installer with pre-download announce, a single ASCII hint line for positional/piped runs, and the CLI-04 friendly-error taxonomy with inline WhatsApp/Telegram export instructions on every failure (exit 1, zero tracebacks)**

## Performance

- **Duration:** 20 min
- **Started:** 2026-08-04T08:50:58Z (local 14:20:58)
- **Completed:** 2026-08-04T09:11:02Z (local 14:41:02)
- **Tasks:** 3 (TDD: test RED + 2 feat GREEN)
- **Files modified:** 3 (2 source, 1 created test file)

## Accomplishments

- **`install_nlp(cpu_only)` in nlp_gate.py** — guarded runtime pip re-install of the already-declared `[nlp]` extras (torch + transformers, `transformers>=4.30,<6`), CPU wheel index for the ~0.6 GB path, output captured, `RuntimeError` on failure so the caller degrades to basic analysis + hint (T-04-10, Pitfall 4 — never a frozen terminal)
- **D-04 tty menu (`_nlp_menu`)** — shows ONLY when NLP is missing AND `sys.stdin.isatty()`; 3 options (full torch ~3GB / CPU-only ~0.6GB default / no download); dispatch announces model name + size BEFORE install (D-05); install failure echoes a friendly warn + continues with basic analysis (Pitfall 4)
- **D-06 single hint line** — printed after `show_summary` (never before "Messages: N") in the positional branch and the piped/no-menu interactive path; `soft_wrap=True` keeps it one line on non-tty consoles; ASCII only, rich-escape-safe
- **CLI-04 error taxonomy** — `_EXPORT_WHATSAPP`/`_EXPORT_TELEGRAM` constants + `_friendly_error` classifier (file-not-found / unsupported type / empty-parse / defensive catch-all); every positional failure exits `typer.Exit(code=1) from None`; interactive loop keeps `continue` (D-15 re-prompt)
- **7-test phase4 CLI suite** — subprocess proofs for hint (positional + piped), menu suppression off-tty, all 3 error classes with export steps + no traceback, re-prompt loop, and an in-process tty-menu unit test (subprocess cannot fake a pty)

## Task Commits

Each task was committed atomically:

1. **Task 1: TDD RED — test_phase4_cli.py (7 tests)** - `6f2dcbf` (test)
2. **Task 2: GREEN — menu dispatch + hint line + guarded installer** - `e639101` (feat)
3. **Task 3: GREEN — friendly-error taxonomy with export instructions** - `f9a381d` (feat)

**Plan metadata:** pending (this SUMMARY commit)

## Files Created/Modified

- `tests/test_phase4_cli.py` - NEW 7-test suite: positional/piped hint (D-06), 3 error classes + export steps + no-traceback (D-13), re-prompt loop (D-15), in-process `_nlp_menu` tty unit test (D-04); `CHAT_ANALYZER_FORCE_NLP=0` + `BROWSER=__none__` env pins (Pitfall 5)
- `src/chat_analyzer/cli/nlp_gate.py` - +`install_nlp(cpu_only)` guarded subprocess installer; module docstring updated
- `src/chat_analyzer/cli/main.py` - +`_EXPORT_WHATSAPP`/`_EXPORT_TELEGRAM` constants, `_friendly_error` classifier, `_nlp_menu`, startup silent availability check, menu gate + dispatch, D-06 hint lines (soft_wrap); smoke token, `--version`, utf-8 reconfigure, and both branch structures untouched

## Decisions Made

- **Menu rendering at agent discretion (D-04):** rich `console.print` option lines + `typer.prompt("Choice", default="2")`; option 2 is the CPU-only default (T-04-13), option 3 always available
- **Rich markup hazard found and neutralized:** `console.print("...chat-analyzer-pro[nlp]")` silently DROPS the `[nlp]` tag text (verified on rich 14.3.3) — all messages containing the package name go through `typer.echo` (RuntimeError warn path) or escape the bracket as `\[nlp]` (hint lines); the hint line also gets `soft_wrap=True` because a bare Console() wraps at 80 cols off-tty, splitting the hint across two lines and breaking the "exactly one hint line" contract
- **install_nlp returns RuntimeError (not exception propagation)** so the interactive menu degrades to basic analysis + continue hint rather than crashing the run (Pitfall 4, T-04-12)
- **`menu_shown` flag** distinguishes the tty-menu path (user saw options — no hint after) from the piped/positional path (D-06 hint applies)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Rich Console word-wrapped the hint line on non-tty consoles**
- **Found during:** Task 2 (GREEN) — `test_positional_hint_line`/`test_piped_noarg_hint` failed: the hint rendered as `"...pip install \nchat-analyzer-pro[nlp]"` because a bare `Console()` defaults to 80 columns and wraps off-tty
- **Issue:** The D-06 "exactly one hint line" contract broke whenever stdout was piped/narrow (the normal positional/piped case), splitting the hint across two lines
- **Fix:** Added `soft_wrap=True` to both hint `console.print` calls — output stays one physical line regardless of console width
- **Files modified:** src/chat_analyzer/cli/main.py
- **Verification:** tests 1 + 2 pass; `out.count("pip install chat-analyzer-pro[nlp]") == 1`
- **Committed in:** `e639101`

**2. [Rule 1 - Bug] Rich markup silently dropped `[nlp]` from console.print output**
- **Found during:** Task 2 (GREEN) — pre-implementation probe showed `console.print('[INFO] ... chat-analyzer-pro[nlp]')` renders `'...chat-analyzer-pro\n'` (the `[nlp]` tag text vanished) on rich 14.3.3
- **Issue:** Any hint/warn printed via console.print with the package name would corrupt the actionable install instruction
- **Fix:** Hint lines escape the bracket as `\[nlp]`; the `install_nlp` RuntimeError message (which contains the package name) is echoed via `typer.echo` (no markup parsing)
- **Files modified:** src/chat_analyzer/cli/main.py, src/chat_analyzer/cli/nlp_gate.py
- **Verification:** hint substrings assert exactly in tests 1-2; visual spot-check of the missing-file message renders fully
- **Committed in:** `e639101`, `f9a381d`

**3. [Rule 3 - Blocking] ruff PLW1510 required explicit `check=False` on subprocess.run**
- **Found during:** Task 2 verification (`python -m ruff check`)
- **Issue:** The plan's contract snippet `subprocess.run(cmd, capture_output=True, text=True)` triggers ruff PLW1510
- **Fix:** Added explicit `check=False` — semantics unchanged (returncode is inspected immediately to raise the friendly RuntimeError)
- **Files modified:** src/chat_analyzer/cli/nlp_gate.py
- **Verification:** ruff clean on touched files
- **Committed in:** `e639101`

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 blocking lint)
**Impact on plan:** All fixes preserve the plan's user-facing contract (one hint line, intact package name, guarded installer). No scope creep.

## Issues Encountered

- **Plan Task 3 ruff acceptance criterion is unsatisfiable at baseline:** `python -m ruff check src/chat_analyzer tests` reports 382 errors — all pre-existing legacy analysis-module errors (STATE.md documented blocker, deferred-items.md #1; 04-01/04-02 same handling). All files touched by this plan are 0-error clean (`main.py`, `nlp_gate.py`, `test_phase4_cli.py`).
- **ruff PLR0402 in test 7:** `import chat_analyzer.cli.nlp_gate as nlp_gate` auto-fixed to `from chat_analyzer.cli import nlp_gate` (same module object — patch target unchanged).
- **Smoke-test repo pollution (pre-existing):** Phase 1 smoke tests run the CLI from the repo root (D-09 cwd reports), generating `whatsapp_sample_report.html` in the working tree during `test_phase1_smoke.py`; removed after each run (untracked, not part of this plan — Phase 2's `test_report_written_next_to_input` snapshot handles it gracefully).
- **Manual tty check (plan verification item):** a real interactive `chat-analyzer` tty session cannot be driven from this executor; the menu path is covered by the in-process test 7 (patched `isatty` + forced probe) and the menu rendering was visually verified via rich Console capture.

## User Setup Required

None - no external service configuration required. `CHAT_ANALYZER_FORCE_NLP=0|1` remains an optional debug/test override, not a user-facing requirement.

## Next Phase Readiness

- **CLI-04 complete:** all four failure classes (missing file, wrong format, empty chat, unparseable) exit 1 positionally / re-prompt interactively with distinct friendly messages + inline WhatsApp/Telegram export steps; zero tracebacks verified
- **D-04/D-06 UX shipped:** tty users get the one 3-option question; positional/piped users get exactly one hint line; the guarded installer announces name+size before any download and degrades gracefully offline
- **04-04 (legacy test rewire)** proceeds on an untouched baseline: `main.py` error strings preserved every substring the Phase 2 error tests assert; the 45 pre-existing legacy test failures (test_parser/test_analysis/test_reporting/test_end_to_end) remain 04-04's scope
- **04-05 (README)**: `[INFO] Tip: ... pip install chat-analyzer-pro[nlp]` hint copy and the menu option text are the canonical strings D-18/D-19's neutral presentation should mirror
- Blocker: none. Deferred: the 382 legacy ruff errors (deferred-items.md #1, unchanged baseline)

## Threat Surface

All new surface sits inside the plan's threat register — no unflagged additions:
- `install_nlp` runs `subprocess.run([sys.executable, "-m", "pip", "install", "torch", "transformers>=4.30,<6", ...])` — no shell=True, fixed audited package names (T-04-10 mitigated); failure raises RuntimeError → caller degrades (T-04-12 mitigated)
- Every failure path composes copy through `_friendly_error` + `typer.Exit(code=1) from None` — exception text wrapped, never dumped (T-04-11 mitigated)
- Menu default is option 2 (CPU ~0.6GB) with option 3 always available (T-04-13 mitigated); only torch/transformers enter the dependency graph at runtime (T-04-SC accepted)

## Known Stubs

None - no placeholder values or unwired data sources introduced by this plan.

---
*Phase: 04-nlp-extras-quality-gate*
*Completed: 2026-08-04*

## Self-Check: PASSED

- FOUND: `tests/test_phase4_cli.py`
- FOUND: `.planning/phases/04-nlp-extras-quality-gate/04-03-SUMMARY.md`
- FOUND: commit `6f2dcbf` (TDD RED)
- FOUND: commit `e639101` (Task 2 GREEN)
- FOUND: commit `f9a381d` (Task 3 GREEN)
- FOUND: commit `57e690b` (plan metadata)
- Verified: `python -m pytest tests/test_phase4_cli.py` 7/7 pass; `python -m pytest tests/test_phase2_cli.py tests/test_phase1_smoke.py` 20/20 pass; ruff 0 errors on all touched files
