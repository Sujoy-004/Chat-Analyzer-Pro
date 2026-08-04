---
phase: 04-nlp-extras-quality-gate
plan: 01
subsystem: cli
tags: [matplotlib, networkx, pandas, jinja2, html-report, tdd, relationship-health, network-graph]

# Dependency graph
requires:
  - phase: 02-one-command-terminal-insights
    provides: run_pipeline + adapt + report_html (AnalysisResults contract, 5-tab template, Phase 2 tests)
provides:
  - Always-on Relationship Health + Network Graph analysis in run_pipeline (no [nlp] gate)
  - Health + Network tabs in the HTML report (lead-ins at insights[5]/[6], base64 charts, scalar tables)
  - Report location resolved to Path.cwd()/<stem>_report.html (D-09), no output flags (D-08/OUT-04/OUT-05)
  - network_figure() figure-returning wrapper; NullHandler import hygiene in relationship_health.py
  - adapt() keyword-only health/network/emotion/summary=None contract slots (04-02 ready)
affects: [04-02-nlp-gate, 04-03-interactive-menu, 04-05-quality-gate, plan 01-02 verification]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pattern 2: figure-returning wrapper (no plt.show) so figures are base64-embeddable"
    - "Pattern 3: adapter extracts serializable scalars only — DataFrame/DiGraph never leak into AnalysisResults"
    - "NullHandler logging at import time (never logging.basicConfig in a library module)"

key-files:
  created:
    - tests/test_phase4_alwayson.py
  modified:
    - src/chat_analyzer/cli/contracts.py
    - src/chat_analyzer/analysis/relationship_health.py
    - src/chat_analyzer/analysis/network_graph.py
    - src/chat_analyzer/cli/adapters.py
    - src/chat_analyzer/cli/pipeline.py
    - src/chat_analyzer/cli/report_html.py
    - tests/test_phase2_cli.py
    - tests/test_phase2_report.py
    - tests/test_phase2_pipeline.py

key-decisions:
  - "D-09 resolved: report lands at Path.cwd()/<stem>_report.html — cwd, not next-to-input; OUT-04/OUT-05 resolution is no flags, always generate"
  - "Health + Network are always-on (D-07/D-07b): pandas/numpy/networkx/matplotlib only, so gating behind [nlp] adds friction with zero benefit"
  - "adapt() grows keyword-only health/network/emotion/summary=None (reconciliation note #2) — Phase 2 direct-call tests stay green; positional EDA param renamed eda_summary so the keyword-only summary slot can exist for 04-02"
  - "network_figure wrapper added (no plt.show) so the graph is base64-embeddable (Pitfall 6)"

patterns-established:
  - "Figure-returning wrapper convention: analysis modules return Figures; CLI serializes via _safe_chart (chart crash never kills the report)"
  - "Adapter boundary: AnalysisResults carries scalars only; DataFrame/prepared_data and networkx DiGraph never cross into the contract"

requirements-completed: [ANAL-07, ANAL-09, OUT-04, OUT-05, QUAL-02]

# Metrics
duration: 17min
completed: 2026-08-04
---

# Phase 4 Plan 1: Always-On Health + Network Analysis Slice Summary

**Relationship Health and Network Graph wired end-to-end as always-on analysis (no `[nlp]` install): new report tabs with grade/density lead-ins and base64 charts, report location moved to the cwd (`Path.cwd()/<stem>_report.html`), Phase 2 tests reconciled, proven by a RED→GREEN e2e test slice**

## Performance

- **Duration:** ~17 min (2026-08-04T05:50Z → 06:08Z)
- **Started:** 2026-08-04T05:50:40Z
- **Completed:** 2026-08-04T06:07:48Z
- **Tasks:** 3 executed + 1 verification (plan has 3 tasks + plan-level verification)
- **Files modified:** 10 (6 src + 3 test reconciled + 1 new test)

## Accomplishments

- `AnalysisResults` contract gains `health` + `network` keys; `adapt()` accepts keyword-only `health/network/emotion/summary=None` (defaults keep all Phase 2 direct-call tests green)
- `run_pipeline` runs `analyze_relationship_health(df)` + `analyze_network(df)` always-on (D-07/D-07b) and adds `health`/`network` base64 chart URIs (health trend reuses `plot_relationship_health_trend`, network uses the new `network_figure` wrapper)
- HTML report gains **Relationship Health** and **Network** tabs with narrative lead-ins (`insights[5]`/`insights[6]`), charts, and scalar tables — all inside the existing autoescape Jinja env + `_CHART_PREFIX` whitelist (T-04-01 mitigated)
- Import hygiene: `relationship_health.py` no longer calls `logging.basicConfig` at import (NullHandler per CONVENTIONS.md)
- Report location resolved to `Path.cwd() / f"{stem}_report.html"` (D-09) — no flags, always generated (OUT-04/OUT-05)
- RED→GREEN proof: `tests/test_phase4_alwayson.py` (in-process pipeline + subprocess cwd report) fails on absent keys before, passes after

## Task Commits

Each task was committed atomically:

1. **Task 1: Failing e2e test for health + network report slice** - `34717d5` (test — RED)
2. **Task 2: Thinnest slice — contracts, analysis fixes, adapters, pipeline, report tabs, cwd location** - `c2e8bdb` (feat — GREEN)
3. **Task 3: Reconcile Phase 2 tests for cwd location, new chart keys, new report sections** - `6596206` (test)

**Plan metadata:** `(docs: complete plan)` — final metadata commit below.

## Files Created/Modified

- `tests/test_phase4_alwayson.py` (created) - RED→GREEN e2e proof: in-process pipeline health/network keys + subprocess cwd-located report with the two new tabs
- `src/chat_analyzer/cli/contracts.py` - `health: dict[str, Any]`, `network: dict[str, Any]` added to `AnalysisResults`
- `src/chat_analyzer/analysis/relationship_health.py` - `logging.basicConfig` → `logging.getLogger(__name__).addHandler(logging.NullHandler())`
- `src/chat_analyzer/analysis/network_graph.py` - `network_figure(df)` module wrapper (returns Figure, no `plt.show`), `import matplotlib` bound for the annotation
- `src/chat_analyzer/cli/adapters.py` - keyword-only `health/network/emotion/summary=None`; `_build_health_block`/`_build_network_block` scalar-only extractors; `build_insights` cap 7→11 with health/network lead-ins at indices 5/6
- `src/chat_analyzer/cli/pipeline.py` - always-on health + network analysis in the redirect_stdout stage; charts gain `health`/`network`; `health=health_res, network=network_res` passed to `adapt`
- `src/chat_analyzer/cli/report_html.py` - `report_path = Path.cwd() / f"{stem}_report.html"`; two new nav buttons + panels (`tab-health`, `tab-network`); defensive `results.get("health", {})` render args
- `tests/test_phase2_cli.py` - `cwd=tmp_path` on all five report-writing CLI runs (LOW #8)
- `tests/test_phase2_report.py` - crafted results gain `health`/`network` keys; `_write` chdirs into tmp_path; `monkeypatch.chdir` in location + skip-note tests
- `tests/test_phase2_pipeline.py` - e2e charts set = 6 always-on keys

## Decisions Made

- **D-09 cwd location:** report resolves against `Path.cwd()`, not the input's parent — enables default-path behavior with no flags (OUT-04/OUT-05). Phase 2 tests reconciled by running the CLI with `cwd=tmp_path`.
- **Always-on gating:** health + network need no torch/transformers, so they ship in the base install (D-07/D-07b). Emotion + summary stay gated for 04-02.
- **Backward-compatible `adapt()`:** new params are keyword-only with `None` defaults; the positional EDA param was renamed `eda_summary` so the keyword-only `summary` slot can exist for 04-02's contract. `test_single_message_no_response_time` needs no edit (verified).
- **Figure-returning wrapper:** `network_figure` returns the Figure so `_safe_chart` can base64-embed it — `plt.show()` would black-hole the chart (Pitfall 6).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] F821 undefined name `matplotlib` in `network_figure` return annotation**
- **Found during:** Task 2 verification (ruff compare: touched-file errors 80 → 81)
- **Issue:** `-> "matplotlib.figure.Figure"` referenced a name the module never binds (`import matplotlib.pyplot as plt` alone does not bind `matplotlib`)
- **Fix:** Added `import matplotlib` at module level (alongside the existing `matplotlib.pyplot` import)
- **Files modified:** `src/chat_analyzer/analysis/network_graph.py`
- **Verification:** ruff on touched files back to baseline 80; `tests/test_phase4_alwayson.py` still 2 passed
- **Committed in:** `c2e8bdb` (Task 2 commit)

**2. [Rule 2 - Correctness] `test_interactive_path` + `test_telegram_roundtrip` wrote reports into the repo root after D-09**
- **Found during:** Task 3 verification (untracked `whatsapp_sample_report.html` + `telegram_sample_report.html` appeared in repo root)
- **Issue:** The plan's Task 3 only listed three `_run` calls to add `cwd=tmp_path` to, but two more CLI runs had no cwd → D-09 made them resolve the report against pytest's cwd (repo root), violating LOW #8
- **Fix:** Added `cwd=tmp_path` to both calls; removed the two generated artifacts
- **Files modified:** `tests/test_phase2_cli.py`
- **Verification:** Both tests pass with cwd; `git status` clean of repo-root artifacts
- **Committed in:** `6596206` (Task 3 commit)

**3. [Rule 2 - Correctness] `_write()` + `test_skip_note_surfacing` in test_phase2_report.py would write reports to the repo root after D-09**
- **Found during:** Task 3 (same root cause as #2 — direct `write_report` calls resolve against pytest cwd)
- **Issue:** Five `_write`-based tests + the skip-note test call `write_report` directly; with the cwd location they would drop `chat_analysis_test_report.html`/`a_report.html` into the repo tree
- **Fix:** `_write` chdirs into `tmp_path` (try/finally restore); `test_skip_note_surfacing` uses `monkeypatch.chdir(tmp_path)`
- **Files modified:** `tests/test_phase2_report.py`
- **Verification:** All 9 report tests pass; no artifacts in repo root after the run
- **Committed in:** `6596206` (Task 3 commit)

---

**Total deviations:** 3 auto-fixed (1 bug, 2 correctness)
**Impact on plan:** All three are direct consequences of the D-09 location change + the new annotation; no scope creep, no behavioral change to shipped code beyond the plan's intent.

## Issues Encountered

- **Ruff acceptance criterion is unsatisfiable at baseline (deferred, not a deviation):** Task 3 requires `python -m ruff check src/chat_analyzer tests` = 0 errors, but the baseline already reports **382 errors** (262 auto-fixable) in legacy analysis modules — verified by stashing all changes and re-running: identical 382/262. Plan 04-01 introduces **zero** new errors (touched-file 80→80, full-tree 382→382). This legacy lint debt is out of scope for this plan's changes and is logged in `deferred-items.md`; a dedicated legacy-lint cleanup plan is needed before any future phase gates on a clean `ruff check`.
- Pre-existing (unrelated) pytest warning during the smoke run (`torch` importable in this env) — unchanged by this plan.
- D-09 relocates `test_phase1_smoke.py`'s generated report from `data/sample_chats/` (next-to-input) to the repo root (its `run_cli` helpers pass no `cwd`). Smoke tests never assert on report location and still pass 10/10; the artifact is deleted after each run and the fix is logged in `deferred-items.md` #3 (per instruction, the smoke test file was not modified).

## Known Stubs

None — health/network tabs render real analysis output; both chart URIs come from actual figures via `_safe_chart`.

## Threat Flags

None — all new surface (two tabs, two chart URIs, two adapter blocks) stays inside the plan's threat model: existing Jinja autoescape environment, `_CHART_PREFIX` whitelist boundary, scalar-only serialization (T-04-01/T-04-04 mitigated). No new network endpoints, auth paths, or file-access patterns beyond the mandated cwd report write.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 04-02 (NLP gate + emotion/summary) can slot into the existing keyword-only `emotion=None, summary=None` adapt params and the charts→root migration noted in `test_phase2_pipeline.py` (7-key charts set with the gate pinned True)
- `test_phase4_alwayson.py`'s subprocess helper is reusable for 04-03's interactive-menu tests
- **Blocker for future phases:** the 382-error legacy ruff debt (see deferred-items.md #1) must be addressed before any phase gates on a clean `ruff check`; ROADMAP/verification wording should not promise 0 errors until then

---
*Phase: 04-nlp-extras-quality-gate*
*Completed: 2026-08-04*

## Self-Check: PASSED

- FOUND: `.planning/phases/04-nlp-extras-quality-gate/04-01-SUMMARY.md`
- FOUND: `.planning/phases/04-nlp-extras-quality-gate/deferred-items.md`
- FOUND: `tests/test_phase4_alwayson.py`
- FOUND: commit `34717d5` (RED)
- FOUND: commit `c2e8bdb` (GREEN)
- FOUND: commit `6596206` (Task 3 reconcile)
