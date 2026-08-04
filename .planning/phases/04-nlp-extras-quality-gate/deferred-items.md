# Deferred Items — Phase 04

Out-of-scope discoveries logged during plan execution (SCOPE BOUNDARY: only
fix issues directly caused by the current task's changes).

| # | Category | Item | Status | Deferred At | Found In |
|---|----------|------|--------|-------------|----------|
| 1 | Lint debt | `python -m ruff check src/chat_analyzer tests` reports 382 pre-existing errors (262 auto-fixable) in legacy analysis modules — e.g. `analysis/relationship_health.py` (50), `analysis/network_graph.py` (30), plus ~302 elsewhere (legacy `src/analysis/*`, legacy tests). The 04-01 Task 3 acceptance criterion "0 errors" is unsatisfiable at BASELINE (verified via `git stash` baseline run: identical 382/262 counts). Plan 04-01 introduces ZERO new errors (touched-file ruff 80 → 80, full-tree 382 → 382). Needs a dedicated legacy-lint cleanup plan before any future phase gates on `ruff check` clean. | Open | 2026-08-04 | 04-01 Task 3 verify |
| 2 | Stale artifact | Untracked `data/sample_chats/whatsapp_sample_report.html` left in the repo by a reverted prior 04-01 run. Not a test failure — `test_report_written_next_to_input` snapshots and tolerates it. Left as-is (do not commit). | Open | 2026-08-04 | 04-01 Task 3 |
| 3 | Report output relocation | `test_phase1_smoke.py`'s `run_cli`/`run_python_m` helpers run with no `cwd`, so the D-09 cwd-location change relocates the smoke suite's generated report from `data/sample_chats/` (next-to-input) to the repo ROOT. Smoke tests never assert on report location and still pass 10/10; the file is deleted after each run. Fix deferred to a smoke-suite cleanup (give the helpers a `cwd=tmp_path` or delete-on-success). Per instruction, `test_phase1_smoke.py` was NOT modified. | Open | 2026-08-04 | 04-01 Task 4 verify |
