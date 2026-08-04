# Deferred Items — Phase 04

Out-of-scope discoveries logged during plan execution (SCOPE BOUNDARY: only
fix issues directly caused by the current task's changes).

| # | Category | Item | Status | Deferred At | Found In |
|---|----------|------|--------|-------------|----------|
| 1 | Lint debt | `python -m ruff check src/chat_analyzer tests` reports 382 pre-existing errors (262 auto-fixable) in legacy analysis modules — e.g. `analysis/relationship_health.py` (50), `analysis/network_graph.py` (30), plus ~302 elsewhere (legacy `src/analysis/*`, legacy tests). The 04-01 Task 3 acceptance criterion "0 errors" is unsatisfiable at BASELINE (verified via `git stash` baseline run: identical 382/262 counts). Plan 04-01 introduces ZERO new errors (touched-file ruff 80 → 80, full-tree 382 → 382). Needs a dedicated legacy-lint cleanup plan before any future phase gates on `ruff check` clean. | Open | 2026-08-04 | 04-01 Task 3 verify |
| 2 | Stale artifact | Untracked `data/sample_chats/whatsapp_sample_report.html` left in the repo by a reverted prior 04-01 run. Not a test failure — `test_report_written_next_to_input` snapshots and tolerates it. Left as-is (do not commit). | Open | 2026-08-04 | 04-01 Task 3 |
