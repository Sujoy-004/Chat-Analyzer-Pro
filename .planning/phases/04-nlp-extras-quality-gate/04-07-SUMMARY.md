---
phase: 04-nlp-extras-quality-gate
plan: 07
subsystem: nlp
tags: [emotion, parallel, multiprocessing, spawn-safety, performance, dedupe, quarterly, env-config, determinism]

# Dependency graph
requires:
  - phase: 04-nlp-extras-quality-gate
    provides: 04-02 gated emotion path (distilbert 6-class EmotionAnalyzer, nlp_gate availability probe) and 04-06 deterministic stratified sampling (sample_cap + emotion_sample_cap) — the parallel exact path slots into _score_unique_texts, which both the exact and sampled paths already share
provides:
  - Parallel exact DistilBERT emotion scoring for large chats: unique-text dedupe + process map-reduce over contiguous fixed-order chunks (default 3 spawn workers, capped min(value, cpu, 8)), byte-identical output to the sequential exact path regardless of worker count
  - CHAT_ANALYZER_EMOTION_WORKERS env knob (absent/empty→3; "0"/"1"/"off"/"false"/parses-to-<2→1 sequential; positive int→max(1,min(value,cpu,8)); garbage→3; never raises) + real-pipeline gate so mocked tests stay sequential by construction (D-17)
  - Failure-degrade contract: any worker-side failure (parse error, non-aligned shape, pipeline exception, BrokenProcessPool) falls back to the identical sequential _score_batch — byte-parity guaranteed by construction, not just by test
  - Quarterly emotion aggregation (per-quarter MEAN of the six emotion scores, oldest first, 6-decimal rounding, honors the emotion_scored column in sampled mode) + ECharts timeline spec rendered in the HTML report
  - Test coverage: test_emotion_parallel.py (7 fast + 1 slow real-model smoke at 07a7b57, +1 fast boundary test in the review-fix ab3da3b) and test_emotion_quarterly.py (14 fast tests)
  - cp1252 smoke-test fix (86faefe): run_cli/run_python_m decode subprocess output as utf-8 — the pre-existing deferred-items #6 failure is no longer triggerable and is marked Resolved
affects: [large-chat exact-scoring performance, CPU oversubscription risk on shared boxes, QUAL-02 regression suite, deferred-items #6]

# Tech tracking
tech-stack:
  added: []  # NO new packages — stdlib concurrent.futures + existing torch/transformers
  patterns:
    - "Spawn-safe worker contract: module-level _score_text_chunk(texts, model_name, batch_size, num_threads) taking ONLY JSON-able args (list[str], str, two ints); the pipeline is built IN the child; torch.set_num_threads(threads) runs before inference; TOKENIZERS_PARALLELISM=false is set before the transformers import; 4.x/5.x shapes normalize via the shared _parse_emotion_scores"
    - "Thread clamp scaled to the ACTUAL pool size (200749f): num_threads = max(1, (os.cpu_count() or 1) // workers) — on the 12-core box, workers=3 → 4 threads/worker (12 total), never N x 10 default threads"
    - "Parity-by-construction degrade: _score_unique_texts_parallel returns None on any exception (logger.exception, BLE001-safe); _score_unique_texts falls back to the identical sequential _score_batch; chunk boundaries cannot change per-text results because scores are pure functions of the text"
    - "Real-pipeline gate: _is_real_pipeline() returns True only for a genuine transformers.Pipeline instance — pytest mocks (plain callables) always take the sequential path, keeping the fast suite fast and offline (D-17)"

key-files:
  created: [tests/test_emotion_parallel.py, tests/test_emotion_quarterly.py, .planning/phases/04-nlp-extras-quality-gate/04-07-SUMMARY.md]
  modified: [src/chat_analyzer/analysis/emotion.py, src/chat_analyzer/cli/nlp_gate.py, src/chat_analyzer/cli/pipeline.py, src/chat_analyzer/cli/adapters.py, src/chat_analyzer/cli/report_html.py, src/chat_analyzer/cli/chart_json.py, tests/test_phase1_smoke.py]

key-decisions:
  - "Threshold kept at VADER's 20_000 unique texts (research recommended 5_000): DistilBERT is ~10x slower per message so the breakeven is lower, but the plan chose to measure with scripts/benchmark.py before tuning down — parity and gating are unaffected by the exact threshold value"
  - "Real-pipeline gate over env-only gating: parallel ONLY for a genuine transformers.Pipeline instance — mocked pipelines always take the sequential path, keeping D-17 tests fast, offline, and deterministic"
  - "Env disable semantics aligned with emotion_sample_cap (MD-01, fixed in 60b42da): 'off'/'false'/parses-to-<2 ('00','-0','-5') map to 1 (sequential) instead of the default 3, so a user's natural 'disable' spellings never silently spawn a 3-worker pool (~4 GB+ peak)"
  - "Explicit workers on the public API are validated + capped (MD-02, fixed in 60b42da): non-numeric falls back to nlp_gate env resolution; ints clamp to max(1, min(value, os.cpu_count() or 1, 8)) — direct analyze_emotions(df, workers=N) callers cannot bypass the RAM bound"
  - "No local_files_only=True in workers (research delta #4, accepted): the parallel path only runs when the parent pipeline exists, so the cache is provably complete; default etag checks are read-only"

patterns-established:
  - "The env-parsing contract lives in nlp_gate.emotion_worker_count() (mirroring emotion_sample_cap() style) — one resolver for the CLI pipeline and any future callers, never raises"
  - "The worker shares the parent's _parse_emotion_scores normalization (module-level; 4.x flat / 5.x nested / whole-batch-wrapped shapes) so parallel and sequential output are structurally identical — the worker must never re-derive shape logic"
  - "Quarterly aggregation rides the same rows the summary used: pipeline computes emotion_summary['quarterly'] = get_emotion_quarterly(df_emo), which honors the emotion_scored column in sampled mode, so report labels and quarterly values always agree"

requirements-completed: []

# Metrics
duration: "~2 sessions (2026-08-14 + 2026-08-15), multi-agent"
completed: 2026-08-15
---

# Phase 4 Plan 7: Parallel Exact DistilBERT Emotion Scoring Summary

**Process-map-reduce exact emotion scoring for large chats — dedupe-by-unique-text, spawn-safe module-level workers with the torch thread clamp scaled to the pool size and TOKENIZERS_PARALLELISM hardened, a CHAT_ANALYZER_EMOTION_WORKERS env knob, failure-degrade to the byte-identical sequential path, quarterly emotion aggregation + timeline chart, and two new test suites (7 fast + 1 slow smoke; 14 fast) — reviewed APPROVE WITH CHANGES (no CRITICAL/HIGH), all in-scope findings fixed.**

## Performance

- **Duration:** ~2 sessions across 2026-08-14 (15:37–15:50Z commits) and 2026-08-15 (08:26–08:31Z commits + 08:47Z review-fix commits)
- **Started:** 2026-08-14
- **Completed:** 2026-08-15
- **Tasks:** 11 (feature/test) + 2 review-fix commits
- **Files changed:** 9 (7 modified, 2 created)

## Accomplishments
- Large-chat exact emotion scoring now fans unique-text inference across a process pool: `_score_unique_texts` dedupes via `dict.fromkeys` (each unique string scored exactly once), and above `_EMOTION_PARALLEL_THRESHOLD` (20_000) with `workers >= 2` the pool map-reduces contiguous fixed-order chunks — output is byte-identical to the sequential exact path regardless of worker count or chunk boundaries
- Spawn-safe worker `_score_text_chunk` (emotion.py:69): module-level, takes only JSON-able args, builds its own `transformers` pipeline in-child (never pickles one), clamps `torch.set_num_threads`, sets `TOKENIZERS_PARALLELISM=false` before the `transformers` import (111a6bd), and normalizes 4.x/5.x shapes through the shared `_parse_emotion_scores` — verified safe on Windows spawn
- Thread clamp scaled to the actual pool size (200749f): `num_threads = max(1, (os.cpu_count() or 1) // workers)` — 12-core box: 3 workers → 4 threads/worker (12 total), no N x 10 oversubscription thrash
- `emotion_worker_count()` env contract (96fb271, semantics hardened in 60b42da): absent/empty→3, `"0"`/`"1"`/`"off"`/`"false"`/parses-to-<2→1 (sequential), positive int→`max(1, min(value, os.cpu_count() or 1, 8))`, garbage→3; never raises; callers parallelize only when `>= 2`
- Real-pipeline gate (`_is_real_pipeline`): parallel only for genuine `transformers.Pipeline` instances, so mocked tests keep exercising the sequential path untouched (D-17 safe)
- Failure-degrade contract: `_score_unique_texts_parallel` wraps `pool.map` in try/except → `logger.exception` → returns `None`; the caller falls back to the identical sequential `_score_batch` (VADER precedent) — per-row output matches today's exact path even under partial worker failures
- Quarterly emotion aggregation + timeline chart: `get_emotion_quarterly` (module-level, per-quarter MEAN of the six emotion columns, oldest first, 6-decimal rounding, honors `emotion_scored` in sampled mode, never raises) wired in pipeline as `emotion_summary["quarterly"]`, consumed by `build_emotion_timeline_spec` (chart_json.py) and rendered as the `emotion-quarterly` ECharts line chart in the HTML report; the contract rides the adapter block (adapters.py `"quarterly"`)
- cp1252 fix (86faefe): `encoding="utf-8"` added to BOTH `run_cli` and `run_python_m` in test_phase1_smoke.py, matching main.py's `stream.reconfigure(encoding="utf-8")` bootstrap — deferred-items #6 (the pre-existing `test_console_script_help` Windows failure) is resolved
- Review verdict **APPROVE WITH CHANGES** (04-07-REVIEW.md): 0 CRITICAL, 0 HIGH, 3 MEDIUM, 2 LOW, 3 NIT. The fixer landed all 5 in-scope findings (MD-01/02/03 + LW-01/02) in commits `60b42da` + `ab3da3b` (see Deviations)

## Task Commits

Each task was committed atomically (chronological order):

1. **Task 1: Quarterly emotion ECharts spec builder (chart_json.py)** - `c8e1264` (feat)
2. **Task 2: Render quarterly emotion chart in the report (report_html.py)** - `9c84d0e` (feat)
3. **Task 3: Carry quarterly emotion through the adapter contract (adapters.py)** - `9c2fc9e` (feat)
4. **Task 4: Dedupe emotion scoring by unique text (emotion.py)** - `dd5fe0a` (perf)
5. **Task 5: Parallel emotion workers via process map-reduce + env knob (emotion.py, nlp_gate.py)** - `96fb271` (perf)
6. **Task 6: Quarterly emotion aggregation + pipeline wiring (emotion.py, pipeline.py)** - `e6eef33` (feat)
7. **Task 7: Scale worker torch threads with actual pool size (emotion.py)** - `200749f` (perf)
8. **Task 8: Disable tokenizers parallelism in spawn workers (emotion.py)** - `111a6bd` (perf)
9. **Task 9: Decode subprocess output as utf-8 in smoke helpers / cp1252 (test_phase1_smoke.py)** - `86faefe` (fix)
10. **Task 10: Quarterly emotion aggregation + timeline spec coverage (tests/test_emotion_quarterly.py)** - `c2e5e6d` (test)
11. **Task 11: Parallel emotion worker + threshold + env coverage (tests/test_emotion_parallel.py)** - `07a7b57` (test)

Review-fix commits (landed 2026-08-15 08:47Z, after the review):

12. **Task 12: Consistent emotion worker count env parsing + worker cap (MD-01/02; nlp_gate.py, emotion.py)** - `60b42da` (fix)
13. **Task 13: Robust worker-count assertions + thread-clamp coverage (MD-03/LW-01/02; tests/test_emotion_parallel.py)** - `ab3da3b` (test)

## Files Created/Modified
- `src/chat_analyzer/analysis/emotion.py` - `_EMOTION_PARALLEL_THRESHOLD = 20_000`; module-level spawn-safe `_score_text_chunk` (69); `_score_unique_texts` dedupe via `dict.fromkeys`; `_is_real_pipeline` (genuine transformers.Pipeline only); `_scoring_workers` (real-pipeline gate, explicit-workers validation/cap); `_score_unique_texts_parallel` (ProcessPoolExecutor + contiguous chunks + `itertools.repeat` args + `num_threads = max(1, cpu // workers)` + degrade-to-None); module-level `get_emotion_quarterly` + `EmotionAnalyzer.get_emotion_quarterly` wrapper
- `src/chat_analyzer/cli/nlp_gate.py` - `_EMOTION_WORKERS_ENV`, `EMOTION_WORKERS_DEFAULT = 3`, `emotion_worker_count()` env resolver (never raises; post-fix semantics in 60b42da)
- `src/chat_analyzer/cli/pipeline.py` - `workers = nlp_gate.emotion_worker_count()` passed to `analyze_emotions(..., workers=workers)`; `emotion_summary["quarterly"] = emo_analyzer.get_emotion_quarterly(df_emo)`; best-effort `build_emotion_timeline_spec` → `charts_json["emotion-quarterly"]`
- `src/chat_analyzer/cli/adapters.py` - `"quarterly": emotion.get("quarterly") or []` in the emotion contract block
- `src/chat_analyzer/cli/report_html.py` - `emotion-quarterly` chart div in the chart keys + render
- `src/chat_analyzer/cli/chart_json.py` - `build_emotion_timeline_spec` (one smooth line per emotion, slider+inside dataZoom, yAxis 0..1, JSON-safe, never raises)
- `tests/test_phase1_smoke.py` - `encoding="utf-8"` in `run_cli` + `run_python_m` (86faefe)
- `tests/test_emotion_parallel.py` - 7 fast tests (env contract, real-pipeline gate, threshold below/above, contiguous chunks, pool-failure degrade, tokenizers-parallelism source order) + 1 `@pytest.mark.slow` real-model spawn-parity smoke (07a7b57); +1 fast boundary test (exact 20_000 stays sequential) and hardened assertions (ab3da3b)
- `tests/test_emotion_quarterly.py` - 14 fast tests (quarterly means oldest-first/rounded, degenerate frames → [], sampled-mode rows-only aggregation, malformed-input never raises, analyzer wrapper delegation, timeline spec None/with-data/malformed/NaN-safe/JSON-safe)

## Decisions Made
- Threshold kept at 20_000 (VADER precedent) — measure with scripts/benchmark.py before tuning down (research delta #2)
- Real-pipeline gate + env knob: parallel only when a genuine transformers pipeline exists AND `emotion_worker_count() >= 2` AND `len(unique_texts) > threshold`
- Env disable semantics aligned with `emotion_sample_cap()` (MD-01): `"off"`/`"false"`/`<2` → sequential 1, never a silent default-3 pool
- Explicit `workers` validated + capped on the public API (MD-02): non-numeric → env resolution; int → `max(1, min(value, cpu, 8))`
- No `local_files_only=True` in workers — cache is provably complete once the parent pipeline exists (research delta #4, accepted)
- Quarterly aggregation honors `emotion_scored` so sampled-mode report labels and chart values agree

## Deviations from Plan

### Auto-fixed Issues (in plan execution)

**1. [Research delta #1 - Oversubscription] Thread clamp hardcoded to `cpu_count // 3`, not scaled to the actual worker count**
- **Found during:** 96fb271 (parallel workers landed) review of the worker
- **Issue:** `_score_text_chunk` used `max(1, cpu_count // 3)` regardless of pool size — with 6 env workers each child still used 4 threads → 24 threads on 12 cores → 2x oversubscription
- **Fix:** Parent computes `num_threads = max(1, (os.cpu_count() or 1) // workers)` in `_score_unique_texts_parallel` and passes it as the 4th worker arg (200749f); later proven end-to-end by the fixer's `seen_threads` assertion (ab3da3b, LW-02)
- **Files modified:** src/chat_analyzer/analysis/emotion.py
- **Committed in:** 200749f

**2. [Research delta #3 - Hardening] No `TOKENIZERS_PARALLELISM=false` in the worker**
- **Found during:** research (Q1/Q4) + implementation review
- **Issue:** tokenizers' rayon pool could oversubscribe per child on top of the torch threads
- **Fix:** `os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")` set BEFORE the `transformers` import inside `_score_text_chunk` (111a6bd); source-order asserted by `test_tokenizers_parallelism_disabled_in_worker`
- **Files modified:** src/chat_analyzer/analysis/emotion.py
- **Committed in:** 111a6bd

**3. [Research delta #5 - Coverage gap] ZERO test coverage for the parallel branch**
- **Found during:** research (grep showed no test referenced `emotion_worker_count` / `_score_text_chunk` / `_score_unique_texts_parallel` / `_EMOTION_PARALLEL_THRESHOLD`)
- **Issue:** the real-pipeline gate means mocked tests cannot reach the pool; the spawn path was only exercisable with a real model
- **Fix:** `tests/test_emotion_parallel.py` (07a7b57) — 7 fast mocked tests (fake pool, inline worker, env contract, gating, chunking, degrade) + 1 `@pytest.mark.slow` real-model spawn-parity smoke that forces the threshold to 1 with the cached pipeline and asserts parity vs the sequential reference (skipped when the [nlp] extra/model cache is absent)
- **Files modified:** tests/test_emotion_parallel.py (new)
- **Committed in:** 07a7b57

### Review Fixes (from 04-07-REVIEW.md, APPROVE WITH CHANGES — fixed by the fixer)

**4. [MD-01 - Medium] `emotion_worker_count()` — negative/garbage values silently enabled the default 3-worker pool, inconsistent with `emotion_sample_cap()`**
- **Issue:** `-5 → 3`, `off`/`false` → 3 (parallel), while the sample knob disables on any ≤ 0 — a user's natural "disable" spellings spawned a 3-worker pool on huge chats
- **Fix (60b42da):** `"off"`/`"false"` and any parses-to-`<2` value → **1 (sequential)**; garbage still → 3; docstring + module comment updated
- **Verification:** extended `test_emotion_worker_count_env_contract` asserts the full table (ab3da3b)

**5. [MD-02 - Medium] `_scoring_workers()` accepted uncapped/unvalidated explicit `workers` — direct API callers bypassed the RAM cap; garbage raised**
- **Issue:** `analyze_emotions(df, workers=32)` → 32 spawned processes (no cap); `workers="abc"` → `int()` ValueError propagating out of the degrade contract
- **Fix (60b42da):** non-numeric → `logger.exception` + fall back to `nlp_gate.emotion_worker_count()`; positive ints → `max(1, min(value, os.cpu_count() or 1, 8))`
- **Verification:** ruff + fast suites green (24 passed, 1 deselected after fix)

**6. [MD-03 - Medium] Test fragility — `test_emotion_worker_count_env_contract` hardcoded `== 4` for env "4", ignoring the cpu cap**
- **Fix (ab3da3b):** asserts `max(1, min(4, os.cpu_count() or 1, 8))`; also added the MD-01 semantics cases and the `test_threshold_at_exactly_20000_stays_sequential` strict-greater boundary test (LW-02)

**7. [LW-01 - Low] Slow spawn-parity smoke could silently pass without exercising the spawn path**
- **Fix (ab3da3b):** skip when `CHAT_ANALYZER_FORCE_NLP` is set; replaced the hardcoded `~/.cache/huggingface/hub/...` path with `nlp_gate.model_cached(nlp_gate.MODEL_ID)` (honors HF_HUB_CACHE/HF_HOME); removed the inert `CHAT_ANALYZER_EMOTION_WORKERS=2` setenv (NT-02)

**8. [LW-02 - Low] Thread-clamp formula and exact 20_000 threshold boundary untested**
- **Fix (ab3da3b):** `_FakePool.map` records `seen_threads`; the chunk test asserts `num_threads == max(1, (os.cpu_count() or 1) // 3)` reaches the worker; new exact-20_000 boundary test

NITs NT-01/NT-03 were out of scope (kept `setdefault` — respecting an explicit user `TOKENIZERS_PARALLELISM`; the dead `or` fallback is harmless); NT-02's dead env line was removed trivially inside the LW-01 edit.

---

**Total deviations:** 3 auto-fixed in-plan (1 oversubscription, 1 hardening, 1 coverage gap) + 5 review findings fixed (MD-01/02/03, LW-01/02); NITs NT-01/03 accepted.
**Impact on plan:** All fixes were correctness/hardening improvements within scope; no scope creep, no new packages, and the exact-path parity contract was preserved throughout.

## Issues Encountered
- **Pre-existing Windows cp1252 test failure — RESOLVED in this phase (86faefe):** `tests/test_phase1_smoke.py::test_console_script_help` failed on Windows with `UnicodeDecodeError: 'charmap' codec can't decode byte 0x90` because the `run_cli`/`run_python_m` subprocess helpers decoded `--help` output with the locale encoding. The fix adds `encoding="utf-8"` to both helpers (matching main.py's `stream.reconfigure(encoding="utf-8")`); the reviewer confirmed the observable failure is no longer triggerable (main.py's Typer help is pure ASCII under utf-8). Deferred-items #6 is now **Resolved**.
- **Review findings (see Deviations #4-#8):** 3 MEDIUM + 2 LOW fixed by the fixer in 60b42da/ab3da3b; 2 NITs accepted (NT-01 `setdefault` respected, NT-03 harmless fallback).

## Stub Scan
- No stubs introduced. The parallel driver returns a fully populated `{text: scores}` map or `None` (which triggers the complete sequential fallback); the quarterly aggregator returns real lists or `[]` — never empty placeholders that could be mistaken for data.

## Threat Surface Scan
- No new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries. The env knob reads one `CHAT_ANALYZER_EMOTION_WORKERS` var with a strict whitelist parser (never `eval`; never raises). Spawn workers receive only JSON-able args and re-use the already-audited local model cache (read-only concurrent loads). No threat flags.

## User Setup Required

None - no external service configuration required. Users may optionally set `CHAT_ANALYZER_EMOTION_WORKERS` (e.g. `1` to force sequential, `2`-`8` for a bigger pool, or `off`/`false`/`0` for sequential). Parallel exact scoring engages automatically above 20_000 unique texts with a real pipeline installed.

## Next Phase Readiness
- Large-chat exact emotion scoring is bounded by dedupe + parallel workers and byte-identical to the sequential path; quarterly emotion chart ships; both new suites + ruff are green
- Review-fix commits `60b42da` + `ab3da3b` landed and verify clean (24 passed, 1 deselected; ruff clean; nlp_gate/alwayson suites 7 passed, 2 deselected)
- Follow-up candidates: the result cache keyed by file hash (repeat run = seconds) is now planned as **04-08** (04-08-PLAN.md + 04-08-RESEARCH.md drafted); benchmark-measure the 20_000 threshold before lowering it; deferred-items #1–#5 remain open and unchanged

---
*Phase: 04-nlp-extras-quality-gate*
*Completed: 2026-08-15*

## Self-Check: PASSED

All 11 task files + SUMMARY.md verified present on disk; all 11 feature/test commit
hashes (c8e1264, 9c84d0e, 9c2fc9e, dd5fe0a, 96fb271, e6eef33, 200749f, 111a6bd,
86faefe, c2e5e6d, 07a7b57) plus the 2 review-fix hashes (60b42da, ab3da3b) verified
in git history. Ruff clean on src + tests; fast suites green (44 passed, 1
deselected pre-fix; 24 passed, 1 deselected post-fix on parallel+sampling; phase1
smoke 7 passed). Review verdict: APPROVE WITH CHANGES (0 CRITICAL, 0 HIGH);
deferred-items #6 (cp1252) marked Resolved.