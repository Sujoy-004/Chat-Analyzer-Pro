---
phase: 04-nlp-extras-quality-gate
fixed_at: 2026-08-15T12:30:00Z
review_path: .planning/phases/04-nlp-extras-quality-gate/04-07-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 04-07: Code Review Fix Report

**Fixed at:** 2026-08-15
**Source review:** `.planning/phases/04-nlp-extras-quality-gate/04-07-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 5 (MD-01, MD-02, MD-03, LW-01, LW-02)
- Fixed: 5
- Skipped: 0

## Fixed Issues

### MD-01: `emotion_worker_count()` — consistent disable/fall-back semantics with `emotion_sample_cap()`

**Files modified:** `src/chat_analyzer/cli/nlp_gate.py`
**Commit:** 60b42da
**Applied fix:** Values parsing to `< 2` ("00", "-0", "-5") now map to **1 (sequential)** instead of the default 3; "off"/"false" (case-insensitive via `.lower()`, matching `emotion_sample_cap()`) also map to **1 (sequential)**. Garbage still falls back to the default 3; never raises. Docstring and the module comment block updated to the new contract.

| Env value | Before | After |
|---|---|---|
| absent / empty | 3 (default) | 3 (default) |
| `"0"` / `"1"` | 1 | 1 |
| `"off"` / `"false"` / `"OFF"` / `"FALSE"` | 3 (default) — silently parallel | **1 (sequential)** |
| `"00"` / `"-0"` / `"-5"` | 3 (default) — silently parallel | **1 (sequential)** |
| `"4"` | min(4, cpu, 8) | min(4, cpu, 8) |
| `"99"` | min(99, cpu, 8) | min(99, cpu, 8) |
| garbage (`"abc"`, `"3x"`) | 3 (default) | 3 (default) — never raises |

### MD-02: `_scoring_workers()` — validate/cap explicit `workers`, never raises

**Files modified:** `src/chat_analyzer/analysis/emotion.py`
**Commit:** 60b42da
**Applied fix:** An explicitly passed `workers` is now validated: non-numeric (TypeError/ValueError) falls back to `nlp_gate.emotion_worker_count()` (env resolution, never raises) instead of raising out of `analyze_emotions`; positive ints are clamped to `max(1, min(value, os.cpu_count() or 1, 8))` — the same cap math nlp_gate uses, so direct library API callers no longer bypass the RAM bound. `_scoring_workers` and the `analyze_emotions` `workers` docstring updated to the new contract.

### MD-03: `test_emotion_worker_count_env_contract` — capped assertion instead of literal 4

**Files modified:** `tests/test_emotion_parallel.py`
**Commit:** ab3da3b
**Applied fix:** The `"4"` case now asserts `max(1, min(4, os.cpu_count() or 1, 8))` (no longer a hardcoded `== 4`). The contract test was extended for the MD-01 semantics: `"off"/"false"/"OFF"/"FALSE"` → 1, `"00"/"-0"/"-5"` → 1, garbage (`"abc"`, `"3x"`) → 3 default. Header docstring updated.

### LW-01: Slow spawn smoke — robust skip (FORCE_NLP gate + canonical cache probe)

**Files modified:** `tests/test_emotion_parallel.py`
**Commit:** ab3da3b
**Applied fix:** Added a skip when `CHAT_ANALYZER_FORCE_NLP` is set (the probe can then report True without transformers, letting the smoke pass without spawning). Replaced the hardcoded `~/.cache/huggingface/hub/...` path with `nlp_gate.model_cached(nlp_gate.MODEL_ID)` (honors `HF_HUB_CACHE`/`HF_HOME`). Kept the `nlp_available()` + model-cache skip. Also removed the inert `CHAT_ANALYZER_EMOTION_WORKERS=2` setenv line (NT-02, trivially free on the same lines).

### LW-02: Direct thread-clamp assertion + exact 20_000 boundary test

**Files modified:** `tests/test_emotion_parallel.py`
**Commit:** ab3da3b
**Applied fix:** `_FakePool.map` now records `seen_threads` (the `num_threads` arg per chunk); `test_parallel_chunks_contiguous_fixed_order` asserts `pool.seen_threads == [max(1, (os.cpu_count() or 1) // 3)] * len(chunks)`, proving the per-worker torch-thread budget reaches the worker. Added `test_threshold_at_exactly_20000_stays_sequential` asserting the strict-greater boundary (`len(unique_texts) > _EMOTION_PARALLEL_THRESHOLD` is False at exactly 20_000).

## Skipped Issues

None — all in-scope findings fixed. NT-01..03 (NITs) were out of scope; NT-02's dead env line was removed trivially while editing the same smoke function.

## Verification

1. `.venv\Scripts\python.exe -m pytest tests/test_emotion_parallel.py tests/test_emotion_sampling.py -m "not slow" -q` → **24 passed, 1 deselected**
2. `.venv\Scripts\python.exe -m ruff check src/chat_analyzer/analysis/emotion.py src/chat_analyzer/cli/nlp_gate.py tests/test_emotion_parallel.py` → **all checks passed**
3. `.venv\Scripts\python.exe -m pytest tests/test_phase4_nlp_gate.py tests/test_phase4_alwayson.py -m "not slow" -q` → **7 passed, 2 deselected**

---

_Fixed: 2026-08-15_
_Fixer: the agent (gsd-code-fixer)_
_Iteration: 1_