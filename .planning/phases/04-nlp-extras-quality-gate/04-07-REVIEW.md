---
phase: 04-nlp-extras-quality-gate
reviewed: 2026-08-15T12:00:00Z
depth: deep
files_reviewed: 6
files_reviewed_list:
  - src/chat_analyzer/analysis/emotion.py
  - src/chat_analyzer/cli/nlp_gate.py
  - src/chat_analyzer/cli/pipeline.py
  - tests/test_emotion_parallel.py
  - tests/test_emotion_quarterly.py
  - tests/test_phase1_smoke.py
findings:
  critical: 0
  high: 0
  medium: 3
  low: 2
  nit: 3
  total: 8
status: issues_found
---

# Phase 04-07: Code Review Report — Parallel Emotion Workers

**Reviewed:** 2026-08-15
**Depth:** deep (cross-file: emotion.py ↔ nlp_gate.py ↔ pipeline.py ↔ tests)
**Files Reviewed:** 6
**Status:** issues_found (no CRITICAL / no HIGH)

## Summary

Reviewed the parallel exact-emotion-scoring work (commits dd5fe0a, 96fb271,
200749f, 111a6bd, 86faefe + test commits 07a7b57, c2e5e6d) at deep depth,
including tracing the full call chain `pipeline.run_pipeline → analyze_emotions
→ _score_unique_texts → _score_unique_texts_parallel → _score_text_chunk
(spawn child) → _parse_emotion_scores` and the quarterly contract through
`get_emotion_quarterly → build_emotion_timeline_spec`.

Verification performed:
- **Fast suites:** `.venv\Scripts\python.exe -m pytest tests/test_emotion_parallel.py tests/test_emotion_quarterly.py tests/test_perf_parity_emotion.py tests/test_emotion_sampling.py -m "not slow" -q` → **44 passed, 1 deselected** (slow smoke correctly deselected); `tests/test_phase1_smoke.py -m "not slow"` → 7 passed.
- **Ruff:** `ruff check src/chat_analyzer/analysis/emotion.py src/chat_analyzer/cli/nlp_gate.py tests/test_emotion_parallel.py tests/test_emotion_quarterly.py` → all checks passed.
- **Parity contract:** The map-reduce is parity-safe *by construction*, not just by test: the parent's `_score_unique_texts` falls back to the exact same sequential `_score_batch` whenever the driver returns `None`, and every worker-side failure (parse error, non-aligned shape, pipeline exception, BrokenProcessPool) funnels into that fallback. Chunk boundaries cannot affect per-text results because scores are pure functions of the text and the alignment check (`len(batch_out) != len(chunk)` → raise) precedes the `zip`. Dedupe order is preserved by `dict.fromkeys` and restored via the `{text: scores}` map. `_parse_emotion_scores` handles 4.x flat / 5.x nested / whole-batch-wrapped shapes identically in parent and worker (worker passes the default `EMOTION_LABELS`, which equals `self.emotions`). No parity divergence found.
- **Windows spawn safety:** module-level `_score_text_chunk`; only JSON-able args (`list[str]`, `str` model id, two `int`s) cross the pool; pipeline built in-child; no global reads (`_emotion_analyzer` untouched); `itertools.repeat` args are consumed parent-side per-task (never pickled). Safe.
- **Oversubscription:** `num_threads = max(1, (os.cpu_count() or 1) // workers)` — verified 12-core box: workers=2 → 6 threads/worker (12 total), workers=3 → 4/worker (12 total). `TOKENIZERS_PARALLELISM=false` is set before the `transformers` import inside the worker (source order verified).
- **Env parsing:** `emotion_worker_count()` never raises; absent→3, "0"/"1"→1, "4"→4, "99"→min(99,cpu,8), garbage→3 (all asserted by tests, and confirmed by live probe).
- **D-17:** fast tests are fully mocked (real-pipeline gate returns 0 workers for plain callables); the slow smoke gates on model-cache presence. See LOW-01 for a silent-pass hole.
- **cp1252 fix:** `encoding="utf-8"` present in BOTH `run_cli` and `run_python_m` (test_phase1_smoke.py:68, 81), matching main.py's `stream.reconfigure(encoding="utf-8")` bootstrap; help tests pass.

The implementation is genuinely solid — the degrade-to-sequential design makes
byte-parity robust even under adversarial conditions (partial worker failures,
5.x shape quirks, per-item parse failures all resolve to the identical
sequential result). The findings below are robustness/consistency gaps and
test-reliability gaps, not correctness defects.

## Critical Findings

None. No security vulnerability, data-loss risk, or incorrect-output path was
found in the reviewed code.

## High Findings

None. The parity contract, spawn safety, and oversubscription math all hold
under adversarial trace (see Summary).

## Medium Findings

### MD-01: `emotion_worker_count()` — negative/garbage values silently enable the default 3-worker pool (parallel), inconsistent with `emotion_sample_cap()`

**File:** `src/chat_analyzer/cli/nlp_gate.py:128-139`

**Issue:** `CHAT_ANALYZER_EMOTION_WORKERS=-5` → **3** (parallel pool enabled),
`CHAT_ANALYZER_EMOTION_WORKERS=off` / `false` / `abc` → **3**, while `"0"`/`"1"`
→ 1 (sequential). Verified live:
`-5 → 3, off → 3, 0 → 1`, versus the sibling knob `emotion_sample_cap()` where
any ≤ 0 value and "off"/"false" **disable** (`-5 → None, off → None`). A user
who tries the natural "disable" spellings (mirroring the documented sample
knob's semantics) gets, on a >20_000-unique-text chat, 3 spawned processes
each loading a ~255 MB model + torch (~4 GB+ peak) — the exact opposite of
their intent. The 04-07-RESEARCH.md (Q5) itself flagged this: "a negative
value silently enables the default 3 workers … flag for consistency" — the
plan was supposed to resolve it and did not.

**Fix:** Route every parsed value through the same disable semantics as the
sample knob:
```python
if raw in ("0", "1", "off", "false", "no"):
    return 1
try:
    value = int(raw)
except ValueError:
    return 1  # garbage -> sequential, not a silent 3-worker pool
if value < 2:  # "00", "-0", "-5"
    return 1
return max(1, min(value, os.cpu_count() or 1, 8))
```
(Update the docstring and `test_emotion_worker_count_env_contract`
accordingly.)

### MD-02: `_scoring_workers()` accepts uncapped/unvalidated explicit `workers` — direct API callers bypass the nlp_gate RAM cap; garbage raises instead of degrading

**File:** `src/chat_analyzer/analysis/emotion.py:503`

**Issue:** For an explicit `workers` argument, `_scoring_workers` returns
`max(1, int(workers))` with no cap and no type guard:
- `analyze_emotions(df, workers=32)` → 32 spawned processes × ~1.3 GB each —
  the nlp_gate cap (`min(value, cpu, 8)`) exists precisely because of this
  RAM bound, but it is bypassed for the public library API
  (`analyze_emotions` is documented usage in the module docstring).
- `workers="abc"` → `int()` raises `ValueError`, which propagates through
  `_score_unique_texts` and out of `analyze_emotions` — contradicting the
  module's own "never crashes the batch" degrade contract.

The production pipeline is safe (it always passes the nlp_gate-resolved
value), so this is a hardening gap, not a shipped-bug.

**Fix:**
```python
try:
    resolved = int(workers)
except (TypeError, ValueError):
    logger.exception("invalid emotion workers %r; using sequential", workers)
    return 0
return max(1, min(resolved, os.cpu_count() or 1, 8))
```

### MD-03: Test fragility — `test_emotion_worker_count_env_contract` hardcodes `== 4` for env "4", ignoring the cpu cap

**File:** `tests/test_emotion_parallel.py:146`

**Issue:** `assert nlp_gate.emotion_worker_count() == 4` fails on any box with
fewer than 4 logical CPUs (the implementation returns
`min(4, os.cpu_count() or 1, 8)`). The "99" case on line 149 correctly
computes the expected cap; the "4" case does not — a latent CI failure on
2-core runners.

**Fix:**
```python
monkeypatch.setenv(env, "4")
assert nlp_gate.emotion_worker_count() == min(4, os.cpu_count() or 1, 8)
```

## Low Findings

### LW-01: Slow spawn-parity smoke can silently pass without exercising the spawn path (FORCE_NLP bypass + hardcoded cache path)

**File:** `tests/test_emotion_parallel.py:309-315`

**Issue:** The skip gate is `not nlp_gate.nlp_available() or not os.path.isdir(model_cache)`.
`nlp_available()` honors `CHAT_ANALYZER_FORCE_NLP=1` even when transformers is
absent — the test then proceeds, `EmotionAnalyzer()` degrades to
`pipeline=None` (rule-based), `_is_real_pipeline()` returns False, both
"parallel" and "sequential" runs take the rule-based branch, and
`assert_frame_equal` passes **without ever spawning a worker**. Additionally,
the hardcoded `~/.cache/huggingface/hub/...` path ignores `HF_HUB_CACHE` /
`HF_HOME` custom roots (the module's own `model_cached()` uses the canonical
hub constant), causing spurious skips on machines with a relocated cache.

**Fix:** Gate on the same probe the production code uses, and assert the real
pipeline actually loaded:
```python
if not nlp_gate.model_cached(nlp_gate.MODEL_ID):
    pytest.skip("real DistilBERT model cache missing (D-17)")
...
analyzer = EmotionAnalyzer()
if not isinstance(analyzer.pipeline, transformers.Pipeline):
    pytest.skip("real pipeline unavailable — cannot exercise spawn path")
```

### LW-02: Thread-clamp formula (200749f) has no direct test; the exact threshold boundary (20_000) is untested

**File:** `tests/test_emotion_parallel.py:216-239` (chunk test), `:180-210` (boundary tests)

**Issue:** `_inline_fake_worker` accepts and ignores `num_threads`, so nothing
asserts the per-worker budget `max(1, cpu_count // workers)` actually reaches
the worker — the 200749f change is only implicitly covered by the slow smoke.
And the threshold gate is tested at 100 (below) and 20_001 (above), but never
at exactly 20_000, where `len(unique_texts) > _EMOTION_PARALLEL_THRESHOLD` is
False (strict-greater semantics is the contract).

**Fix:** In `test_parallel_chunks_contiguous_fixed_order`, capture the args
the fake pool receives and assert
`num_threads == max(1, (os.cpu_count() or 1) // 3)`; add
`_pending(20_000)` with `workers=2` asserting the driver is not invoked.

## Nits

### NT-01: `TOKENIZERS_PARALLELISM` uses `setdefault` — an inherited "true" wins; the test only source-sniffs

**File:** `src/chat_analyzer/analysis/emotion.py:88`

Spawn children inherit the parent environ; if the parent environment carries
`TOKENIZERS_PARALLELISM=true` (e.g., user/CI export), the setdefault does not
override it and each worker's tokenizers rayon pool oversubscribes the box —
the exact failure 111a6bd was meant to prevent. `setdefault` is defensible
(respecting an explicit user setting), but consider a hard
`os.environ["TOKENIZERS_PARALLELISM"] = "false"`, and note
`test_tokenizers_parallelism_disabled_in_worker` asserts source *order* only
— it cannot catch a regression where the env var is set after the import.

### NT-02: Slow smoke sets `CHAT_ANALYZER_EMOTION_WORKERS=2` but passes `workers=2` explicitly — the env line is dead

**File:** `tests/test_emotion_parallel.py:317`

`analyze_emotions(df, workers=2)` never reads the env knob; the
`monkeypatch.setenv` is inert. Remove it or pass `workers=None` to actually
exercise the env-resolution path.

### NT-03: `threads = max(1, num_threads or ((os.cpu_count() or 1) // 3))` — dead fallback in the pool path

**File:** `src/chat_analyzer/analysis/emotion.py:92`

The parent always sends `num_threads >= 1`, so the `or` fallback only fires on
direct (non-pool) calls. Harmless, but the fallback arithmetic
(`cpu // 3`) is now orphaned documentation — either drop it or keep the
worker's `num_threads` parameter `int` (non-optional) since the pool always
provides it.

## Verdict

**APPROVE WITH CHANGES** — no CRITICAL and no HIGH findings. The parallel
emotion scorer is correctness-sound: byte-parity with the sequential path is
guaranteed by construction (any worker failure degrades to the identical
sequential `_score_batch`), Windows spawn safety holds, the thread-clamp math
is verified, D-17 is respected, and both fast suites plus ruff are green.
The three MEDIUM findings (env-knob disable semantics, uncapped explicit
workers on the public API, and the cpu-count-fragile assertion) should be
addressed before merge; the LOW/NIT items are test-reliability and hardening
improvements that can ride along.

---

_Reviewed: 2026-08-15_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: deep_