# Phase 4 (04-07): Parallel Exact DistilBERT Emotion Scoring - Research

**Researched:** 2026-08-14
**Domain:** Spawn-safe multiprocessing map-reduce for exact transformers pipeline scoring (Windows, CPU torch)
**Confidence:** HIGH (repo facts verified by code reads + live probes; external facts checked against PyTorch/HF docs)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions (relevant subset, verbatim)
- **D-02:** Silent availability check: transformers/torch importable AND emotion model cached; if present use NLP silently.
- **D-07c:** Emotion model is `bhadresh-savani/distilbert-base-uncased-emotion` (~255 MB, 6 labels: joy/sadness/anger/fear/surprise/love).
- **D-17:** Heavy model load mocked with unittest.mock in tests - suite fast and offline-safe; real-model inference in tests REJECTED (deferred).
- **Feature (AGREED, 04-07):** Parallel exact DistilBERT emotion scoring via multiprocessing map-reduce over partitions, following the VADER pattern (`sentiment.py` `_score_vader_parallel`). Exact path only - sampled path (04-06) unchanged.

### the agent's Discretion
- Worker count default, threshold, chunk partitioning, exact test organization.
</user_constraints>

<phase_requirements>
## Phase Requirements

- **QUAL-02** (real-module tests, fast + offline per D-17): the mocked-scorer spawn test design below keeps D-17.
- **ANAL-06** (6-class emotion classification): this is a perf refactor of the existing pipeline - output parity is the contract.
</phase_requirements>

## Summary

The VADER pattern (`sentiment.py:192-249`) transfers to emotion scoring with three deltas: (1) the worker constructs its own `transformers` pipeline from a model-id string (a pipeline is NOT picklable); (2) each worker must clamp its torch thread pool or N x 10 default threads thrash a 12-thread machine; (3) parsing/normalization must be module-level so parent and workers share one code path.

**HEAD state (committed during this research window):** `EMOTION_LABELS` (emotion.py:61), module-level `_parse_emotion_scores` (emotion.py:281 - documented as "Shared by the parent's per-message/batch paths and the spawn-child worker"), `_score_unique_texts` dedupe seam (emotion.py:434), and the full parallel stack in `96fb271` - see Implementation Status below.

**Primary recommendation:** module-level `_emotion_score_batch(strings, model_id, batch_size, threads, scorer=None)` + `_score_emotion_parallel(...)` mirroring `sentiment.py:192-249`; the pool slots into `_score_unique_texts` (replace its `self._score_batch` call when `emotion_worker_count() >= 2` and `len(unique_texts) > _EMOTION_PARALLEL_THRESHOLD`); `torch.set_num_threads(max(1, cpu_count // workers))` per worker; threshold ~5000 unique texts; any worker failure degrades to the existing sequential `_score_batch`.

## Implementation Status (research-time observation)

The design below was ALREADY implemented and committed during this research window: `96fb271 perf(nlp): parallel emotion workers via process map-reduce (env knob)` (emotion.py + nlp_gate.py; HEAD). Landed code matches this research on: spawn-safe module-level `_score_text_chunk(texts, model_name, batch_size)` (emotion.py:69-115) with pipeline built in-child and 4.x/5.x normalization via the shared `_parse_emotion_scores`; `_score_unique_texts_parallel` driver (emotion.py:499) with ProcessPoolExecutor + `pool.map`; contiguous fixed-order chunks (deterministic output regardless of worker count); failure degrade to sequential (VADER precedent); env knob `emotion_worker_count()` (default 3, "0"/"1"->1, cap `min(value, cpu_count, 8)`, garbage->default); real-pipeline gate (parallel ONLY for `transformers.Pipeline` instances - mocked tests stay sequential by construction, D-17-safe).

**Deltas vs this research - verify/harden in the plan:**
1. **Thread clamp is hardcoded `cpu_count // 3` (emotion.py:87), not scaled to the actual worker count.** With the env knob set to 6 workers, each child still uses 4 threads -> 24 threads on 12 cores -> 2x oversubscription. Recommend the parent compute `max(1, (os.cpu_count() or 1) // workers)` and pass it as an arg (research Q2 formula).
2. **Threshold is 20_000 (VADER's value), not the recommended 5_000** - DistilBERT is ~10x slower per message, so the breakeven is lower; measure with scripts/benchmark.py before tuning down.
3. **No `os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")`** in the worker - tokenizers' rayon pool can oversubscribe per child; cheap hardening.
4. **No `local_files_only=True`** - acceptable per Q3 (cache is provably complete once the parent pipeline exists; default etag checks are read-only).
5. **ZERO test coverage for the parallel branch** - no test references `emotion_worker_count` / `_score_text_chunk` / `_score_unique_texts_parallel` / `_EMOTION_PARALLEL_THRESHOLD`. The real-pipeline gate means mocked tests cannot reach the pool; the spawn path is only exercisable with a real model. Recommended: a `slow`-marked smoke test (marker exists in pyproject.toml) that forces the threshold to 1 with the real cached pipeline and asserts parity vs the sequential reference (mirrors test_perf_parity_sentiment.py:137-151, but skipped when the [nlp] extra / model cache is absent).

## Q1: Windows spawn safety

- `multiprocessing.get_start_method()` is `spawn` on Windows (verified on this box). Children re-import `__main__`; every worker arg is pickled (by reference or value).
- Worker must be a module-level top-level function (pickled by reference, like `_vader_score_batch`); never a method/closure. Pipeline/model must NOT be an argument - pass only picklables (`list[str]`, model-id string, ints).
- Worker builds its own pipeline, imports inside the function:
  ```python
  def _emotion_score_batch(strings, model_id, batch_size=32, threads=None, scorer=None):
      """Spawn-safe worker - builds its OWN pipeline from the model-id string."""
      import os
      os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # before transformers
      import torch
      torch.set_num_threads(threads or max(1, (os.cpu_count() or 1) // 3))
      if scorer is None:  # tests inject a picklable module-level mock
          from transformers import pipeline
          scorer = pipeline("text-classification", model=model_id, top_k=None,
                            device=-1, local_files_only=True)
      return _score_texts_with(scorer, strings, batch_size)  # shared core (Q4)
  ```
- Module top level must stay light: `emotion.py` already imports only numpy/pandas/matplotlib at top (torch/transformers are lazy inside `_initialize_model`) - children re-import cheaply. Keep it so. Copy the 6-line `_in_spawn_child()` helper (sentiment.py:10-16) into emotion.py if any future import-time print/import is added. The `if __name__ == "__main__":` guard is correct.

## Q2: Thread balancing and worker count

- torch defaults to `get_num_threads() == 10` on this box (live probe; os.cpu_count() == 12). N untamed workers -> N x 10 threads on 12 logical cores -> oversubscription thrashing. Clamping is mandatory.
- Per-worker budget (parent computes, passes as arg):
  ```python
  threads = max(1, (os.cpu_count() or 1) // workers)   # 12 // 3 = 4; 12 // 4 = 3
  ```
- Recommended N = **3 (default)**, hard cap `min(N, os.cpu_count())`. ~1-1.3 GB RSS per torch worker (255 MB weights + runtime + OpenMP arenas) plus the parent's pipeline (~1.3 GB, reused for fallback - never build a second parent pipeline) + DataFrame. N=3 ~5-6 GB peak; N=4 ~6.5-7 GB - safe on 15.7 GB; N=8 ~13+ GB - OOM risk. VADER caps at 4 (sentiment.py:194-198, ME-02); default 3 respects that ceiling.
- Realistic speedup: **~1.5-3x wall-clock on >=5000 scorable messages - NOT Nx, never 8x.** The single process already saturates 10 threads; the win is parallelizing Python/tokenization/pipeline overhead, while each worker's throughput drops with thread splitting. Each worker pays ~10-30 s model load + import (sentiment.py:41-42 ME-01) - amortized only above the threshold.

## Q3: Model cache / concurrency

- Model is cached on this box (verified: `~/.cache/huggingface/hub/models--bhadresh-savani--distilbert-base-uncased-emotion` exists). Cached loads are read-only - N processes reading concurrently is safe, no corruption.
- First-load concurrency (fresh install): huggingface_hub writes blobs to temp, atomic-moves, and uses a per-repo file lock, so concurrent `from_pretrained` cannot corrupt the cache - worst case is a duplicated download. [ASSUMED - lock mechanics from knowledge of huggingface_hub source, not re-verified; mitigated by D-02/download_models ensuring presence before the emotion stage]
- Gotcha - per-worker network revalidation: default `from_pretrained` etag-checks the Hub per file. Since the parallel path only runs when `self.pipeline is not None`, the cache is provably complete -> workers may pass `local_files_only=True` (no network, no offline retry stalls). Plan decision: if the parent ever constructs without downloading, revert.
- top_k=None in workers: pass `top_k=None` at construction AND per-call (the landed worker does both, emotion.py:88-95); transformers 5.14.1 returns nested shapes - handled via the shared `_parse_emotion_scores` plus the alignment/unwrap logic the landed `_score_text_chunk` mirrors from `_score_batch`.

## Q4: Degrade + determinism

- Per-message scores are a pure function of (text, model, device, thread-count): identical pipeline kwargs, identical `str(text)[:512]` truncation (emotion.py:372), identical softmax over the 6 labels. Chunk boundaries/order cannot change per-row results; `_score_unique_texts` (emotion.py:434) already dedupes via `dict.fromkeys` - score each unique string exactly once, map back by (idx, text).
- Shared parser already landed: the worker MUST call the module-level `_parse_emotion_scores` (emotion.py:281) per item - it normalizes both 4.x flat and 5.x nested shapes and is explicitly designed for worker reuse. The parent's `_score_batch` (emotion.py:412) alignment/unwrap logic (list vs nested vs whole-batch-wrapped, lines 412-435) is the behavioral contract the worker's chunk loop must mirror:
  ```python
  def _score_texts_with(scorer, texts, batch_size):
      """Shared per-message core: sequential parent AND spawn workers call this,
      so parallel == sequential by construction. Mirrors _score_batch (412-435):
      chunk; scorer(chunk, batch_size=.., top_k=None); align/unwrap 4.x/5.x
      shapes; normalize via _parse_emotion_scores. (If the worker keeps its own
      loop instead, it must reproduce exactly this alignment logic.)"""
  ```
- Parity tests assert exact equality on mocks (mirror test_perf_parity_sentiment.py:137-151, rtol=1e-9/atol=1e-12). Real-model runs may show last-bit float noise (<1e-6, thread-dependent reductions) - document a tolerance.
- Failure degrade: wrap `pool.map` in try/except Exception -> `logger.exception(...)` (BLE001-safe pattern, sentiment.py:242-245) -> score the whole unique-text list through the parent's sequential `self._score_batch`. Never crash the batch; per-row output identical to today's exact path.
- Spawn re-import: children get fresh globals (`_emotion_analyzer`/`_emotion_model_loaded` None/False) - the worker MUST NOT touch them (build a local pipeline only, same contract as `_vader_score_batch`'s docstring, sentiment.py:202-203). No heavy work at module import (Q1).
- Driver mirrors `_score_vader_parallel` (210-249): dedupe is already done by the caller - the pool scores the unique-text list partitioned into `workers` chunks -> `ProcessPoolExecutor(max_workers=workers)` + `pool.map(partial(_emotion_score_batch, model_id=..., batch_size=..., threads=...), chunks, chunksize=1)` -> rebuild `{text: scores}` -> caller maps rows in original order. `_EMOTION_PARALLEL_THRESHOLD = 5000` (VADER uses 20k; DistilBERT is ~10x slower per message, lower breakeven; 04-06 caps work at 50k where the win concentrates).

## Q5: Env knob

`CHAT_ANALYZER_EMOTION_WORKERS` - committed `nlp_gate.emotion_worker_count()` (96fb271, mirroring `emotion_sample_cap()` style) implements the contract. Returns int >= 1 (1 = sequential); never raises:

- absent/empty -> `EMOTION_WORKERS_DEFAULT` (3)
- "0"/"1" -> 1 (sequential, no pool; "1" buys nothing but spawn overhead)
- positive int -> `max(1, min(value, os.cpu_count() or 1, 8))` - capped at 8 (RAM-bound: 8 x ~1.3 GB workers + parent ~2-3 GB ~ 13 GB on a 15.7 GB box)
- any other string (garbage, "off", "false") or a parsed int < 2 (e.g. "00", "-5") -> `EMOTION_WORKERS_DEFAULT` (3)
- callers parallelize only when the resolved value is >= 2

Deltas vs this research's draft semantics - verify intent during the plan:
- **Cap 8 vs cpu_count:** the user spec said `min(value, cpu_count)`; the landed code adds the tighter RAM cap of 8. N=8 approaches the 15.7 GB limit - acceptable as a user-forced ceiling, default stays 3.
- **Parsed int < 2 -> default 3, not 1:** `emotion_sample_cap()` maps any <= 0 value to "disabled"; here a negative value silently enables the default 3 workers. Harmless (never raises) but flag for consistency.
- **"off"/"false" not special-cased** (falls to default 3, where `emotion_sample_cap` disables). Acceptable; document in the README.

Caller: `workers = nlp_gate.emotion_worker_count(); if workers > 1 and len(unique_texts) > _EMOTION_PARALLEL_THRESHOLD: parallel else: self._score_batch(...)` (inside `_score_unique_texts`, emotion.py:434).

## Don't Hand-Roll

- **Map-reduce:** use `ProcessPoolExecutor` + `pool.map` (audited VADER pattern), not raw `Process`/queues.
- **Thread control:** `torch.set_num_threads` per worker (official API), not custom schedulers.
- **Parse/unwrap:** share `_score_texts_with` core, don't re-derive 4.x/5.x shape logic.
- **Env parsing:** `nlp_gate.emotion_worker_count()`, not inline `int(os.getenv(...))`.
- No new packages: stdlib `concurrent.futures` + existing torch/transformers. Package legitimacy audit: N/A (zero new installs).

## Common Pitfalls

- **Oversubscription:** forgetting `set_num_threads` -> N x 10 threads on 12 cores. Set it FIRST in the worker.
- **Pickling the pipeline:** passing `self.pipeline` or a MagicMock as an arg (not picklable). Worker takes only strings/ints; tests inject a **module-level function** mock (`_classifier`, test_perf_parity_emotion.py:64 - picklable by reference; `tests/` is a package with `__init__.py`, spawn children can import it). `functools.partial` of the worker with picklable args is fine.
- **Worker touching module globals:** `_emotion_analyzer` is None in children - never read/write it.
- **First-load stampede:** N+1 processes downloading 255 MB. Safe (locks/atomic moves) but slow; `local_files_only=True` in workers avoids it once the parent's pipeline exists.

## Environment Availability

torch 2.13.0+cpu + transformers 5.14.1 (within pin >=4.30,<5.15) in `.venv` - verified. Model cached - verified. `multiprocessing` spawn default on Windows - verified. `scripts/benchmark.py` harness available for perf validation of the speedup claim. No missing dependencies.

## Security Domain

No new trust boundary: only input is the locked model-id constant and the whitelist-parsed env knob (never eval'd - mirrors 04-06 threat scan). No network/auth/file-access changes; chat text flows to the model exactly as today.

## Assumptions Log

| # | Claim | Risk if Wrong |
|---|-------|---------------|
| A1 | torch CPU inference bit-deterministic per (input, model, threads) | Last-bit float noise only; parity test uses mocks (exact) |
| A2 | huggingface_hub first-load concurrency lock-safe | Wasted duplicate download, no corruption; model already cached here |
| A3 | ~1-1.3 GB RSS per torch worker | N=3 leaves ~2.5x headroom; N=4 safe on 15.7 GB |
| A4 | Expected 1.5-3x speedup above ~5k messages | Measure via scripts/benchmark.py in the plan before locking threshold |

## Sources

- [VERIFIED: live probe] torch 2.13.0+cpu / transformers 5.14.1 / start=spawn / get_num_threads()=10 / cpu_count()=12 / model cache dir / tests `__init__.py` / `slow` marker
- [VERIFIED: repo read] sentiment.py 10-16, 30-51, 192-249; emotion.py HEAD 96fb271: `_score_text_chunk` (69-115), `_parse_emotion_scores` (281), `_score_unique_texts` (434), `_score_unique_texts_parallel` (499-520); nlp_gate.py `emotion_sample_cap` (74-101) + `emotion_worker_count`; test_perf_parity_sentiment.py 137-151; test_perf_parity_emotion.py 60-96; zero parallel-branch test coverage (grep, session-end)
- [VERIFIED: PyTorch docs] torch.set_num_threads - intraop parallelism; must precede eager/JIT/autograd - https://docs.pytorch.org/docs/2.13/generated/torch.set_num_threads.html
- [CITED: HF docs] Cache blobs/snapshots structure, Windows symlink caveat - https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache

## Metadata

- Standard stack: HIGH - no new packages; stdlib multiprocessing + existing torch/transformers
- Architecture: HIGH - direct transfer of the audited VADER pattern
- Pitfalls: HIGH - repo-verified; external claims flagged in Assumptions Log
- **Valid until:** 2026-09-14 (pins locked in nlp_gate.py)