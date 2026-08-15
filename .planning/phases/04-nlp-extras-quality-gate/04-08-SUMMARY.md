---
phase: 04-nlp-extras-quality-gate
plan: 08
subsystem: nlp
tags: [result-cache, sha256, opt-in, env-config, performance, privacy, json-only, ttl, determinism]

# Dependency graph
requires:
  - phase: 04-nlp-extras-quality-gate
    provides: 04-02 gated emotion path (distilbert 6-class EmotionAnalyzer, nlp_gate availability flag), 04-06 Option C sample-cap resolution (emotion_sample_cap + tty y/N prompt, moved up inside run_pipeline), and 04-07 emotion_worker_count() — the effective sample cap and worker count fold into the cache key
provides:
  - Opt-in result cache keyed by sha256 of the input file bytes + a config signature: CHAT_ANALYZER_RESULT_CACHE env var (default OFF); on a repeat run of the same file with the same settings run_pipeline prints "[INFO] Loaded analysis from cache" and returns the stored AnalysisResults payload — the compute/NLP/narrative stages never re-run
  - NEW stdlib-only src/chat_analyzer/cli/result_cache.py: sha256_file (1 MiB chunked), cache_key (file hash + {schema, app_version, nlp_on, effective sample cap, emotion_workers, sorted chosen transcripts}), recursive _sanitize (numpy -> .item()/tolist, NaN/Inf -> None, non-serializable dict keys -> str), load (json-only, self-heals corrupt files, schema-check, report_path forced ""), store (atomic tmp + os.replace, prune-on-store, never raises), _prune (30-day mtime TTL)
  - Cache dir OUTSIDE the repo (privacy lock): %LOCALAPPDATA%\chat-analyzer\cache on Windows, ~/.cache/chat-analyzer elsewhere, or any explicit path value; no cache dir is ever created while the feature is disabled
  - zip transcript selection surfaced to the pipeline (parse_zip_with_report now returns a 4-tuple with chosen_names) so the key distinguishes transcript picks; benchmark.py hard-pins the cache off for honest first-run timings
  - 15 fast mocked tests (a-o) driving the REAL run_pipeline with mocked model callables (D-17, QUAL-02): hit/miss, key invalidation (file bytes/config/tier/tty y-N/zip selection), corruption + pickle -> miss-not-crash, schema mismatch, sanitizer round-trip, TTL prune, report_path reset, default OFF
affects: [large-chat repeat-run iteration time, personal derived data at rest (outside the repo), future AnalysisResults contract changes (must bump RESULT_CACHE_SCHEMA), benchmark timing honesty]

# Tech tracking
tech-stack:
  added: []  # NO new packages — stdlib hashlib/json/os/tempfile/time/datetime/importlib.metadata/pathlib only (04-07 zero-new-packages precedent)
  patterns:
    - "Whole-input-bytes key: sha256_file reads 1 MiB chunks (mtime/size never touched — a `touch` still hits); zip inputs hash the WHOLE archive bytes so media-member changes invalidate by construction (research Q1.2)"
    - "Config signature folded into the key with json.dumps(sort_keys=True): {schema, app_version, nlp_on, effective sample cap (tty y/N collapsed in), emotion_workers, sorted chosen_transcripts} — any config change = miss = full re-analysis, never a stale serve"
    - "Central recursive sanitizer at the store boundary: numpy scalar -> .item(), ndarray -> tolist(), NaN/+-Inf -> None (chart_json _float_or_none precedent), and non-serializable dict keys (verified leak: sentiment.daily_avg keyed by datetime.date) -> str() — one walker that cannot miss a field added later (research A8)"
    - "Miss-on-any-failure load: json.load only (never pickle/eval — T-04-20); corrupt file unlinked (self-heal), schema-mismatch file KEPT (newer-format stays), any failure returns None, never raises"
    - "Atomic store: same-dir tempfile + os.replace (atomic on the same volume, Windows-safe; last-write-wins for concurrent same-file runs, Q5.6); best-effort try/except with logger.exception first (BLE001-safe)"

key-files:
  created: [src/chat_analyzer/cli/result_cache.py, tests/test_result_cache.py, .planning/phases/04-nlp-extras-quality-gate/04-08-SUMMARY.md]
  modified: [src/chat_analyzer/cli/nlp_gate.py, src/chat_analyzer/cli/pipeline.py, src/chat_analyzer/cli/zip_input.py, scripts/benchmark.py, README.md]

key-decisions:
  - "Default OFF (opt-in) via CHAT_ANALYZER_RESULT_CACHE — matches the conservative tty-default-NO UX and the 04-06 decision record ('a file-hash cache (rejected by plan, could revisit as opt-in)'); with the var unset, behavior is byte-identical to today and no cache dir is created anywhere"
  - "The EFFECTIVE sample cap (the tty y/N answer collapsed in) rides in the key, and the sample prompt is re-asked on hits — y and N on the same file produce two distinct entries (research A1/Q1.4)"
  - "report_path forced to '' on load — a stale cwd path is never served; main.py always fills it fresh after write_report (D-09 unchanged)"
  - "json.load only, hex-only key filenames (no user input in paths -> traversal impossible); corrupt entries degrade to a miss, never a crash; 30-day mtime TTL pruned on every store"
  - "Cache dir OUTSIDE the repo by construction (privacy lock — DEFERRED.md 'no personal data in the repo'); the payload carries derived data (sender names, top words, charts) but never raw message text; README documents cache-directory deletion for full erase"

patterns-established:
  - "The env-parsing contract lives in nlp_gate (result_cache_enabled/result_cache_dir, mirroring emotion_sample_cap/emotion_worker_count style) — one resolver for the pipeline, never raises, side-effect-free (dir creation belongs to result_cache.store)"
  - "The cache hook lives INSIDE run_pipeline after parse + after the Option C sample decision, before 'Computing insights' — the key needs the effective cap (needs the parse + prompt), and the check must precede the ~90-110 s compute stage (Q6.2); the finally: progress.stop() already handles the early-return-on-hit path"
  - "pipeline owns narration (render.py:6-8 single-source rule): the '[INFO] Loaded analysis from cache' label and the tip hint are printed by run_pipeline, never by the cache module"

requirements-completed: [QUAL-02]

# Metrics
duration: ~50min (single session, 2026-08-15)
completed: 2026-08-15
---

# Phase 4 Plan 8: Result Cache Keyed by File Hash Summary

**Opt-in repeat-run result cache keyed by sha256 of the input file bytes + a config signature (effective sample cap, tier, workers, zip transcript picks all fold in) — stdlib-only `result_cache.py`, cache dir OUTSIDE the repo (`%LOCALAPPDATA%\chat-analyzer\cache` / `~/.cache/chat-analyzer`), json-only payload with a recursive numpy/date-key sanitizer, 30-day TTL prune, corruption → miss-never-crash, and 15 fast mocked tests driving the REAL `run_pipeline` — default OFF so a clone with the var unset sees zero behavior change and zero data at rest.**

## Performance

- **Duration:** ~50 min
- **Started:** 2026-08-15 (immediately after 04-07 completed)
- **Completed:** 2026-08-15
- **Tasks:** 6 (feature/test/docs) + 1 auto-fixed sanitizer bug (see Deviations)
- **Files modified:** 7 (5 modified, 2 created) + README

## Accomplishments
- A repeat run of the same chat file with `CHAT_ANALYZER_RESULT_CACHE` set now completes in seconds: `run_pipeline` resolves the effective sample cap, computes the cache key (sha256 of the whole input bytes + `{schema, app_version, nlp_on, sample_cap, emotion_workers, chosen_transcripts}`), and on a hit prints `[INFO] Loaded analysis from cache` and returns the stored `AnalysisResults` payload — the ~90-110 s compute stage and the NLP/narrative stages never re-run (the parse narration and the sample prompt still run; the `finally: progress.stop()` handles the early return)
- The payload is the ENTIRE post-`adapt()` contract (charts are base64 PNG strings, `charts_json` is JSON-safe by construction) — a hit reconstructs every tab/chart/insight with zero NLP re-runs; the one hard gap, numpy scalars leaking into the contract (`stats.peak_hour` int64, `stats.avg_response_time` float64, emotion distribution counts, sentiment means), is closed by the recursive `_sanitize`
- Verified leak found and fixed during Task 5: `sentiment.daily_avg` is keyed by `datetime.date` — the sanitizer now converts non-serializable dict keys to `str()` (the same normalization json applies to int keys); `json.dump` on the real `adapt()` payload previously raised `TypeError` (see Deviations #1)
- `nlp_gate` parsers mirror the `emotion_sample_cap` style exactly: `result_cache_enabled()` (whitelist, never raises: absent/empty/off-words → False, else True) and `result_cache_dir()` (None when disabled; on-words → `%LOCALAPPDATA%\chat-analyzer\cache` on nt with `~/.cache/chat-analyzer` fallback, `~/.cache/chat-analyzer` elsewhere; any other value IS the dir; side-effect-free — no mkdir)
- Cache correctness is conservative by construction: file edits (any byte), tier changes, sample-cap choices (tty y/N collapse into the effective cap), worker-count changes, zip transcript selections, app upgrades, and schema bumps all invalidate the key → miss → full re-analysis; `touch` (mtime-only) still hits
- Corruption/schema-mismatch/expired entries degrade to a miss, never a crash: `load()` is json-only (T-04-20), self-heals corrupt files (unlinked), keeps schema-mismatch files (newer-format stays), and forces `report_path=""` on every successful load; `store()` is atomic (same-dir tmp + `os.replace`), prunes entries older than 30 days on every store, and never raises into the pipeline
- `parse_zip_with_report` now returns a 4-tuple `(rows, counts, source, chosen_names)` — the sorted chosen transcript member names ride in the cache key, so picking different transcripts on the same zip produces a distinct entry; non-zip paths pass `[]`
- `scripts/benchmark.py` hard-pins `CHAT_ANALYZER_RESULT_CACHE=0` in the worker env (Q6.5/Pitfall 7) so a user with the var set globally never gets cache-hit seconds in the README timing table
- 15 fast mocked tests (research Q7 a-o) drive the REAL `run_pipeline` with `nlp_enabled=True` on a generated WhatsApp fixture with all model callables mocked (D-17): default OFF, env parser (+ privacy-lock assertion that the default dir is never under the repo), miss-stores, hit-skips-NLP (mock call counter unchanged), Loaded label, file-edit/config/tier/sample-choice invalidation, corrupt text + pickle payload miss-not-crash (json-only proof), schema-mismatch miss (file kept), sanitizer round-trip (numpy + date keys), 30-day TTL prune, zip selection in key (mtime-only → same key), report_path reset on load

## Task Commits

Each task was committed atomically (chronological order):

1. **Task 1: Env parsers + schema constant (nlp_gate.py)** - `1666ad3` (feat)
2. **Task 2: File-hash result cache store (result_cache.py, NEW)** - `663d534` (feat)
3. **Task 3: Cache lookup inside run_pipeline after sample resolution (pipeline.py, zip_input.py)** - `7585d8a` (feat)
4. **Task 4: Force result cache off for first-run timings (scripts/benchmark.py)** - `dffae77` (chore)
5. **Task 5: Result cache hit/miss/key/corruption/TTL suite (tests/test_result_cache.py, NEW)** - `ada2cf6` (test)
6. **Sanitizer fix: non-serializable dict keys (result_cache.py)** - `63e499a` (fix) — see Deviations #1
7. **Task 6: README env var + privacy note; 04-08 SUMMARY (README.md, 04-08-SUMMARY.md)** - *(this commit)* (docs)

## Files Created/Modified
- `src/chat_analyzer/cli/nlp_gate.py` - `_RESULT_CACHE_ENV`, `RESULT_CACHE_SCHEMA = 1` (public; bump on contract/key-composition changes), `result_cache_enabled()` (whitelist, never raises), `result_cache_dir()` (on-words → OS default dir, path values verbatim, None when disabled, side-effect-free)
- `src/chat_analyzer/cli/result_cache.py` - NEW stdlib-only module: `RESULT_CACHE_TTL_DAYS = 30`, `sha256_file` (1 MiB chunked), `_app_version` (guarded `importlib.metadata`), `cache_key` (file hash + sorted signature, hex-only filename), `_sanitize` (recursive: numpy → `.item()`/tolist, NaN/Inf → None, dict keys → str; guarded numpy import), `load` (json-only, self-heal, schema-check, `report_path` → ""), `store` (atomic tmp + `os.replace`, prune-on-store, never raises), `_prune` (30-day mtime TTL, best-effort)
- `src/chat_analyzer/cli/pipeline.py` - `chosen_names` captured at parse (txt/json → `[]`, zip → 4th tuple element); Option C sample-decision block MOVED up to right after the df build (identical logic/prompt, same `if nlp_on:` guard, `sample_cap` resolved once); cache lookup after sample resolution (`cache_dir = nlp_gate.result_cache_dir()`, lazy `result_cache` import, hit → `[INFO] Loaded analysis from cache` + early return); store after `adapt()` on a miss; tip hint on the NLP path when the cache is disabled; stage labels and order unchanged
- `src/chat_analyzer/cli/zip_input.py` - `parse_zip_with_report` returns `(rows, counts, source, chosen_names)` with the sorted chosen transcript member names; docstring updated
- `scripts/benchmark.py` - `env["CHAT_ANALYZER_RESULT_CACHE"] = "0"` in `_spawn_and_run` (after the `CHAT_ANALYZER_NO_OPEN` line)
- `tests/test_result_cache.py` - NEW: 15 fast mocked tests (a-o) with `_mocked_models`/`_CountingClassifier`/`_big_whatsapp_file`/`_non_tty_console` helpers, `MPLBACKEND=Agg` headless-first, per-test cache-dir isolation via `monkeypatch.setenv("CHAT_ANALYZER_RESULT_CACHE", str(tmp_path))` (zero repo pollution)
- `README.md` - `CHAT_ANALYZER_RESULT_CACHE` env-var bullet (default OFF, keyword/path semantics, 30-day TTL, Loaded label, fresh report) appended AFTER the uncommitted `CHAT_ANALYZER_EMOTION_WORKERS` bullet (preserved verbatim); Privacy section sentence (cache stored outside the repo, delete the dir to erase) + dev-staleness note (static dev version → delete the cache dir after code changes; the schema constant is the manual override)

## Decisions Made
- Default OFF (opt-in) — matches the locked conservative UX and the 04-06 "could revisit as opt-in" decision record; unset → byte-identical behavior, no cache dir created anywhere
- Effective sample cap in the key (tty y/N collapsed in) + re-ask the prompt on hits (research A1)
- `report_path` blanked on load; main.py fills it fresh after `write_report` (D-09 unchanged)
- json-only payload, hex-only key filenames, corruption → miss-never-crash, corrupt file unlinked, schema-mismatch file kept, 30-day TTL prune-on-store
- Cache dir outside the repo by construction (privacy lock); README documents cache-directory deletion
- `emotion_worker_count()` in the key (parity-identical output but a config change SHOULD invalidate — research A3, conservative)
- Benchmark env pin `CHAT_ANALYZER_RESULT_CACHE=0` so first-run timings stay honest (Pitfall 7)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `_sanitize` missed non-serializable dict KEYS — `json.dump` raised TypeError on the real payload**
- **Found during:** Task 5 verification (test_miss_runs_and_stores)
- **Issue:** The plan's sanitizer spec covered numpy scalars/arrays and NaN/Inf values, but the real `adapt()` payload carries `sentiment.daily_avg` keyed by `datetime.date` (research Q2.3 flagged the field's numpy floats but not its date keys) — `json.dump` raised `TypeError: keys must be str, int, float, bool or None, not date`, so `store()` logged and swallowed every write (the cache never persisted)
- **Fix:** `_sanitize`'s dict branch now normalizes keys: numpy scalars → `.item()`, anything else not JSON-serializable → `str()` (the same normalization json applies to int keys); the central walker covers any future key type too (research A8)
- **Files modified:** src/chat_analyzer/cli/result_cache.py
- **Verification:** all 15 cache tests green, including the extended `test_sanitizer_roundtrip` (date-key case) and `test_hit_skips_nlp_stages` (deep-equal through `_sanitize`)
- **Committed in:** `63e499a`

**2. [Test-spec adjustment - config-change test] `CHAT_ANALYZER_EMOTION_SAMPLE=0` vs unset collapse to the same effective cap below the 50000 default**
- **Found during:** Task 5 (test g as literally specified)
- **Issue:** On the 40-message fixture, env `0` (cap None) and env unset (cap 50000, 40 < 50000 → not over) both resolve to effective `sample_cap=None` — the plan's literal pair would produce IDENTICAL keys, failing the "distinct keys" assertion without a 50k-message fixture (too slow for a fast test)
- **Fix:** test g uses `CHAT_ANALYZER_EMOTION_SAMPLE=5` (over-cap → AUTO-SAMPLE, effective cap 5) vs unset (exact) — the same env-driven cap-resolution code path under test, genuinely distinct keys, fast; commented in the test
- **Files modified:** tests/test_result_cache.py
- **Committed in:** `ada2cf6`

**3. [Test-spec adjustment - corruption test] The corrupt file is deleted by `load()` but re-created by the miss `store()`**
- **Found during:** Task 5 (test j)
- **Issue:** After a corrupt-load miss, the pipeline runs fully and `store()` writes a fresh VALID entry at the same key path — `assert not target.exists()` was wrong (the file exists again, with valid content)
- **Fix:** test j now proves the contract honestly: a unit-level `load()` on garbage returns None AND deletes the file (self-heal), then the end-to-end runs assert the corrupt bytes never surface (the file is valid JSON with the current schema after the miss store); the pickle payload variant proves json-only (inert data → miss, replaced by a valid entry)
- **Files modified:** tests/test_result_cache.py
- **Committed in:** `ada2cf6`

**4. [Test-spec adjustment - hit deep-equality] Cached payload equals the first run modulo JSON normalization**
- **Found during:** Task 5 (test d)
- **Issue:** The plan's "output deep-equal to the first run" is not literally achievable: the sanitizer necessarily normalizes numpy scalars to plain numbers and `datetime.date` keys to str, so `second == first` (raw) cannot hold on the daily_avg field
- **Fix:** test d asserts `second == result_cache._sanitize(first)` — the cached payload equals the first run through the JSON normalization every cache must apply; the call-counter assertion (NLP stages never re-ran) is unchanged and is the primary correctness proof
- **Files modified:** tests/test_result_cache.py
- **Committed in:** `ada2cf6`

---

**Total deviations:** 1 auto-fixed bug (Rule 1) + 3 test-spec adjustments that made the plan's tests honest/implementable without changing any plan behavior.
**Impact on plan:** The sanitizer fix was REQUIRED for the feature to work at all (the store silently failed on every real payload); the test adjustments preserve each test's intent (config invalidation, corruption self-heal, hit equality) without scope creep. No new packages; no parser/analysis/adapters/render/report_html changes.

## Issues Encountered
- The Task 5 suite initially failed 7/15 because of the sanitizer date-key bug (Deviations #1) — the only real bug found; fixed in `63e499a`
- No pre-existing failures: the full fast suite is **302 passed, 26 deselected, 0 failed** — the 04-06-era cp1252 failure (deferred-items #6) stayed resolved (86faefe), and the Task 3 relocation broke nothing (stage narration + Option C sampling suites green)

## Stub Scan
- No stubs introduced. `load()` returns the stored results dict or None (miss → the full pipeline runs); `store()` writes the complete envelope or logs-and-continues; `_sanitize` leaves every pass-through value intact (nothing is replaced with empty placeholders).

## Threat Surface Scan
- New local file access patterns at a trust boundary (the cache dir), all mitigated per the plan's threat register: T-04-20 (Tampering) — `json.load` only, never pickle/eval (proven by test j's pickle payload), hex-only key filenames (no path traversal), corrupt files unlinked; T-04-22 (Information disclosure) — cache dir OUTSIDE the repo by construction (asserted in test b), opt-in default OFF, README documents deletion; T-04-23 (Tampering) — env-driven dir values are used verbatim (documented), side-effect-free parser, hex filenames confine writes; T-04-24 (DoS) — 30-day TTL pruned on every store, best-effort. T-04-21 (Spoofing) accepted per plan (user-owned dir, opt-in default). No new network endpoints or auth paths. No threat flags beyond the plan's register.

## User Setup Required

None - no external service configuration. Users may optionally set `CHAT_ANALYZER_RESULT_CACHE` (`1`/`on`/`true`/`yes` for the default directory, or any other value as an explicit cache directory; `0`/`off`/`false`/`no` disables). On a repeat run of the same file with the same settings the terminal prints `[INFO] Loaded analysis from cache` and the run completes in seconds; the HTML report is always regenerated fresh in the cwd. To erase all stored analysis data, delete the cache directory (`%LOCALAPPDATA%\chat-analyzer\cache` on Windows, `~/.cache/chat-analyzer` elsewhere, or the explicit path chosen).

## Next Phase Readiness
- Repeat runs of large chats are now seconds with the opt-in cache; the cache is default-OFF, json-only, privacy-locked outside the repo, and invalidation-correct across file bytes/config/tier/sample/workers/zip picks
- A future phase that changes the AnalysisResults contract shape or the key composition MUST bump `RESULT_CACHE_SCHEMA` (documented on the constant and in README)
- Deferred items #1-#5 remain open and unchanged; #6 (cp1252) stays resolved

---
*Phase: 04-nlp-extras-quality-gate*
*Completed: 2026-08-15*

## Self-Check: PASSED

All 6 task files + SUMMARY.md verified present on disk; all commit hashes
(1666ad3, 663d534, 7585d8a, dffae77, ada2cf6, 63e499a, plus the Task 6 docs
commit) verified in git history. Ruff clean on all touched files. Verification
battery: 15/15 cache tests, 28 passed (phase4+sampling+gate), 22 passed
(04-07 suites), full fast suite 302 passed / 26 deselected / 0 failed, and
`git status` clean of cache artifacts (per-test monkeypatch dir isolation).