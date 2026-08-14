---
phase: 04-nlp-extras-quality-gate
plan: 06
subsystem: nlp
tags: [emotion, sampling, option-c, performance, stratified-sampling, env-config, pandas, determinism]

# Dependency graph
requires:
  - phase: 04-nlp-extras-quality-gate
    provides: 04-02 gated emotion path (distilbert 6-class EmotionAnalyzer, emotion_scored history column, nlp_gate availability flag) and 04-03 always-on interactive menu — the sampled path plugs into this existing analyzer with zero API change
provides:
  - Deterministic stratified emotion sampling for large chats: when a chat exceeds a configurable cap (default 50 000 scorable messages), a representative subset is model-scored and the rest get neutral 1/6, so huge exports finish in bounded time instead of scoring every message
  - CHAT_ANALYZER_EMOTION_SAMPLE env knob (positive int = cap; "0"/"off"/"false" = disable sampling = exact; garbage = default) with a tty prompt for interactive runs and AUTO-SAMPLE for piped/CI runs
  - Honest reporting: the HTML report and terminal summary state "based on a sample of N of M messages" only when a sample actually ran (gate on the sampled flag, not presence of the key)
  - 9 fast mocked tests proving exact-below-cap, exact-at-cap, sampled-above-cap, determinism, sampled-vs-exact tolerance, sender stratification, env parsing, and pipeline gating
affects: [emotion analysis scalability, future large-chat performance plans, QUAL-02 regression suite]

# Tech tracking
tech-stack:
  added: []  # NO new packages — pandas qcut/random_state + stdlib os env parsing only
  patterns:
    - "Largest-remainder seat allocation (Hamilton-style) across senders with floor-1 per active sender, refill round-robin by largest fraction, tie-break by sender name — guarantees every sender with scorable messages appears in the sample"
    - "Stratified time buckets: per-sender pd.qcut over astype(int64) datetimes (min(50, len) buckets), floor-1 per bucket, random_state=42 draws — keeps temporal representativeness"
    - "Exact-path invariance: when sample_cap is None or n_scorable <= cap, the code path is byte-for-byte the legacy loop — no emotion_scored column, no attrs, no sampling imports exercised"

key-files:
  created: [tests/test_emotion_sampling.py, .planning/phases/04-nlp-extras-quality-gate/04-06-SUMMARY.md]
  modified: [src/chat_analyzer/cli/nlp_gate.py, src/chat_analyzer/analysis/emotion.py, src/chat_analyzer/cli/pipeline.py, src/chat_analyzer/cli/adapters.py, src/chat_analyzer/cli/report_html.py, src/chat_analyzer/cli/render.py]

key-decisions:
  - "attrs carry an additive 'sampled': True/False key beyond the plan's three (scored/total/cap): the report and terminal labels gate on emotion.sample.sampled so the failure-degrade path (sampling exception -> exact scoring + sampled=False + note) never mislabels output as sampled"
  - "The tty prompt default is NO (exact) — sampling is opt-in on interactive terminals, auto-enabled only off-tty; matches the locked UX decision and keeps interactive behavior conservative"
  - "No file-hash cache (per plan): sampling is deterministic via random_state=42 rather than memoized, so re-runs over the same file are stable without storing state"

patterns-established:
  - "Sampling cap resolution lives in nlp_gate.emotion_sample_cap() (env parse -> int | None) so the CLI gate and any future callers share one parsing contract"
  - "Sampling metadata rides DataFrame.attrs through the analysis, is copied into the emotion summary dict (emotion_summary['sampled']), then into the AnalysisResults contract (emotion.sample) for render/report — one hop per layer, no globals"

requirements-completed: []

# Metrics
duration: 51min
completed: 2026-08-14
---

# Phase 4 Plan 6: Sampled Emotion Inference for Large Chats (Option C) Summary

**Deterministic stratified emotion sampling capped at 50k messages via `CHAT_ANALYZER_EMOTION_SAMPLE`, with tty prompt / off-tty AUTO-SAMPLE gating, honest "sample of N of M" labels in the HTML report and terminal, and 9 new fast mocked tests — small chats stay byte-for-byte exact.**

## Performance

- **Duration:** 51 min
- **Started:** 2026-08-14T09:25:00Z
- **Completed:** 2026-08-14T10:16:00Z
- **Tasks:** 7
- **Files modified:** 7

## Accomplishments
- Large-chat emotion analysis now bounds work: above the cap, exactly `cap` scorable messages are model-scored (stratified by sender + time, deterministic via `random_state=42`); everything else gets neutral 1/6 — no more full-corpus scoring of 500k-message exports
- Exact-path invariance verified by regression tests: below/at cap output equals the sequential reference byte-for-byte (no `emotion_scored` column, no attrs) — the 04-02 parity suite passes untouched
- `emotion_sample_cap()` env contract: absent/empty→50000, `0`/`off`/`false` (case-insensitive) and any value parsing to a non-positive int (e.g. `00`, `-5`)→`None` (disable), positive int→int, garbage→50000, never raises
- Pipeline gating: interactive tty prompts (default NO = exact, y/yes = sampled); piped/CI/tests AUTO-SAMPLE without asking; sampling failures degrade to exact scoring with a `note` and `sampled=False`
- Representative-sample tolerance proven: on a 240-message fixture, the 60-row sampled summary's per-class means and distribution shares stay within 0.15 of the exact summary

## Task Commits

Each task was committed atomically:

1. **Task 1: Cap resolver (nlp_gate.py)** - `99706b9` (feat)
2. **Task 2: Sampled inference core (emotion.py)** - `1f59982` (perf)
3. **Task 3: Pipeline gating + sampled summary (pipeline.py)** - `bbceff2` (feat)
4. **Task 4: Adapter metadata (adapters.py)** - `0c02c5d` (feat)
5. **Task 5: HTML report sample label (report_html.py)** - `dd91111` (feat)
6. **Task 6: Terminal sample label (render.py)** - `5c887c1` (feat)
7. **Task 7: Sampling test suite (tests/test_emotion_sampling.py)** - `d456d13` (test)

## Files Created/Modified
- `src/chat_analyzer/cli/nlp_gate.py` - `emotion_sample_cap()` env parser (returns int cap or None to disable)
- `src/chat_analyzer/analysis/emotion.py` - `sample_cap` param on `analyze_emotions`; `_allocate_seats` (largest-remainder), `_global_time_edges`, `_pick_in_sender` (per-sender qcut buckets), `_stratified_sample_indices` (boolean mask); `_EMOTION_SAMPLE_RNG = 42`, `_TIME_BUCKETS = 50`; module logger
- `src/chat_analyzer/cli/pipeline.py` - cap resolution before "Analyzing emotions" stage, tty prompt vs off-tty auto-sample, summary over scored rows only + `emotion_summary["sampled"]`
- `src/chat_analyzer/cli/adapters.py` - `_build_emotion_block` adds `"sample": emotion.get("sampled")` to the results contract
- `src/chat_analyzer/cli/report_html.py` - emotion tab renders "Emotion scores based on a sample of N of M messages" when `emotion.sample.sampled`
- `src/chat_analyzer/cli/render.py` - terminal `[INFO] Emotions based on a sample of {scored} of {total} messages.` gated on the sampled flag
- `tests/test_emotion_sampling.py` - 9 fast mocked tests (a–h): exact below/at cap, sampled above cap (exactly cap scored, neutrals 1/6), determinism, sampled-vs-exact tolerance < 0.15, imbalanced-sender representation, env parsing, pipeline non-tty auto-sample + tty y/N/empty, sampling-disabled-via-env

## Decisions Made
- Additive `"sampled"` key in the attrs/summary/contract so labels gate on whether sampling actually ran (honest failure degrade) — see Deviations #1
- Sampling opt-in on tty (default NO), auto on pipes/CI — matches the locked UX decision
- No file-hash cache; determinism from a fixed RNG seed (per plan)
- Env `0`/`off`/`false` disable sampling entirely rather than clamping to a tiny cap

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `set(selected)` on a boolean Series yielded values, not labels**
- **Found during:** Task 2 (sampled inference core)
- **Issue:** `selected_set = set(selected)` iterated the boolean mask's values ({True/False}) instead of the DataFrame index labels, so the row lookup `df.loc[selected_set]` could not find rows — sampling would have crashed or mis-selected
- **Fix:** `selected_set = set(df_copy.index[selected])` — take the index positions the mask marks
- **Files modified:** src/chat_analyzer/analysis/emotion.py
- **Verification:** 47-test emotion/perf/analysis run passed; sampled scored exactly `cap` rows in manual checks
- **Committed in:** 1f59982 (Task 2 commit)

**2. [Rule 2 - Missing Critical] Render/report labels would mislabel failure-degrade output as sampled**
- **Found during:** Tasks 5–6 (report + terminal labels)
- **Issue:** The plan's verbatim `if emotion.sample:` / `if sample:` gates would also fire on the failure-degrade path (sampling exception → exact scoring with `sampled: False`), telling users "based on a sample of N of M" when no sample ran
- **Fix:** Added `"sampled": True/False` to the attrs/summary/contract; render and report gate on `emotion.sample.sampled`; added the `sampled` key to adapters and the pipeline summary
- **Files modified:** src/chat_analyzer/analysis/emotion.py, src/chat_analyzer/cli/pipeline.py, src/chat_analyzer/cli/adapters.py, src/chat_analyzer/cli/report_html.py, src/chat_analyzer/cli/render.py
- **Verification:** pipeline tests assert `sample["sampled"] is True` when sampled and `sample is None` for exact; manual pipeline run showed `{'scored': 10, 'total': 40, 'cap': 10, 'sampled': True}`
- **Committed in:** 1f59982, bbceff2, 0c02c5d, dd91111, 5c887c1 (tasks 2–6)

**3. [Rule 1 - Bug] Stale `# noqa: BLE001` on the sampling-failure except clause**
- **Found during:** Task 2 (ruff check)
- **Issue:** Ruff's BLE001 (default-on in 0.16.1) does NOT fire when an except body's first statement is a `*.exception(...)` logging call; the copied `# noqa: BLE001` was dead code that RUF100 flagged
- **Fix:** Removed the noqa from the `logger.exception(...)` except clause; left the existing `# noqa: BLE001` on `except Exception as e:` lines (those DO fire)
- **Files modified:** src/chat_analyzer/analysis/emotion.py
- **Verification:** `ruff check src/chat_analyzer tests` → "All checks passed!"
- **Committed in:** 1f59982 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 missing critical)
**Impact on plan:** All three were necessary for correctness/honesty; no scope creep. The `sampled` key is additive and backward-compatible (exact path never sets it).

## Issues Encountered
- **Pre-existing Windows cp1252 test failure (out of scope):** `tests/test_phase1_smoke.py::test_console_script_help` fails with `UnicodeDecodeError: 'charmap' codec can't decode byte 0x90` because the test's `run_cli` subprocess decodes `--help` output with the locale encoding, which cannot decode the em-dashes in `main.py`'s help docstring. Verified pre-existing: the em-dashes exist at `e05618a` (before this plan's first commit), `git diff e05618a..HEAD -- main.py test_phase1_smoke.py` is empty, and the test passes under `PYTHONUTF8=1`. Logged to `deferred-items.md` #6 per the scope boundary rule — NOT fixed (no 04-06 file touches the CLI help path). Full-suite result: **250 passed, 1 failed** (that one failure only).

## Stub Scan
- No stubs introduced. The `emotion_sample` attrs dict is always fully populated when sampling runs; the failure-degrade path sets explicit `sampled: False` + `note` rather than empty placeholders.

## Threat Surface Scan
- No new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries. The env knob reads one `CHAT_ANALYZER_EMOTION_SAMPLE` var with a strict whitelist parser (never `eval`/`int` of arbitrary input beyond a positive-int check). No threat flags.

## User Setup Required

None - no external service configuration required. Users may optionally set `CHAT_ANALYZER_EMOTION_SAMPLE` to tune the sampling cap.

## Next Phase Readiness
- Large-chat emotion analysis is bounded and deterministic; exact behavior for small chats is regression-protected
- Follow-up candidates: a file-hash cache (rejected by plan, could revisit as opt-in), tuning the default cap, and the deferred cp1252 smoke-test fix (deferred-items #6)
- Existing deferred items #1–#5 (legacy lint debt, stale artifact, report relocation, WR-03/WR-04) remain open and unchanged

---
*Phase: 04-nlp-extras-quality-gate*
*Completed: 2026-08-14*

## Self-Check: PASSED

All 7 task files + SUMMARY.md verified present on disk; all 7 commit hashes
(99706b9, 1f59982, bbceff2, 0c02c5d, dd91111, 5c887c1, d456d13) verified in
git history. Ruff clean on src + tests; full fast suite 250/251 (the single
failure is the pre-existing cp1252 `test_console_script_help`, deferred-items #6).
