---
phase: 04-nlp-extras-quality-gate
plan: 02
subsystem: cli-pipeline
tags: [nlp-gate, transformers, emotion, summarization, rich-progress, jinja2, huggingface]

# Dependency graph
requires:
  - phase: 04-nlp-extras-quality-gate
    provides: 04-01 always-on health/network blocks, keyword-only adapt() contract slots (emotion/summary=None), cwd report location
provides:
  - nlp_gate.py silent availability probe (transformers+torch importable AND emotion model cached) with locked model constants and CHAT_ANALYZER_FORCE_NLP env override
  - Fixed 6-class emotion path: flat list-of-dicts pipeline parse (Pitfall 1 [0] bug), locked bhadresh-savani/distilbert-base-uncased-emotion default (D-07c), figure-returning emotion_figure
  - Gated emotion + summary pipeline stages (silent degrade to None, announce-before-download, try/except → never crash)
  - Emotion + Summary HTML report tabs with pip-install unavailable note (Jinja autoescape, no |safe)
  - D-12 live determinate progress bar on tty, plain [OK] stage lines off-tty
affects: [04-03-interactive-menu (extends nlp_gate), 04-04-legacy-tests, 04-05-quality-gate, tests/test_phase2_pipeline.py 7-key set]

# Tech tracking
tech-stack:
  added: [sentencepiece (nlp extra), rich Progress (existing dep, new usage), huggingface_hub cache probe]
  patterns:
    - "Silent availability probe: never raises, never prompts; env override for determinism (Pitfall 5)"
    - "Pattern 3 scalar-only adapter extraction for gated NLP blocks"
    - "Figure-returning chart helper (Pattern 2) for base64 embedding"
    - "tty branch narration with shared stage() helper falling back to stage_status"

key-files:
  created: [src/chat_analyzer/cli/nlp_gate.py, tests/test_phase4_nlp.py]
  modified: [src/chat_analyzer/analysis/emotion.py, src/chat_analyzer/cli/pipeline.py, src/chat_analyzer/cli/adapters.py, src/chat_analyzer/cli/contracts.py, src/chat_analyzer/cli/report_html.py, tests/test_phase2_pipeline.py, tests/test_phase1_smoke.py, pyproject.toml]

key-decisions:
  - "Silent degrade is the gate contract (D-02/D-06): the pipeline never prompts or hints — the interactive download menu + positional hint are main.py's job in 04-03"
  - "nlp_gate uses HF cache existence (huggingface_hub HF_HUB_CACHE, fallback ~/.cache/huggingface/hub) — no import-time model construction; model name+size announced before from_pretrained (D-05/Pitfall 4)"
  - "CHAT_ANALYZER_FORCE_NLP=0/1 env override makes the gate deterministic in tests — the dev machine has transformers but no cached emotion model (Pitfall 5)"
  - "Dominant emotion is derived in the adapter (argmax of distribution) — get_emotion_summary has no dominant key"

patterns-established:
  - "Gated heavy stages: probe first → announce → construct lazily inside try/except → degrade to None, never crash (T-04-05/T-04-06)"
  - "One stage() helper for tty/off-tty narration so labels stay verbatim ('Parsing chat', 'Computing insights', 'Analyzing emotions', 'Summarizing conversation')"

requirements-completed: [ANAL-06, ANAL-08, QUAL-02]

# Metrics
duration: 45min
completed: 2026-08-04
---

# Phase 4 Plan 2: Gated NLP Emotion + Summary Slice Summary

**6-class emotion pipeline parse bug fixed (Pitfall 1 `[0]` indexing), silent availability-gate module (`nlp_gate`) with locked D-07c model, gated emotion+summary pipeline stages that degrade silently, Emotion+Summary report tabs, and a D-12 live progress bar that degrades to plain `[OK]` lines off-tty**

## Performance

- **Duration:** 45 min
- **Started:** 2026-08-04T07:35:05Z
- **Completed:** 2026-08-04T08:20:00Z
- **Tasks:** 3 (TDD: test → feat → feat)
- **Files modified:** 8 (plus 1 created test file, 1 created module)

## Accomplishments
- `nlp_gate.py`: silent availability probe (`transformers`+`torch` importable AND emotion model cached in the HF hub cache) + locked model constants (`bhadresh-savani/distilbert-base-uncased-emotion` ~255MB, `t5-small` ~231MB) + `CHAT_ANALYZER_FORCE_NLP` env override for deterministic test branching (Pitfall 5)
- Fixed the CRITICAL emotion parse bug: transformers 4.x `top_k=None` returns a FLAT list of `{"label","score"}` dicts; the old `self.pipeline(text[:512])[0]` grabbed one dict, iterated its keys, raised, and silently degraded every message to uniform 1/6 scores. Now consumes the whole list; regression-trapped by a faithful mock asserting non-uniform scores (T-04-08)
- Emotion + Summary pipeline stages run ONLY when the gate passes; both wrapped in try/except → degrade to None, never crash (T-04-05, Pitfall 7); model name + size announced via console.print BEFORE construction (T-04-06, Pitfall 4), outside the redirect capture so piped output sees it (Pitfall 8)
- Report gains Emotion (distribution table + base64 bar chart + dominant label) and Summary (text + message count) tabs; unavailable state shows `pip install chat-analyzer-pro[nlp]` — both rendered through the existing autoescape Jinja env, no `|safe` on model output (T-04-07)
- D-12 real-time determinate progress bar on a real terminal (`rich.progress.Progress`, one advance per stage, total matches gated stages); off-tty degrades to the unchanged `[OK] <label>...` stage lines via a shared `stage()` helper (Pitfall 8)

## Task Commits

Each task was committed atomically:

1. **Task 1: TDD RED — test_phase4_nlp.py (3 tests, faithful list-of-dicts mocks)** - `901d867` (test)
2. **Task 2: GREEN — nlp_gate + emotion fix + gated stages + tabs** - `4a2c142` (feat)
3. **Task 3: D-12 progress bar + deterministic charts-set pin** - `806ba83` (feat)

**Plan metadata:** pending (this SUMMARY commit)

## Files Created/Modified
- `src/chat_analyzer/cli/nlp_gate.py` - NEW silent availability probe + locked model constants + env override
- `src/chat_analyzer/analysis/emotion.py` - flat list-of-dicts parse fix (Pitfall 1), locked model default (D-07c), `emotion_figure` helper (Pattern 2)
- `src/chat_analyzer/cli/pipeline.py` - gate computed up front, gated emotion/summary stages with announce-before-download, `stage()` helper + `Progress` for D-12
- `src/chat_analyzer/cli/adapters.py` - `_build_emotion_block`/`_build_summary_block` (dominant derived from distribution), lead-ins at tab indices 7/8 (D-11)
- `src/chat_analyzer/cli/contracts.py` - `AnalysisResults` gains `emotion`/`summary` slots (None when gate OFF)
- `src/chat_analyzer/cli/report_html.py` - Emotion + Summary tabs, unavailable pip-install note, render context
- `tests/test_phase4_nlp.py` - NEW 3-test suite (gate ON/OFF/report) with faithful D-17 mocks + singleton reset fixture
- `tests/test_phase2_pipeline.py` - `test_whatsapp_e2e` pins gate OFF, asserts deterministic 6-key charts set (Pitfall 5)
- `tests/test_phase1_smoke.py` - `[nlp]` extra pin updated for sentencepiece (consequence of the Rule 2 fix)
- `pyproject.toml` - `[nlp]` extra gains `sentencepiece>=0.1.99`

## Decisions Made
- **Silent degrade is the gate contract (D-02/D-06):** the pipeline never prompts/hints; the interactive download menu + positional hint land in 04-03 (main.py)
- **nlp_gate probe = importable transformers/torch AND cached model files** (huggingface_hub `HF_HUB_CACHE` with `~/.cache/huggingface/hub` fallback) — no construction, no network at probe time; the announce print happens before any `from_pretrained` (D-05/Pitfall 4)
- **`CHAT_ANALYZER_FORCE_NLP=0|1` env override** — the dev machine has transformers but no cached emotion model, so the raw probe would never report available; tests force both branches deterministically (Pitfall 5)
- **Dominant emotion derived in the adapter** (argmax of `emotion_distribution`) — `get_emotion_summary` has no dominant key; the adapter is the only place that knows module dict shapes
- **Progress task total is dynamic** (4 with gate ON, 3 OFF) — the gate is computed once up front so the bar matches the stages that actually run

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] `[nlp]` extra missing sentencepiece (T5Tokenizer requirement)**
- **Found during:** Task 2 (GREEN) — `test_emotion_summary_with_mocked_nlp` failed: `transformers.T5Tokenizer` raised `requires_backends` ImportError (sentencepiece absent)
- **Issue:** transformers does not auto-install sentencepiece, yet `ConversationSummarizer` (ANAL-08) imports `T5Tokenizer` in its constructor. A real `pip install chat-analyzer-pro[nlp]` would silently degrade summarization forever.
- **Fix:** Added `sentencepiece>=0.1.99` to the `[nlp]` extra in pyproject.toml (heavy dep stays gated behind the extra — PKG-03 preserved); installed sentencepiece 0.2.2 in the dev env so the real summary path is exercised
- **Files modified:** pyproject.toml (+ tests/test_phase1_smoke.py pin updated in Task 3)
- **Verification:** 3/3 test_phase4_nlp tests pass; `test_lean_base_structural` nlp list assertion updated
- **Committed in:** `4a2c142` (pyproject), `806ba83` (smoke pin)

**2. [Rule 2 - Missing Critical] `AnalysisResults` TypedDict lacked emotion/summary keys**
- **Found during:** Task 2 — the plan's `files_modified` lists adapters.py/report_html.py but not contracts.py; the contract needs the two slots for the new blocks
- **Issue:** Tests and adapters index `results["emotion"]`/`results["summary"]`; the TypedDict (the pipeline contract's single source of truth) must declare them (None when gate OFF)
- **Fix:** Added `emotion: dict[str, Any] | None` and `summary: dict[str, Any] | None` to `AnalysisResults` with a docstring noting the silent-degrade semantics
- **Files modified:** src/chat_analyzer/cli/contracts.py
- **Verification:** 30-test verification batch green; ruff clean
- **Committed in:** `4a2c142`

**3. [Rule 1 - Bug] Adapter looked for a nonexistent `dominant_emotion` summary key**
- **Found during:** Task 2 (GREEN) — `test_emotion_summary_with_mocked_nlp` failed: `emotion["dominant"]` was None with a real distribution `{joy: 26, love: 1}`
- **Issue:** `get_emotion_summary` (emotion.py:220-276) returns `emotion_distribution`/`average_emotion_scores` but NO `dominant_emotion` key; the adapter's `.get("dominant_emotion")` always returned None
- **Fix:** Derive dominant as argmax of the distribution inside `_build_emotion_block` (the adapter's documented role — the only place that knows the module's dict shape)
- **Files modified:** src/chat_analyzer/cli/adapters.py
- **Verification:** Test A now passes (`dominant in dist`); report tab shows the dominant label
- **Committed in:** `4a2c142`

**4. [Rule 3 - Blocking] `PYTHONIOENCODING=utf-8` leaked into subprocess tests**
- **Found during:** Task 3 verification — `test_console_script_help` failed with `TypeError: argument of type 'NoneType' is not iterable` (cp1252 parent could not decode UTF-8 child output byte 0x90)
- **Issue:** NOT a code bug — my shell `$env:PYTHONIOENCODING="utf-8"` propagated to `chat-analyzer.exe` subprocesses; the child wrote UTF-8, the parent decoded with locale cp1252 → UnicodeDecodeError in the reader thread
- **Fix:** Stop setting `PYTHONIOENCODING` for pytest runs on this box (cp1252 round-trip is the working convention); confirmed the test passes without it and is unrelated to 04-02 code (main.py untouched since Phase 2)
- **Files modified:** none (test-harness convention)
- **Verification:** `test_console_script_help` 1 passed without the env var
- **Committed in:** n/a (no code change)

---

**Total deviations:** 4 auto-fixed (2 missing critical, 1 bug, 1 blocking — the last a harness-environment artifact)
**Impact on plan:** All fixes necessary for the feature to operate correctly (ANAL-08's T5 dependency, the contract, and the dominant-emotion data flow). No scope creep.

## Issues Encountered
- **Ruff F401 on the probe's `import torch`:** the availability import is intentionally unused — resolved with `# noqa: F401` on both heavy imports (same pattern 04-01 used)
- **Ruff F821 on `emotion_figure`'s `matplotlib.figure.Figure` annotation:** the module-level `import matplotlib` bind was missing — added at the top of emotion.py (same fix 04-01 made for network_graph.py)
- **Ruff I001 unsorted imports in nlp_gate.py:** resolved by ordering torch before transformers
- **Rich Progress vs console.is_terminal:** the bar only engages on a real tty; all 26 phase-2 + 10 smoke + 3 new tests run off-tty and assert the deterministic `[OK]` lines — no regressions

## User Setup Required
None - no external service configuration required. (`CHAT_ANALYZER_FORCE_NLP=0|1` is an optional debug override, not a user-facing requirement.)

## Next Phase Readiness
- `nlp_gate` now carries `MODEL_ID`/`SUMMARY_MODEL_ID` + sizes + the force-env — 04-03's interactive download menu extends this module (the plan's `install_nlp(cpu_only)` guard signature is designed but unimplemented, as planned)
- The 6-class emotion path is regression-trapped: the faithful list-of-dicts mock (`_fake_emotion_classifier`) asserts non-uniform scores, so the `[0]` bug cannot silently return (T-04-08)
- `stage()`/`stage_status` narration contract is pinned by `test_stage_narration_and_order` — 04-03's menu and hint must keep the stage labels verbatim
- 04-04 (legacy test rewire) will touch emotion.py signatures — the locked model default and `analyze_emotions`/`get_emotion_summary` signatures are the post-04-02 state the 04-04 plan expects
- Blocker: none. Deferred: 382 legacy ruff errors (deferred-items.md #1, unchanged baseline)

## Threat Surface
All new surface sits inside the plan's threat model (T-04-05..T-04-09, T-04-SC):
- New env override `CHAT_ANALYZER_FORCE_NLP` can force model `from_pretrained` (network download) on an uncached machine — opt-in, mitigated by the announce-before-download print (T-04-06); documented as a test/debug affordance
- `nlp_gate` reads the local HF cache directory (~/.cache/huggingface/hub) — read-only existence probe, matches the plan's declared trust boundary
- Summary text + emotion labels reach HTML only through the autoescape Jinja env with the `_CHART_PREFIX` whitelist — no `|safe` on model output (T-04-07)

## Known Stubs
- `conv_summary = {"summary": "Summary unavailable.", "messages_summarized": 0}` (pipeline.py) is the documented silent-degrade value when summarization fails at runtime (Pitfall 7) — distinct from the gate-OFF state (summary=None → pip-install note). Intentional.

---
*Phase: 04-nlp-extras-quality-gate*
*Completed: 2026-08-04*

## Self-Check: PASSED

- FOUND: `src/chat_analyzer/cli/nlp_gate.py`
- FOUND: `tests/test_phase4_nlp.py`
- FOUND: `04-02-SUMMARY.md`
- FOUND: commit `901d867` (TDD RED)
- FOUND: commit `4a2c142` (Task 2 GREEN)
- FOUND: commit `806ba83` (Task 3 progress bar + pin)
