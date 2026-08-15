# Phase 4 (04-08): Result Cache Keyed by File Hash - Research

**Researched:** 2026-08-15
**Domain:** Opt-in user-local result cache for the deterministic analysis pipeline (repeat run = seconds); JSON payload, sha256 file key, privacy-locked location
**Confidence:** HIGH for repo facts (code reads + probes); MEDIUM for design choices (flagged [ASSUMED])

<user_constraints>
## User Constraints (from CONTEXT.md + DEFERRED.md)

### Locked Decisions (relevant subset, verbatim)
- **D-08:** NO CLI flags ship in Phase 4. (Cache must be env-gated, never a flag.)
- **D-09:** Report is saved to the current working directory, named after the chat file — always generated, always auto-opened (D-10). A cache hit must still write + open the report.
- **D-02/D-06:** Silent NLP availability check; non-interactive/positional runs never prompt (auto-degrade to basic analysis). The cache key must capture the resolved NLP-on state.
- **D-17:** Heavy model load (transformers/torch pipelines) mocked with unittest.mock in tests — fast and offline-safe; real-model inference in tests REJECTED.
- **DEFERRED.md (locked by user):** "No real chats, no synthetic fixture, no names/photos/voices anywhere in the repo... only the resulting timings table ships. No sample-data file is committed." — the result cache MUST live OUTSIDE the repo and must never ship personal chat data.
- **DEFERRED.md OPEN follow-up:** "[ ] **Result cache keyed by file hash** (repeat run = seconds) — still an OPEN follow-up; deliberately out of this plan's scope." — THIS is the phase's charter.
- **04-06 decision record:** "No file-hash cache (per plan): sampling is deterministic via random_state=42 rather than memoized" + "a file-hash cache (rejected by plan, could revisit as opt-in)" — the follow-up lands as OPT-IN.

### the agent's Discretion
- Cache default on/off, cache directory path, TTL/prune policy, exact key composition, hook placement, test organization.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| (DEFERRED.md follow-up) | Repeat run of the same chat file completes in seconds instead of minutes/hours | Q1 key, Q2 payload, Q6 hook — cache hit skips compute/NLP/narrative, leaving parse + JSON load + HTML render |
| (Locked) Never ship personal chat data | Cache outside repo; payload carries no raw messages; opt-in | Q3 location, Q2 payload contents, Q4 opt-in default |
| D-17 | All cache tests mocked, no real-model inference | Q7 test design mirrors test_emotion_sampling.py mocking pattern |
| QUAL-02 | Tests exercise the real `chat_analyzer.*` modules | Q7: cache tests drive the REAL run_pipeline with mocked model callables |
</phase_requirements>

## Summary

The full `AnalysisResults` dict produced by `adapt()` is **already JSON-serializable after a recursive numpy-scalar sanitizer** — `charts` are base64 PNG *strings* (`fig_to_data_uri`, pipeline.py:33-41), `charts_json` is guaranteed JSON-safe (chart_json.py:12-14 + `_validate_charts_json` in report_html.py:374), and the DataFrames/figures/DiGraphs never reach the contract (adapters.py Pattern 3: only serializable scalars are extracted). So a cache can persist the *final contract* — render.py, report_html.py and main.py consume nothing else, and a cache hit reconstructs every tab/chart/insight with zero NLP re-runs. The one hard finding: numpy scalars **do** leak into the contract (`stats.peak_hour` = `groupby('hour').size().idxmax()` → numpy int, eda.py:116; `stats.avg_response_time` = `np.mean(...)` → numpy float, eda.py:70; emotion `distribution` counts → numpy int64, emotion.py:754; sentiment `avg_compound`), so a recursive sanitizer (numpy → `.item()`, NaN/Inf → `None`) is mandatory — `json.dumps` without it raises `TypeError`.

**Primary recommendation:** a new `src/chat_analyzer/cli/result_cache.py` (stdlib only) + a whitelist parser `nlp_gate.result_cache_dir()` mirroring `emotion_sample_cap()`'s never-raises style. The hook lives **inside `run_pipeline`**, after the parse stage and after the Option C sample decision is resolved, before the "Computing insights" stage. Key = sha256 of the input file bytes + a JSON config signature {cache schema, app version, resolved nlp_on, effective sample cap (the prompt answer collapses into this), emotion worker count, chosen zip transcript list}. `CHAT_ANALYZER_RESULT_CACHE` is **default OFF (opt-in)**: absent/empty/off-words → disabled; on-words → default dir (`%LOCALAPPDATA%\chat-analyzer\cache` on Windows, `~/.cache/chat-analyzer` elsewhere); any other value → explicit dir. On a hit: print `[INFO] Loaded analysis from cache`, return the payload with `report_path` forced to `""` (main.py fills it after `write_report` — D-09 unchanged). On a miss: run today's stages with the pre-resolved sample cap, then best-effort `store()`. Corruption/bad JSON/schema mismatch → miss, never crash, corrupt file deleted. Age-based TTL prune (30 days) on store. `scripts/benchmark.py` must set `CHAT_ANALYZER_RESULT_CACHE=0` so cached hits never pollute first-run timings. No new packages.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Cache key (file hash + config signature) | API/Backend (CLI pipeline) | — | Key inputs (file bytes, nlp_on, sample cap) are all resolved inside `run_pipeline`; no browser/DB tier exists in this tool |
| Cache storage (dir resolution, read/write, prune) | API/Backend (result_cache module) | — | User-local disk under the OS profile — never the repo, never a network service |
| Cache invalidation (schema version, file-edit hash, config change) | API/Backend | — | sha256 of input bytes + signature fold-in at the pipeline boundary |
| Honest UX labels | API/Backend (pipeline narration) | Terminal (render.py) | "Loaded from cache" is a stage line; pipeline owns stage lines (render.py:6-8 single-source narration rule) |
| Sample-consent decision (tty y/N) | API/Backend (pipeline, moved earlier) | — | The answer folds into the key; must resolve BEFORE the cache check (Q1/Q6) |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `hashlib` (stdlib) | 3.11+ (project floor) | sha256 of input file bytes | The tool's only honest cache key: any byte change invalidates |
| `json` (stdlib) | 3.11+ | Payload serialize/deserialize | `json.load` only — never pickle/eval (code-execution risk, Security Domain) |
| `os`/`pathlib`/`tempfile` (stdlib) | 3.11+ | Cache dir resolution, atomic write | `tempfile` + `os.replace` for corruption-free writes |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `platformdirs` | — | NOT adopted | Would resolve %LOCALAPPDATA%/~/.cache canonically, but the locked lean-base rule (PKG-02/03) + 04-07 "zero new packages" precedent favor 6 stdlib lines |
| `pytest` (dev) | >=7.4 (existing) | Cache hit/miss/key-change tests | Existing suite convention (unittest.mock + pytest) — D-16 keeps unittest style |

**Version verification:** stdlib only — no registry lookups needed; verified present in `.venv` (Python 3.14.2, live probe 2026-08-15).

## Package Legitimacy Audit

> N/A — this phase installs ZERO new packages (stdlib `hashlib`/`json`/`os`/`tempfile` only), matching the 04-07 precedent. No slopcheck run required; no registry-verified names to audit.

## Q1: Cache key — what exactly is hashed, and what else must be in the key

**Findings [VERIFIED: repo read]:**

1. **Hash the input file's raw bytes** (`sha256` via `hashlib`, chunked read for the 20 MB Suraj export). `run_pipeline`'s only input is `path` (pipeline.py:123) — parse results are a pure function of file bytes, so bytes-hash is the correct identity. File mtime/size are NOT part of the key: re-running the same file after a `touch` must still hit.
2. **Zip inputs: hash the whole zip's bytes** — NOT extracted content. Reasons [VERIFIED: zip_input.py:38-76]: the transcript list and the media-file count (`count_zip_media_members`) are zip-level properties; `media_messages` in the contract depends on the archive's media members (adapters.py:81-86 `max(marker_rows, media_files)`), so two zips with identical transcripts but different media MUST produce different keys. Zip bytes hash captures all of it.
3. **The zip transcript SELECTION is interactive** (`_select_transcripts`, zip_input.py:79-119) and changes the merged rows — the sorted chosen-member-name list must ride in the config signature [ASSUMED — design choice; nothing in repo suggests otherwise].
4. **Config signature (folded into the key, JSON `sort_keys=True`)** — what changes the analysis output:
   - `schema` (cache schema version int) + `app_version` (`importlib.metadata.version('chat-analyzer-pro')`, the same call main.py:85 uses) — invalidates when analysis code changes. [ASSUMED]
   - `nlp_on` (resolved `nlp_enabled`/probe result, pipeline.py:139) — a tier-1 result has `emotion: None` and a different narrative status; serving it to a tier-3 run would silently drop the Emotion tab. MUST be in the key.
   - **effective sample cap** (the resolved `sample_cap`, `None` = exact) — this is where the **tty y/N answer collapses in**: the prompt (pipeline.py:308-317) turns the env cap into either exact (`None`) or sampled (`cap`). The key uses the *effective* decision, not the raw env string, so `y` vs `N` on the same file produce two distinct entries. [ASSUMED — design choice]
   - `emotion_worker_count` (`nlp_gate.emotion_worker_count()`) — output is parity-identical by contract (04-07), but including it is conservative: a config change SHOULD invalidate, and it costs one int. [ASSUMED — the research question itself lists it]
   - `chosen_transcripts` (sorted member names) for zips.
   - NOT in the key: `CHAT_ANALYZER_NO_OPEN` (only affects auto-open), `CHAT_ANALYZER_ALLOW_LONG_PATH` (install-time warning only), encoding vars (stdout only). [VERIFIED: their only consumers are open_report/main.py/install guards]
5. **Key file format:** `<sha256(json.dumps({"file_sha256":..., "signature":...}, sort_keys=True))>.json` — hex filename, no user input in the path (path-traversal impossible). [ASSUMED]

## Q2: Cache payload — what serializes, what must be rebuilt

**Findings [VERIFIED: repo read]:**

1. **The payload is the ENTIRE post-`adapt()` `AnalysisResults`** (contracts.py:25-53). Every consumer reads only this dict:
   - `render.show_summary` reads `parse`, `stats`, `emotion.sample`, `narrative` (render.py:19-83).
   - `report_html.write_report` reads `parse/stats/participants/content/sentiment/charts/charts_json/insights/health/network/emotion/narrative` (report_html.py:432-450).
   - `main._analyze_path` reads `parse.parsed_messages` + sets `report_path` (main.py:196-199).
   → A cache hit reconstructs EVERYTHING the report + terminal need. Nothing must be re-run.
2. **What is NOT serializable never reaches the contract:** matplotlib figures become base64 strings in `fig_to_data_uri` (pipeline.py:33-41) before `adapt()`; the `prepared_data` DataFrame, the networkx `DiGraph`, and interaction-matrix DataFrame are deliberately extracted to scalars only (adapters.py:126-131 Pattern 3). Charts dict = `dict[str, str]` base64 data URIs — JSON-safe. `charts_json` = `dict[str, dict]` — JSON-safe by construction (chart_json.py:12-14) and boundary-validated at write time (report_html.py:374-386).
3. **The ONE hard gap: numpy scalars leak into the contract** (verified):
   - `stats.peak_hour` — `groupby('hour').size().idxmax()` → numpy int (eda.py:116), no conversion in adapters.py:72.
   - `stats.avg_response_time` — `np.mean(...)` → numpy float (eda.py:70), only NaN-checked (adapters.py:58-60).
   - `emotion.distribution` — `value_counts().to_dict()` → numpy int64 counts (emotion.py:754), passed through (adapters.py:258-268).
   - `emotion.average_scores` → numpy floats; `sentiment.avg_compound`/`by_sender`/`daily_avg` → numpy floats (adapters.py:113-123); `health`/`network` scalars likely numpy floats; `narrative` observation confidences (narrative.py:426-467) can be numpy floats.
   → `json.dumps` raises `TypeError` on any of these. **A recursive sanitizer is REQUIRED:** dict/list/tuple walk, numpy scalar → `.item()`, `NaN`/`±Inf` → `None` (precedent: `chart_json._float_or_none`, chart_json.py:542-550). ~20 lines, stdlib.
4. **`report_path` must be forced to `""` on load** — the stored value from a previous cwd would point at a stale file; main.py sets it after `write_report` (main.py:199). [VERIFIED contract: contracts.py:52 "report_path is filled by main.py after write_report succeeds"]
5. **Payload envelope:** `{"schema": 1, "app_version": "...", "created_at": ISO, "results": {AnalysisResults}}`. [ASSUMED]
6. **Raw chat text is NOT in the contract** (no messages column anywhere in AnalysisResults) — verified by the TypedDict shape. Derived personal data (sender names, top words, narrative observations, insights) IS present — see Q3/Security.

## Q3: Cache location — outside the repo (privacy lock)

**Findings [VERIFIED: env probe] + recommendation [ASSUMED]:**

1. The privacy lock (DEFERRED.md: "No real chats, no synthetic fixture, no names/photos/voices anywhere in the repo") is absolute: the cache dir must be **outside the repo tree**, because the payload contains *derived* personal data (sender names in `participants`/`insights`/`narrative.observations`, top words/emojis, base64 charts). This also means no `.gitignore` strategy is sufficient — the lock requires the files to never be in the working tree at all.
2. **Recommended default path resolution** (in `nlp_gate.result_cache_dir()`, mirroring `model_cached`'s home-dir pattern at nlp_gate.py:69-79):
   - Windows: `%LOCALAPPDATA%\chat-analyzer\cache` → verified present on this box (`C:\Users\KIIT0001\AppData\Local`, probe 2026-08-15). Falls back to `Path.home() / ".cache" / "chat-analyzer"` if `LOCALAPPDATA` is unset.
   - macOS/Linux: `~/.cache/chat-analyzer` (matches the existing `~/.cache/huggingface` convention the project already uses, nlp_gate.py:78).
   - Explicit override: a non-keyword `CHAT_ANALYZER_RESULT_CACHE` value IS the dir path. [ASSUMED]
3. `%LOCALAPPDATA%` is the correct Windows choice vs `%APPDATA%`: it holds non-roaming machine-local data (roaming `%APPDATA%` would sync caches across machines via OneDrive — bad). [ASSUMED — Windows platform knowledge]
4. The cache is per-user and per-machine by construction (user-profile path) — no multi-user sharing, no server, nothing ships. README should document "delete the cache directory to erase all stored analysis data" (privacy self-service). [ASSUMED]

## Q4: Env contract — `CHAT_ANALYZER_RESULT_CACHE`

**Findings [VERIFIED: nlp_gate.py whitelist style] + recommendation [ASSUMED]:**

1. Mirror `emotion_sample_cap()` (nlp_gate.py:82-109) exactly: **never raises, garbage → safe default, whitelist semantics**:
   - absent/empty → **disabled** (cache OFF)
   - `"0"`/`"off"`/`"false"`/`"no"` (case-insensitive) → disabled
   - `"1"`/`"on"`/`"true"`/`"yes"` → enabled with the default dir (Q3)
   - any other value → enabled with that value as the cache dir path (paths are arbitrary strings, so "garbage" cannot be distinguished — document that non-keyword values are dirs). Never raises.
2. **Default OFF (opt-in).** Rationale:
   - Matches the locked conservative UX ("tty prompt default NO / auto-off", 04-06 SUMMARY: "sampling is opt-in on interactive terminals... keeps interactive behavior conservative").
   - No surprise disk growth, no surprise personal-data persistence for a friend who just wants a report.
   - 04-06 explicitly recorded the cache as "could revisit as opt-in" — the locked framing.
   - Default-on would also silently serve potentially stale results after code upgrades (mitigated by schema/version in the key, but opt-in keeps the surprise surface zero).
   - The `[INFO] Tip:` hint line precedent (main.py:275-278) can surface the feature: on a slow miss, print `[INFO] Tip: set CHAT_ANALYZER_RESULT_CACHE=1 to make repeat runs of this file take seconds.` [ASSUMED — recommended, matches the D-06 hint pattern]
3. **Honest label (locked requirement):** on a hit, `run_pipeline` prints `[INFO] Loaded analysis from cache` (pipeline owns stage/narration lines per render.py:6-8). The cached payload's own sample metadata (`emotion.sample.sampled`) still drives the "based on a sample of N of M" labels — a cached sampled result labels itself honestly in terminal AND report. [VERIFIED: render.py:67-73 + report_html.py:299/307 gate on the stored payload's sample dict]

## Q5: Correctness / invalidation

**Findings [VERIFIED] + recommendation [ASSUMED]:**

1. **File edit → new hash → miss.** Any byte change in the input invalidates; `touch` does not. Hashing a 20 MB file is ~0.1-0.3 s (chunked read) — well inside the "seconds" budget.
2. **Config change → miss** via the signature (Q1.4). Tier change (`nlp_on`), sample decision, workers, app upgrade, schema bump all invalidate.
3. **Corruption handling:** `load()` wraps `json.loads` in `try/except (json.JSONDecodeError, OSError, TypeError, ValueError)` → return `None` (miss) AND `unlink` the corrupt file (self-healing). Never raises, never crashes the run. Schema mismatch / missing `results` key / wrong types → same miss path. [ASSUMED]
4. **Never eval cached data:** `json.load` only. Pickle is explicitly forbidden (arbitrary code execution on unpickle — Security Domain). The payload's `charts` base64 strings are validated against the `data:image/png;base64,` prefix at write time anyway (report_html.py:397-400), and jinja autoescapes all payload text in the report. [VERIFIED: report_html.py:27-29, 397-400]
5. **TTL / disk growth:** prune on `store()` — `os.scandir`, delete `.json` entries with `mtime` older than 30 days, best-effort (`try/except OSError`, never raises). Sizing: each entry is dominated by base64 charts (~1-6 MB for a large chat; a few hundred KB typical), so 30 days at a handful of chats/week stays in the tens of MB. [ASSUMED]
6. **Atomic write:** serialize to `<key>.json.tmp` in the same dir, then `os.replace` (atomic on the same volume, Windows-safe). Concurrent runs of the same file → last-write-wins, no torn files. [ASSUMED]

## Q6: Where the hook lives

**Findings [VERIFIED: main.py + pipeline.py structure] + recommendation [ASSUMED]:**

1. **NOT in `main.py` before the tier menu.** The tier menu is the model-download **consent point** (main.py:90-104 + "tier selection is the model-download consent point", DEFERRED.md). Skipping it on a hit would silently skip consent and, worse, the key needs `nlp_enabled` which only exists AFTER the menu resolves. The menu must always appear on interactive runs (cheap: one prompt).
2. **Inside `run_pipeline`, after parse + sample-decision resolution, before "Computing insights".** Three verified constraints force this ordering:
   - The key needs the **effective sample cap**, which needs `n_scorable(df)` (emotion.py:406-416) + the tty prompt (pipeline.py:308-317) — i.e., the parse must have run and the prompt must have been asked BEFORE the cache lookup.
   - The **compute stage is the second-most expensive thing** (~90-110 s idle on 424k messages, DEFERRED.md "Compute stage ~90-110 s") — the check must precede it, or a hit still burns minutes.
   - The **parse stage is cheap** (seconds for 424k) and also produces the honest "Parsed N messages from M participants" line (pipeline.py:186-188) that should print on a hit anyway.
3. **Concrete restructure of `run_pipeline`:** (a) parse stage unchanged; (b) **move the Option C sample-decision block up** (cap resolution + tty prompt / off-tty auto-sample — identical logic and prompt text, pipeline.py:301-317) to right after the `ParseReport`/`df` build; (c) resolve `cache_key(...)`; (d) on hit → `console.print("[INFO] Loaded analysis from cache")`, force `report_path=""`, return the payload dict; (e) on miss → existing "Computing insights" / "Analyzing emotions" / "Generating narrative" stages with the pre-resolved `sample_cap` threaded in, then `store(key, adapt(...))` best-effort before return. The `finally: progress.stop()` already handles the early-return path (pipeline.py:467-469). [ASSUMED — code move, no behavior change: stage labels and order unchanged, so `test_stage_narration_and_order` (labels "Parsing chat"/"Computing insights") stays green]
4. **UX on a hit (honest):** tier menu (tty) → parse narration + parsed count → (if over cap, tty) the sample prompt — because the answer selects which cache entry — → `[INFO] Loaded analysis from cache` → `Messages: N` (main.py:196) → report written to cwd + auto-opened (D-09/D-10 unchanged) → summary panel. The user sees the same outputs; the NLP stages just don't narrate. [ASSUMED]
5. **benchmark.py must set `CHAT_ANALYZER_RESULT_CACHE=0`** in `_spawn_and_run`'s env (alongside `CHAT_ANALYZER_TIER`/`CHAT_ANALYZER_NO_OPEN`, benchmark.py:223-227): a user with the env var set would otherwise measure cache hits as "first-run" timings and pollute the README table. One-line hardening. [VERIFIED: benchmark.py env block]

## Q7: Test design (fast, mocked — D-17)

**Findings [VERIFIED: test_emotion_sampling.py + test_phase4_nlp.py patterns] + recommendation [ASSUMED]:**

1. **Follow the established `_mocked_models()` pattern** (test_emotion_sampling.py:327-361): patch `transformers.pipeline`, the `_emotion_analyzer`/`_emotion_model_loaded` module singletons, and the T5 summarizer classes — the REAL `run_pipeline` runs end-to-end with mocked models. Cache tests exercise the real cache module + real pipeline, no `[nlp]` extra needed, fast.
2. **Cache dir isolation:** `monkeypatch.setenv("CHAT_ANALYZER_RESULT_CACHE", str(tmp_path))` per test — parallel-safe, no repo pollution, exercises the real env parser.
3. **Recommended test list (`tests/test_result_cache.py`, ~14 tests):**
   - a. `test_cache_disabled_by_default` — no env var → run → no cache dir created.
   - b. `test_env_parser` — absent/empty/0/off/no → disabled; 1/on/true/yes → default dir; path → that dir; garbage string → that string as dir (documented); never raises.
   - c. `test_miss_runs_and_stores` — run with mocked models → entry file exists; NLP mock call counter incremented.
   - d. `test_hit_skips_nlp_stages` — second run, same file+config → NLP mock call counter UNCHANGED (proves no re-run); output deep-equal to first run.
   - e. `test_hit_prints_loaded_label` — captured console output contains "Loaded analysis from cache".
   - f. `test_file_edit_invalidates` — append a byte → new hash → miss → mocks called again.
   - g. `test_config_change_invalidates` — `CHAT_ANALYZER_EMOTION_SAMPLE=0` vs unset → different keys.
   - h. `test_tier_change_invalidates` — `nlp_enabled=True` vs `False` → different keys (emotion block presence).
   - i. `test_sample_choice_in_key` — tty `y` vs `N` (patch `console.input`, pattern from test_emotion_sampling.py:408-430) → two entries, distinct keys.
   - j. `test_corrupt_cache_is_miss_not_crash` — write garbage at the expected path → run succeeds, full pipeline runs, corrupt file deleted.
   - k. `test_schema_mismatch_is_miss` — payload with wrong `schema` → miss.
   - l. `test_sanitizer_roundtrip` — a fixture payload containing numpy int64/float64/NaN round-trips through `json.dumps/loads` losslessly (NaN → None).
   - m. `test_ttl_prune` — back-dated entry (mtime −31 days) removed on store.
   - n. `test_zip_transcript_selection_in_key` — key composition unit test: same zip bytes + different chosen member lists → different keys.
   - o. `test_report_path_reset_on_load` — stored `report_path` never surfaces; loaded payload has `""`.
4. **Zip-hash test** (optionally): zip fixture with transcripts — assert key changes when a media member is added (zip-level property) [VERIFIED contract: adapters.py:81-86].
5. All tests fast (no `slow` marker needed; the marker exists in pyproject.toml for subprocess spawns only).

## Q8: Scope estimate

**Findings [VERIFIED: repo structure] + recommendation:**

| File | Change | Est. size |
|------|--------|-----------|
| `src/chat_analyzer/cli/result_cache.py` | **NEW** — `sha256_file`, `cache_key`, `load`, `store`, `_sanitize`, `_prune` | ~180-260 lines |
| `src/chat_analyzer/cli/nlp_gate.py` | **EDIT** — `result_cache_dir()` whitelist parser (mirrors `emotion_sample_cap`) | +~25 lines |
| `src/chat_analyzer/cli/pipeline.py` | **EDIT** — move sample-decision block up; cache check + `[INFO]` label on hit; `store()` on miss | +~45-65 lines (mostly relocation) |
| `scripts/benchmark.py` | **EDIT** — `CHAT_ANALYZER_RESULT_CACHE=0` in worker env | +1 line |
| `tests/test_result_cache.py` | **NEW** — Q7 test list | ~400-550 lines |
| `README.md` | **EDIT** — env-var doc (04-06 precedent documents `CHAT_ANALYZER_EMOTION_SAMPLE`/`WORKERS`) + cache-deletion privacy note | ~15 lines |

**No new packages.** No parser/analysis/adapters/render/report_html changes (payload is the existing contract). No CLI flag surface (D-08).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| File hashing | Custom hash | `hashlib.sha256` chunked read | Stdlib, correct, fast; no alternatives worth a dependency |
| Payload serialization | pickle/marshal/custom formats | stdlib `json` | Pickle = arbitrary code execution on load (Security); json is human-inspectable and validated |
| Atomic cache write | Write-in-place | `tempfile` + `os.replace` | Torn files on crash would corrupt the cache; replace is atomic on same volume (Windows-safe) |
| Env parsing | Inline `os.getenv` in pipeline | `nlp_gate.result_cache_dir()` | One parsing contract (04-06 pattern: "cap resolution lives in nlp_gate... callers share one contract"); never raises |
| Numpy → JSON coercion | Ad-hoc per-field conversion | One recursive `_sanitize` | 9+ fields verified to leak numpy scalars; a central walker can't miss a field added later (precedent: `chart_json._float_or_none`) |
| Cache dir resolution | Hardcoded paths | `%LOCALAPPDATA%` / `~/.cache` convention | Matches the project's existing `~/.cache/huggingface` convention (nlp_gate.py:78) and OS norms |

**Key insight:** everything this cache needs is stdlib. The only genuinely tricky code is the numpy sanitizer (verified necessary) and the key-composition ordering (sample decision before cache check); both are small and testable.

## Common Pitfalls

### Pitfall 1: `json.dumps` TypeError on numpy scalars
**What goes wrong:** the first real cache write crashes with `TypeError: Object of type int64/float64 is not JSON serializable` on `stats.peak_hour`, `stats.avg_response_time`, emotion distribution counts, or sentiment means.
**Why:** pandas/numpy results flow into the contract unconverted (verified eda.py:116/70, emotion.py:754, adapters.py:72/58).
**How to avoid:** recursive `_sanitize` (numpy → `.item()`, NaN/Inf → None) applied at `store()`; test l covers round-trip.
**Warning signs:** `TypeError: Object of type numpy.int64 is not JSON serializable`.

### Pitfall 2: Cache check before the sample decision → wrong-variant results
**What goes wrong:** the key can't include the tty y/N answer if the lookup happens before the prompt; a cached exact result gets served to a user who just asked for sampled (or vice versa) — the report label would honestly say what it is, but the user's explicit choice is silently ignored.
**Why:** the prompt needs `n_scorable(df)` which needs the parse; the naive "check at the very top" placement skips it.
**How to avoid:** resolve the sample decision (move the existing block up) BEFORE the cache check; fold the *effective* cap into the key (Q6.3).
**Warning signs:** cache hit returns while a prompt the user never saw was skipped.

### Pitfall 3: Hook in `main.py` before the tier menu
**What goes wrong:** the consent flow (tier menu = model-download consent) is skipped on hits, and `nlp_enabled` doesn't exist yet, so the key can't distinguish tier 1 from tier 3 — a basic result could be served to an NLP-tier run (missing Emotion tab).
**How to avoid:** hook stays in `run_pipeline`, key includes `nlp_on`.
**Warning signs:** a tier-3 interactive run shows "NLP enabled" but the report has no emotion tab.

### Pitfall 4: Cache anywhere in the repo
**What goes wrong:** violates the locked "no personal data in the repo" rule — the payload contains sender names, top words, narrative observations, and charts.
**Why:** `%LOCALAPPDATA%`/`~/.cache` are outside the tree; any repo-relative path is not.
**How to avoid:** Q3 location contract; add an assertion in tests that the default dir is never under the repo.
**Warning signs:** cache files appearing in `git status`.

### Pitfall 5: Corrupt cache crashes the run
**What goes wrong:** a truncated/tampered JSON file raises on `json.loads`, killing a run that could have just re-analyzed.
**Why:** no try/except on the load path.
**How to avoid:** miss-on-any-failure + `unlink` the corrupt file; never raise (Q5.3).
**Warning signs:** run dies with a JSONDecodeError from a cache file.

### Pitfall 6: `report_path` baked into the cached payload
**What goes wrong:** a hit renders `Report: C:\old\cwd\report.html` (a stale path) and auto-opens a file that no longer matches the current cwd.
**Why:** the stored results came from a previous run's cwd.
**How to avoid:** force `""` on load; main.py sets it after `write_report` (Q2.4).
**Warning signs:** summary panel prints a report path that predates this run.

### Pitfall 7: Benchmark pollution
**What goes wrong:** with `CHAT_ANALYZER_RESULT_CACHE` set globally, `scripts/benchmark.py` measures cache-hit seconds and the README timing table becomes fake.
**Why:** `run_one` calls `run_pipeline` directly and inherits the env.
**How to avoid:** `CHAT_ANALYZER_RESULT_CACHE=0` in the worker env (Q6.5).
**Warning signs:** benchmark seconds collapse to <5 s for the 424k chat.

### Pitfall 8: Unbounded disk growth
**What goes wrong:** every analyzed file (incl. 20 MB chats → multi-MB payloads) accumulates forever.
**How to avoid:** 30-day mtime TTL pruned on store, best-effort (Q5.5); document manual deletion.
**Warning signs:** cache dir size grows without bound.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | Everything | ✓ | 3.14.2 (`.venv`; floor >=3.11 per pyproject) | — |
| stdlib `hashlib`/`json`/`os`/`tempfile`/`pathlib` | Cache module | ✓ | 3.14.2 stdlib | — |
| Windows (win32) | `%LOCALAPPDATA%` default dir | ✓ | Windows 11; `LOCALAPPDATA=C:\Users\KIIT0001\AppData\Local` (probe) | `~/.cache/chat-analyzer` |
| `pytest` + `unittest.mock` | Tests | ✓ | existing dev extra (>=7.4) | — |
| Mocked `transformers`/T5 (D-17) | Pipeline-level cache tests | ✓ (in `.venv`) | transformers 5.14.1 per 04-07 | tests skip if `[nlp]` absent (test_emotion_sampling.py:53-56 pattern) |

**Missing dependencies:** none. **Step 2.6 note:** no new tools/services introduced by this phase — code + env var only.

## Security Domain

No new network endpoints, auth paths, or trust boundaries — the cache is a local file store. Controls:

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes | `nlp_gate.result_cache_dir()` whitelist env parser (never raises, garbage → safe default — mirrors emotion_sample_cap); cache payload loaded with `json.load` ONLY; schema/type check on load; hex-only key filenames (no user input in paths → no traversal) |
| V6 Cryptography | yes | `hashlib.sha256` used as an integrity/identity key (not authentication) — stdlib, correct usage; never hand-rolled |
| V2/V3/V4 | no | No authentication/session/access-control surface |

**Known threat patterns:**

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Cached-file code execution (pickle/eval of crafted payload) | Tampering | Hard rule: `json.load` only, never `pickle`/`eval`; payload is inert data consumed by autoescaping jinja templates (report_html.py:421) and validated PNG-prefix chart strings (report_html.py:397-400) |
| Cache poisoning (attacker crafts a plausible payload) | Spoofing | Threat model: the cache dir is user-owned (`%LOCALAPPDATA%`); a local attacker who can write there already has full user access — the mitigation is the opt-in default (no cache exists unless enabled) and the "delete the cache dir" privacy escape hatch, not access control |
| Personal data at rest | Information disclosure | Cache OUTSIDE the repo (Q3), opt-in only (Q4), documented manual deletion; payload carries derived data (sender names, top words) but never raw message text (verified contract shape) |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Effective sample cap (prompt answer collapsed) belongs in the key; the prompt is re-asked on hits | Q1.4/Q6.3 | If a future UX drops the re-prompt, key semantics shift — planner should lock the UX first |
| A2 | App version + cache schema in the key (invalidate on code changes) | Q1.4 | Dev-mode staleness: version 0.1.0 is static during development, so an editable-install dev with the cache ON sees stale results after code edits — mitigated by the opt-in default + README note; schema constant is the manual override |
| A3 | `emotion_worker_count` in the key despite parity-identical output | Q1.4 | Spurious misses when the user changes workers; conservative choice per the research question |
| A4 | 30-day mtime TTL, prune-on-store | Q5.5 | Too aggressive → frequent misses; too lax → disk growth. Constant is trivially tunable |
| A5 | Default cache dir `%LOCALAPPDATA%\chat-analyzer\cache` / `~/.cache/chat-analyzer` | Q3 | Wrong-dir choice would only misplace (not lose) data; explicit env path overrides |
| A6 | `result_cache_dir()` semantics: non-keyword value = explicit path (garbage indistinguishable from a path) | Q4 | A typo'd value silently creates a cache dir at that name — documented; never raises |
| A7 | Moving the sample-decision block earlier in `run_pipeline` changes no observable behavior (same prompt text, same stage order) | Q6.3 | If wrong, `test_stage_narration_and_order` or the sampling prompt tests fail — caught in the plan's verification loop |

## Sources

### Primary (HIGH confidence — repo-verified this session)
- [VERIFIED: repo read] `pipeline.py` — run_pipeline structure (123-469), `fig_to_data_uri` (33-41), sample prompt (301-317), `nlp_on` resolution (139), `adapt()` call (449-466), `finally: progress.stop()` (467-469)
- [VERIFIED: repo read] `contracts.py` — AnalysisResults TypedDict (25-53), `report_path` filled by main.py (52)
- [VERIFIED: repo read] `adapters.py` — Pattern-3 scalar extraction (126-131), numpy leaks at 58-60/72, emotion block (247-268)
- [VERIFIED: repo read] `eda.py:70,116` — numpy mean / idxmax peak_hour; `emotion.py:406-416` n_scorable; `emotion.py:754` value_counts distribution; `narrative.py` observations embed sender names
- [VERIFIED: repo read] `report_html.py` — consumers (432-450), PNG prefix validation (397-400), autoescape (421), `_validate_charts_json` (374-386)
- [VERIFIED: repo read] `nlp_gate.py` — `emotion_sample_cap` whitelist style (82-109), `model_cached` home-dir pattern (69-79), version() usage in main.py:85
- [VERIFIED: repo read] `zip_input.py` — transcript selection (79-119), zip-level media count (57-76)
- [VERIFIED: repo read] `main.py` — tier menu = consent point (90-104, 231-250), `_analyze_path` (182-203), report_path set (199)
- [VERIFIED: repo read] `render.py` — single-source narration rule (6-8), consumers (19-83)
- [VERIFIED: repo read] `scripts/benchmark.py` — worker env block (223-227), `run_one` calls `run_pipeline` directly (158-195)
- [VERIFIED: repo read] `tests/test_emotion_sampling.py` — `_mocked_models` pattern (327-361), console.input patching (408-430)
- [VERIFIED: repo read] `.planning/config.json` — `workflow.nyquist_validation: false` → Validation Architecture section omitted
- [VERIFIED: live probe] Python 3.14.2 in `.venv`; Windows 11 / win32; `LOCALAPPDATA` set; stdlib hashlib/json importable (2026-08-15)
- [VERIFIED: git] HEAD `200749f`; working tree clean except untracked 04-07-RESEARCH.md (this research is NOT committed per instruction)

### Secondary (MEDIUM confidence)
- [VERIFIED: DEFERRED.md] benchmark timing facts (compute ~90-110 s, Suraj tier-1 ~10 min / sampled ~33 min / exact hours); open follow-up wording; "no real chats" lock
- [VERIFIED: 04-06 SUMMARY] "revisit as opt-in" record; conservative tty-default-NO UX; `sampled` key contract

### Tertiary (LOW confidence)
- None — no external/unverified claims are load-bearing; design choices are flagged [ASSUMED] in the Assumptions Log

## Metadata

- Standard stack: HIGH — stdlib only, zero new packages; every library verified present in the environment
- Architecture: HIGH for repo facts (contract serializability, ordering constraints, hook placement verified by reads); MEDIUM for the sample-decision relocation (A7) and key composition choices (A1-A3)
- Pitfalls: HIGH — all 8 verified against repo code
- **Valid until:** 2026-09-15 (analysis code at HEAD is stable; a future phase that changes the AnalysisResults contract must bump the cache schema constant and re-read Q2)
