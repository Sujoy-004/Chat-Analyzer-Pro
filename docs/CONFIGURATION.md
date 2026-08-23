<!-- generated-by: gsd-doc-writer -->
# Configuration

Chat-Analyzer-Pro is a local, pip-installable CLI tool. There is no config file
(no JSON/YAML/TOML settings file) and no `.env` file — configuration is done in
three places: **the package's declared dependencies in `pyproject.toml`** (which
determine what features are installed), **environment variables** (runtime
behavior), and **the working directory** (where reports land). Python >= 3.11 is
the enforced floor (`requires-python = ">=3.11"` in `pyproject.toml`).

## Environment variables

All environment variables are **optional**. None cause startup to fail if
missing — each one only changes a specific behavior. The word-list variables
(`CHAT_ANALYZER_EMOTION_SAMPLE`, `CHAT_ANALYZER_EMOTION_WORKERS`,
`CHAT_ANALYZER_RESULT_CACHE`) are matched case-insensitively after stripping
(`.strip().lower()` in `nlp_gate.py`; the same normalization applies to
`CHAT_ANALYZER_EMOTION_QUANT` in `analysis/emotion.py`); the boolean flags
(`CHAT_ANALYZER_NO_OPEN`, `CHAT_ANALYZER_ALLOW_LONG_PATH`) match the exact
string `"1"`.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `CHAT_ANALYZER_TIER` | No | unset (silent tier 1) | Non-tty NLP tier override: `"1"` = without NLP, `"2"` = minimal (~0.6 GB), `"3"` = full-fledged (~3 GB). Only consulted when stdin is **not** a TTY — interactive runs always show the tier menu instead (see [NLP tier selection](#nlp-tier-selection)). Values outside `1`/`2`/`3` are ignored and resolution falls through to the legacy `CHAT_ANALYZER_FORCE_NLP` mapping, then silent tier 1 (`src/chat_analyzer/cli/main.py:107-120`). |
| `CHAT_ANALYZER_EMOTION_SAMPLE` | No | `50000` | Cap on how many messages the emotion model scores on large chats. `0`/`off`/`false` (or any value parsing to `<= 0`, e.g. `"00"`, `"-5"`) → sampling disabled (always exact); positive integer → that cap; anything else (garbage) → `50000`. Never raises (`src/chat_analyzer/cli/nlp_gate.py:96-123`). See [Emotion analysis tuning](#emotion-analysis-tuning). |
| `CHAT_ANALYZER_EMOTION_WORKERS` | No | `3` | Parallel emotion worker count. `0`/`1`/`off`/`false` (or any value parsing to `< 2`) → `1` (sequential, no pool); positive integer → `max(1, min(value, os.cpu_count() or 1, 8))`; anything else (garbage) → `3`. Never raises, always `>= 1` (`nlp_gate.py:126-155`). Parallel scoring only engages with a **real** transformers pipeline, `>= 2` workers, and more than 20,000 unique texts (`_EMOTION_PARALLEL_THRESHOLD`, `analysis/emotion.py:66`). |
| `CHAT_ANALYZER_EMOTION_QUANT` | No | OFF (`0`) | Opt-in INT8 dynamic quantization of the emotion classifier's Linear layers (`torch.ao.quantization.quantize_dynamic` → `qint8`, `analysis/emotion.py:160-187`). `1`/`on`/`true`/`yes` enable it; any other value keeps fp32. Applies only when a genuine transformers pipeline runs — no effect on the rule-based fallback or mocked scorers. Faster but approximate: developer-reported (unaudited, not CI-benchmarked) ~12 min vs ~18 min wall-clock on the 424k-message exact-score run, with only ~74.5% agreement between quantized and fp32 outputs on a stratified probe sample. See [Emotion analysis tuning](#emotion-analysis-tuning). |
| `CHAT_ANALYZER_RESULT_CACHE` | No | OFF (unset) | Opt-in repeat-run cache keyed by the input file's sha256 + a config signature. `0`/`off`/`false`/`no` → off; `1`/`on`/`true`/`yes` → the OS default dir (`%LOCALAPPDATA%\chat-analyzer\cache` on Windows, `~/.cache/chat-analyzer` elsewhere); **any other non-empty value IS the cache directory** (a path, or garbage treated as a path). 30-day TTL, schema-based invalidation (`nlp_gate.py:162-207`). See [Result cache](#result-cache). |
| `CHAT_ANALYZER_NO_OPEN` | No | unset (report auto-opens) | `"1"` suppresses auto-opening the HTML report in the browser. Honors the opt-out before `webbrowser` is ever touched — no exception, no log (`src/chat_analyzer/cli/report_html.py:463`). |
| `CHAT_ANALYZER_ALLOW_LONG_PATH` | No | unset (guard active on Windows) | `"1"` bypasses the Windows deep-path MAX_PATH guard before a multi-GB torch download starts (`src/chat_analyzer/cli/nlp_gate.py:371`). Power-user override; see [Windows deep-path guard](#windows-deep-path-guard). |
| `CHAT_ANALYZER_FORCE_NLP` (legacy) | No | unset (auto-detect) | Legacy override, superseded by `CHAT_ANALYZER_TIER` for tier selection. Two roles: (1) forces the NLP availability probe deterministically — `"1"` → NLP available, `"0"` → not installed, any other value ignored (the probe falls through to the import check) (`nlp_gate.py:222-227`); (2) in non-tty tier resolution it maps `0` → tier 1 and `1` → tier 3 (`main.py:117-120`). Primarily a test/debug affordance. |

Other variables are honored indirectly:

- `BROWSER` — the standard Python `webbrowser` module variable. CI sets it to
  `"__none__"` so subprocess runs can't pop a browser
  (`.github/workflows/ci.yml:29,51,77`).
- `TEMP` — used by `scripts/make_nlp_env.ps1` as the default base directory for
  the short-path NLP virtualenv (see below).
- `LOCALAPPDATA` — read on Windows to build the default result-cache directory
  (`%LOCALAPPDATA%\chat-analyzer\cache`; falls back to `~/.cache/chat-analyzer`
  when unset/empty — `nlp_gate.py:200-206`).
- `TOKENIZERS_PARALLELISM` — set (via `setdefault`) to `"false"` inside the
  emotion worker child processes so tokenizer forks never emit warnings
  (`_score_text_chunk`, `analysis/emotion.py:95`). Not a user-facing knob.

## Config file format

There is no config file format. Behavior is controlled exclusively by:

1. **`pyproject.toml`** — dependency groups:

   | Install command | What you get |
   | --- | --- |
   | `pip install -e .` | Base install — core analysis: parsing, stats, sentiment, relationship health, network graph, EDA, visualization, single-file HTML report. No emotion analysis: the emotion stage is gated behind the NLP availability check (`if nlp_on:`, `pipeline.py:345`), so on a base install it is skipped and the report shows "Emotion analysis unavailable" (`report_html.py:309`). |
   | `pip install -e ".[nlp]"` | Adds `torch>=2.0`, `transformers>=4.30,<5.15`, `sentencepiece>=0.1.99`. Enables lazy-imported heavy features: emotion classification and the Tier B generative written narrative. |
   | `pip install -e ".[dev]"` | Adds `pytest>=7.4`, `pytest-cov>=4.1`, `ruff>=0.16.1` for development. |
   | `pip install -e ".[dev,nlp]"` | Combined dev + nlp extras (used by the CI `nlp` job). |

   Note: `sentencepiece` is declared in the `[nlp]` extra because transformers'
   `T5Tokenizer` requires it but does not install it automatically — without it
   the Tier B narrative (flan-t5-small) would silently degrade (comment in `pyproject.toml:26-30`).

2. **`src/chat_analyzer/cli/nlp_gate.py`** — the locked model constants (not
   user-configurable, documented here for reference):

   | Constant | Value |
   | --- | --- |
   | `MODEL_ID` | `bhadresh-savani/distilbert-base-uncased-emotion` (~255 MB) |
   | `TIER_B_MODEL_ID` | `google/flan-t5-small` (~340 MB) |

   Model *weights* are not installed by pip — they download at tier selection
   (or on first use for manually installed extras) and are cached in the
   Hugging Face cache (`HF_HUB_CACHE`, falling back to
   `~/.cache/huggingface/hub` — `nlp_gate.py:83-93`).

## Windows deep-path guard

`pip install torch` on Windows crashes with WinError 206 when the active
venv's `site-packages` folder is deep enough that torch's wheel extraction
crosses the 260-character `MAX_PATH` limit (torch 2.x reserves a large amount of
extraction depth). The CLI checks this before any multi-GB download begins
(`src/chat_analyzer/cli/nlp_gate.py:358-385`):

- The check is **heuristic**: `len(site_packages) + 180 (TORCH_PATH_RESERVE) > 260 (WINDOWS_MAX_PATH)` → the path is too deep. A global Python3xx user-site install is not measured, which is why the override exists.
- **Suppressed automatically** when any of these hold:
  - The OS is not Windows.
  - `CHAT_ANALYZER_ALLOW_LONG_PATH=1`.
  - The Windows `LongPathsEnabled` system flag is on — read from the registry key
    `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`, DWORD value 1
    (`nlp_gate.py:336-355`). <!-- VERIFY: the LongPathsEnabled registry flag is a Windows OS setting described in the README (HKLM\SYSTEM\CurrentControlSet\Control\FileSystem); enabling it requires admin rights and a shell restart, and its exact availability depends on the Windows version. -->
  - The site-packages path leaves ≥ 180 characters of headroom under the 260 limit.
- **When triggered**: the terminal prints one `[WARN]` line before the tier
  menu on an interactive run; the runtime installer (`install_nlp` / `update_nlp`)
  refuses with a `RuntimeError` that still leaves basic analysis working.

**The reliable fix** is a short-path venv. `scripts/make_nlp_env.ps1` creates a
disposable venv at `%TEMP%\chat-analyzer-nlp` (or a path you pass as
`-BasePath`) and installs the package with `[nlp]` extras into it, keeping the
path well under 260 characters (`scripts/make_nlp_env.ps1:8-14`).

## NLP tier selection

### Interactive runs (TTY): the always-on tier menu

On a real TTY (`sys.stdin.isatty()`) **every** interactive run shows the 3-option
tier menu regardless of the current install state
(`src/chat_analyzer/cli/main.py:90-105`, `231-250`):

1. **Without NLP** — the default choice
2. **Minimal (~0.6 GB)** — CPU-only torch + the models
3. **Full-fledged (~3 GB)** — full torch + the models

If the Windows deep-path guard has a message, it is printed as a `[WARN]` line
*before* the menu offers a multi-GB download. Tier 1 runs with NLP off. Tiers
2/3 call `_ensure_nlp_for_tier` (`main.py:135-179`) to make NLP ready before
analyzing:

- **READY** → announces "NLP ready." and proceeds.
- **MISSING** → announces the install (name + size first), runs the guarded
  runtime installer (`install_nlp`, `nlp_gate.py:388-414` — a subprocess
  `pip install`, never `shell=True`), then `download_models` (`nlp_gate.py:436-457`)
  downloads **both** model weight sets immediately — tier selection is the
  consent point, not first pipeline use. CPU-only mode (tier 2) installs torch
  from the PyTorch CPU wheel index (`https://download.pytorch.org/whl/cpu`)
  first, then transformers separately from PyPI; full mode (tier 3) installs
  torch + transformers together. The install has a 900-second timeout and only
  installs torch + transformers — it never installs sentencepiece, which only
  arrives via `pip install ".[nlp]"`.
- **OUTDATED** → on a TTY it asks **"Update NLP packages/models now?"**
  (default no); yes runs `update_nlp` (`--upgrade` to the pinned ranges,
  `nlp_gate.py:417-433`) plus `download_models`, otherwise it proceeds with
  what's installed.

Any install/download failure (offline, no pip, timeout, deep path) degrades to
basic analysis with a friendly note — never a frozen terminal.

### Non-interactive runs (piped/CI): env override, else silent tier 1

When the tool can't prompt (positional argument with piped stdin, CI), it never
shows the menu (`main.py:247-250`):

- `CHAT_ANALYZER_TIER=1|2|3` wins when set.
- Otherwise the legacy `CHAT_ANALYZER_FORCE_NLP` maps `0` → tier 1 and `1` →
  tier 3.
- Otherwise the run is **silent tier 1** (no NLP) and prints a single hint
  line instead: `pip install chat-analyzer-pro\[nlp]`.

A non-tty tier 2/3 still goes through `_ensure_nlp_for_tier` — so it may
attempt an install/download — but the OUTDATED branch can't prompt and simply
proceeds with the current install.

### `nlp_status()` — READY / OUTDATED / MISSING

`nlp_gate.nlp_status()` (`nlp_gate.py:279-302`) resolves the environment into
one of three states (packages = torch, transformers, sentencepiece; pins in
`nlp_gate.py:240-244`, mirrored from `pyproject.toml`):

- **MISSING** — at least one `[nlp]` package is not installed.
- **OUTDATED** — all packages installed but a version falls outside the pins,
  or the model weights are not cached (a fresh install is not ready until the
  download step runs).
- **READY** — packages satisfy the pins **and** both model weight sets are
  cached.

Separately, the silent importability probe (`nlp_available`, `nlp_gate.py:210-235`)
returns True when transformers + torch import — it never requires cached
weights — and `CHAT_ANALYZER_FORCE_NLP` overrides both branches for
deterministic tests.

The terminal prints an **NLP status line** on every run ("NLP enabled..." or
"NLP not installed - basic analysis only...", `src/chat_analyzer/cli/render.py:54-66`), and the report's "What's going on" tab states which tier produced it — nothing is silently skipped.

## Emotion analysis tuning

Three environment knobs tune the emotion stage on large chats (two resolved in
`nlp_gate.py`, one in `analysis/emotion.py`; all **never raise**):

### Sampling cap — `CHAT_ANALYZER_EMOTION_SAMPLE`

`emotion_sample_cap()` (`nlp_gate.py:96-123`) returns the cap or `None`:

- unset/empty → `50000`
- `0` / `off` / `false`, or any value parsing to `<= 0` → `None` (**sampling
  disabled: always exact**)
- positive integer → that value as the cap
- anything else (garbage) → `50000`

When NLP is on, the cap resolves, and the **scorable** count (same
`_is_scorable` rule the analyzer uses, `emotion.py:467-486`) exceeds it, the
pipeline decides between prompt and auto (`pipeline.py:201-226`):

- **TTY** → a prompt asks: *"N scorable messages — exact emotion scoring can
  take ~2 hours; sampled (up to N) takes minutes. Sample? [y/N]"* — default
  **NO = exact**; `y`/`yes` samples.
- **Non-tty** (piped/CI/tests) → **auto-samples** (no prompt).

Sampled scoring is a **deterministic** (participant, time)-stratified sample
(random_state=42, ~50 time buckets per sender — `emotion.py:190-350`), so the
same file + cap always produces the same sample. Below the cap (or with the cap
disabled) behavior is byte-for-byte the exact path. When a sample runs, the
report and the terminal label it "based on a sample of N of M messages" and the
summary is computed over the scored rows only (`render.py:67-73`,
`report_html.py:299,307`). The **effective** cap (a tty y/N answer collapsed in)
rides in the result-cache key, so y and N on the same file produce two distinct
cache entries.

### Parallel workers — `CHAT_ANALYZER_EMOTION_WORKERS`

`emotion_worker_count()` (`nlp_gate.py:126-155`) returns a worker count `>= 1`:

- unset/empty → `3`
- `0` / `1` / `off` / `false`, or any value parsing to `< 2` → `1` (sequential,
  no pool)
- positive integer → `max(1, min(value, os.cpu_count() or 1, 8))` — capped at 8
  because every worker loads its own ~255 MB model copy plus a torch runtime
  (RAM bound)
- anything else (garbage) → `3`

Parallel exact scoring fans unique-text inference out to a process pool
(`_score_unique_texts_parallel`, `emotion.py:588-688`) — each unique text is
scored once and mapped back, so output is byte-identical to sequential
regardless of worker count. It engages only when **all** of these hold: a real
transformers pipeline is present (mocked pipelines always stay sequential), the
resolved count is `>= 2`, and there are more than 20,000 unique scorable texts
(`_EMOTION_PARALLEL_THRESHOLD`, `emotion.py:66`; gate in
`_score_unique_texts`, `emotion.py:514-539`). Worker failures are **partially
rescued** rather than discarding the whole pool: chunks are consumed as they
complete, completed chunks' scores are kept, and only the lost chunks are
re-scored sequentially in the parent and merged back in original order — output
still matches the sequential path byte-for-byte. Only when nothing comes back
from the pool at all does the entire stage fall back to fully sequential
scoring.

### INT8 quantization — `CHAT_ANALYZER_EMOTION_QUANT`

Opt-in (**default off**) INT8 dynamic quantization of the emotion classifier's
Linear layers: when enabled, `_maybe_quantize_pipeline`
(`analysis/emotion.py:160-187`) rewrites the pipeline model's `torch.nn.Linear`
layers to `qint8` via `torch.ao.quantization.quantize_dynamic`. Parsing mirrors
the other knobs — `.get("CHAT_ANALYZER_EMOTION_QUANT", "0").strip().lower()`
(`emotion.py:174`) — with the truthy values `1`/`on`/`true`/`yes`; anything
else keeps fp32. It is applied identically in the parent process and inside
every spawn worker, so sequential and parallel outputs stay comparable within
a mode, and it has no effect on the rule-based fallback or mocked scorers
(nothing genuine to quantize).

The speedup trades accuracy. These figures are developer-reported (unaudited,
not CI-benchmarked):

- ~12 min vs ~18 min wall-clock on the 424k-message exact-score run.
- ~74.5% dominant-label agreement between quantized and fp32 outputs on a
  stratified probe sample (the source notes systematic surprise/love → joy
  flips).

Note that the result-cache key does **not** include this flag
(`result_cache.py:78-105`), so toggling quantization neither invalidates nor
re-keys existing entries — delete the cache directory to force a re-score when
switching modes on an already-cached file.

## Result cache

`CHAT_ANALYZER_RESULT_CACHE` opts into a repeat-run cache (`nlp_gate.py:162-207`,
`cli/result_cache.py`):

- **Off words**: `0` / `off` / `false` / `no` (or unset) → cache disabled (the
  default).
- **On words**: `1` / `on` / `true` / `yes` → the OS default directory:
  `%LOCALAPPDATA%\chat-analyzer\cache` on Windows (falling back to
  `~/.cache/chat-analyzer` when `LOCALAPPDATA` is unset/empty), and
  `~/.cache/chat-analyzer` elsewhere.
- **Any other non-empty value** → that value is the cache directory, normalized by strip + lowercase (`.strip().lower()` before `Path(raw)` — `nlp_gate.py:200-207`)
  (garbage is indistinguishable from a path — documented in the README).

On a cache hit the whole compute/NLP/narrative is skipped: the terminal prints
`[INFO] Loaded analysis from cache` and the run completes in seconds — the HTML
report is always regenerated fresh. On a miss, the result is stored best-effort
(never raises). When NLP is on and the cache is disabled, the pipeline prints a
tip: `[INFO] Tip: set CHAT_ANALYZER_RESULT_CACHE=1 to make repeat runs of this
file take seconds.` (`pipeline.py:504-508`).

Cache mechanics (`result_cache.py`):

- **Key**: sha256 of the input file's bytes **plus** a JSON config signature
  (`{schema, app_version, nlp_on, sample_cap, emotion_workers,
  chosen_transcripts}` — `result_cache.py:78-105`). The `RESULT_CACHE_SCHEMA`
  constant (`nlp_gate.py:74`) folds into every key, so a schema bump or an app
  upgrade invalidates all existing entries by construction.
- **TTL**: entries older than **30 days** are pruned on every store
  (`result_cache.py:43`, `219-238`).
- **Format**: `json.load` only — never pickle, never eval. Corrupt or
  schema-mismatched entries degrade to a miss (corrupt files self-heal by
  deletion); a stale `report_path` never survives a load.
- **Privacy**: the cache lives **outside the repo tree** — the default directory
  is always a user-profile path, never the working tree. The payload is the
  derived `AnalysisResults` contract only (sender names, top words, charts);
  raw chat text never enters it, and key filenames are hex digests only, so no
  user input ever enters a path. **To erase all stored analysis, delete the
  cache directory** (the default `%LOCALAPPDATA%\chat-analyzer\cache` /
  `~/.cache/chat-analyzer`, or whatever path you set). Note that cached entries
  are invalidated by app-version/schema changes, but during development the app
  version is static — if you develop with the cache on, delete the cache
  directory after code changes.

## Report output configuration

- **Location**: the report is always written to the **current working
  directory** (the folder where you run the command), not the input's directory
  (`report_path = Path.cwd() / "<chat_name>_report.html"`,
  `src/chat_analyzer/cli/report_html.py:452`).
- **Filename**: `<chat_name>_report.html`, where `<chat_name>` is the
  sanitized bare stem of the input file (e.g., `my-chat.txt` →
  `my-chat_report.html`).
- **Format**: a single self-contained HTML file — all charts/assets are
  base64-embedded; the file is written UTF-8 (`report_html.py:453`).
- **Interactive charts**: charts render as interactive ECharts — hover
  tooltips, dataZoom zoom-to-detail on the timeline/sentiment/health trends,
  and a 3D drag-to-rotate conversation network — driven by `charts_json`
  specs (`cli/chart_json.py`). The ECharts bundles are inlined into the file,
  so it stays single-file and offline (a **~1.7 MB** size is normal). Any
  chart without a buildable spec — or any WebGL-less browser — falls back to
  its static PNG; nothing is ever missing.
- **Auto-open**: after writing, the report opens in the default browser unless
  `CHAT_ANALYZER_NO_OPEN=1`; if the browser can't open, the absolute path is
  printed instead. It also honors the `BROWSER` env var (CI sets
  `"__none__"`).

## Testing and tooling

- **Framework**: `pytest` (registered in `pyproject.toml`) with pytest-cov. The
  tests exercise the real `chat_analyzer.*` module (never mocks of the CLI when
  asserting user-visible contract lines).
- **Markers**: one custom marker is registered, `slow`, for wall-clock-long
  tests (subprocess spawns / full pipeline renders):
  `markers = ["slow: long wall-clock tests (subprocess spawns / full pipeline renders)"]`
  in `[tool.pytest.ini_options]` (`pyproject.toml:40-43`). The fast suite runs
  with `pytest -m "not slow"`; the slow suite with `pytest -m "slow"`.
- **Coverage**: no coverage threshold is configured in `pyproject.toml` —
  coverage gates are not enforced.
- **Linting**: `ruff` (in the `[dev]` extra) with `ruff check src tests`.
  There is no `[tool.ruff]` section — ruff runs with its defaults.
- Circuit tests set `CHAT_ANALYZER_NO_OPEN=1` and `CHAT_ANALYZER_FORCE_NLP=0`
  in env so subprocess runs never open a browser or download models.

## Per-environment behavior

There are no dev/staging/production config files. The only difference between
environments is *which extras are installed* and *which env vars are set*:

| Environment | Install | Key env vars |
| --- | --- | --- |
| Local base | `pip install -e .` | none (auto-open on by default) |
| Local NLP | `pip install -e ".[nlp]"` | `CHAT_ANALYZER_ALLOW_LONG_PATH=1` (only if the venv site-packages path trips the MAX_PATH guard) |
| Tests / CI (`test` job) | `pip install -e ".[dev]"` | `CHAT_ANALYZER_NO_OPEN=1`, `CHAT_ANALYZER_FORCE_NLP=0`, `BROWSER="__none__"` — see `.github/workflows/ci.yml:27-31` |
| CI `slow` job | `pip install -e ".[dev]"` | same as above (`ci.yml:50-53`) |
| CI `nlp` job | `pip install -e ".[dev,nlp]"` | same as above (`ci.yml:75-79`) |

The CI workflow pumps `push` and `pull_request` events; the fast `test` job
runs across ubuntu/windows × Python 3.11/3.12 with `-m "not slow"`, the `slow`
job runs `-m "slow"` plus one ruff pass, and the `nlp` job runs the
NLP-gated test file against `. [dev,nlp]`.
