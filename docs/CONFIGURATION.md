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
missing — each one only changes a specific behavior. Values are matched exactly
as `"1"`/`"0"` strings.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `CHAT_ANALYZER_NO_OPEN` | No | unset (report auto-opens) | `"1"` suppresses auto-opening the HTML report in the browser. Honors the opt-out before `webbrowser` is ever touched — no exception, no log (`src/chat_analyzer/cli/report_html.py:306`). |
| `CHAT_ANALYZER_ALLOW_LONG_PATH` | No | unset (guard active on Windows) | `"1"` bypasses the Windows deep-path MAX_PATH guard before a multi-GB torch download starts (`src/chat_analyzer/cli/nlp_gate.py:163`). Power-user override; see [Windows deep-path guard](#windows-deep-path-guard). |
| `CHAT_ANALYZER_FORCE_NLP` | No | unset (importability probe) | Forces the NLP availability branch deterministically: `"1"` → NLP treats as available, `"0"` → NLP treated as not installed. Any other value is ignored (the probe falls through to the import check). Latched in `src/chat_analyzer/cli/nlp_gate.py:81-86`. Primarily a test/debug affordance. |

Two variables are also honored indirectly:

- `BROWSER` — the standard Python `webbrowser` module variable. CI sets it to
  `"__none__"` so subprocess runs can't pop a browser
  (`.github/workflows/ci.yml:29`).
- `TEMP` — used by `scripts/make_nlp_env.ps1` as the default base directory for
  the short-path NLP virtualenv (see below).

## Config file format

There is no config file format. Behavior is controlled exclusively by:

1. **`pyproject.toml`** — dependency groups:

   | Install command | What you get |
   | --- | --- |
   | `pip install -e .` | Base install — core analysis: parsing, stats, sentiment, relationship health, network graph, EDA, visualization, single-file HTML report. No emotion analysis: the emotion stage is gated behind the NLP availability check (`if nlp_on:`, `pipeline.py:269`), so on a base install it is skipped and the report shows "Emotion analysis unavailable" (`report_html.py:184`). |
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

   Model *weights* are not installed by pip — they download on first use and are
   cached in the Hugging Face cache (`HF_HUB_CACHE`, falling back to
   `~/.cache/huggingface/hub` — `nlp_gate.py:56-66`).

## Windows deep-path guard

`pip install torch` on Windows crashes with WinError 206 when the active
venv's `site-packages` folder is deep enough that torch's wheel extraction
crosses the 260-character `MAX_PATH` limit (torch 2.x reserves a large amount of
extraction depth). The CLI checks this before any multi-GB download begins
(`src/chat_analyzer/cli/nlp_gate.py:150-177`):

- The check is **heuristic**: `len(site_packages) + 180 (TORCH_PATH_RESERVE) > 260 (WINDOWS_MAX_PATH)` → the path is too deep. A global Python3xx user-site install is not measured, which is why the override exists.
- **Suppressed automatically** when any of these hold:
  - The OS is not Windows.
  - `CHAT_ANALYZER_ALLOW_LONG_PATH=1`.
  - The Windows `LongPathsEnabled` system flag is on — read from the registry key
    `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`, DWORD value 1
    (`nlp_gate.py:128-147`). <!-- VERIFY: the LongPathsEnabled registry flag is a Windows OS setting described in the README (HKLM\SYSTEM\CurrentControlSet\Control\FileSystem); enabling it requires admin rights and a shell restart, and its exact availability depends on the Windows version. -->
  - The site-packages path leaves ≥ 180 characters of headroom under the 260 limit.
- **When triggered**: the terminal prints one `[WARN]` line before the download
  menu; the runtime installer (`install_nlp`) refuses with a `RuntimeError`
  that still leaves basic analysis working.

**The reliable fix** is a short-path venv. `scripts/make_nlp_env.ps1` creates a
disposable venv at `%TEMP%\chat-analyzer-nlp` (or a path you pass as
`-BasePath`) and installs the package with `[nlp]` extras into it, keeping the
path well under 260 characters (`scripts/make_nlp_env.ps1:8-46`).

## NLP download prompt and tier auto-detection

### The one-time interactive menu

On a real TTY (`sys.stdin.isatty()`) with NLP not installed, the CLI shows a
3-option menu once (`src/chat_analyzer/cli/main.py:84-100`):

1. **Download full torch (~3 GB)** — best quality
2. **Download CPU-only torch + model (~0.6 GB)** — the default choice
3. **No download** — run basic analysis

Choice 1 or 2 dispatches the guarded runtime installer
(`install_nlp`, `nlp_gate.py:180-206`, ending at file end) which runs `pip install` in a
subprocess (never shell=True): CPU-only mode installs torch from the PyTorch CPU
wheel index (`https://download.pytorch.org/whl/cpu`) first, then transforms
separately from PyPI; full mode installs torch + transforms together. The
install has a 900-second timeout and only installs torch + transformers — it
never installs sentencepiece, which only arrives via `pip install ".[nlp]"`.
Any failure (offline, no pip, timeout, deep path) degrades to basic analysis
with a friendly note, never a frozen terminal.

If the tool can't prompt (positional argument, piped stdin, or the user picks
option 3), it prints a single hint line instead:
`pip install chat-analyzer-pro\[nlp]`.

### Tier auto-detection

At startup the CLI runs a **silent probe** (`nlp_available`) that never
requests: `transformers` + `torch` import == NLP available; the probe no longer
requires the weights to be cached (they download on first use). The
`CHAT_ANALYZER_FORCE_NLP` variable overrides both branches for deterministic
tests.

- **Tier A** — pandas-only heuristic narrative (arc, driver, reciprocity,
  engagement). Labeled speculative; always runs when NLP is off.
- **Tier B** — the generative written paragraph from `google/flan-t5-small`,
  summarizing a compact *signal digest* (never raw messages). Only runs when
  NLP is available.

The terminal prints an **NLP status line** on every run ("NLP enabled..." or
"NLP not installed - basic analysis only...", `src/chat_analyzer/cli/render.py:56-66`), and the report's "What's going on" tab states which tier produced it — nothing is silently skipped.

## Report output configuration

- **Location**: the report is always written to the **current working
  directory** (the folder where you run the command), not the input's directory
  (`report_path = Path.cwd() / "<chat_name>_report.html"`,
  `src/chat_analyzer/cli/report_html.py:295-296`).
- **Filename**: `<chat_name>_report.html`, where `<chat_name>` is the
  sanitized bare stem of the input file (e.g., `my-chat.txt` →
  `my-chat_report.html`).
- **Format**: a single self-contained HTML file — all charts/assets are
  base64-embedded; the file is written UTF-8 (`report_html.py:296`).
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
  in `[tool.pytest.ini_options]` (`pyproject.toml:39-42`). The fast suite runs
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
| CI `nlp` job | `pip install -e ".[dev,nlp]"` | same as above (`ci.yml:74-79`) |

The CI workflow pumps `push` and `pull_request` events; the fast `test` job
runs across ubuntu/windows × Python 3.11/3.12 with `-m "not slow"`, the `slow`
job runs `-m "slow"` plus one ruff pass, and the `nlp` job runs the
NLP-gated test file against `. [dev,nlp]`.