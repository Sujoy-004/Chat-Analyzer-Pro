<!-- generated-by: gsd-doc-writer -->
# Getting Started

Chat-Analyzer-Pro turns a WhatsApp `.txt` or Telegram `.json` chat export into
real insights about the conversation — printed in the terminal and wrapped in a
self-contained HTML report. Everything runs locally: no accounts, nothing
uploaded. This guide walks a first-time user from prerequisites to first report.

## Prerequisites

- **Python 3.11 or newer** — enforced by the package
  (`requires-python = ">=3.11"` in `pyproject.toml`). Check your version with
  `python --version` (Windows) or `python3 --version` (macOS / Linux).
- **A chat export** — a WhatsApp `.txt`, a Telegram `.json`, or a `.zip`
  export archive (see [Export your chat](#2-export-your-chat)).
- No accounts, no API keys, no telemetry — the tool is fully local.

## 1. Get the code

```bash
git clone https://github.com/Sujoy-004/Chat-Analyzer-Pro.git
cd Chat-Analyzer-Pro
```

## 2. Export your chat

The tool accepts three input types:

- **WhatsApp `.txt`:** open the chat in WhatsApp → **⋮** menu → **More** →
  **Export chat** → save the `.txt` file.
- **Telegram `.json`:** Telegram Desktop → **Settings** → **Advanced** →
  **Export Telegram data** → choose **Messages only** → export as **JSON**.
- **`.zip` archive:** both WhatsApp "Export chat" and Telegram "Export
  Telegram data" can produce a zip. When a zip contains several chat
  transcripts, the tool lists them and lets you choose which to analyze
  (press **Enter** to analyze all). Media files inside the zip (images,
  videos, stickers) are ignored — only the conversation text is analyzed.

Not sure your file is valid? Two small sample exports ship with the repo in
`data/sample_chats/` (`whatsapp_sample.txt` and `telegram_sample.json`) — you
can run your first analysis on one of those.

## 3. Install

Create a virtual environment and install the package in editable mode.

**Windows (PowerShell):**

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -e .
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The base install includes all the core analysis: message statistics,
participants, activity trends, top words and emojis, sentiment, relationship
health, the conversation network, and a pandas-only **"What's going on"**
narrative (heuristic observations — arc, who drives the chat, reciprocity,
engagement — always labelled as speculative).

For **emotion classification** and an understanding **generative** narrative
(a short written paragraph), also install the NLP extras (torch + transformers):

```bash
pip install -e ".[nlp]"   # quotes needed on some shells (e.g. zsh)
```

### Deep mode on Windows (long paths)

Installing the NLP extras on Windows downloads multi-gigabyte torch wheels.
Windows caps file paths at 260 characters (`MAX_PATH`), and torch's extraction
crashes with WinError 206 when the active virtual environment's `site-packages`
folder sits at a deep path — the `Chat-Analyzer-Pro` checkout itself often
qualifies.

- First, try enabling Windows' long-path support: set the `LongPathsEnabled`
  DWORD to `1` under `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`, then
  restart your shell (admin rights required). The CLI honors this flag: once
  it's on, the guard stops warning.
- The more reliable fix is a short-path environment: run
  `scripts/make_nlp_env.ps1`, which builds a short-path NLP venv (for example
  `%TEMP%\chat-analyzer-nlp`) in your temp folder and installs the `[nlp]`
  extras there.
- The CLI guards this **before any download starts**: when the environment's
  path is too deep it prints a `[WARN]` line — or refuses the install with
  these same instructions — so you never begin a multi-gigabyte install that
  cannot complete. The check is heuristic, so power users can override it with
  `CHAT_ANALYZER_ALLOW_LONG_PATH=1`.

Full detail lives in [docs/CONFIGURATION.md](CONFIGURATION.md) under
**"Windows deep-path guard"**.

## 4. Run your first analysis

```bash
chat-analyzer path/to/your-chat-export.txt
```

If the `chat-analyzer` command isn't found (for example, your Python scripts
directory isn't on `PATH`), use the module form instead:

```bash
python -m chat_analyzer path/to/your-chat-export.txt
```

No flags are required — one command does everything (the CLI adds only
`--version` and typer's built-in `--help` for introspection). The terminal
shows progress as the analysis runs, then a summary of what it found,
including a `Messages: N` count and an **NLP status line** (`NLP enabled` or
`NLP not installed`) so the tool never picks a tier silently.

### The NLP tier menu — shown on every run

On **every** interactive run, the tool shows a 3-option NLP tier menu
**before analyzing**, regardless of what's already installed — you choose how
heavy this run should be:

```
NLP tier:
  1) Without NLP
  2) Minimal (~0.6 GB)
  3) Full-fledged (~3 GB)
Choice [1]:
```

1. **Without NLP** (default) — forces NLP off for *this* run (even if the
   extras are installed) and runs basic analysis: every other feature still
   works. Press **Enter** to accept the default.
2. **Minimal** — CPU-only torch (~0.6 GB) plus both the emotion and
   narrative models.
3. **Full-fledged** — full torch (~3 GB) plus the same models.

Choosing tier 2 or 3 **is the download consent point**: any missing packages
are installed and any missing model weights are downloaded immediately (sizes
are announced first), so the tier is ready before analysis starts. On a first
run, start with **tier 2** if you want emotion classification and the
generative narrative without the multi-gigabyte full-torch build; pick
**tier 3** for the full torch install.

When the menu can't be shown (output is piped, or the run is scripted/CI),
the tool never prompts: it defaults to **tier 1** silently and prints a single
hint line (`pip install chat-analyzer-pro\[nlp]`) instead. Automation can
pick a tier explicitly with `CHAT_ANALYZER_TIER=1|2|3` — see
[docs/CONFIGURATION.md](CONFIGURATION.md) for the full environment-variable
reference.

The report's **"What's going on"** tab always states which tier produced it —
nothing is silently skipped. Basic analysis uses pandas-only heuristic
observations (**Tier A**); the generative written paragraph (**Tier B**) only
appears once the models are installed.

### First-run tips

- **Very large chats and emotion:** on a tier 2/3 run, chats above the
  sampling cap ask `Sample? [y/N]` on an interactive terminal (default **No**
  = score every message exactly); on piped/CI runs they auto-sample instead.
  Sampled scoring is deterministic and labels the result "based on a sample
  of N of M messages".
- **Repeat runs are instant with the cache:** `CHAT_ANALYZER_RESULT_CACHE=1`
  opts into a repeat-run result cache (off by default) — re-analyzing the
  same file takes seconds. Full semantics in
  [docs/CONFIGURATION.md](CONFIGURATION.md).

## Where the report is written

The report is always saved to the **current working directory** (the folder
where you ran the command) as `<chat_name>_report.html` — where `<chat_name>`
is the sanitized filename of your input (e.g. `my-chat.txt` →
`my-chat_report.html`). It auto-opens in your browser; if the browser can't
open, the report's absolute path is printed instead. Set
`CHAT_ANALYZER_NO_OPEN=1` to suppress auto-opening.

The report is a single self-contained HTML file — all charts and assets are
embedded, so you can share it or archive it as-is. Its charts are
**interactive ECharts**: hover for tooltips, scroll or drag to zoom into the
timeline, sentiment and health trends, and a true **3D drag-to-rotate
conversation network**. The embedded runtime keeps the file offline; a report
size of **~1.7 MB** is normal. Any chart that can't be built falls back to
its static image, so the report always shows all charts.

## Common setup issues

- **`chat-analyzer` command not found** — your Python scripts directory isn't
  on `PATH`. Use `python -m chat_analyzer <file>` instead (or add the
  scripts directory to `PATH`).
- **NLP install fails on Windows with WinError 206** — the venv's path is too
  deep for torch's wheel extraction. Use `scripts/make_nlp_env.ps1` for a
  short-path NLP venv, or set the `LongPathsEnabled` registry flag (see
  [Step 3](#3-install)).
- **The report won't open in a browser** — the absolute path is printed
  instead; open it manually. Set `CHAT_ANALYZER_NO_OPEN=1` if you don't want
  auto-open behavior.
- **"File not found" or "Unsupported file type"** — the error message includes
  the exact export instructions for both WhatsApp and Telegram; re-export the
  chat and try again.

## Next steps

- [docs/ARCHITECTURE.md](ARCHITECTURE.md) — system overview, components, and
  data flow
- [docs/CONFIGURATION.md](CONFIGURATION.md) — environment variables, the
  `[nlp]` extras, tier auto-detection, and report output details
- [docs/TESTING.md](TESTING.md) — test framework, how to run the test suite,
  and CI integration
- [README.md](../README.md) — features, privacy notes, and narrative-honesty
  caveats