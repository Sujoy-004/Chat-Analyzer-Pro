# Chat-Analyzer-Pro

One command turns a WhatsApp `.txt` or Telegram `.json` chat export into real insights about the conversation — printed in the terminal and wrapped in a self-contained HTML report. Everything runs locally: no accounts, nothing uploaded.

## Quickstart

### 1. Export your chat

- **WhatsApp:** open the chat → **⋮** menu → **More** → **Export chat** → save the `.txt` file
- **Telegram:** Telegram Desktop → **Settings** → **Advanced** → **Export Telegram data** → choose **Messages only** → export as **JSON**

### 2. Get the code

```bash
git clone https://github.com/Sujoy-004/Chat-Analyzer-Pro.git
cd Chat-Analyzer-Pro
```

### 3. Install

Requires **Python 3.11 or newer**.

```bash
# Windows (PowerShell)
py -m venv .venv
.venv\Scripts\activate
pip install -e .
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The base install includes all the core analysis: message statistics, participants, activity trends, top words and emojis, sentiment, relationship health, the conversation network, and a pandas-only **"What's going on"** narrative (heuristic observations — arc, who drives the chat, reciprocity, engagement — always labelled as speculative).

For emotion classification and an understanding conversation **generative** narrative (a short written paragraph), also install the NLP extras (torch + transformers):

```bash
pip install -e ".[nlp]"   # quotes needed on some shells (e.g. zsh)
```

You don't have to install them by hand: on an interactive run the tool's **NLP tier menu** can install them for you (see below).

#### Deep mode on Windows (long paths)

Installing the NLP extras on Windows downloads multi-gigabyte torch wheels. Windows caps file paths at 260 characters (`MAX_PATH`), and torch's extraction crashes with WinError 206 when the active virtual environment's `site-packages` folder sits at a deep path — the `Chat-Analyzer-Pro` checkout itself often qualifies.

If you hit that error, first try enabling Windows' long-path support: set the `LongPathsEnabled` DWORD to `1` under `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`, then restart your shell (admin rights required). The CLI honors this flag: once it's on, the guard stops warning. The more reliable fix is a short-path environment: create a virtualenv at a path well under 260 characters and run the `scripts/make_nlp_env.ps1` helper, which builds a short-path NLP venv (for example `%TEMP%\chat-analyzer-nlp`) inside your temp folder and installs the `[nlp]` extras there.

The CLI guards this before any download starts: when the environment's path is too deep it prints a `[WARN]` line — or refuses the install with these same instructions — so you never begin a multi-gigabyte install that cannot complete. The check is heuristic (an extended user-site path isn't measured), so power users can override it with `CHAT_ANALYZER_ALLOW_LONG_PATH=1`.

### 4. Run it

```bash
chat-analyzer path/to/your-chat-export.txt
```

If the `chat-analyzer` command isn't found (for example, your Python scripts directory isn't on `PATH`), use the module form instead:

```bash
python -m chat_analyzer path/to/your-chat-export.txt
```

No flags are required — one command does everything (the CLI adds only `--version` and typer's built-in `--help` for introspection). The terminal shows progress as the analysis runs, then a summary of what it found. The report is always saved to the **current working directory** (the folder where you run the command) as `<chat_name>_report.html` and auto-opens in your browser (if the browser can't open, the absolute path is printed instead).

Every chart in the report is **interactive ECharts** (the runtime is inlined, so the file stays fully offline): hover for tooltips, scroll or drag to zoom into the timeline, sentiment and health trends, and a true **3D drag-to-rotate conversation network**. Any chart that can't be built still renders its static PNG, so everything always shows. Because the JavaScript is embedded, a report is larger than before — **~1.7 MB is normal** for the single-file format.

The tool detects automatically whether the NLP models are installed and never picks a tier silently: the terminal always prints an **NLP status line** (`NLP enabled` or `NLP not installed`), and the report's **"What's going on"** tab states which tier produced it. Set `CHAT_ANALYZER_NO_OPEN=1` to stop the report from auto-opening in a browser.

**Inputs:** a WhatsApp `.txt`, a Telegram `.json`, or a `.zip` export archive (WhatsApp "Export chat" and Telegram "Export Telegram data" can both produce one). When a `.zip` contains several chat transcripts, the tool lists them and lets you choose which to analyze (press Enter to analyze all). Media files inside the zip (images, videos, stickers) are ignored — only the conversation text is analyzed.

## What does the NLP tier menu mean?

On **every** interactive run, the tool shows a 3-option NLP tier menu **before analyzing**, regardless of what's already installed — you choose how heavy the run should be:

```
NLP tier:
  1) Without NLP
  2) Minimal (~0.6 GB)
  3) Full-fledged (~3 GB)
Choice [1]:
```

1. **Without NLP** — forces NLP off for *this* run (even if the extras are installed) and runs basic analysis: every other feature still works.
2. **Minimal** — CPU-only torch (~0.6 GB) plus both models. Downloads anything missing, then analyzes with NLP on.
3. **Full-fledged** — full torch (~3 GB) plus both models. Downloads anything missing, then analyzes with NLP on.

Choosing tier 2 or 3 **is the consent point**: any missing packages are installed and any missing model weights are downloaded immediately (sizes are announced first), not deferred to first use. The tool makes sure your chosen tier is ready before it starts analyzing.

**When the menu can't be shown** (output is piped, or the run is driven by a script/CI), the tool never prompts: it defaults to **tier 1** silently and prints a single hint line. Automation can pick a tier explicitly with the environment override:

- `CHAT_ANALYZER_TIER=1` — without NLP (explicit)
- `CHAT_ANALYZER_TIER=2` — minimal
- `CHAT_ANALYZER_TIER=3` — full-fledged

## How the tiers are enforced

The tool checks — at startup — whether `torch` + `transformers` are installed and within the pinned versions, and whether both model weight sets are already cached:

- **READY** → tier 2/3 proceeds immediately ("NLP ready").
- **MISSING** → the packages install first (tier 2 = CPU-only torch, tier 3 = full torch), then both models download.
- **OUTDATED** → a version is outside the pins or the weights aren't cached: on an interactive terminal it asks **Update now / Go with current**; on a pipe it proceeds with what's installed.

- **Tier 1 / NLP off** → the **Tier A narrative** still runs: pandas-only heuristic observations ("What's going on"). **Tier B** (the written paragraph from a small local flan-t5-small model, summarizing a compact *signal digest* of your chat — never raw messages) only appears when NLP is enabled.

The **"What's going on"** tab in the report and the terminal status line always tell you which analysis actually ran — nothing is silently skipped.

## On narrative honesty

Relationship-health grades, emotion labels, and narrative observations are **statistical inference, not verdicts**. The "What's going on" tab labels every observation *speculative*, with a *low / medium / high* confidence tag, and the generative paragraph carries an explicit "small local model" disclaimer. Take them as conversationally interesting guesses, not relationship advice.

## Privacy

Everything runs **entirely on your machine** — no accounts, no server, no telemetry. `pip install` pulls public model weights (downloaded at tier selection, or on first use if you installed the extras manually, and cached locally); the model, your chat data, and the generated report never leave your device. The terminal messages say this on every NLP run.

## Features

One `chat-analyzer <chat-file>` command runs the whole pipeline — no configuration, no flags required (beyond the standard `--version` / `--help`):

| Area | What it covers |
|------|----------------|
| Input formats | WhatsApp `.txt`, Telegram `.json`, and `.zip` archives; a multi-chat zip asks which chat to analyze (or all of them) |
| Statistics | Message counts, participants, activity trends, top words and emojis |
| Sentiment | VADER-based sentiment per message and across the conversation |
| Relationship analysis | Relationship-health scoring and the conversation network (both always on) |
| "What's going on" narrative | Heuristic, pandas-based observations (arc, who drives the chat, reciprocity, engagement) — every observation labelled speculative with a confidence tag |
| NLP extra (`[nlp]`) | Emotion classification and a generative written summary driven by a small local model (flan-t5) |
| Output | Terminal progress + summary, plus a self-contained single-file HTML report (~1.7 MB, offline) with interactive ECharts — hover tooltips, dataZoom zoom-to-detail, a 3D drag-to-rotate network — and static PNG chart fallbacks |
| Local-first | No accounts, no server, no telemetry; model weights downloadable on demand and cached locally |

## Documentation

Further docs live in the `docs/` directory:

| Doc | Purpose |
|-----|---------|
| [GETTING-STARTED.md](docs/GETTING-STARTED.md) | Prerequisites, installation steps, and first-run setup |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System overview, components, and data flow |
| [DEVELOPMENT.md](docs/DEVELOPMENT.md) | Development setup, build commands, code style, and the PR process |
| [TESTING.md](docs/TESTING.md) | Test framework, how to run tests, and CI integration |

## Project status

v1.0 is complete — the full CLI pipeline (parse → analyze → terminal summary → HTML report) is implemented and verified. The test suite is green (210 tests, pytest) and the code base is clean under `ruff check` for `src/chat_analyzer` and `tests`.

Distribution today is clone-and-install from source (see Quickstart above); the package is not yet published to PyPI.