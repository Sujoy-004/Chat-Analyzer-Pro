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
- `CHAT_ANALYZER_FORCE_NLP` — legacy override kept for existing automation and tests: `1` maps to tier 3 (full-fledged), `0` to tier 1 (without NLP); `CHAT_ANALYZER_TIER` takes precedence when both are set

Emotion scoring on very large chats follows the same prompt-vs-automation split. When a tier 2/3 run exceeds the emotion sampling cap, an interactive terminal is asked `Sample? [y/N]` (default **No** = score every message exactly); on piped/CI runs it auto-samples instead. Sampling scores a deterministic, participant- and time-stratified subset — the same file always samples identically — and when it runs the terminal prints `Emotions based on a sample of N of M messages.` while the report shows *"Emotion scores based on a sample of N of M messages."* under the emotion table.

- `CHAT_ANALYZER_EMOTION_SAMPLE=<n>` — emotion sampling cap (default **50000**): `0`, `off`, or `false` disables sampling (always exact); a value that parses to a non-positive integer (e.g. `-5`, `00`) also disables sampling; a positive integer sets the cap; anything else (garbage) falls back to the default.
- `CHAT_ANALYZER_EMOTION_WORKERS=<n>` — parallel emotion worker count (default **3**): exact scoring fans unique-text inference out to a process pool on very large chats (each worker builds its own DistilBERT pipeline); `0` or `1` forces sequential (no pool); a positive integer sets the count but is capped at **8** (RAM-bound — each worker loads its own ~255 MB model copy plus a torch runtime); anything else falls back to the default. Parallel scoring only engages above the internal unique-text threshold.
- `CHAT_ANALYZER_EMOTION_QUANT=<on|off>` — int8 dynamic quantization of the emotion model (default **on**): roughly 1.5–2.5x faster CPU inference with negligible score drift, applied identically in the parent process and every pool worker; `0`, `off`, `false`, or `no` keep full fp32 weights.
- `CHAT_ANALYZER_RESULT_CACHE=<dir>` — opt-in result cache keyed by the input file's sha256 hash + your settings (default **OFF**): `0`, `off`, `false`, or `no` disable it; `1`, `on`, `true`, or `yes` enable it with the default directory (`%LOCALAPPDATA%\chat-analyzer\cache` on Windows, `~/.cache/chat-analyzer` elsewhere); any other value IS the cache directory. On a repeat run of the same file with the same settings, the terminal prints `[INFO] Loaded analysis from cache` and the run completes in seconds — the HTML report is always regenerated fresh. Entries expire after 30 days (pruned automatically on each store).

## How the tiers are enforced

The tool checks — at startup — whether `torch` + `transformers` are installed and within the pinned versions, and whether both model weight sets are already cached:

- **READY** → tier 2/3 proceeds immediately ("NLP ready").
- **MISSING** → the packages install first (tier 2 = CPU-only torch, tier 3 = full torch), then both models download.
- **OUTDATED** → a version is outside the pins or the weights aren't cached: on an interactive terminal it asks **Update now / Go with current**; on a pipe it proceeds with what's installed.

- **Tier 1 / NLP off** → the **Tier A narrative** still runs: pandas-only heuristic observations ("What's going on"). **Tier B** (the written paragraph from a small local flan-t5-small model, summarizing a compact *signal digest* of your chat — never raw messages) only appears when NLP is enabled.

The **"What's going on"** tab in the report and the terminal status line always tell you which analysis actually ran — nothing is silently skipped.

## Expected runtimes

Real wall-clock times measured via `scripts/benchmark.py` on the author's Windows dev machine (12 cores) for the four benchmark chats, at full scale:

| Chat size (~messages) | Tier 1 | Tier 2/3 (sampled for large chats) |
|---|---|---|
| 424,826 | ~10 min (~589 s) | **~13 min (~758 s)** — **sampled** (50,000 of 424,826) |
| 31,218 | ~2 min (~119 s) | ~14 min (~834 s)* — exact |
| 3,644 | ~46 s | ~2 min (~134 s)* — exact |
| 1,014 | ~19 s | ~1.5 min (~85 s)* — exact |

\* Rows marked with an asterisk predate the August 2026 emotion-path overhaul (int8 dynamic quantization of the DistilBERT model, resilient bounded-chunk process pooling, vectorized score write-back); the 424k row was re-measured after that work. Expect the smaller exact rows to improve by a similar inference-speedup factor (roughly 1.5–2.5x on the emotion stage).

These are **conservative upper bounds**: the machine was not idle (VS Code and editors were running throughout), so a quiet machine runs faster. For chats above the sampling cap, the emotion stage scores a deterministic stratified sample instead of every message and the report and terminal label it — and exact (unsampled) tier 2/3 on a 424k chat is the pre-Option-C behavior, which takes hours; that wall is exactly why sampled mode exists (see `CHAT_ANALYZER_EMOTION_SAMPLE` above).

Emotion inference uses int8 dynamic quantization by default (faster CPU scoring with negligible score drift). Set `CHAT_ANALYZER_EMOTION_QUANT=off` to keep full fp32 weights; parallel and sequential paths quantize identically so results stay comparable either way.

## On narrative honesty

Relationship-health grades, emotion labels, and narrative observations are **statistical inference, not verdicts**. The "What's going on" tab labels every observation *speculative*, with a *low / medium / high* confidence tag, and the generative paragraph carries an explicit "small local model" disclaimer. Take them as conversationally interesting guesses, not relationship advice.

## Privacy

Everything runs **entirely on your machine** — no accounts, no server, no telemetry. `pip install` pulls public model weights (downloaded at tier selection, or on first use if you installed the extras manually, and cached locally); the model, your chat data, and the generated report never leave your device. The terminal messages say this on every NLP run.

With the optional result cache enabled (`CHAT_ANALYZER_RESULT_CACHE`), analysis results are stored **outside the repo** (never in the working tree) under your user profile — delete that cache directory to erase all stored analysis data. Note that cached entries are invalidated by app-version/schema changes, but during development the app version is static: if you are developing with the cache on, delete the cache directory after code changes (the schema constant is the manual override).

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
| Output | Terminal progress + summary, plus a self-contained single-file HTML report (~1.7 MB, offline) with interactive ECharts — hover tooltips, dataZoom zoom-to-detail, a quarterly emotion timeline, a 3D drag-to-rotate network — and static PNG chart fallbacks |
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

v1.0 is complete — the full CLI pipeline (parse → analyze → terminal summary → HTML report) is implemented and verified. The test suite is green (302 fast + 26 slow tests, pytest — fast, golden-parity, and emotion sampling-parity) and the code base is clean under `ruff check` for `src/chat_analyzer` and `tests`.

Distribution today is clone-and-install from source (see Quickstart above); the package is not yet published to PyPI.