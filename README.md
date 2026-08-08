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

There are no flags — one command does everything. The terminal shows progress as the analysis runs, then a summary of what it found. The report is always saved to the **current working directory** (the folder where you run the command) as `<chat_name>_report.html` and auto-opens in your browser (if the browser can't open, the absolute path is printed instead).

The tool detects automatically whether the NLP models are installed and never picks a tier silently: the terminal always prints an **NLP status line** (`NLP enabled` or `NLP not installed`), and the report's **"What's going on"** tab states which tier produced it. Set `CHAT_ANALYZER_NO_OPEN=1` to stop the report from auto-opening in a browser.

**Inputs:** a WhatsApp `.txt`, a Telegram `.json`, or a `.zip` export archive (WhatsApp "Export chat" and Telegram "Export Telegram data" can both produce one). When a `.zip` contains several chat transcripts, the tool lists them and lets you choose which to analyze (press Enter to analyze all). Media files inside the zip (images, videos, stickers) are ignored — only the conversation text is analyzed.

## What does the NLP download question mean?

The first time you run the tool on an interactive terminal and the NLP models aren't installed, it asks one question about downloading them — it's only asked once. The options are:

1. **Full torch (~3 GB)** — best quality
2. **CPU-only torch + model (~0.6 GB)**
3. **No download** — run basic analysis

If you choose no download, the tool runs basic analysis — every other feature still works, and you can install the extras any time with `pip install -e ".[nlp]"`.

If the tool can't ask (for example, when output is piped), it never prompts — it just prints a single hint line instead.

## How the auto-detected tiers work

The tool silently checks — at startup — whether `torch` + `transformers` are installed. There are no flags to remember:

- **NLP available** → everything runs automatically: emotion, the conversation summary, **and** the written narrative paragraph (a small local model, flan-t5-small, summarizing a compact *signal digest* of your chat — never raw messages).
- **NLP not available** → you're offered the download menu (or, off a terminal, a hint line). If you decline, the **Tier A narrative** still runs: pandas-only heuristic observations. **Tier B** (the written paragraph) only appears once the models are installed.

The **"What's going on"** tab in the report and the terminal status line always tell you which analysis actually ran — nothing is silently skipped.

## On narrative honesty

Relationship-health grades, emotion labels, and narrative observations are **statistical inference, not verdicts**. The "What's going on" tab labels every observation *speculative*, with a *low / medium / high* confidence tag, and the generative paragraph carries an explicit "small local model" disclaimer. Take them as conversationally interesting guesses, not relationship advice.

## Privacy

Everything runs **entirely on your machine** — no accounts, no server, no telemetry. `pip install` pulls public model weights (downloaded on first use and cached locally); the model, your chat data, and the generated report never leave your device. The terminal messages say this on every NLP run.
