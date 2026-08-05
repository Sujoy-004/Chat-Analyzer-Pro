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

The base install includes all the core analysis: message statistics, participants, activity trends, top words and emojis, sentiment, relationship health, and the conversation network.

For emotion classification and conversation summarization, also install the NLP extras (torch + transformers):

```bash
pip install -e ".[nlp]"   # quotes needed on some shells (e.g. zsh)
```

### 4. Run it

```bash
chat-analyzer path/to/your-chat-export.txt
```

If the `chat-analyzer` command isn't found (for example, your Python scripts directory isn't on `PATH`), use the module form instead:

```bash
python -m chat_analyzer path/to/your-chat-export.txt
```

There are no flags — one command does everything. The terminal shows progress as the analysis runs, then a summary of what it found. The report is always saved to the **current working directory** (the folder where you run the command) as `<chat_name>_report.html` and auto-opens in your browser (if the browser can't open, the absolute path is printed instead).

## What does the NLP download question mean?

The first time you run the tool on an interactive terminal and the NLP models aren't installed, it asks one question about downloading them — it's only asked once. The options are:

1. **Full torch (~3 GB)** — best quality
2. **CPU-only torch + model (~0.6 GB)**
3. **No download** — run basic analysis

If you choose no download, the tool runs basic analysis — every other feature still works, and you can install the extras any time with `pip install -e ".[nlp]"`.

If the tool can't ask (for example, when output is piped), it never prompts — it just prints a single hint line instead.
