<!-- generated-by: gsd-doc-writer -->

# Chat-Analyzer-Pro

**chat-analyzer-pro** (v0.1.0) turns a WhatsApp `.txt`, Telegram `.json`, or `.zip` chat export into terminal insights plus a single self-contained HTML report — one command, entirely on your machine. No accounts, no uploads, no telemetry. Requires Python >= 3.11.

## Features

### Always on (base install)

- Message stats, participant breakdowns, and activity/timeline charts
- Top words and emojis
- VADER sentiment consensus
- Relationship health — score + grade with a gamified friendship index and streaks
- Directed interaction network — centrality, community detection, and a drag-to-rotate 3D ECharts view
- Tier A heuristic narrative ("What's going on") — hedged arc/driver/reciprocity/engagement observations, always labelled speculative

Interactive charts are powered by vendored ECharts + echarts-gl assets that are **inlined** into the HTML, so the report is one portable file that works offline.

### NLP-gated (optional `[nlp]` extra)

- Six-class emotion classification plus a quarterly emotion-timeline chart
- Tier B generative narrative paragraph — flan-T5 reads an ASCII digest of derived statistics, never your raw chat text

## Quickstart

1. Export a chat:
   - **WhatsApp:** open the chat → ⋮ menu → **More → Export chat** → save the `.txt`
   - **Telegram:** Telegram Desktop → **Settings → Advanced → Export Telegram data** → select only *Messages* → JSON format
2. Clone, create a virtualenv, and install:

   ```bash
   git clone https://github.com/Sujoy-004/Chat-Analyzer-Pro.git
   cd Chat-Analyzer-Pro
   python -m venv .venv
   .venv\Scripts\activate        # Windows PowerShell (source .venv/bin/activate on POSIX)
   pip install -e .
   ```

3. Analyze the bundled sample (no export needed yet):

   ```bash
   chat-analyzer data/sample_chats/whatsapp_sample.txt
   ```

4. The report opens in your browser automatically and is written to the current directory as `whatsapp_sample_report.html`.

No arguments? `chat-analyzer` runs an interactive loop that re-prompts until you give it a valid export path. `--version` and `--help` work as expected, and `python -m chat_analyzer` is equivalent to the console script. On success the exit code is 0; failures exit 1 with a friendly, instructive message — never a traceback.

Try the Telegram sample too: `data/sample_chats/telegram_sample.json`.

## Input formats

| Format | Recognized as | Notes |
| --- | --- | --- |
| `.txt` | WhatsApp export | System/skip lines counted honestly, never silently dropped |
| `.json` | Telegram Desktop export | Messages-only JSON from the official exporter |
| `.zip` | Full WhatsApp/Telegram archive | Transcripts inside are listed and you pick which to analyze on a terminal (merged if you select several); real media files are counted once while text transcripts are analyzed |

The report file name is derived from your input with sanitization (`<stem>_report.html`) and is always written to the current working directory. Auto-open is best-effort — set `CHAT_ANALYZER_NO_OPEN=1` to opt out.

## NLP tiers & models

Every interactive run shows this menu; the default is tier 1:

| Tier | Name | What you get | Download size |
| --- | --- | --- | --- |
| 1 | Without NLP *(default)* | All always-on features above | none |
| 2 | Minimal | + emotion classification & quarterly emotion timeline | CPU-only torch (~0.6 GB) + both models below |
| 3 | Full-fledged | + everything in tier 2 and the generative narrative paragraph | full torch (~3 GB) + both models below |

Models downloaded once, cached locally by Hugging Face:

- Emotion: [`bhadresh-savani/distilbert-base-uncased-emotion`](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) (~255 MB, 6 emotion classes)
- Narrative: [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small) (~340 MB)

Names and sizes are announced **before** any install/download starts, and anything missing is installed before analysis runs — never mid-report. If an install or download fails, the tool degrades to basic analysis with a hint instead of freezing.

Non-interactive runs (piped output, CI) never prompt: set `CHAT_ANALYZER_TIER=1|2|3`, otherwise tier 1 runs silently and prints a single hint line afterward. The legacy `CHAT_ANALYZER_FORCE_NLP` flag (`0` → tier 1, `1` → tier 3) still works; `CHAT_ANALYZER_TIER` takes precedence.

> **Windows deep-path warning:** before offering multi-GB `[nlp]` downloads, the tool checks your active path depth against MAX_PATH and warns/refuses if the venv sits too deep. Fix it by enabling Windows' LongPathsEnabled registry flag, or create a short-path venv with `scripts/make_nlp_env.ps1`. Power users can override the heuristic with `CHAT_ANALYZER_ALLOW_LONG_PATH=1`.

## Large chats

Three mechanisms keep hundred-thousand-message exports tractable:

- **Deterministic stratified sampling.** When a chat exceeds `CHAT_ANALYZER_EMOTION_SAMPLE` (default 50,000), emotion scoring offers Option C sampling: participant-stratified largest-remainder seats across ~50 time buckets per sender with `random_state=42`, so results are reproducible. Sampled results are explicitly labelled *"based on a sample of N of M messages"*. On a terminal you're asked `Sample? [y/N]` — answering No means exact scoring. Piped/non-interactive runs auto-sample. Set the cap to `0`/`off` to disable sampling entirely.
- **Parallel exact scoring.** Below the cap (or when you decline sampling), unique texts are deduplicated and scored through a process pool above 20,000 unique texts. Workers default to 3 (max 8, tunable via `CHAT_ANALYZER_EMOTION_WORKERS`; values below 2 go sequential). Batch sizes are truncation-safe with bounded chunk counts, a worker failure falls back to partial-rescue rather than losing the run, and results are written back vectorized.
- **INT8 quantization — strictly opt-in.** Setting `CHAT_ANALYZER_EMOTION_QUANT=1/on/true/yes` enables dynamic INT8 quantization of the emotion model for roughly 1.7x faster inference. This is **off by default**: quantization measurably distorts scores, so accuracy-first fp32 remains the default behavior.

## Performance

Measured numbers come from `scripts/benchmark.py`: each `(input, tier)` pair runs in a fresh subprocess through the real user-facing pipeline with the result cache forced off.

### Final authoritative snapshot (2026-09-25)

Current tier-1 (no-NLP) figures, captured in a single locked run of the real user-facing pipeline (parse → insights → report) on the 424,826-message WhatsApp export with the result cache forced off, so every stage reconciles against the same thermal state:

| Metric | Final | Original baseline | Speedup |
| --- | --- | --- | --- |
| Full pipeline | **64.3 s** | 243.7 s | **3.79×** |
| Parsing chat | **3.1 s** | 128.5 s | **41×** |
| Computing insights | **54.0 s** | 198.0 s | **3.7×** |
| Sentiment (VADER) | **37.6 s** | 266.5 s | **7.1×** |
| Throughput | **6,607 messages/s** | — | — |
| Peak memory (insights) | **3.41 GB** | 5.7 GB | −40% |
| `strptime()` calls per parse | **0** | 3,823,434 | — |

> **Benchmark caveat:** the CPU thermally throttles (i7-1255U, ~1.7 GHz under load), so cross-session absolute timings can spread roughly ±40%. Treat this single-run snapshot as the authoritative figure; re-run `scripts/benchmark.py` for current numbers.

Historical developer-reported figures (predate the final snapshot; tier-3 NLP/quant, not re-measured here):

- Large-chat pipeline improved 2401 s → 758 s in commit `8122df9` (2026-08-22 performance overhaul); the current no-NLP pipeline is 64.3 s
- ~1084 s (~18 min) re-measured fp32 full-exact tier-3 run on the 424k-message chat
- ~12 min on the same chat with `CHAT_ANALYZER_EMOTION_QUANT=1` (int8), with ~74.5% agreement between int8 and fp32 dominant labels on a stratified probe sample
- ~34 s end-to-end small-chat tier-3 run

## Privacy & honest reporting

- Everything runs locally — parsing, analysis, and model inference. Nothing is uploaded, and there are no accounts or telemetry.
- The optional result cache stores **derived scalars only**, never raw chat text.
- Report filenames are sanitized, and chat content is treated as untrusted input (Jinja2 autoescaping is on).
- Narrative outputs — both the Tier A heuristics and the Tier B generated paragraph — always carry speculative/speculative-inference disclaimers.
- Sampled emotion results are labelled as such; skipped/system lines are counted honestly instead of being silently dropped.

## Configuration

All knobs are environment variables; every one is optional.

| Variable | Default | Effect |
| --- | --- | --- |
| `CHAT_ANALYZER_TIER` | unset | Non-interactive NLP tier override: `1`, `2`, or `3`. Wins over `CHAT_ANALYZER_FORCE_NLP`. Interactive runs ignore it and show the menu. |
| `CHAT_ANALYZER_FORCE_NLP` | unset | Legacy automation flag: `0` → tier 1, `1` → tier 3. Ignored when `CHAT_ANALYZER_TIER` is set. |
| `CHAT_ANALYZER_EMOTION_SAMPLE` | `50000` | Emotion-scoring cap above which stratified sampling is offered. `0`/`off` disables sampling (always exact). |
| `CHAT_ANALYZER_EMOTION_WORKERS` | `3` | Parallel workers for exact emotion scoring (>20,000 unique texts); capped at 8. Values `<2` force sequential scoring. |
| `CHAT_ANALYZER_EMOTION_QUANT` | off | **Opt-in** INT8 dynamic quantization (`1`/`on`/`true`/`yes`). Faster inference, distorted scores — off by default. |
| `CHAT_ANALYZER_RESULT_CACHE` | off | Opt-in result cache. On-values (`1`/`on`/`true`/`yes`) enable it with the default location; any other value enables it **and** overrides the cache directory. |
| `CHAT_ANALYZER_NO_OPEN` | unset | Set to `1` to stop the report auto-opening in your browser. |
| `CHAT_ANALYZER_ALLOW_LONG_PATH` | unset | Set to `1` to bypass the Windows MAX_PATH guard heuristic before `[nlp]` downloads. |

### Result cache details

When enabled via `CHAT_ANALYZER_RESULT_CACHE`, pipeline results are keyed by the SHA-256 of the file bytes plus a config signature (schema version, app version, NLP on/off, sample cap, worker count, chosen zip transcripts). Cache files live **outside the repo** — `%LOCALAPPDATA%\chat-analyzer\cache` on Windows, `~/.cache/chat-analyzer` elsewhere — and are loaded as JSON only, written atomically, pruned after 30 days, and self-healed if corrupt. Even on a cache hit the HTML report is regenerated fresh.

## Development

```bash
pip install -e ".[dev]"
python -m ruff check src tests
python -m pytest              # full suite; add -m "not slow" for the fast subset
```

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for local setup, conventions, and CI jobs, and [docs/TESTING.md](docs/TESTING.md) for the test layout and golden-parity harness.

## Documentation

- [Getting started](docs/GETTING-STARTED.md) — prerequisites, install walkthrough, first run, troubleshooting
- [Architecture](docs/ARCHITECTURE.md) — pipeline stages and module tour
- [Configuration](docs/CONFIGURATION.md) — every env var and knob in depth
- [Development](docs/DEVELOPMENT.md) — local setup, code style, branch/PR conventions
- [Testing](docs/TESTING.md) — suite structure, markers, coverage

## License

No license has been declared for this project yet — there is no `LICENSE` file at the repository root and no license field in `pyproject.toml` as of v0.1.0. Vendored third-party assets ship under their own licenses (`src/chat_analyzer/assets/LICENSE.echarts`, `src/chat_analyzer/assets/LICENSE.echarts-gl`).

<!-- VERIFY: license status — confirm intended licensing before publishing; currently undeclared in pyproject.toml and no root LICENSE file -->
