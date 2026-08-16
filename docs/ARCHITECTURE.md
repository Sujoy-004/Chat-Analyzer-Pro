<!-- generated-by: gsd-doc-writer -->
# Architecture — chat-analyzer-pro

**Version documented:** 0.1.0  
**Package:** `chat-analyzer-pro` (`src/` layout, package `chat_analyzer`)  
**Runtime:** Python >= 3.11

## System overview

`chat-analyzer` is a pip-installable CLI tool that turns a WhatsApp `.txt`,
Telegram `.json`, or `.zip` chat export into terminal insights plus a single
self-contained HTML report. One command — `chat-analyzer <chat_file>` (the
single command on the typer app is collapsed into the root — there is no
`analyze` subcommand; calling the console script with a positional path runs
it directly, and calling it with no argument runs an interactive re-prompt
loop, while `--version`/`--help` stay available).

The architecture is a **layered pipeline** with a strict data contract boundary:

```
chat export file
      │
      ▼
┌────────────────────────────────────────────────────────────────────┐
│  PARSERS (chat_analyzer/parser/)                                    │
│  WhatsAppParser .txt  ·  telegram_parser .json                      │
│  produce (rows: list[dict], counts: dict) — never a DataFrame;      │
│  .zip → cli/zip_input.py (pick → merged rows/counts, 4-tuple)       │
└───────────────────────────────┬────────────────────────────────────┘
                                │ ParseReport (dataclass)
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  INGEST (chat_analyzer/ingest/)                                     │
│  messages_to_dataframe() → canonical DataFrame                      │
│  (datetime / timestamp / date / hour / sender / message /           │
│   message_length / source / uid)  — tz-naive UTC                    │
└───────────────────────────────┬────────────────────────────────────┘
                                │ canonical DataFrame
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  SAMPLE DECISION (Option C — nlp_on only)                           │
│  n_scorable > CHAT_ANALYZER_EMOTION_SAMPLE cap (default 50_000)?    │
│  tty prompt (default NO = exact) · off-tty AUTO-SAMPLE              │
└───────────────────────────────┬────────────────────────────────────┘
                                │ effective sample cap (rides cache key)
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  RESULT CACHE (cli/result_cache.py — opt-in)                        │
│  key = sha256(file bytes) + {schema, app_version, nlp_on,           │
│        sample_cap, emotion_workers, chosen_transcripts}             │
│  hit → "[INFO] Loaded analysis from cache" → early return           │
└───────────────────────────────┬────────────────────────────────────┘
                                │ miss → continue
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  ALWAYS-ON ANALYSIS (pandas/numpy/networkx/matplotlib only)         │
│  ChatEDA (summary/volume/dynamics/content)                          │
│  add_sentiment_analysis (VADER, consensus)                          │
│  analyze_relationship_health → health dict                           │
│  analyze_network → network dict                                      │
│  ChatVisualizer → matplotlib figures                                  │
│  analyze_narrative → Tier A heuristic narrative (always-on)         │
└───────────────────┬─────────────────────────────────────┬──────────┘
                    │                                     │
             [nlp] gate OFF                        [nlp] gate ON (probe)
                    │                                     │
                    ▼                                     ▼
┌──────────────────────────────┐   ┌──────────────────────────────────────┐
│ emotion stays None;           │   │  EmotionAnalyzer (distilbert emotion)│
│ report renders "unavailable"  │   │    sampled ≤ cap or exact scoring,   │
└──────────────────────────────┘   │    unique-text dedupe + process pool  │
                                   │    above 20_000 unique texts          │
                                   │  → emotion dict + emotion chart +     │
                                   │    quarterly means → emotion-quarterly│
                                   │    ECharts timeline                   │
                                   │  Tier B narrative (flan-t5-small over  │
                                   │    compact ASCII digest)               │
                                   └──────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  ADAPTER (chat_analyzer/cli/adapters.py)                             │
│  adapt(…) → AnalysisResults TypedDict                                │
│  extracts only serializable scalars (no DataFrames / DiGraph leak)   │
│  a cache miss is stored back best-effort before return              │
└───────────────────────────────┬────────────────────────────────────┘
                                │ AnalysisResults
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  REPORTING OUTPUTS                                                │
│  terminal: "Messages: N" line (main.py), ASCII summary panel       │
│            (render.py, rich), stage narration (pipeline.py)         │
│  file:     <stem>_report.html in cwd (report_html.py, jinja2,      │
│            autoescape) — ALWAYS regenerated, even on a cache hit;   │
│            base64 PNG data URIs + interactive ECharts specs         │
│            (cli/chart_json.py, incl. emotion-quarterly), vendored   │
│            JS bundles inlined from assets/ (importlib.resources)    │
│  browser:  best-effort auto-open (CHAT_ANALYZER_NO_OPEN=1 opt-out) │
└────────────────────────────────────────────────────────────────────┘
```

A `.zip` export may contain multiple transcripts (`.txt` = WhatsApp,
`.json` = Telegram); `cli/zip_input.py` lets the user pick which to
analyze (all on a non-tty run) and merges their rows + counts into one
`ParseReport`. `parse_zip_with_report()` returns a 4-tuple —
`(rows, counts, source, chosen_names)` — where `chosen_names` is the
sorted list of chosen transcript member names that rides in the
result-cache key, so a different selection on the same zip invalidates
the cache. The zip's real media *files* are counted once at the
archive level and folded into the report's "Media messages" stat.

## Module inventory

Every module that exists in `src/chat_analyzer/`, with its responsibility:

| Module | Responsibility |
|---|---|
| `__init__.py` | Package metadata (`__version__ = "0.1.0"`), `__all__`, package-level logging `NullHandler` |
| `__main__.py` | `python -m chat_analyzer` entry — runs the typer app directly |
| `cli/__init__.py` | Exposes `app` (from `cli.main`) |
| `cli/main.py` | The typer app: the single command is the root — no `analyze` subcommand exists — plus the interactive re-prompt loop, `--version` eager callback, `_friendly_error` classification, the always-on 3-option NLP tier menu (PH2) with `CHAT_ANALYZER_TIER` env resolution, exit codes 0/1 |
| `cli/pipeline.py` | The single orchestration path — `run_pipeline(path, console, nlp_enabled)` — parse → canonical df → Option C sample decision → result-cache lookup → insights → charts → gated emotion/quarterly/narrative → `AnalysisResults`, with `stage()`/`stage_status()` narration, `_safe_chart()` degradation, and best-effort cache store on a miss |
| `cli/contracts.py` | `ParseReport` dataclass + `AnalysisResults` TypedDict — the pipeline's single output contract; core modules never import it |
| `cli/adapters.py` | `adapt(...)` — the ONLY place that knows each analysis module's internal dict shape; extracts serializable scalars; `build_insights()` generates the narrative lead-in sentences |
| `cli/result_cache.py` | Opt-in repeat-run result cache (04-08): sha256 of the input file bytes + config signature → `<key>.json` outside the repo; recursive numpy/date-key sanitizer; json-only load that self-heals corrupt files; atomic store + 30-day TTL prune; never raises |
| `cli/render.py` | Terminal end-of-run rendering: skipped/system notes, rich ASCII "Summary" panel, Option C "sample of N of M" label (no charts in the terminal) |
| `cli/report_html.py` | Single-file HTML report: jinja2 autoescape template, sanitized `<stem>_report.html` written to cwd, eight chart slots (incl. `emotion-quarterly`), best-effort browser open (`CHAT_ANALYZER_NO_OPEN=1`) |
| `cli/chart_json.py` | `build_chart_specs()` (timeline/activity/participants/sentiment/health/network) + `build_emotion_spec()` + `build_emotion_timeline_spec()` (quarterly line) — ECharts option dicts behind the `charts_json` contract; strictly JSON-serializable, never raises |
| `cli/zip_input.py` | `.zip` export support: transcript discovery, interactive transcript selection, zip-level media-file counting, merged parse; `parse_zip_with_report()` returns `(rows, counts, source, chosen_names)` for the cache key |
| `cli/nlp_gate.py` | NLP availability + tier resolution: `nlp_status()` (READY/OUTDATED/MISSING), locked model constants, `model_cached()`, guarded `install_nlp()`/`update_nlp()`/`download_models()`, env knobs (`CHAT_ANALYZER_EMOTION_SAMPLE`, `CHAT_ANALYZER_EMOTION_WORKERS`, `CHAT_ANALYZER_RESULT_CACHE`), Windows `MAX_PATH` guard |
| `ingest/__init__.py` | Package marker |
| `ingest/ingestion.py` | `messages_to_dataframe()` (canonical df builder), `normalize_message()` contract, legacy `process_uploaded_file()` library API, WhatsApp regex/text + JSON message parse helpers, optional OCR/PDF dependency gates (PIL, pytesseract, pdfplumber, pdf2image) |
| `parser/__init__.py` | Package marker |
| `parser/whatsapp_parser.py` | `WhatsAppParser` class — strict line regex parsing + `parse_file_with_report()` returning `(rows, counts)`; system-line classification, honest skip counters; legacy `parse_file()`/`parse_whatsapp_chat()` DataFrame paths |
| `parser/telegram_parser.py` | `parse_telegram_chat_with_report()` (file path or URL) + `parse_telegram_chat()` DataFrame path — both Telegram JSON shapes, entity-array text joining, service-message filtering, tz→naive UTC |
| `analysis/__init__.py` | Package marker |
| `analysis/eda.py` | `ChatEDA` — volume, dynamics (avg response time, balance ratio), content (word/emoji frequency), comprehensive summary |
| `analysis/sentiment.py` | Multi-engine sentiment: VADER (`vaderSentiment`), textblob/transformers gated behind `*_AVAILABLE` flags; `add_sentiment_analysis()` + `get_sentiment_summary()`; consensus majority-vote |
| `analysis/emotion.py` | `EmotionAnalyzer` — 6-class emotion via HF pipeline (`distilbert-base-uncased-emotion`) with rule-based fallback; Option C deterministic stratified sampling (`emotion_scored` column + `attrs["emotion_sample"]`); unique-text dedupe + parallel exact scoring above 20_000 unique texts (spawn-safe module-level `_score_text_chunk`, torch thread clamp); `get_emotion_quarterly()` per-quarter means; `emotion_figure()` returns a chart for base64 embedding |
| `analysis/narrative.py` | Tier A heuristic narrative — `analyze_narrative()`; hedged observations (arc/driver/reciprocity/engagement) from pandas+re only; never imports CLI |
| `analysis/relationship_health.py` | `analyze_relationship_health()` — starters, initiator ratio, response patterns, dominance, weighted health score + gamification (friendship index, streaks, emoji personality, milestones, rolling health) |
| `analysis/network_graph.py` | `analyze_network()` + `network_figure()` — NetworkX directed interaction graph, centrality metrics, community detection |
| `analysis/summarizer.py` | `ConversationSummarizer` — T5 abstractive summarization (direct `T5ForConditionalGeneration.generate`), group-dynamics extras; Tier B backend |
| `utils/__init__.py` | Re-exports `clean_messages`, `extract_emojis`, `preprocess_text`, `ChatVisualizer` |
| `utils/preprocessing.py` | Text helpers: `preprocess_text`, `clean_messages`, `extract_emojis`, URLs, tokenization, whitespace normalization |
| `utils/visualization.py` | `ChatVisualizer` chart library (timeline, heatmap, wordcloud, sentiment, activity, response time, relationship-health trend, summary dashboard) + module-level `quick_*` wrappers |
| `reporting/__init__.py` | Re-exports PDF generator |
| `reporting/pdf_report.py` | `ChatAnalysisPDFGenerator` (ReportLab) + `generate_chat_analysis_pdf()` — **shipped but NOT wired into the CLI** (deferred to v2) |
| `reporting/weekly_digest.py` | `WeeklyDigestBot` (SMTP + Telegram bot digest) — **shipped but NOT wired into the CLI** (deferred to v2) |

## Data flow / message schema

### Parser boundary (rows + counts)

Each parser returns `(rows: list[dict], counts: dict)`; `counts` carries
`total_lines`, `parsed_messages`, `skipped_lines`, `system_messages`
(`whatsapp_parser.parse_file_with_report`, `telegram_parser.parse_telegram_chat_with_report`,
`zip_input.parse_zip_with_report`).

Parser-level row shape (whatsapp, `parse_line_strict`):

```python
{
  'datetime': datetime,   # parsed strictly, no fabricated fallback
  'sender': str,
  'message': str,
  'message_length': int,
  'type': 'message',
  'date': date, 'time': time, 'hour': int,
  'day_of_week': str, 'word_count': int,
}
```

Telegram rows add an optional `message_id` and normalize datetimes to
naive UTC (`_to_naive_utc`). System/service rows and unparseable lines are
counted, never dropped silently.

### Canonical DataFrame (the shared analysis input)

`messages_to_dataframe(messages)` in `ingest/ingestion.py` is the ONLY
dict→df builder. Every row carries:

```
datetime  timestamp  date  hour  sender  message  message_length  source  uid
```

- `datetime` is always normalized to **tz-naive UTC** (`_to_naive_utc`).
- `timestamp` is an alias column the chart visualizer requires.
- Rows with no parseable datetime are dropped (the caller owns skip counting).
- Nothing is fabricated (`datetime` and `message` fallbacks only).

### Report contract

`AnalysisResults` (in `cli/contracts.py`) is the TypedDict consumed by
`render.py` and `report_html.py` — still a plain dict, safe for the Jinja2
template:

| Key | Content |
|---|---|
| `source` | `"whatsapp"` / `"telegram"` / `"mixed"` |
| `parse` | `{total_lines, parsed_messages, skipped_lines, system_messages}` |
| `stats` | total, participant count/list, date range, duration, busiest day, peak hour, avg response, media count |
| `participants` | per-sender `{messages, avg_message_length, share_pct}`, sorted desc |
| `content` | top-15 words/emojis, total/unique word counts |
| `sentiment` | distribution, avg compound, by-sender, daily avg |
| `health` | **always-on** scalars (overall score, grade, components, initiator balance, avg response minutes, response balance, dominance) |
| `network` | **always-on** scalars (nodes/edges/density/reciprocity, strongest connections, key participants, subgroup count) |
| `emotion` | **None when the NLP gate is OFF** (silent degrade); when on: `{distribution, dominant, average_scores, sample, quarterly}` — `sample` carries the Option C metadata (`scored`/`total`/`cap`) so the report/terminal can label "based on a sample of N of M messages"; `quarterly` carries the per-quarter mean rows |
| `narrative` | always present — Tier A observations; Tier B `narrative_summary` when generated |
| `charts` | `{name: "data:image/png;base64,…"}` — six always, `"emotion"` when NLP on |
| `charts_json` | `{name: ECharts option dict}` — interactive specs for the same charts (`"emotion"` + `"emotion-quarterly"` when NLP on); a chart with no buildable spec renders its PNG from `charts` instead |
| `insights` | up to 11 narrative lead-in sentences |
| `report_path` | filled by `main.py` after the HTML write — forced to `""` on a cache load so the report is always regenerated |

## Key design decisions

### CLI-first (typer), web-app patterns removed

The previous Streamlit app (`app/streamlit_app.py`) and its runtime
`exec()` of downloaded modules was deleted entirely. The package ships only
the `chat_analyzer` package (`chat_analyzer` console script → `chat_analyzer.cli:app`
in `pyproject.toml`, plus the `python -m chat_analyzer` fallback). There is
no `unsafe_allow_html`, no remote code fetch, no web server.

- Interactive (no path): re-prompt loop; `ValueError` on a bad file re-prompts
  on a real tty.
- Positional: `chat-analyzer <path>` exits 0 on success, 1 with a friendly
  line (every failure path ends in `typer.Exit(code=1) from None`, never a
  traceback).
- `--version`: eager callback reading `importlib.metadata.version('chat-analyzer-pro')`
  (typer 0.27 has no built-in flag).

### Lazy heavy imports

`--help`/`--version` stay instant: only typer + stdlib are imported at the
module level in `cli/`. The analysis modules and matplotlib are imported
*inside* `run_pipeline`/`_analyze_path`, and matplotlib anchors the **Agg
backend before any pyplot import** (`matplotlib.use("Agg")` at the top of the
pipeline).

### The `[nlp]` gate — silent availability probe

The base install **must not pull torch/transformers/streamlit** — heavy deps
live only behind the optional `[nlp]` extra (`torch>=2.0`,
`transformers>=4.30,<5.15`, `sentencepiece>=0.1.99`) — and the pipeline makes
no assumption the models are present:

- `nlp_gate.nlp_available(MODEL_ID)` is a **pure importability probe**
  (`import torch, transformers`) — never raises, never prompts. It is
  consumed up front by `main()`'s tier resolution: choosing tier 2/3
  resolves `nlp_status()` (READY / OUTDATED / MISSING), and the forced
  `nlp_enabled` flag is passed into `pipeline.run_pipeline()`. Direct/test
  callers that pass `nlp_enabled=None` still get the auto-detect probe.
- `CHAT_ANALYZER_FORCE_NLP=0|1` env var forces either branch deterministically
  in tests (a dev box may have transformers installed but no cached weights);
  `CHAT_ANALYZER_TIER=1|2|3` is the non-interactive tier override (legacy
  `CHAT_ANALYZER_FORCE_NLP` maps 0→1 and 1→3).
- Model **weights are not required up front** — they download on first use;
  `model_cached()` announces when a download is about to happen and the model
  and size are printed **before** any `from_pretrained` call (announce-then-
  construct D-05 rule). Choosing tier 2/3 makes the tier ready *before*
  analyzing: MISSING runs `install_nlp(cpu_only)` + `download_models()`,
  OUTDATED offers "Update now / Go with current" (`update_nlp()`), READY
  proceeds — so both model weight sets are in the HF cache before the
  pipeline's first model use.
- Gated stages degrade, they never crash: `EmotionAnalyzer` falls back to a
  rule-based keyword classifier when the model load fails; the Tier B narrative
  degrades to the Tier A lead when its model fails.

When the gate is OFF, the emotion stage is **skipped silently** — `emotion`
stays `None` in the contract and the report renders its `unavailable` note with
an install hint. When the gate is ON, emotion
(`distilbert-base-uncased-emotion`, ~255 MB) and the Tier B narrative
(`google/flan-t5-small`, ~340 MB) both run locally.

Every interactive run (positional or in the no-arg loop) shows the
always-on **3-option NLP tier menu** — 1 = without NLP, 2 = minimal
(CPU-only torch, ~0.6 GB), 3 = full-fledged (~3 GB) — regardless of the
current install state; non-interactive runs never prompt (env override,
else silent tier 1 plus a one-line hint). `nlp_gate.install_nlp(cpu_only)`
re-installs the already-declared extras via a guarded subprocess pip
(nothing new enters the dependency graph) and raises `RuntimeError` on
failure so the run degrades to basic analysis. On Windows, a `MAX_PATH`
guard (WinError 206 with torch wheels) warns before any multi-GB download and
points to `scripts/make_nlp_env.ps1` (creates a short-path venv in `%TEMP%`);
`CHAT_ANALYZER_ALLOW_LONG_PATH=1` overrides the heuristic check.

### Sampled vs exact emotion scoring (Option C) + parallel workers

The emotion stage scales two ways for large chats — by **sampling** (fewer
messages scored) and by **parallelizing** exact scoring (score faster, same
result) — and both knobs live in `nlp_gate`/`analysis/emotion.py`, not in
the CLI:

- **Sampled path (Option C).** `CHAT_ANALYZER_EMOTION_SAMPLE` sets the cap
  (default `50000`; `0`/`off`/`false` disables sampling = always exact).
  The cap is resolved **once** in `run_pipeline` right after the parse
  (before the cache lookup): when `EmotionAnalyzer.n_scorable(df)` exceeds
  it, an interactive terminal is prompted (default NO = exact) and piped/CI
  runs **auto-sample**. The sample itself is deterministic and
  (participant, time)-stratified — largest-remainder seat allocation across
  senders, ~50 time buckets per sender, `random_state=42` everywhere — so
  the same file + cap always yields the same sample. The returned frame
  gains an `emotion_scored` boolean column and
  `df.attrs["emotion_sample"]` metadata; the pipeline computes its summary
  over the **scored sample rows only** and the terminal/report label the
  result "based on a sample of N of M messages". At/below the cap (or with
  `sample_cap=None`) the behavior is byte-for-byte the exact path.
- **Parallel exact scoring.** Every unique text is scored exactly once
  (dedupe via `dict.fromkeys` — emotion scoring is a pure function of the
  text) and mapped back to all rows, so the output is identical to scoring
  every message. Above `_EMOTION_PARALLEL_THRESHOLD` (20_000 unique texts),
  with a genuine transformers pipeline and
  `CHAT_ANALYZER_EMOTION_WORKERS` ≥ 2 (default 3, capped at 8 — each worker
  loads its own ~255 MB model copy), a `ProcessPoolExecutor` map-reduces
  contiguous fixed-order chunks via the module-level, spawn-safe
  `_score_text_chunk(texts, model_name, batch_size, num_threads)` — JSON-able
  args only, the pipeline is built **inside** the child, torch threads are
  clamped to `max(1, cpu_count // workers)`, and
  `TOKENIZERS_PARALLELISM=false` is set before the transformers import.
  Any worker failure degrades to the identical sequential `_score_batch`
  (byte-parity by construction). The pool lives in the emotion analysis
  module, never the CLI.
- **Quarterly aggregation.** `get_emotion_quarterly(df_emo)` computes the
  per-quarter MEAN of the six emotion scores (oldest first, honors the
  `emotion_scored` column in sampled mode); the pipeline attaches it as
  `emotion_summary["quarterly"]`, `chart_json.build_emotion_timeline_spec()`
  turns it into the `emotion-quarterly` ECharts line spec (one smooth line
  per emotion, slider+inside `dataZoom`), and `report_html.py` renders the
  chart when the spec exists.

### Opt-in result cache (keyed by file hash)

`CHAT_ANALYZER_RESULT_CACHE` (default **OFF**) turns repeat runs of the same
chat file into seconds: the whole post-`adapt()` `AnalysisResults` payload is
persisted and served on a hit, so the compute and NLP/narrative stages never
re-run.

- **Key** = `sha256` of the input file's raw bytes (1 MiB chunks; a `touch`
  still hits; zip archives hash the whole container, media included) folded
  with a config signature via `json.dumps(sort_keys=True)`:
  `{schema: RESULT_CACHE_SCHEMA, app_version, nlp_on, sample_cap,
  emotion_workers, chosen_transcripts}`. The **effective** sample cap (the
  tty y/N answer collapsed in) rides in the key, and the sample prompt is
  re-asked on hits — so file edits, tier changes, cap choices, worker counts,
  zip transcript picks, app upgrades, and schema bumps all invalidate the
  key by construction. Key filenames are hex digests only — no user input
  ever enters a path.
- **Storage** lives OUTSIDE the repo (privacy lock): `%LOCALAPPDATA%\chat-analyzer\cache`
  on Windows, `~/.cache/chat-analyzer` elsewhere, or any explicit path value.
  The payload is derived data only (sender names, top words, charts) — raw
  chat text never enters it. Writes are atomic (same-dir tempfile +
  `os.replace`), pruned by a **30-day TTL** on every store, and pass through
  a recursive `_sanitize` (numpy scalars → `.item()`, ndarrays → `tolist()`,
  NaN/±Inf → `None`, non-serializable dict keys — the verified
  `sentiment.daily_avg` `datetime.date` leak — → `str()`).
- **Load** is json-only (never pickle/eval); a corrupt file is self-healed
  (unlinked), a schema-mismatch file is kept, and `report_path` is forced to
  `""` so the HTML report is **always regenerated fresh** in the cwd.
  A hit prints `[INFO] Loaded analysis from cache` and returns early; a miss
  stores best-effort after `adapt()` (never raises, and the NLP path hints
  once when the cache is disabled).

### Tier A narrative vs Tier B generative

- **Tier A (always on, pandas + re only)** — `analysis/narrative.py`
  `analyze_narrative(df)` turns basic signals into hedged, clearly-disclaimed
  observations (kinds `arc`, `driver`, `reciprocity`, `engagement`). Every
  observation starts with a hedge ("Possibly", "May", …) and confidence comes
  from a conservative table (high: n≥30 & |signal|≥0.5; medium: n≥10 and
  |signal|≥0.3; else low). With the gate OFF it reports
  `status = {nlp_available: False, tier_b_generated: False}`.
- **Tier B (only when the gate is ON)** — `pipeline._tier_b_digest()` builds a
  compact **ASCII signal digest from the Tier A observations** (never raw chat
  text, keeping the model input tiny and private), and a `ConversationSummarizer`
  instance on `google/flan-t5-small` (max 90 / min 20 tokens) summarizes that
  digest into `narrative["narrative_summary"]`, with
  `tier_b_generated = True`. Any failure keeps the Tier A observations — the
  run never crashes. The report's narrative tab lead reflects the real gate
  state ("Tier B enabled (local generative model) — statistical inference is
  speculative" vs "Tier A (statistical inference, speculative)").

### Single-file HTML report (no external assets)

`cli/report_html.py` always produces `./<sanitized-stem>_report.html` in the
current working directory (no `--output`/`--no-report` flags in v1, per the
decision log):

- The Jinja2 template is an inline module constant; `autoescape` is
  **explicitly enabled** via `select_autoescape(["html","xml"])` because chat
  content is untrusted input (plain jinja2 defaults to `False`).
- Charts are **interactive ECharts** by default — hover tooltips, `dataZoom`
  (inside + slider) zoom-to-detail on the timeline/sentiment/health lines,
  per-cell heatmap tooltips, and a true **3D network** (`scatter3D` +
  `lines3D` on `grid3D`, drag-to-rotate with `autoRotate`). WebGL-less
  browsers get the static PNG network instead.
- `echarts.min.js` (~1 MB) and `echarts-gl.min.js` (~0.6 MB) are **vendored**
  in `src/chat_analyzer/assets/` (shipped via the wheel's `artifacts` glob)
  and **inlined** at render time with `importlib.resources` — the report stays
  a single offline file, no CDN, and grows to ~1.6–2 MB (expected). A missing
  bundle degrades to `""` and the report still renders the PNGs.
- Chart images are `data:image/png;base64,…` data URIs (the **fallback** when
  a chart has no interactive spec), **validated at the boundary** (only URIs
  with the `data:image/png;base64,` prefix reach the template — no `|safe`
  filter needed).
- Interactive specs come from `cli/chart_json.py` (the `charts_json`
  contract) and are injected via Jinja `|tojson` (escapes `<`, `&`, etc.);
  every inlined bundle is scrubbed for `</script`/`<!--` so chat content can
  never escape the inline `<script>`. Boundary validation
  (`_validate_charts_json`) drops any spec that cannot JSON round-trip, which
  re-triggers its PNG fallback.
- Chart encoding failures degrade to an empty string (`_safe_chart`) — a
  crash in one plot can never kill the report. Spec building never raises
  either (`build_chart_specs` logs and omits a failing chart, which falls
  back to PNG).
- Filenames are sanitized with an explicit invalid-char regex
  (`[<>:"/\\|?*\x00-\x1f\x7f]`), falling back to `chat_analysis` when empty.
- Written with UTF-8 (explicit `utf-8` encoding, never platform default).
- Browser auto-open is best-effort and never crashes; `CHAT_ANALYZER_NO_OPEN=1`
  opts out (used by tests).

### Contract boundary & data-exchange rules

- **Core never imports CLI contracts.** Parsers and analysis modules return
  plain dicts/DataFrames; `cli/contracts.py` is where the boundary lives and
  only the CLI layer imports it. So the analysis core stays CLI-agnostic
  (notebook/tests can use it directly).
- **One adapter knows everything.** `cli/adapters.py::adapt()` is the only
  place that understands each module's internal shape; every access is a
  defensive `.get()` so empty edge-case dicts (e.g., a 1-message chat with
  no avg response time) never `KeyError` here.
- **Serializability is enforced.** The health module returns a
  `prepared_data` DataFrame and the network module a `networkx.DiGraph`; the
  adapter extracts **only serializable scalars** (scores, grades, counts,
  strongest connections) — the DataFrame/DiGraph never leak into
  `AnalysisResults` (Jinja2-consumed).
- **Always-on vs gated** — relationship health and network are always-on
  (pandas/numpy/networkx/matplotlib only); emotion is gated.
- **One "Messages: N" smoke token**: printed once by `main.py` after
  `run_pipeline` (never again in `pipeline.py`/`render.py`) — pinned by the
  Phase-1 smoke test.

### Windows / console hardening

- `main()` reconfigures `sys.stdout`/`sys.stderr` to UTF-8 with
  `errors="replace"` so cp1252 CMD consoles never crash the tool on emoji.
- Terminal output stays ASCII-only by design (no box-drawing glyphs beyond
  `+-|`), with soft-wrapped hint lines so a single hint stays one physical
  line on narrow piped consoles.
- Analysis-stage prints (`print(...)` in legacy energy modules) are captured
  with `contextlib.redirect_stdout(io.StringIO())` so loader spam never
  reaches the user; the package-level logging `NullHandler` keeps
  `logger.exception` noise silent.

## How heavy NLP runs vs the pandas-only fallback

| Aspect | pandas-only (base install, gate OFF) | NLP (with `[nlp]` extra, gate ON) |
|---|---|---|
| Sentiment | VADER via `vaderSentiment` (pinned `TRANSFORMERS_AVAILABLE = FALSE`), consensus over VADER label | same VADER path — CLI intentionally pins to VADER so pre-existing transformers in the env cannot trigger per-message HF inference |
| Emotion | Not available — report tab shows the install hint | `EmotionAnalyzer` (distilbert emotion, 6 classes) → `emotion` block + chart; sampled (Option C) or exact scoring with unique-text dedupe + a process pool above 20_000 unique texts; quarterly means → `emotion-quarterly` timeline |
| Narrative | Tier A heuristic observations always render | Tier A + Tier B generative paragraph (flan-t5-small over ASCII digest) |
| Health/Network | Always-on (pandas/networkx) | Always-on (identical — no extra deps) |
| Terminal behavior | runs basic, hints once at the end | models run locally, progress bar total = 4 |

Availability is decided by the tier selection — the interactive menu or
`CHAT_ANALYZER_TIER` — resolved through `nlp_gate.nlp_status()`
(READY / OUTDATED / MISSING) and forced into `run_pipeline` as
`nlp_enabled`; tiers 2/3 download any missing model weights at selection
time (`download_models()`), announced with the model + size before
construction.

## Tech stack

| Layer | Tool | Notes |
|---|---|---|
| Language | Python >= 3.11 | enforced in `pyproject.toml`, `requires-python = ">=3.11"` |
| Build/packaging | hatchling (wheel), src/ layout | `[project.scripts] chat-analyzer = "chat_analyzer.cli:app"` |
| CLI | typer, rich | typer app + rich Console/Progress/Status/Panel; ASCII-first |
| Data | pandas, numpy | canonical DataFrame everywhere |
| Charts | matplotlib (Agg backend), seaborn | figure-returning wrappers encoded to base64 PNG |
| Interactive charts | echarts + echarts-gl (vendored `assets/`, inlined) | option specs via `cli/chart_json.py`; PNG figures stay the fallback |
| Word cloud | wordcloud | lazy import inside `plot_wordcloud`; degrades to a text note |
| Sentiment | vaderSentiment | base install; consensus path |
| Graph | networkx | directed interaction graph, centrality, communities |
| HTML | jinja2 | autoescape enabled; inline template constant |
| HTTP | requests | Telegram JSON loading from a URL (parser) |
| PDF / images | reportlab, Pillow | `reporting/pdf_report.py` — shipped, not wired into CLI (v2) |
| NLP (optional) | torch, transformers, sentencepiece | `[nlp]` extra; lazy imports; parallel emotion scoring via stdlib `concurrent.futures` |
| Result cache | stdlib (hashlib, json, os, tempfile) | opt-in `cli/result_cache.py` — file-hash + config-signature key, 30-day TTL |
| Dev/test | pytest, pytest-cov, ruff | `[dev]` extra; CI-quality gates |
| Not shipped | plotext (dropped), Streamlit (deleted), plotly (ECharts used instead) | charts exist only in the HTML report |

## What's not implemented yet

Per the project state (`.planning/STATE.md`), these are **deferred to v2** and
not wired into the CLI in v0.1.0, even though the modules ship:

- PDF reporting — `reporting/pdf_report.py` exists and is importable but is
  **not invoked by the CLI pipeline** (only HTML reports are produced).
- Telegram-weekly digest — `reporting/weekly_digest.py` (`WeeklyDigestBot`
  SMTP/Telegram) is likewise unconnected to the CLI.
- Additional export formats (Instagram/Messenger/Discord) — not parsed.
- `--output`/`--no-report` flags — no flags in v1; report always written to cwd.