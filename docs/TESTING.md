<!-- generated-by: gsd-doc-writer -->
# Testing

Chat-Analyzer-Pro is tested with **pytest** (the dev extra in `pyproject.toml`
declares `pytest>=7.4`, `pytest-cov>=4.1`, and `ruff>=0.16.1`). The suite has
**330 tests across 31 files** — 304 fast tests plus 26 wall-clock-slow tests.
Wall-clock-slow tests (CLI subprocess spawns, full pipeline/report renders, and
the one real-model emotion spawn-parity smoke) are gated behind the `slow`
marker (`[tool.pytest.ini_options]` in `pyproject.toml`) so the fast suite
stays lean for every commit. The pytest config sets no `addopts`: plain
`python -m pytest` runs all 330 tests, and the fast-only selection is an
explicit `-m "not slow"`.

Tests exercise the **real shipped modules** — `chat_analyzer.parser.*`,
`chat_analyzer.analysis.*`, `chat_analyzer.cli.*`, etc. — not copies of the
logic. Only heavyweight external callables (transformers pipelines, T5 model
loads, `webbrowser`) are mocked, and mocks are shaped to produce the exact
output the real libraries return. Real-model inference happens in exactly one
place: the slow-marked emotion spawn-parity smoke in `test_emotion_parallel.py`
(D-17 — the fast suite must never hit the network or load weights).

## Test framework and setup

- **Runner:** pytest. The `slow` marker is registered in
  `[tool.pytest.ini_options]` of `pyproject.toml`; no `addopts` are set.
- **Legacy files:** four earlier files (`test_parser.py`, `test_analysis.py`,
  `test_reporting.py`, `test_end_to_end.py`) still use `unittest.TestCase`
  classes with module-level `run_*_tests()` runners and
  `if __name__ == '__main__': unittest.main()` blocks. pytest collects them
  normally — keep them compatible; new tests are plain pytest functions.
- **Package under test:** tests import `from chat_analyzer...` (the `src/`
  layout), so the package must be installed before running anything.

Install the test tooling (editable, with the dev extra):

```bash
pip install -e ".[dev]"
```

The `[nlp]` extra (`torch>=2.0`, `transformers>=4.30,<5.15`,
`sentencepiece>=0.1.99`) is needed only for the NLP-gated tests, which skip
cleanly when transformers is absent:

```bash
pip install -e ".[dev,nlp]"
```

## Running tests

All commands below ran from the repo root and were verified against the suite:

| Command | What it runs |
| --- | --- |
| `python -m pytest` | Full suite — 330 tests (304 fast + 26 slow). |
| `python -m pytest -m "not slow"` | Fast suite only — 304 tests (this is the CI default; CI adds `-q -p no:cacheprovider`). |
| `python -m pytest -m "slow"` | Slow suite only — 26 tests (CLI subprocess spawns, full pipeline + report renders, real-model emotion spawn-parity smoke). |
| `python -m pytest tests/test_phase2_telegram.py` | A single test file. |
| `python -m pytest tests/test_phase2_whatsapp.py::test_exact_fixture_counts` | A single test function. |
| `python -m pytest -k "not_a_chat or empty"` | Tests matching a keyword expression (the Telegram `test_not_a_chat_export_raises` parametrization is a good target). |
| `python -m ruff check src tests` | Lint gate used by CI (must pass before tests in the `slow` job). |

The slow suite needs real model weights before it can run: the emotion
spawn-parity smoke uses the cached
`bhadresh-savani/distilbert-base-uncased-emotion` checkpoint, and the tier-B
narrative paths use `google/flan-t5-small`. Those tests **skip** (they do not
fail) when the `[nlp]` extra or the model cache is missing, or when
`CHAT_ANALYZER_FORCE_NLP` is set — real-model inference is allowed only in slow
tests (D-17), and the fast suite must never touch the network or the HF Hub.

The four legacy unittest files also run standalone:

```bash
python tests/test_parser.py                 # unittest.main() entry point
python -m unittest tests.test_parser -v     # single legacy module
```

## Test data and fixtures

All file-backed fixtures live in either `data/sample_chats/` (real exports) or
`tests/fixtures/` (adversarial exports crafted for parse-report assertions),
plus the committed golden CSVs in `tests/golden/`. There is no `conftest.py`
and no shared fixture factory — most tests build inline data, and file I/O uses
`tmp_path` with UTF-8 encoding (`encoding="utf-8"` — the Windows cp1252 console
cannot encode the emoji-heavy fixtures).

| Path | Contents / purpose |
| --- | --- |
| `data/sample_chats/whatsapp_sample.txt` | Real-style WhatsApp export, **27 messages** (`KNOWN_WHATSAPP_COUNT = 27`), 2 senders over 3 days, emojis, multiline continuation, 2 `<Media omitted>` markers. Locked-in count used by CLI e2e tests (`"Messages: 27"`). |
| `data/sample_chats/telegram_sample.json` | 5-message Telegram export (bare Chat shape). CLI e2e asserts `"Messages: 5"`. |
| `tests/fixtures/whatsapp_system_skip.txt` | 7 lines: 3 parsed messages, 2 system lines (encryption notice + "Alice added Bob"), 1 bad-date line, 1 continuation line. Locked-in counts `{total_lines: 7, parsed_messages: 3, skipped_lines: 1, system_messages: 2}`. |
| `tests/fixtures/whatsapp_all_skipped.txt` | 3 lines, zero parseable — drives the friendly `"No messages could be parsed"` error path (exit 1, no traceback). |
| `tests/fixtures/telegram_full_export.json` | `chats.list[]` shape with entity-array text, a service message, a tz-aware `Z` row (id 3), a bad date, and a `forwarded` row → locked counts `{6, 3, 2, 1}` (the `+05:30` offset row lives in `telegram_bare_entity.json`, id 2). |
| `tests/fixtures/telegram_bare_entity.json` | Bare-Chat shape with entity-array text (`@team` + text) and a `+05:30` offset row → 2 parsed, 0 skipped. |
| `tests/golden/whatsapp_sample.csv`, `tests/golden/telegram_sample.csv` | Long-format `metric,value` golden CSVs captured on the author machine from a real tier-1 (`nlp_enabled=False`, no torch/transformers) pipeline run. The parity harness re-runs the live pipeline and asserts the same projection matches — floats with `pytest.approx(rel=1e-4, abs=1e-6)`, ints/strings exactly, and the metric-key set must match exactly. |

CLI-facing tests copy a sample into `tmp_path` and run with `cwd=tmp_path` so a
report run never writes into the repo tree.

## What the suite covers

| Area | Files | Coverage |
| --- | --- | --- |
| WhatsApp parser (`.txt`) | `tests/test_parser.py`, `tests/test_phase2_whatsapp.py` | Basic parsing, sender extraction, datetime parsing, emoji preservation, multiline joins, system-message classification (encryption notice, "added Bob"), strict date parsing with **no** `datetime.now()` fabrication, US 12h / EU 24h / iOS bracket / 4-digit-year formats, DD/MM-tried-first regression, empty-body messages, honest skip/system counters. |
| Telegram parser (`.json`) | `tests/test_parser.py`, `tests/test_phase2_telegram.py` | Bare-Chat + `chats.list[]` shapes, recursive entity-array text joins, service-message filtering, tz-aware → naive UTC normalization, honest malformed-drop counting, `ValueError("Not a Telegram chat export")` on empty/missing-key exports, `parse_telegram_chat` DataFrame contract. |
| Ingestion (canonical DataFrame) | `tests/test_phase2_builder.py` | `messages_to_dataframe` 9-column schema + defaults, tz-aware → naive UTC, Telegram full-ISO path, WhatsApp date+time path, unparseable rows dropped. |
| Sentiment (VADER) | `tests/test_analysis.py`, `tests/test_phase1_smoke.py`, `tests/test_perf_parity_sentiment.py` | Real VADER columns (`vader_compound`/`vader_sentiment`) on positive/negative/neutral fixtures, `[-1, 1]` score range, sentiment distribution summary; the transformers path pinned off (`TRANSFORMERS_AVAILABLE = False`); the dedupe + parallel `_score_vader_parallel` path is bit-identical to the per-message reference. |
| Emotion classification (NLP-gated) | `tests/test_analysis.py`, `tests/test_phase4_nlp.py` | Real `EmotionAnalyzer` with the transformers pipeline mocked — label/score shape, non-uniform dominant emotion, transformers 5.x nested-list compat, locked default model name; pipeline-level emotion block + `emotion` chart with the gate on/off. |
| Emotion sampling (Option C) | `tests/test_emotion_sampling.py` | Deterministic stratified-sample path in `EmotionAnalyzer` and the pipeline gate that drives it — all model callables mocked (D-17), mirroring `test_perf_parity_emotion.py`; report labels and sample stats agree. |
| Emotion parallel scoring | `tests/test_emotion_parallel.py` | `CHAT_ANALYZER_EMOTION_WORKERS` env contract (absent → 3, off-words/`<2` → 1, capped by cpu & 8, garbage → 3, never raises); the real-pipeline gate (a mocked classifier is not a `transformers.Pipeline`, so `_scoring_workers` returns 0 and mocked tests stay sequential); the `_EMOTION_PARALLEL_THRESHOLD` (20 000) boundary both directions plus exactly-at; **bounded** contiguous fixed-order chunking with a `_FakePool` (`_MIN_CHUNK_TEXTS` floor → several chunks per worker, the default floor collapses to one big chunk, and the per-worker torch-thread budget reaches every chunk); partial pool-failure rescue — completed chunks are kept and only the `BrokenProcessPool`-lost slices are re-scored sequentially; whole-pool degrade to sequential with identical output; an empty frame flows through the vectorized write-back as a noop; `TOKENIZERS_PARALLELISM=false` set before the transformers import in the worker; plus **one `@pytest.mark.slow` real-model spawn-parity smoke** (skips without the cached DistilBERT weights or when `CHAT_ANALYZER_FORCE_NLP` is set). |
| Quarterly emotion aggregation | `tests/test_emotion_quarterly.py` | Per-quarter MEAN of the six `emotion_*` columns, oldest quarter first, rounded to 6 decimals; degenerate/malformed frames (empty, None, missing `datetime`, bad columns) → `[]` without raising; sampled mode aggregates only `emotion_scored` rows; analyzer method wrapper delegation; `build_emotion_timeline_spec` chart contract (one line series per emotion in fixed order, yAxis 0–1, NaN → None, numpy scalars coerced, JSON-safe). |
| Result cache | `tests/test_result_cache.py` | Drives the **real `run_pipeline`** with mocked models: default OFF (no env var → no cache dir, no `Loaded` label); env parser whitelist (off-words disable, on-words → default dir, path values used as-is, never raises, default dir never under the repo tree); miss runs all stages and stores; hit skips the NLP stages (mock call counter unchanged) and prints the honest `[INFO] Loaded analysis from cache` label; file-edit / config (sampling knob) / tier (`nlp_on`) / tty y-N choice invalidation; corrupt (garbage text + pickle payload) → miss with self-heal; schema-mismatch → miss, file kept; sanitizer round-trips numpy scalars/arrays, NaN/Inf → None, `datetime.date` keys → str; 30-day TTL prune on the next store; zip transcript selection rides in the key while mtime does not; `report_path` reset to `''` on load. |
| Summarizer (NLP-gated) | `tests/test_phase4_nlp.py` | Real `ConversationSummarizer` with T5 pipeline/tokenizer/model mocked — non-empty summary text, `summary` block present only with the gate on. |
| EDA | `tests/test_analysis.py` | `ChatEDA` message volume, hourly/daily activity, top senders, comprehensive summary schema. |
| Relationship health | `tests/test_analysis.py`, `tests/test_phase4_alwayson.py` | Conversation starters, initiator balance score, dominance, `[0, 1]` overall health score + grade, streaks, friendship index `[0, 100]` + tier, milestones, emoji personality, rolling window scores with a minimum-message threshold; health block in the pipeline/report. |
| Network graph | `tests/test_phase4_alwayson.py`, `tests/test_perf_parity_network.py` | Density float, `network` chart key with a base64 PNG URI in the report; the vectorized sender-switch edge counting is bit-identical to the original per-row loop. |
| Narrative (Tier A) | `tests/test_narrative.py` | Arc/driver/reciprocity/engagement observation kinds, hedged summary, question-ratio extractors (Banglish + English wh-words), empty/single-sender/NaN-sender degradation. |
| Visualization | `tests/test_analysis.py` | Real `ChatVisualizer` timeline + heatmap return matplotlib `Figure` objects (Agg backend pinned). |
| Interactive charts (ECharts) | `tests/test_interactive_charts.py` | Real `build_chart_specs` on a synthetic canonical chat DataFrame — all six always-on specs (timeline/activity/participants/sentiment/health/network) are returned and each is JSON-serializable (`json.dumps(..., allow_nan=False)`); the network spec is true 3D (`scatter3D` + `lines3D` on `grid3D`, auto-rotate, degree-sized nodes); `build_emotion_spec` pie shape; `write_report` renders interactive chart divs + `CHART_SPECS` when specs exist; PNG fallback when the spec builder raises or a spec is non-serializable; a `</script>` payload in a participant name is `|tojson`-escaped and never survives into the inlined JS. |
| HTML report | `tests/test_phase2_report.py`, `tests/test_phase2_cli.py`, `tests/test_phase4_nlp.py`, `tests/test_phase4_alwayson.py` | Single-file self-containment (no external refs), `utf-8` + emoji integrity, content escaping (`<script>` payload & `Alice <3 Bob` sender), filename sanitization, report lands in cwd, auto-open degrade & `CHAT_ANALYZER_NO_OPEN`, skip-note surfacing, tab structure (overview/participants/flow/words/sentiment/narrative, health, network, emotion, summary), Tier A/B narrative rendering. |
| CLI (end-to-end) | `tests/test_phase1_smoke.py`, `tests/test_phase2_cli.py`, `tests/test_phase4_cli.py` | `chat-analyzer` console script + `python -m chat_analyzer` fallback, `--help`/`--version`, prompt happy path + invalid-path re-prompt, stage narration order, `"Messages: 27"` smoke token, friendly exit-1 taxonomy (missing file / wrong format / unparseable) with no tracebacks, NLP hint line and tty download menu, `CHAT_ANALYZER_FORCE_NLP` determinism. |
| Pipeline orchestration | `tests/test_phase2_pipeline.py` | Stage sequencing and contract of `run_pipeline`, runtime Agg backend assertion, no stale report files left in the repo tree. |
| ZIP export input | `tests/test_phase4_zip.py`, `tests/test_phase4_zip_media.py` | Single/multi-transcript zips (non-tty auto-select-ALL merge), empty/corrupted zip exit-1 messages, interactive transcript selection, media file counting (`max` of `<Media omitted>` markers vs zip media members). |
| Perf-parity (pure refactors) | `tests/test_perf_parity.py`, `tests/test_perf_parity_sentiment.py`, `tests/test_perf_parity_emotion.py`, `tests/test_perf_parity_network.py`, `tests/test_perf_parity_dynamics.py` | Prove behavior-identical output after each pure-performance refactor: dominance scores, VADER dedupe + parallel scoring, emotion batch inference (honors `batch_size`), interaction-network groupby, conversation dynamics. |
| Golden parity (clone-friendly) | `tests/test_golden_parity.py` | Re-runs the live tier-1 pipeline (`nlp_enabled=False` — no torch, no transformers) on the committed sample chats and asserts the parse/stats/sentiment projection matches `tests/golden/*.csv`; a contract rename trips the exact key-set assert first; an autouse fixture stands in for `transformers` in `sys.modules` so a clean-clone ImportError is reproduced even on machines with the `[nlp]` extra installed. |
| NLP installer / long paths | `tests/test_long_path_guard.py` | Windows long-path guard (`LongPathsEnabled` registry, `CHAT_ANALYZER_ALLOW_LONG_PATH` override, pip recorder proves the guard fires before any install), non-Windows silence. |
| Tier resolution + honesty | `tests/test_phase4_nlp_gate.py`, `tests/test_ws_honesty.py` | Pure-logic tests for the always-on tier menu (`nlp_status` version satisfaction + status mapping, no subprocess/network) and the honesty workstreams (WS-3 network wording, WS-4 non-Latin disclaimer). |
| Packaging / smoke | `tests/test_phase1_smoke.py` | Full `chat_analyzer.*` import matrix, no web-app tokens (`exec(code` / `unsafe_allow_html` / streamlit / plotly), lean base deps (heavy deps confined to `[nlp]`), reporting modules importable but not wired into the CLI. |

See `docs/GETTING-STARTED.md` for the sample chats and run instructions.

## Writing new tests

- **Import the real modules** — never re-implement parser/analysis logic in the
  test:

  ```python
  from chat_analyzer.parser.whatsapp_parser import WhatsAppParser
  from chat_analyzer.cli.pipeline import run_pipeline
  from chat_analyzer.ingest.ingestion import messages_to_dataframe
  ```

- **Mock only the heavy stuff.** The transformers pipeline/T5 model-load use
  `unittest.mock.patch` on the module-level caches (`_emotion_analyzer`,
  `_emotion_model_loaded`, `transformers.pipeline`, `T5Tokenizer` /
  `T5ForConditionalGeneration.from_pretrained`) — see
  `tests/test_phase4_nlp.py::_mocked_nlp` and
  `tests/test_result_cache.py::_mocked_models`. `nlp_availability` is forced via
  the gate or `CHAT_ANALYZER_FORCE_NLP`. Add a fresh-load reset fixture when you
  touch the emotion singletons (mirror `_reset_emotion_singletons`).
- **Pin the backend before importing chat_analyzer** when a test renders
  figures — importing `chat_analyzer.analysis.emotion` pulls matplotlib, so set
  `os.environ.setdefault("MPLBACKEND", "Agg")` at module top *before* the
  `from chat_analyzer...` imports. `tests/test_analysis.py`,
  `tests/test_emotion_sampling.py`, `tests/test_emotion_parallel.py`,
  `tests/test_emotion_quarterly.py`, `tests/test_result_cache.py`, and all five
  `test_perf_parity*.py` files do this; `tests/test_phase2_pipeline.py` instead
  asserts the runtime backend is Agg
  (`matplotlib.get_backend().upper() == "AGG"`). New visualization tests must
  pin it too (or the TkAgg default crashes headless).
- **Env-var tests set/restore `os.environ` with `monkeypatch`** (`setenv` /
  `delenv`) around the `nlp_gate` readers — e.g.
  `CHAT_ANALYZER_EMOTION_WORKERS`, `CHAT_ANALYZER_RESULT_CACHE`,
  `CHAT_ANALYZER_EMOTION_SAMPLE` — so a mutated environment never leaks between
  tests.
- **Gate thread/process pools so mocked unit tests stay sequential.**
  `tests/test_emotion_parallel.py` exercises the `ProcessPoolExecutor` driver
  with a `_FakePool` stand-in (no real processes are spawned) and asserts a
  mocked classifier is not a real pipeline, so `_scoring_workers` returns 0;
  the real multiprocess spawn path runs only in the `slow` smoke. Do the same
  for any new parallel code.
- **Redirect noisy analysis stdout** where print-emotes would break a cp1252
  console (`redirect_stdout(StringIO())` around module import / analyzer
  calls).
- **Keep reports out of the repo** — copy the sample into `tmp_path`, run with
  `cwd=tmp_path`, and assert the repo tree has no new `*_report.html` files.
- **Run commands from repo root in tests that spawn subprocesses** (like
  `_cli_cmd`), pass `encoding="utf-8"` to `subprocess.run`, and pin
  deterministic environment vars: `BROWSER=__none__`, `CHAT_ANALYZER_NO_OPEN=1`,
  `CHAT_ANALYZER_FORCE_NLP=0`.
- **Range assertions are idiomatic:** health scores in `[0, 1]`, friendship
  index in `[0, 100]`, VADER compound in `[-1, 1]`, counts as exact equality
  with a message (`Should parse 6 messages...`).
- **Mark wall-clock-heavy tests `@pytest.mark.slow`** so they don't run in the
  fast CI job (subprocess spawns, full-pipeline renders, zip e2e, and the
  real-model emotion spawn-parity smoke). Real-model inference is allowed only
  in slow tests (D-17); the fast suite must never hit the network.

## Coverage requirements

No minimum coverage thresholds are configured — `pyproject.toml` has no
`coverage` section and CI has no coverage step. Coverage artifacts are
gitignored (`.coverage`, `htmlcov/`, `coverage.xml`) but not enforced.

`pytest-cov` is part of the dev extra, so you can inspect coverage yourself:

```bash
python -m pytest --cov=src --cov-report=term-missing
```

## CI integration

The repo's `.github/workflows/ci.yml` ("CI" workflow) runs pytest on **every
`push` and `pull_request`** to any branch, split across three jobs (all with
`timeout-minutes: 30` and the env vars `BROWSER=__none__`,
`CHAT_ANALYZER_NO_OPEN=1`, `CHAT_ANALYZER_FORCE_NLP=0`):

| Job | OS / Python | Install | Steps |
| --- | --- | --- | --- |
| `test` | matrix `{ubuntu-latest, windows-latest}` × `{3.11, 3.12}` (`fail-fast: false`) | `python -m pip install -e ".[dev]"` | `python -m pytest -m "not slow" -q -p no:cacheprovider` (fast suite) |
| `slow` | `ubuntu-latest` / 3.11 | `python -m pip install -e ".[dev]"` | `python -m ruff check src tests` then `python -m pytest -m "slow" -q -p no:cacheprovider` |
| `nlp` | `ubuntu-latest` / 3.11 | `python -m pip install -e ".[dev,nlp]"` | `python -m pytest tests/test_phase4_nlp.py -q -p no:cacheprovider` (proves the real [nlp] extra + model-load mock path) |

Updates on the same ref cancel in-progress runs
(`concurrency.cancel-in-progress: true`); `actions/checkout@v4` +
`actions/setup-python@v5` (with the pip cache) are used throughout. Note that
the slow-marked real-model spawn-parity smoke does **not** run in CI — the
`slow` job has a base install (no torch) and that test skips without the
cached model weights, so it is intended as a local verification only.