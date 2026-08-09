<!-- generated-by: gsd-doc-writer -->
# Testing

Chat-Analyzer-Pro is tested with **pytest** (the dev extra in `pyproject.toml`
declares `pytest>=7.4`, `pytest-cov>=4.1`, and `ruff>=0.16.1`). The suite has
**210 tests across 18 files**. Wall-clock-slow tests (CLI subprocess spawns and
full pipeline/report renders) are gated behind the `slow` marker
(`[tool.pytest.ini_options]` in `pyproject.toml`) so the fast suite stays lean
for every commit.

Tests exercise the **real shipped modules** — `chat_analyzer.parser.*`,
`chat_analyzer.analysis.*`, `chat_analyzer.cli.*`, etc. — not copies of the
logic. Only heavyweight external callables (transformers pipelines, T5 model
loads, `webbrowser`) are mocked, and mocks are shaped to produce the exact
output the real libraries return.

## Test framework and setup

- **Runner:** pytest. The `slow` marker is registered in
  `[tool.pytest.ini_options]` of `pyproject.toml`.
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
| `python -m pytest` | Full suite — 210 tests (fast + slow). |
| `python -m pytest -m "not slow"` | Fast suite only — 185 tests (this is the CI default). |
| `python -m pytest -m "slow"` | Slow suite only — 25 wall-clock tests (CLI subprocess spawns, full pipeline + report renders). |
| `python -m pytest tests/test_phase2_telegram.py` | A single test file. |
| `python -m pytest tests/test_phase2_whatsapp.py::test_exact_fixture_counts` | A single test function. |
| `python -m pytest -k "not_a_chat or empty"` | Tests matching a keyword expression (the Telegram `test_not_a_chat_export_raises` parametrization is a good target). |
| `python -m ruff check src tests` | Lint gate used by CI (must pass before tests in the `slow` job). |

The four legacy unittest files also run standalone:

```bash
python tests/test_parser.py                 # unittest.main() entry point
python -m unittest tests.test_parser -v     # single legacy module
```

## Test data and fixtures

All file-backed fixtures live in either `data/sample_chats/` (real exports) or
`tests/fixtures/` (adversarial exports crafted for parse-report assertions).
There is no `conftest.py` and no shared fixture factory — most tests build
inline data, and file I/O uses `tmp_path` with UTF-8 encoding
(`encoding="utf-8"` — the Windows cp1252 console cannot encode the emoji-heavy
fixtures).

| Path | Contents / purpose |
| --- | --- |
| `data/sample_chats/whatsapp_sample.txt` | Real-style WhatsApp export, **27 messages** (`KNOWN_WHATSAPP_COUNT = 27`), 2 senders over 3 days, emojis, multiline continuation, 2 `<Media omitted>` markers. Locked-in count used by CLI e2e tests (`"Messages: 27"`). |
| `data/sample_chats/telegram_sample.json` | 5-message Telegram export (bare Chat shape). CLI e2e asserts `"Messages: 5"`. |
| `tests/fixtures/whatsapp_system_skip.txt` | 7 lines: 3 parsed messages, 2 system lines (encryption notice + "Alice added Bob"), 1 bad-date line, 1 continuation line. Locked-in counts `{total_lines: 7, parsed_messages: 3, skipped_lines: 1, system_messages: 2}`. |
| `tests/fixtures/whatsapp_all_skipped.txt` | 3 lines, zero parseable — drives the friendly `"No messages could be parsed"` error path (exit 1, no traceback). |
| `tests/fixtures/telegram_full_export.json` | `chats.list[]` shape with entity-array text, a service message, a tz-aware `Z` row (id 3), a bad date, and a `forwarded` row → locked counts `{6, 3, 2, 1}` (the `+05:30` offset row lives in `telegram_bare_entity.json`, id 2). |
| `tests/fixtures/telegram_bare_entity.json` | Bare-Chat shape with entity-array text (`@team` + text) and a `+05:30` offset row → 2 parsed, 0 skipped. |

CLI-facing tests copy a sample into `tmp_path` and run with `cwd=tmp_path` so a
report run never writes into the repo tree.

## What the suite covers

| Area | Files | Coverage |
| --- | --- | --- |
| WhatsApp parser (`.txt`) | `tests/test_parser.py`, `tests/test_phase2_whatsapp.py` | Basic parsing, sender extraction, datetime parsing, emoji preservation, multiline joins, system-message classification (encryption notice, "added Bob"), strict date parsing with **no** `datetime.now()` fabrication, US 12h / EU 24h / iOS bracket / 4-digit-year formats, DD/MM-tried-first regression, empty-body messages, honest skip/system counters. |
| Telegram parser (`.json`) | `tests/test_parser.py`, `tests/test_phase2_telegram.py` | Bare-Chat + `chats.list[]` shapes, recursive entity-array text joins, service-message filtering, tz-aware → naive UTC normalization, honest malformed-drop counting, `ValueError("Not a Telegram chat export")` on empty/missing-key exports, `parse_telegram_chat` DataFrame contract. |
| Ingestion (canonical DataFrame) | `tests/test_phase2_builder.py` | `messages_to_dataframe` 9-column schema + defaults, tz-aware → naive UTC, Telegram full-ISO path, WhatsApp date+time path, unparseable rows dropped. |
| Sentiment (VADER) | `tests/test_analysis.py`, `tests/test_phase1_smoke.py` | Real VADER columns (`vader_compound`/`vader_sentiment`) on positive/negative/neutral fixtures, `[-1, 1]` score range, sentiment distribution summary; the transformers path pinned off (`TRANSFORMERS_AVAILABLE = False`). |
| Emotion classification (NLP-gated) | `tests/test_analysis.py`, `tests/test_phase4_nlp.py` | Real `EmotionAnalyzer` with the transformers pipeline mocked — label/score shape, non-uniform dominant emotion, transformers 5.x nested-list compat, locked default model name; pipeline-level emotion block + `emotion` chart with the gate on/off. |
| Summarizer (NLP-gated) | `tests/test_phase4_nlp.py` | Real `ConversationSummarizer` with T5 pipeline/tokenizer/model mocked — non-empty summary text, `summary` block present only with the gate on. |
| EDA | `tests/test_analysis.py` | `ChatEDA` message volume, hourly/daily activity, top senders, comprehensive summary schema. |
| Relationship health | `tests/test_analysis.py`, `tests/test_phase4_alwayson.py` | Conversation starters, initiator balance score, dominance, `[0, 1]` overall health score + grade, streaks, friendship index `[0, 100]` + tier, milestones, emoji personality, rolling window scores with a minimum-message threshold; health block in the pipeline/report. |
| Network graph | `tests/test_phase4_alwayson.py` | Density float, `network` chart key with a base64 PNG URI in the report. |
| Narrative (Tier A) | `tests/test_narrative.py` | Arc/driver/reciprocity/engagement observation kinds, hedged summary, question-ratio extractors (Banglish + English wh-words), empty/single-sender/NaN-sender degradation. |
| Visualization | `tests/test_analysis.py` | Real `ChatVisualizer` timeline + heatmap return matplotlib `Figure` objects (Agg backend pinned). |
| HTML report | `tests/test_phase2_report.py`, `tests/test_phase2_cli.py`, `tests/test_phase4_nlp.py`, `tests/test_phase4_alwayson.py` | Single-file self-containment (no external refs), `utf-8` + emoji integrity, content escaping (`<script>` payload & `Alice <3 Bob` sender), filename sanitization, report lands in cwd, auto-open degrade & `CHAT_ANALYZER_NO_OPEN`, skip-note surfacing, tab structure (overview/participants/flow/words/sentiment/narrative, health, network, emotion, summary), Tier A/B narrative rendering. |
| CLI (end-to-end) | `tests/test_phase1_smoke.py`, `tests/test_phase2_cli.py`, `tests/test_phase4_cli.py` | `chat-analyzer` console script + `python -m chat_analyzer` fallback, `--help`/`--version`, prompt happy path + invalid-path re-prompt, stage narration order, `"Messages: 27"` smoke token, friendly exit-1 taxonomy (missing file / wrong format / unparseable) with no tracebacks, NLP hint line and tty download menu, `CHAT_ANALYZER_FORCE_NLP` determinism. |
| ZIP export input | `tests/test_phase4_zip.py`, `tests/test_phase4_zip_media.py` | Single/multi-transcript zips (non-tty auto-select-ALL merge), empty/corrupted zip exit-1 messages, interactive transcript selection, media file counting (`max` of `<Media omitted>` markers vs zip media members). |
| NLP installer / long paths | `tests/test_long_path_guard.py` | Windows long-path guard (`LongPathsEnabled` registry, `CHAT_ANALYZER_ALLOW_LONG_PATH` override, pip recorder proves the guard fires before any install), non-Windows silence. |
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
  `tests/test_phase4_nlp.py::_mocked_nlp`. `nlp_availability` is forced via the
  gate or `CHAT_ANALYZER_FORCE_NLP`. Add a fresh-load reset fixture when you
  touch the emotion singletons (mirror `_reset_emotion_singletons`).
- **Pin the backend before importing matplotlib** when a test renders figures:

  ```python
  import os
  os.environ.setdefault("MPLBACKEND", "Agg")
  ```

  Only `tests/test_analysis.py` does this; `tests/test_phase2_pipeline.py`
  instead asserts the runtime backend is Agg (`matplotlib.get_backend().upper() == "AGG"`).
  New visualization tests must pin it too (or the TkAgg default crashes headless).
- **Redirect noisy analysis stdout** where print-emotes would break a cp1252
  console (`redirect_stdout(StringIO())` around module import / analyzer
  calls).
- **Keep reports out of the repo** — copy the sample into `tmp_path`, run with
  `cwd=tmp_path`, and assert the repo tree has no new `*_report.html` files.
- **Run commands from repo root in tests that spawn subprocesses** (like
  `_cli_cmd`), and pin deterministic environment vars:
  `BROWSER=__none__`, `CHAT_ANALYZER_NO_OPEN=1`, `CHAT_ANALYZER_FORCE_NLP=0`.
- **Range assertions are idiomatic:** health scores in `[0, 1]`, friendship
  index in `[0, 100]`, VADER compound in `[-1, 1]`, counts as exact equality
  with a message (`Should parse 6 messages...`).
- **Mark wall-clock-heavy tests `@pytest.mark.slow`** so they don't run in the
  fast CI job (subprocess spawns, full-pipeline renders, zip e2e).

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
| `test` | matrix `{ubuntu-latest, windows-latest}` × `{3.11, 3.12}` | `python -m pip install -e ".[dev]"` | `python -m pytest -m "not slow"` (fast suite) |
| `slow` | `ubuntu-latest` / 3.11 | `python -m pip install -e ".[dev]"` | `python -m ruff check src tests` then `python -m pytest -m "slow"` |
| `nlp` | `ubuntu-latest` / 3.11 | `python -m pip install -e ".[dev,nlp]"` | `python -m pytest tests/test_phase4_nlp.py` (proves the real [nlp] extra + model-load mock path) |

Updates on the same ref cancel in-progress runs
(`concurrency.cancel-in-progress: true`).