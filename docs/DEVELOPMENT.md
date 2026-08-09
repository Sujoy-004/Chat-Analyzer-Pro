<!-- generated-by: gsd-doc-writer -->
# Development

This guide covers working on the Chat-Analyzer-Pro codebase: setting up a development
environment, running the CLI from a checkout, understanding the `src/chat_analyzer`
module layout, the coding conventions to follow, and how to add a new parser or
feature.

New to the repo? Start with [GETTING-STARTED.md](GETTING-STARTED.md) for
prerequisites and first-run setup. For the test suite, see
[TESTING.md](TESTING.md).

---

## Local setup

### Prerequisites

- **Python >= 3.11** — enforced by `requires-python = ">=3.11"` in `pyproject.toml`.
  The CI matrix runs 3.11 and 3.12, so anything in that range is safe to develop on.
- **pip** — bundled with modern Python installs. No other system packages are
  required for the base tool. <!-- VERIFY: depends on the local Python install; on
  a machine where `pip` or the `py` launcher is missing, use the full
  `python3 -m pip` / `python.exe -m pip` form. -->

### Clone and install

```bash
git clone https://github.com/Sujoy-004/Chat-Analyzer-Pro.git
cd Chat-Analyzer-Pro
```

Create a virtual environment and install the package in editable mode so changes
to `src/` are picked up immediately:

```bash
# Windows (PowerShell)
py -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

- `-e .` performs an editable install of the `chat-analyzer-pro` package and the
  `chat-analyzer` console script (defined as `chat_analyzer.cli:app`).
- `[dev]` adds the dev-only tools: `pytest`, `pytest-cov`, and `ruff`. Install the
  dev extra now; it keeps `pytest` and `ruff` in lockstep with the versions CI uses.
- The **base** install deliberately excludes `torch`/`transformers`. Those heavy
  dependencies are gated behind the optional `[nlp]` extra (see
  [Coding conventions](#coding-conventions)); install them only if you are working
  on the emotion/summarizer modules:

  ```bash
  python -m pip install -e ".[nlp]"
  ```

### Recommended editor (VS Code)

The repo does not commit editor settings (`.vscode/` is gitignored). The
recommended setup is:

1. Install the **Python** extension from Microsoft.
   `<!-- VERIFY: the "Python" and "Ruff" VS Code extensions are installed from
   the VS Code Marketplace; extension availability and exact versions depend on
   the local VS Code install. -->`
2. Select the project interpreter: **Python: Select Interpreter → Enter
   interpreter path →** `.venv\Scripts\python.exe` (Windows) or
   `.venv/bin/python` (macOS/Linux).
3. Optionally install the **Ruff** extension so lint problems surface inline
   while editing (same rules as `ruff check`, see
   [Code quality tooling](#code-quality-tooling)).
4. Enable the Python test integration: set the test framework to
   `pytest` and point it at the repo root — tests live in `tests/`.

### Windows long-path note (NLP only)

Installing the `[nlp]` extra on Windows downloads multi-gigabyte torch wheels.
Windows caps paths at 260 characters (`MAX_PATH`), and torch's wheel extraction
fails with WinError 206 when the venv's `site-packages` folder sits at a deep
path. The CLI detects this before any download and prints a `[WARN]` line or
refuses the install. Two ways around it:

- Enable the `LongPathsEnabled` DWORD under
  `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem` (admin rights required), or
- create a short-path venv with the committed helper
  `scripts/make_nlp_env.ps1` (e.g. `%TEMP%\chat-analyzer-nlp`), which installs
  the `[nlp]` extras there.

You can also silence the guard explicitly with `CHAT_ANALYZER_ALLOW_LONG_PATH=1`
(heuristically measured — use the short-path venv for the reliable fix).

---

## Running the CLI from a checkout

Once the editable install is in place, the console script is available in the
venv. From the repo root, use the bundled sample chats:

```bash
chat-analyzer data/sample_chats/whatsapp_sample.txt          # WhatsApp .txt
chat-analyzer data/sample_chats/telegram_sample.json         # Telegram .json
```

If the `chat-analyzer` script is not on your `PATH`, the module form is
equivalent:

```bash
python -m chat_analyzer data/sample_chats/whatsapp_sample.txt
```

Behavior to expect:

- Accepted inputs: WhatsApp `.txt`, Telegram `.json`, and `.zip` export archives.
- Interactive, no-arg runs (`chat-analyzer` with no path) re-prompt for a file.
- The report is written to the **current working directory** as
  `<chat_name>_report.html` and best-effort auto-opened in the browser. Set
  `CHAT_ANALYZER_NO_OPEN=1` to suppress auto-open (also set by CI).
- The NLP gate (`CHAT_ANALYZER_FORCE_NLP=0/1`) deterministically forces the
  NLP-off / NLP-on branches in tests; without it, the gate probes whether
  `torch`+`transformers` import.

`--version` prints the installed package version and exits:
`python -m chat_analyzer --version`.

---

## Module layout

The package root is `src/chat_analyzer/` (hatchling wheel target:
`packages = ["src/chat_analyzer"]`).

```
src/chat_analyzer/
├── __init__.py            # package metadata + NullHandler, no heavy imports
├── __main__.py            # `python -m chat_analyzer` entry point -> cli.app
├── cli/                   # the CLI application layer (Typer front end)
│   ├── main.py            # Typer app: positional/interactive analyze, --version, friendly errors
│   ├── pipeline.py        # run_pipeline: parse → DataFrame → EDA/VADER/health/network/charts → AnalysisResults
│   ├── contracts.py       # AnalysisResults (TypedDict) + ParseReport (dataclass) — the shared CLI contract
│   ├── adapters.py        # analysis-module dicts → AnalysisResults contract (defensive .get())
│   ├── render.py          # terminal summary panel (ASCII only, no business logic)
│   ├── report_html.py     # single-file HTML report: Jinja2 autoescape, base64 PNG charts, auto-open
│   ├── zip_input.py       # .zip archives: enumerate transcripts, count media, select, parse, merge
│   └── nlp_gate.py        # silent NLP availability probe, locked model IDs/sizes, guarded installer
├── parser/
│   ├── whatsapp_parser.py # WhatsAppParser + parse_whatsapp_chat (`.txt`)
│   └── telegram_parser.py # parse_telegram_chat / parse_telegram_chat_with_report (`.json`, both export shapes)
├── ingest/
│   └── ingestion.py       # normalization + messages_to_dataframe(rows) → canonical DataFrame
├── analysis/
│   ├── eda.py             # ChatEDA: summary, volume, dynamics, content stats
│   ├── sentiment.py       # VADER sentiment (always on)
│   ├── relationship_health.py  # always-on relationship-health scoring
│   ├── network_graph.py   # always-on network analysis + figure
│   ├── narrative.py       # Tier A heuristic "what is going on" observations (pandas only)
│   ├── emotion.py         # EmotionAnalyzer — gated behind [nlp]; lazy torch/transformers
│   └── summarizer.py      # ConversationSummarizer (t5-small) — gated behind [nlp]
├── reporting/
│   ├── pdf_report.py      # legacy PDF report generator (importable, not wired into the CLI)
│   └── weekly_digest.py   # legacy automated weekly digest (not wired into the CLI)
└── utils/
    ├── preprocessing.py   # text cleaning helpers
    └── visualization.py   # ChatVisualizer — matplotlib/Ag base64 charts for the report
```

Key relationships:

- `cli/main.py` → `cli/pipeline.py` → `parser/*` (per file suffix) →
  `ingest.ingestion.messages_to_dataframe` → `analysis/*` (EDA, sentiment,
  relationship health, network, narrative; emotion + summarizer only when the
  `nlp_gate` probe passes) → `cli/adapters.adapt` → `AnalysisResults` →
  `cli/render.py` (terminal) and `cli/report_html.py` (HTML file).
- `cli/contracts.py` is the single source of truth between the CLI modules; the
  analysis core (`parser`, `analysis`, `ingest`) never imports it.
- The legacy `reporting/` modules and the old Streamlit web app (`app/`) are not
  part of the CLI. Do not treat them as the reference for new code.

---

## Coding conventions

Conventions below come from `.planning/codebase/CONVENTIONS.md` and the repo
rules in `AGENTS.md`. The planning docs predate the CLI pivot in places — when
they disagree with the policies here, the policies here win.

### Architecture rules (hard constraints)

- **Package layout:** `src/chat_analyzer/` with proper `__init__.py` markers in
  every subpackage (required for PyPI shipping). The old misnamed `_init_.py`
  markers from the pre-pivot layout are gone — never reintroduce them.
- **Lean base install:** the base install must NOT pull `torch`,
  `transformers`, `streamlit`, or `plotly`. Heavy analysis (emotion,
  summarizer) is gated behind the `[nlp]` optional extra with **lazy imports**
  and graceful-availability flags (e.g. `try/except ImportError` → module-level
  `_AVAILABLE` flag → functions return safe defaults when the flag is False).
- **No web-app-only code:** do not reintroduce remote-`exec()` patterns or
  `unsafe_allow_html`-style rendering. Specifically never port the old
  `app/streamlit_app.py` practice of fetching modules from GitHub URLs and
  running them via `exec(code, namespace)`.
- **Reuse the existing analysis modules:** do not rewrite analysis logic in the
  pipeline. The CLI wraps modules like `ChatEDA`, `add_sentiment_analysis`,
  `analyze_relationship_health`, `analyze_network`, and `analyze_narrative`.
- **Single-file HTML report:** charts are embedded as base64 PNG data URIs,
  the template is an inline Jinja2 constant with `select_autoescape` enabled —
  chat content is UNTRUSTED input, never rendered unescaped.

## Style rules

- `snake_case.py` for module and test file names; tests are `test_<area>.py`
  and live in `tests/`.
- Classes: PascalCase, one primary class per module. Functions: `snake_case`,
  verb-first. Private helpers prefixed with a single underscore (`_`).
- **Double quotes** in new code (the current majority style; older modules use
  single quotes — don't propagate it).
- Google-style docstrings with `Args:` / `Returns:` (for type-hinted signatures
  omit the `(str)`-style parenthesized annotations found in older modules). New
  code is expected to carry full type hints (use `src/chat_analyzer/analysis/
  network_graph.py` or `src/chat_analyzer/cli/contracts.py` as references).
- Imports: stdlib → third party → local. Optional heavy deps are imported
  lazily inside functions or protected `try/except` blocks, never at the top of
  the import block.
- Error handling: prefer **return-safe-defaults** over raising; analysis
  functions on empty/invalid data return `{'error': ...}`-style dicts. Never
  write bare `except:` — catch `(ValueError, TypeError)` or `except Exception
  as e:` explicitly.
- Logging: `logger = logging.getLogger(__name__)` for production modules;
  keep console narration (`rich`) in the CLI layer only.

---

## How to add a new parser

A parser turns one chat-export format into the message rows the pipeline
consumes. The wall-clock happens in `cli/pipeline.py`, `cli/main.py`, and —
for zip archives — `cli/zip_input.py`.

Steps:

1. **Create the parser module:**
   `src/chat_analyzer/parser/<format>_parser.py`. Follow the contract of the
   existing parsers, which return a `(rows, counts)` tuple:
   - `rows`: a list of message dicts (other keys include at least a datetime /
      timestamp plus the text; see `parse_telegram_chat()`/`WhatsAppParser` for the
     exact keys).
   - `counts`: a dict with `total_lines`, `parsed_messages`, `skipped_lines`,
     `system_messages` — no `media_messages` (only `cli/zip_input.py` adds
     that key; the `ParseReport` dataclass in `cli/contracts.py` fills it via
     default 0 — the shapes must align).
   Also provide a convenience `parse_<format>_chat()` (like
   `parse_whatsapp_chat`) and a `parse_<format>_chat_with_report()` that
   returns the `(rows, counts)` shape. Wrap confusing per-line failures
   honestly — never bare-`except: continue`.
2. **Wire the suffix in the dispatcher:** in
   `src/chat_analyzer/cli/pipeline.py`, `run_pipeline` dispatches by
   `path.suffix.lower()`. Add an `elif` for your suffix that imports the parser
   lazily and produces `(rows, counts, source)`.
3. **Add the extension to the positional path allowlist** in
   `src/chat_analyzer/cli/main.py` (currently `.txt`, `.json`, `.zip`).
   Keep the "Unsupported file type" check in sync.
4. **Zip archives** — if the export format commonly arrives inside a `.zip`,
   extend `_list_transcripts` in `cli/zip_input.py` (member suffix → kind
   mapping) so `parse_zip_with_report` recognizes it.
5. **Re-export** the new `parse_<format>_chat` function from
   `src/chat_analyzer/parser/__init__.py` for programmatic use.
6. **Add fixtures + tests** under `tests/`: a realistic export sample in
   `tests/fixtures/` and a `tests/test_phase<n>_<format>.py` exercising the
   parser through the real CLI (integration) and the parser functions directly
   (unit). Run the fast suite before pushing.

---

## How to add a new feature/analysis

1. **New analysis module:** `src/chat_analyzer/analysis/<feature>.py`.
   Write functions that take the canonical DataFrame (columns include
   `timestamp`, `sender`, `message`) and return a `Dict`. Keep it always-on if
   it is pandas/numpy-only; gate it behind the `[nlp]` availability probe
   (lazy torch/transformers imports + safefallbacks) if it needs heavy models.
   See `network_graph.py` and `narrative.py` as reference patterns.
2. **Wire it into the pipeline:** in `cli/pipeline.py` (`run_pipeline`) call
   the new function inside the "Computing insights" stage (under the existing
   `redirect_stdout` capture) and thread the result into
   `cli/adapters.adapt(...)` as a keyword argument with a `None` default.
3. **Extend the contract** in `cli/contracts.py` (`AnalysisResults`) if the
   result should be part of the report or terminal summary, then surface it in
   the `adapt()`. The terminal keeps an ASCII-only summary; charts are
   matplotlib → base64 PNG data URIs (pipelined via `fig_to_data_uri` /
   `_safe_chart`) and rendered by `report_html.py`.
4. **Tests** — add a `test_<feature>.py` that exercises the real
   `chat_analyzer.<module>` entry points (and the CLI path if user-visible);
   use `@pytest.mark.slow` for any test that spawns subprocesses or renders the
   full pipeline.
5. Run `ruff check src tests` and the relevant tests before committing.

---

## Code quality tooling

- **Ruff** is the linter for this project. `ruff>=0.16.1` is pinned in the
  `[dev]` optional extra — `pip install -e ".[dev]"` installs it.
- There is **no `[tool.ruff]` section** in `pyproject.toml`, so linting runs
  on Ruff's default rule set. CI runs exactly one lint command:

```bash
python -m ruff check src tests
```

Keep `src/chat_analyzer` and `tests` clean under this command before pushing.

- No code formatter (Black/`ruff format`) is enforced. Keep lines within the
  existing style (PEP 8 with ~120-char tolerance in long analysis modules).

---

## Test commands

The suite lives in `tests/` (pytest, with `[tool.pytest.ini_options]`
configured in `pyproject.toml`). The `slow` marker is registered there and marks
tests
that spawn subprocesses or render the full pipeline (long wall-clock).

```bash
# Full suite (fast + marked-slow tests)
python -m pytest

# Fast suite only — this is what the CI "test" job runs
python -m pytest -m "not slow"

# Slow subset only: CLI subprocess + e2e renders
python -m pytest -m "slow"

# Specific file / single test
python -m pytest tests/test_phase2_cli.py
python -m pytest tests/test_phase2_cli.py::test_version

# Coverage (pytest-cov is in [dev])
python -m pytest --cov=chat_analyzer
```

CI sets the environment variables `BROWSER="__none__"` and
`CHAT_ANALYZER_NO_OPEN=1` to prevent browser auto-open during tests, and
`CHAT_ANALYZER_FORCE_NLP=0` to keep the NLP gate off. Run tests the same way
locally if your machine has `torch`+`transformers` installed (which would
otherwise make the gate probe take the NLP-on branch).

---

## CI

`.github/workflows/ci.yml` runs on every push and pull request on any branch:

| Job | Runs | Command |
|-----|------|---------|
| `test` | Ubuntu + Windows × Python 3.11/3.12 | `python -m pytest -m "not slow" -q` |
| `slow` | Ubuntu, Python 3.11 | `python -m ruff check src tests` + `python -m pytest -m "slow" -q` |
| `nlp` | Ubuntu, Python 3.11 | `pip install -e ".[dev,nlp]"` + `python -m pytest tests/test_phase4_nlp.py -q` |

No workflow job publishes artifacts; the jobs are gates only.

---

## Branch conventions

No branch naming convention is documented in the repo (there is no
`CONTRIBUTING.md` or PR description in `.github/` — only `ci.yml`). The
default branch is `main` and CI runs against every branch, so any named
branch (e.g. `feat/xyz`, `fix/xyz`) works as long as the fast suite + `ruff
check` are green before opening a PR.

---

## Pull request process

- Push a feature branch and open a PR against `main` via GitHub.
- With the checks: the `test` matrix (base install, fast suite) and the
  `slow` job (which also runs `ruff check src tests`) must pass; the `nlp`
  job exercises the gated extra.
- Keep the base install lean: if your change adds a dependency, only add it
  to `[project.optional-dependencies]` (or `[project.dependencies]` if it is
  genuinely `[nlp]`-free lightweight core requirement); never add
  `torch`/`transformers` to the base.
- Region PR scope small, mirroring the module boundary you touched — parsers
  and analysis modules are already per-format/per-feature.
- If your change touches the CLI surface (new suffix, new flag, new output),
  update the relevant docs in `docs/`.

---

## Next steps

- [GETTING-STARTED.md](GETTING-STARTED.md) — prerequisites and first-run setup
- [ARCHITECTURE.md](ARCHITECTURE.md) — system overview, components, data flow
- [CONFIGURATION.md](CONFIGURATION.md) — environment variables and settings
- [TESTING.md](TESTING.md) — full testing reference (framework, commands, CI)