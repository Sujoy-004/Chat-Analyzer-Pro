# Stack Research

**Domain:** pip-installable Python CLI tool — file-in, pipeline, terminal + HTML output
**Project:** Chat-Analyzer-Pro (greenfield CLI pivot from Streamlit app)
**Researched:** 2026-07-31
**Confidence:** HIGH (all versions verified against the PyPI JSON API and devguide.python.org on 2026-07-31; see Sources)

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | `>=3.11` (floor) | Runtime | **The project's stated "3.8+" floor is obsolete.** Python 3.8 and 3.9 are end-of-life (Oct 2024 / Oct 2025); 3.10 is security-only and EOLs **Oct 2026** (2 months away). The modern analysis stack forces 3.11: pandas 3.0.5 (`>=3.11`), matplotlib 3.11.1 (`>=3.11`), networkx 3.6.1 (`>=3.11`). 3.11 has security support to Oct 2027; everything below passes resolution checks. |
| typer | 0.27.0 | CLI framework | The modern standard for Python CLIs (FastAPI's author). Type-hint-driven arguments/options, auto `--help`/`--version`, automatic type coercion, shell completion built in. **Verified critical fact: typer 0.26.0+ dropped click entirely** — PyPI metadata shows 0.25.1 depends on `click>=8.2.1`; 0.26.0/0.27.0 have NO click dep and ship `rich>=13.8.0` as a hard dependency. One install buys you CLI + terminal rendering. Requires `>=3.10` — satisfied by the 3.11 floor. |
| rich | 15.0.0 | Terminal rendering | Tables, panels, rules, markdown, progress/spinners, syntax highlighting. Auto-resizes columns to terminal width, handles Windows ANSI natively (colorama auto-installed on Windows via typer). Pulled in automatically by typer — no separate decision needed. Released 2026-04-12, actively maintained. |
| plotext | 5.3.2 | Inline terminal charts | Pure Python, **zero runtime dependencies**, works on Python 3.5+. `bar()` (message counts per user, hourly/weekday activity), `plot()` line charts (sentiment timeline), `hist()`, `scatter()`, color themes. The only maintained, dependency-free ASCII-chart library with bar+line+scatter coverage. Low churn (last release 2024-09-24) = stable. |
| jinja2 | 3.1.6 | HTML report templating | Proper templating for the HTML report: template inheritance, `{{ value|e }}` autoescaping, `{% for %}` loops over report sections. **Autoescape matters**: chat messages are user content and get interpolated into HTML — f-string reports are an injection/rendering hazard. Released 2025-03-05, universally known. |
| hatchling | 1.31.0 | Build backend | PEP 517/621-standard build backend for `pyproject.toml`. Zero boilerplate for a flat/src-layout package, auto-detects package discovery, no `setup.py`/`setup.cfg` needed. **uv — the fastest-growing Python tooling — scaffolds new packages with hatchling by default**, making it the de-facto standard backend for new projects in 2025/2026. |
| uv | 0.12.0 | Dev environment tool (not shipped) | Creates venvs + resolves dependencies an order of magnitude faster than pip, with a proper `uv.lock` lockfile. Used only by developers — end users still `pip install chat-analyzer-pro`. This matters here because torch/transformers resolution is the slowest part of any Python install; uv makes dev iteration tolerable. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | 3.0.5 (`>=2.0` bound) | All data handling — core of `src/` | Already in the stack. **Do not raise the bound to `>=3.0` blindly**: pandas 3.x changed defaults (copy-on-write, string dtype) and the existing `src/` code was written for the 2.x era. Ship `pandas>=2.0`, then test the analysis core against 3.0.5 in a dedicated verification task before bumping. |
| numpy | `>=1.24` (floor only) | Numeric computation | Keep the existing floor and let the resolver pick: 3.11 users get numpy 2.4.x, 3.12+ users get 2.5.1. **Never pin `numpy==2.5.1`** — it requires `>=3.12` and would break the 3.11 floor. |
| matplotlib | 3.11.1 | Chart generation → PNG for HTML report | The core's `ChatVisualizer` already produces all figures and uses `plt.savefig(dpi=300, bbox_inches='tight')` (proven path). Reuse as-is; render to `io.BytesIO` → base64 data URI for the HTML report (single-file, no assets dir, no hosting). |
| seaborn | 0.13.2 | Styled matplotlib plots | Already in the stack. In maintenance mode (last release Jan 2024) but stable — keep, do not upgrade-hunt. |
| vaderSentiment | 3.3.2 | Sentiment analysis | Already in the stack. Stale (May 2020) but **zero dependencies and zero breakage risk** — VADER is complete as-is. Keep. |
| transformers | `>=4.30,<6` | Emotion classification + summarization | **Pinning `<6` is critical**: transformers 5.14.1 is current but 5.x made breaking API changes vs 4.x, and `src/analysis/emotion.py`/`summarizer.py` were written against 4.x. Ship the floor `<6`, verify against 5.x in the pipeline phase, and only widen after tests pass. Requires `>=3.10`. |
| torch | 2.13.0 (`>=2.0`) | Transformer backend | Already in the stack. Requires `>=3.10` — fine under the 3.11 floor. This is the 2 GB install cost the project has accepted. |
| networkx | 3.6.1 | Conversation network graph | Already in the stack. Requires `>=3.11` — one more reason the Python floor must rise. |
| tqdm | 4.70.0 | Progress bars in existing core loops | Already used inside `src/` loops. Keep for core progress; use rich `Status`/`Progress` in the new CLI layer for pipeline stages (already available via rich — no new dep). |
| reportlab | 5.0.0 | PDF report | **Do not install.** PDF report is explicitly out of scope (`src/reporting/pdf_report.py` deferred). Removes a dep and keeps install lean. |
| emoji, wordcloud, regex, python-dateutil, pytz | 2.15.0 / 1.9.6 / 2026.7.19 / current / current | Emoji analysis, wordcloud PNG, parsing utilities | Keep as-is — all already in the stack, all Python floors ≤ 3.9, no changes needed. wordcloud renders to PNG → usable in both terminal (no) and HTML report (yes, base64). |
| importlib.metadata | stdlib (3.11+) | Read package version for `--version` | `importlib.metadata.version("chat-analyzer-pro")` — no extra dependency needed on the 3.11 floor. |

### Development Tools

| Tool | Version | Purpose | Notes |
|------|---------|---------|-------|
| uv | 0.12.0 | venv + dependency resolution + lockfile (`uv.lock`) | Replace pip/venv dance in dev. `uv sync` after cloning. End users don't need it. |
| pytest | 9.1.1 (`>=7.4`) | Test runner | Already declared in the repo. Requires `>=3.10` — fine. Keep existing `tests/` (unittest-style classes run under pytest). |
| ruff | 0.16.1 | Lint + format | Modern single-tool replacement for the repo's declared black + flake8 + pylint. Fast, one config block in `pyproject.toml`, zero setup.py hooks. Runs on any Python — no constraint. |
| mypy | 2.3.0 | Static typing (optional) | Already declared. Keep for CI only; the core is untyped and that's acceptable. |
| twine | current | PyPI upload | Only when actually publishing. `uv build` → `twine upload dist/*`. |

## Installation

`pyproject.toml` — the entire packaging story lives here (replaces `requirements.txt` + Dockerfile dependency lines for the CLI distribution):

```toml
[build-system]
requires = ["hatchling>=1.31.0"]
build-backend = "hatchling.build"

[project]
name = "chat-analyzer-pro"
version = "0.1.0"
description = "Analyze WhatsApp and Telegram chat exports from the terminal"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "vaderSentiment>=3.3.2",
    "transformers>=4.30,<6",
    "torch>=2.0",
    "networkx>=3.1",
    "emoji>=2.8",
    "wordcloud>=1.9",
    "regex>=2023.8.8",
    "python-dateutil>=2.8.2",
    "pytz>=2023.3",
    "tqdm>=4.66",
    "typer>=0.27",
    "plotext>=5.3",
    "jinja2>=3.1",
]

[project.scripts]
analyze = "chat_analyzer.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/chat_analyzer"]
```

```bash
# Developers
pip install uv            # or via your preferred installer
uv sync                   # creates .venv, resolves everything incl. torch
uv run analyze data/sample_chats/whatsapp_sample.txt

# End users
pip install chat-analyzer-pro
analyze chat.txt
```

**Note on the import root:** the existing core is importable as `src.parser.whatsapp_parser` etc. only because Python 3's PEP 420 namespace packages tolerate it — the `__init__.py` files are actually misnamed `_init_.py` (single underscores) and are NOT valid package markers. For distribution, restructure to a proper src-layout: move `src/{analysis,ingest,parser,reporting,utils}` → `src/chat_analyzer/`, create real `__init__.py` files, and mechanically rewrite `from src.X` → `from chat_analyzer.X` (9 import sites found in the core today). A top-level package named `src` cannot ship on PyPI — it collides with the universal src-layout convention where `src/` is the directory *containing* the package.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| typer 0.27.0 | click 8.4.2 | Only if you need click's plugin system (entry-point group discovery) or legacy click codebases. Requires `>=3.10` either way. For a single-command tool, typer's type-hint model + bundled rich wins. |
| typer 0.27.0 | argparse (stdlib) | Only if you refuse all dependencies. argparse is verbose (manual type coercion, manual `--version`, no rich integration) and the project already accepts heavy deps (torch). Zero-dep purity is not this project's goal. |
| typer 0.27.0 | rich-argparse 1.8.0 + argparse | If the codebase were already argparse-shaped and adding typer meant rewriting. It's greenfield — start with typer. |
| plotext 5.3.2 | asciichartpy 1.5.25 | Only if you need only line charts and want a tinier API. asciichartpy has no bar charts, no axis labeling, no colors. |
| jinja2 3.1.6 | Plain f-string HTML | Only for a <100-line throwaway report with no user content interpolation. Chat messages ARE user content — autoescape is a correctness feature, not a luxury. |
| hatchling 1.31.0 | setuptools 83.0.0 | Only if the project must support ancient build tooling or relies on setuptools-only features (data files with complex globs, custom build steps). Both require `>=3.10` now, so setuptools has no compatibility advantage left. |
| hatchling 1.31.0 | poetry 2.4.1 (poetry-core) | Only if you want poetry's full dependency-management workflow for contributors. For a single-package CLI, poetry is heavier than the problem. uv covers resolution without owning the build. |
| matplotlib PNG → base64 (HTML) | plotly `fig.to_html()` (interactive HTML) | Only as a v2 enhancement. Verified: plotly is used **exclusively** in `app/streamlit_app.py`, never in the analysis core. Building the report on matplotlib means reusing working code; interactive plotly charts would mean writing a second chart layer. Static PNGs keep the report a single self-contained file. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Python 3.8/3.9/3.10 floors | 3.8 and 3.9 are end-of-life (Oct 2024/2025); 3.10 is security-only, EOL **Oct 2026**. pandas 3.0.5, matplotlib 3.11.1, networkx 3.6.1, typer 0.27, click 8.4 all require `>=3.10`/`>=3.11`. The PROJECT.md "3.8+ runtime" constraint is uninstallable with today's dependencies. | `requires-python = ">=3.11"` |
| setuptools as build backend | Legacy baggage (`setup.py`/`setup.cfg`), slower, requires `>=3.10` anyway. No advantage remains over hatchling. | hatchling |
| poetry as the project's tool | Full dependency-manager workflow overkill for one package; slower resolution; locks contributors into poetry commands. | uv for dev + hatchling for build |
| click 8.4.2 directly | typer 0.26+ no longer wraps click; adding click means maintaining a separate CLI layer and integrating rich yourself (rich-click). | typer 0.27.0 |
| argparse as primary framework | Verbose, manual coercion, no rich output, manual `--version`, manual completion. | typer 0.27.0 |
| Google Fire | Magic-string API, hard to debug, no typing, poor `--help`. | typer |
| cleo | Laravel-style command objects, heavyweight, niche in the Python data world. | typer |
| asciichartpy | Line-only; no bars, no labels, unmaintained-feeling. | plotext |
| unicodeplots | **Not on PyPI (404)** — dead project. | plotext |
| rich-pixels | Renders bitmap images to the terminal — wrong tool for data charts. | plotext |
| Textual / textual-plotext | A full interactive TUI framework. This tool is one-shot: `analyze file → output`. No event loop needed. | plotext + rich (print-only) |
| plotly for the HTML report (v1) | Would require duplicating the matplotlib chart code from the core; plotly today lives only in the Streamlit app being deleted. | matplotlib figures → PNG → base64 in jinja2 template |
| `transformers>=5` unconstrained | 5.x made breaking API changes; `src/analysis/emotion.py`/`summarizer.py` target 4.x. | `transformers>=4.30,<6`, widen after testing |
| `numpy==2.5.1` pin | Requires `>=3.12`; breaks the 3.11 floor. | `numpy>=1.24` floor, let resolver choose |
| reportlab / PDF report | Explicitly out of scope; keeps install lean. | — (defer; revisit only if asked) |
| requirements.txt for the CLI distribution | No metadata, no version constraints, no entry points. | pyproject.toml `[project.dependencies]` + `[project.scripts]` |
| Streamlit + streamlit-option-menu + gunicorn + python-multipart + scikit-learn + python-dotenv | Web-app-only deps being deleted with `app/`. None are used by the analysis core. | Drop from the CLI dependency list entirely |
| `pytesseract`/`pdfplumber`/`pdf2image` (OCR path) | txt/json-only v1 scope; these are optional-import probes with system-level deps (tesseract, poppler) that are painful on Windows end-user installs. | Keep the try/except probe pattern if kept at all; do not make them required deps |

## Stack Patterns by Variant

**If the user is on Windows (the dev machine is win32):**
- Use typer's automatic colorama on Windows; rich handles ANSI rendering natively.
- Test plotext output at 80-column width — its ASCII grids wrap badly in narrow/legacy Windows terminals. Call `plt.theme("clear")` or document `--width` usage.
- wordcloud/torch wheels: both ship Windows wheels (torch on win32 with CUDA CPU variant by default via pip) — no source builds expected for the core deps.

**If torch/transformers install becomes the bottleneck for contributors:**
- Use `uv` (resolves torch far faster than pip) or document `pip install torch --index-url https://download.pytorch.org/whl/cpu` for CPU-only dev.

**If the HTML report must include wordclouds/heatmaps:**
- These are matplotlib figures already in the core — same base64 path. Do NOT try to render them as plotext in the terminal; heatmaps/wordclouds are report-only.

**If `--version` output should not hit the network:**
- `importlib.metadata.version(...)` is local and instant. Never call the PyPI API.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| typer 0.27.0 | rich >=13.8.0 (15.0.0 current) | rich is a hard dep of typer 0.26+; no version conflict. |
| typer 0.27.0 | Python >=3.10 | 3.11 floor satisfies. |
| typer 0.27.0 | click — **none** | 0.26+ removed click; do not add click "for typer". |
| pandas 3.0.5 | Python >=3.11 | Floor driver. Test existing core code against 3.x — copy-on-write and string-dtype defaults changed vs 2.x. |
| numpy 2.5.1 | Python >=3.12 only | Use floor `>=1.24`; resolver picks 2.4.x for 3.11 users. |
| matplotlib 3.11.1 | Python >=3.11 | Current savefig→PNG path in `visualization.py` is unaffected. |
| transformers 5.14.1 | Python >=3.10; **breaking vs 4.x** | Pin `<6`; verify `emotion.py`/`summarizer.py` against 5.x before widening. |
| torch 2.13.0 | Python >=3.10 | Fine under 3.11 floor; 2GB install accepted per project decision. |
| plotext 5.3.2 | Any Python >=3.5; zero deps | Oldest, most stable dependency in the new stack. |
| jinja2 3.1.6 | Python >=3.7 | No constraints. |
| hatchling 1.31.0 | Python >=3.10 | Build-time only, not a runtime dep. |
| NLTK | — | **Not imported anywhere in `src/`** (verified by grep). The `nltk` requirement + Dockerfile punkt downloads are web-app/legacy baggage; the nltk→punkt_tab breaking change in NLTK 3.9+ is therefore NOT a risk here. vaderSentiment is self-contained. |
| reportlab 5.0.0 | Python >=3.9 | Do not install — PDF is out of scope. |

## Sources

- [PyPI JSON API — typer 0.27.0 metadata](https://pypi.org/pypi/typer/0.27.0/json) — click-free deps confirmed (`shellingham`, `rich>=13.8.0`, `annotated-doc`, `colorama`), requires `>=3.10`, released 2026-07-15. HIGH confidence.
- [PyPI JSON API — version/release-date checks](https://pypi.org/pypi/rich/json) — rich 15.0.0 (2026-04-12), click 8.4.2 (2026-06-24), plotext 5.3.2 (2024-09-24), jinja2 3.1.6, hatchling 1.31.0 (2026-07-08), setuptools 83.0.0, poetry 2.4.1, uv 0.12.0, torch 2.13.0, pandas 3.0.5, numpy 2.5.1, matplotlib 3.11.1, networkx 3.6.1, transformers 5.14.1, nltk 3.10.0, reportlab 5.0.0, pytest 9.1.1, ruff 0.16.1, mypy 2.3.0, argcomplete 3.7.0, rich-argparse 1.8.0, asciichartpy 1.5.25, unicodeplots (404 = dead). HIGH confidence.
- [devguide.python.org — Status of Python versions](https://devguide.python.org/versions/) — 3.8 EOL 2024-10-07, 3.9 EOL 2025-10-31, 3.10 EOL 2026-10, 3.11 EOL 2027-10, 3.12 EOL 2028-10. HIGH confidence.
- [Context7 — Typer docs (/fastapi/typer)](https://ctx7.dev) — CLI pattern `app = typer.Typer()`, type-hint arguments, `--version` callback via `is_eager`. MEDIUM-HIGH (Context7's listed version 0.21.1 is stale vs PyPI 0.27.0; patterns still current).
- [Context7 — Rich docs (/textualize/rich)](https://ctx7.dev) — Table/Panel/Markdown/Console usage. HIGH.
- [Context7 — Plotext docs (/piccolomo/plotext)](https://ctx7.dev) — `plt.bar()`/`plt.plot()`/`plt.show()` API, zero core deps. HIGH.
- Repo verification: `.planning/codebase/STACK.md`, `src/` layout scan, grep of import sites (`from src.X` × 9), `_init_.py` misnaming, plotly confined to `app/streamlit_app.py`, matplotlib-only core visualizations (`visualization.py`), no NLTK imports in `src/`. HIGH confidence (direct file inspection).

---
*Stack research for: Chat-Analyzer-Pro CLI pivot*
*Researched: 2026-07-31*
