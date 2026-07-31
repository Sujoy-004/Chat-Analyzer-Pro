# Walking Skeleton — Chat-Analyzer-Pro

**Phase:** 1
**Generated:** 2026-07-31

## Capability Proven End-to-End

> One sentence: the smallest user-visible capability that exercises the full stack.

"A friend clones the repo, runs `pip install .`, gets a `chat-analyzer` command (with a working `python -m chat_analyzer` fallback) that answers `--help` instantly, and — running it with no arguments — is asked 'Enter path to chat export'; typing the path loads the existing analysis engine (`process_uploaded_file`) and reports how many messages were processed."

This is a Python CLI repackaging phase, not a web app, so the skeleton proves the **packaging stack** end-to-end: `pip install .` works → `chat-analyzer` console script exists → `python -m chat_analyzer` works → existing analysis modules import as `chat_analyzer.*` → the interactive prompt hands a real export to the existing ingestion code with zero new file-reading logic.

## Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| CLI framework | typer 0.27 (rich 15 bundled; click-free) | Verified 2026 standard for type-hinted CLIs; one install buys CLI + terminal rendering; instant auto `--help` |
| Build backend | hatchling 1.31 (PEP 517/621), src-layout | Zero-boilerplate package discovery for `src/chat_analyzer`; de-facto standard (uv scaffolds with it) |
| Python floor | `>=3.11` | Stack-verified: pandas 3.0.5 / matplotlib 3.11.1 / networkx 3.6.1 / typer 0.27 all require it; 3.11 has security support to Oct 2027; replaces the obsolete "3.8+" constraint (D-09) |
| Package layout | `src/chat_analyzer/` as the single top-level import package | Kills the `src.parser` / `src.analysis` / `src.utils` PEP 420 namespace collision; `parser`/`utils` are guaranteed collision names on PyPI; one import root per project (PyPA src-layout convention) |
| Entry points | Console script `chat-analyzer = chat_analyzer.cli:app` (D-01) + package-level `src/chat_analyzer/__main__.py` (D-02) | `python -m chat_analyzer` works even when Scripts/ isn't on PATH; both defer to the same Typer app |
| CLI UX | Interactive prompt ("Enter path to chat export:"), zero flags | Target user is a non-technical friend; D-03/D-04; no positional-arg support in v1 |
| Dependency strategy | Lean base deps + `[nlp]` extra (`torch>=2.0`, `transformers>=4.30,<6`) + lazy imports | Base install must not pull torch/transformers/streamlit/plotly (PKG-02/03); `<6` pin is critical (transformers 5.x breaks the 4.x-era core code) |
| Heavy-dep gating | try/except ImportError + `*_AVAILABLE` flags (existing codebase convention) | `import chat_analyzer.analysis.summarizer` succeeds without transformers (import moved into `__init__`); degrade-not-crash, never a traceback |
| CLI startup | Light package markers; no eager subpackage imports; analysis modules imported inside the command handler | `import chat_analyzer` loads no pandas/matplotlib/reportlab → `--help` is instant (research Anti-Pattern 2 / Pitfall 8) |
| Windows robustness | `sys.stdout/stderr.reconfigure(encoding="utf-8", errors="replace")` bootstrap + ASCII-first output | Default CMD cp1252 must never crash the tool or its own error messages (research Pitfall 5) |
| Web app | Deleted: `app/`, `deployment/`, `.streamlit/`, `apt.txt`, `packages.txt`; requirements.txt superseded by pyproject.toml | D-05/D-06; removes the `exec()` remote-module-fetcher and `unsafe_allow_html` injection vectors (CONCERNS.md); single dependency manifest |
| Reporting modules | Ship in the package, importable, NOT wired into the CLI | D-10 ships everything; D-11 defers PDF/digest CLI exposure to v2 |

## Stack Touched in Phase 1

Adapted from the web-app skeleton template to the CLI packaging stack (the "stack" here = pyproject + package markers + CLI entry + import verification):

- [x] Project scaffold — pyproject.toml (hatchling src-layout), `src/chat_analyzer/` package with valid `__init__.py` markers, Python `>=3.11` floor
- [x] "Routing" — console script `chat-analyzer` + package-level `__main__.py` (`python -m chat_analyzer`), both wired to the same Typer app
- [x] "Database" (→ import verification) — all 20 `chat_analyzer.*` modules (analysis/parser/ingest/utils/reporting/cli) import in a clean subprocess; the existing analysis core runs unchanged (`process_uploaded_file` → 27 messages from whatsapp_sample.txt)
- [x] "UI" (→ interactive interaction) — one real interaction wired to the core: the path prompt feeds `process_uploaded_file` (already accepts `str` paths; zero new file-reading code)
- [x] "Deployment" (→ documented local run) — `pip install .` then `chat-analyzer` (or `python -m chat_analyzer`); no hosting, no accounts, no telemetry

## Out of Scope (Deferred to Later Slices)

> Anything that is *not* in the skeleton. Explicit — prevents later phases from re-litigating Phase 1's minimalism.

- Parser hardening (strict date parsing, skip counters, naive-UTC normalization, Telegram shape handling) — Phase 2
- `AnalysisResults` pipeline contract + adapters — Phase 2
- Terminal rendering (rich tables/panels, plotext ASCII charts) — Phase 2
- Single-file HTML report (jinja2 autoescape, base64 PNGs) — Phase 3
- `[nlp]` extra wiring end-to-end (emotion, summarization, model-download notices) — Phase 4
- Friendly "how to export chats" error guidance, `--light`, `--output`, `--quiet` — Phase 4
- Tests rewired to exercise real modules at scale + clean-venv CI install smoke — Phase 4
- PyPI publication / `pip install git+...` instructions — post-v1 (D-08)
- PDF report + Telegram digest CLI exposure — v2 (D-11)
- Instagram/Messenger/Discord import — v2

## Subsequent Slice Plan

Each later phase adds one vertical slice on top of this skeleton without altering its architectural decisions:

- Phase 2: One-command terminal insights — parser hardening + pipeline + rich/plotext terminal output ("correct insights in one command")
- Phase 3: Shareable HTML report — self-contained single-file report with `--output` and `--no-report`
- Phase 4: NLP extras & quality gate — `[nlp]`-gated emotion/health/summary/network, friendly errors, real tests, README quickstart
