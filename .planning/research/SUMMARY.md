# Project Research Summary

**Project:** Chat-Analyzer-Pro — CLI pivot from Streamlit app
**Domain:** pip-installable chat-analysis CLI (WhatsApp `.txt` / Telegram `.json` → terminal insights + inline charts + self-contained HTML report)
**Researched:** 2026-07-31
**Confidence:** HIGH

## Executive Summary

This is a greenfield pivot: repackage the existing Streamlit web app's analysis core (`src/`) as a pip-installable CLI — one command, `analyze chat.txt`, turns a raw chat export into in-terminal insights plus a single-file HTML report. The analysis engine already exists and is ~85% correct; **v1 is not an analysis problem, it is an exposure, packaging, and robustness problem.** The recommended approach is a thin Typer-based CLI over the untouched library core: one canonical `AnalysisResults` contract (built by an adapter layer) feeds two independent renderers — rich/plotext terminal output and a base64-embedded, zero-JS HTML report. The differentiators (emotion classification, relationship health score, summarization, network graph, Telegram support) are nearly free because the engine is already built; the work is in the CLI surface and data correctness.

All four research streams converge on the same first move and the same top risks. Before any feature work: repackage `src/` → `src/chat_analyzer/` (the misnamed `_init_.py` markers and generic top-level names make the package unshippable on PyPI today), raise the Python floor to `>=3.11` (the project's stated "3.8+" floor is uninstallable with current dependencies — pandas 3.0.5, matplotlib 3.11.1, and networkx 3.6.1 all require it), and split heavy ML deps into an `[nlp]` optional extra so base installs stay fast. **Top risks:** (1) parser correctness — the existing parser silently fabricates timestamps on date-format failures, so wrong numbers ship with exit code 0; (2) Windows console encoding — cp1252 crashes kill the tool (and its own error messages) for default CMD users; (3) first-run UX lies — a 2 GB CUDA-bundled torch install plus silent model downloads would destroy the non-technical user's first impression.

Recommended roadmap: **7 phases** — package surgery + CLI skeleton → parser hardening → pipeline + `AnalysisResults` contract → terminal output + inline charts → HTML report → NLP extras gate + error-handling UX → CI + real tests. Each phase maps to specific pitfalls (full table in PITFALLS.md) and specific features (v1 cut in FEATURES.md). Overall confidence is HIGH — stack versions verified against PyPI JSON + CPython lifecycle data, architecture verified against the actual `src/` code — with a handful of decision flags for planning: the `analyze` command name collides with existing PyPI tools, jinja2-vs-stdlib templating conflict between STACK and ARCHITECTURE research, plotext's maintenance risk, and iOS `.zip` scope.

## Key Findings

### Recommended Stack

The stack is settled (STACK.md, HIGH confidence — every version verified against PyPI JSON API and devguide.python.org on 2026-07-31). The one non-negotiable change: **Python `>=3.11` floor**. 3.8/3.9 are EOL; 3.10 is security-only and EOLs Oct 2026; the modern analysis stack (pandas 3.0.5, matplotlib 3.11.1, networkx 3.6.1, typer 0.27) all require `>=3.10`/`>=3.11`. The PROJECT.md "3.8+" constraint must be updated.

**Core technologies:**
- **Python `>=3.11`** — runtime floor; 3.11 has security support to Oct 2027. Everything else in the stack resolves on it.
- **typer 0.27.0** — CLI framework. Verified: 0.26+ dropped click entirely and bundles rich as a hard dep — one install buys CLI + terminal rendering. Entry point: `analyze = "chat_analyzer.cli.main:app"`.
- **rich 15.0.0** — terminal rendering (tables, panels, progress, status). Comes free with typer.
- **plotext 5.3.2** — inline ASCII charts (bar/line/hist). Zero deps. Known maintenance risk (author announced rewrite) and Windows Terminal distortion — plan an ASCII-fallback path (see Pitfalls).
- **jinja2 3.1.6** — HTML templating with `{{ value|e }}` autoescape. Chat messages are user content interpolated into HTML — autoescape is a correctness/security feature, not a luxury. (ARCHITECTURE.md proposed stdlib `string.Template`; STACK's security rationale wins — see Gaps.)
- **hatchling 1.31.0** (build backend) + **uv 0.12.0** (dev-only venv/lockfile — end users still `pip install`) — the 2026 standard pairing.
- **pandas `>=2.0`** (do NOT blind-bump to 3.x — copy-on-write/string-dtype default changes; test against 3.0.5 in a dedicated task later) · **numpy `>=1.24`** floor (never pin 2.5.1 — requires 3.12) · **transformers `>=4.30,<6`** (5.x is breaking vs the 4.x-era core code) · **torch `>=2.0`** (only via `[nlp]` extra) · **networkx `>=3.1`**.

**Repackaging (blocks everything):** the core is importable today only via PEP 420 namespace tolerance — package markers are misnamed `_init_.py` and `src` cannot ship as a top-level package. Move `src/{analysis,ingest,parser,reporting,utils}` → `src/chat_analyzer/`, create real `__init__.py` files, rewrite 9 `from src.X` import sites. Drop: reportlab (PDF out of scope), all Streamlit-era web deps, NLTK (never imported in `src/`), pytesseract/pdfplumber (OCR probes).

### Expected Features

The analysis engine exists, so all complexity ratings reflect **CLI-wrapping cost** — the differentiators are nearly free. Feature research confidence: HIGH (ecosystem-verified) except CLI-UX conventions (LOW, flagged).

**Must have (table stakes):**
- One-command pipeline — `analyze <file>` = full run, exit 0/1. The highest-priority feature in the product.
- Summary + per-participant stats, timeline trends, hour-of-day activity — all exist in the core (`ChatEDA`).
- Word frequency + emoji breakdown — exist; wordcloud/heatmap render HTML-only.
- Sentiment (pos/neg/neutral + over time) — VADER path always works; HF optional behind `--light`.
- Terminal output with inline charts (the pivot's stated differentiator) + self-contained HTML report (base64 PNG, no JS, offline).
- Friendly errors with export instructions (wrong file/format/locale → remediation + exit code 1), progress indication, `--help`/`--output`/`--quiet`.
- WhatsApp `.txt` AND Telegram `.json` (auto-detect by extension + content sniff).

**Should have (differentiators — nearly free, ship in v1):**
- Emotion classification (6 classes, HF + rule-based fallback) — almost nobody in the ecosystem has it.
- **Relationship health score** — the signature hook; include in v1 as the headline panel.
- Conversation summarization (slowest stage; runs last; skipped by `--light`) · Network graph for group chats (HTML-only) · Privacy story ("100% on your device" — structurally true for a local CLI) · `--light` fast path · Telegram support (ecosystem is ~all WhatsApp-only).
- v1.x: `--user`/`--from`/`--to` filters, `--json` output, iOS `.zip` (cheap — ingestion already handles it; decide v1 vs v1.x in planning).

**Anti-features (never):** interactive TUI, cloud/accounts/telemetry (contradicts the privacy core value), PDF report, Excel export, OCR/PDF ingestion, runtime `exec()` of downloaded code (critical vulnerability carried from the web app — delete it), "wrapped" slideshows, live chat monitoring.

**Terminal vs HTML split (design decision):** terminal = curated executive summary (rich tables/panels + plotext); HTML = full exploration (heatmaps, wordcloud, network graph, quote blocks). Both consume the same `AnalysisResults` — analysis runs once, two renderers.

### Architecture Approach

The existing `src/` is 85% correct — DataFrame-centric, UI-independent modules. The fix (ARCHITECTURE.md, HIGH — verified against actual code + PyPA docs): nest everything under one real import package `chat_analyzer`, add a `cli/` subpackage, wire a Typer entry point. The core is **reused as-is**; the CLI is a thin orchestration + rendering layer. Only four surgical core changes: rename+clean package markers, fix 3 import sites, add `messages_to_dataframe()` (moved from `app/streamlit_app.py:313` into `ingest/ingestion.py`), make `summarizer.py`'s transformers import lazy.

**Major components (package layout):**
```
src/chat_analyzer/
├── parser/      # whatsapp_parser, telegram_parser — unchanged
├── ingest/      # ingestion.py + messages_to_dataframe() (moved from app)
├── analysis/    # eda · sentiment · emotion · relationship_health · network_graph · summarizer — unchanged
├── reporting/   # pdf_report, weekly_digest — deferred, kept importable
├── utils/       # preprocessing · visualization (ChatVisualizer → PNG for HTML)
└── cli/         # ALL CLI-specific code (new)
    ├── main.py        # Typer app + console script target
    ├── __main__.py    # python -m chat_analyzer
    ├── pipeline.py    # run_pipeline(path, opts) → AnalysisResults — the only orchestration
    ├── contracts.py   # AnalysisResults TypedDict — canonical contract
    ├── adapters.py    # module dicts → AnalysisResults (pure functions, only place knowing module dict shapes)
    ├── render.py      # rich + plotext → terminal
    ├── report_html.py # single-file HTML + base64 PNGs
    └── errors.py      # exit codes + friendly errors + export instructions
```

**Key patterns:**
1. **Thin CLI over a library core (Facade + Pipeline)** — no business logic in the CLI; the old app's duplicated-logic drift (health dict with a *different shape*) is the anti-pattern to never repeat.
2. **Canonical `AnalysisResults` TypedDict via adapters** — one contract, two renderers; adding an output means adding a renderer, never changing the pipeline.
3. **Optional heavy deps: `[nlp]` extra + lazy imports + degrade-not-crash** — base install = pandas + vaderSentiment + rich/typer + plotext; `pip install chat-analyzer-pro[nlp]` unlocks emotion + summarization. Missing extra → actionable hint, never a traceback.
4. **Single-file HTML via data URIs** — `ChatVisualizer` figures → PNG → base64; no assets, no CDN, opens by double-click.

Data flow: `main.py` (validate) → `pipeline.py` (ingest → df → EDA/sentiment/health [+NLP] → `adapt()`) → both renderers → exit 0 / 1 (analysis) / 2 (usage). Process-local state only; keep module-level model singletons (load once per run); normalize all timestamps to **naive UTC** at the parser boundary (Telegram is tz-aware UTC, WhatsApp is naive local — mixing crashes `sort_values`).

### Critical Pitfalls

13 pitfalls documented in PITFALLS.md (MEDIUM-HIGH confidence, grounded in actual code lines). The ones that decide the roadmap:

1. **Silent `datetime.now()` fallback fabricates timestamps** (`whatsapp_parser.py:61,63,77,79`) — every unhandled date variant becomes "today". Invisible corruption with exit code 0. Fix in Phase 2: strict parse + `skipped_lines` counter surfaced to the user. **Never fabricate a timestamp.**
2. **WhatsApp regional date ambiguity (M/D vs D/M)** — "the single biggest source of parser bugs." Fix: sample-based format detection (AM/PM → 12h, 4-digit year → `%Y`, month > 12 resolves day/month), support iOS-bracket + Android-dash variants, `--locale`/`--date-format` override.
3. **System messages + localized media placeholders leak into content** — "Messages and calls are end-to-end encrypted", `Alex added Sam`, `<Média absent>` (fr), `<Medien ausgeschlossen>` (de). English-only string matching misses non-English exports. Fix: classify structurally (no `sender:` part → system; empty/localized-placeholder body → media); skip the line-2 encryption notice.
4. **Telegram JSON shape drift** — single-chat export ≠ full export (`chats.list[]`); `text` is a string OR array of entity dicts; `except: continue` silently drops malformed messages. Fix: support all three top-level shapes, recursively join `text`, count-and-report drops.
5. **Windows console encoding (cp1252)** — `UnicodeEncodeError` crashes the CLI, and it also kills the *error handler* (fallback message itself prints `❌`), and redirection (`> file`) crashes where interactive worked. Fix in Phase 1, day one: `sys.stdout/stderr.reconfigure(encoding='utf-8', errors='replace')` + ASCII-first output (`[OK]`/`[ERROR]`, `|`/`-`).
6. **Heavy install + first-run model download** — PyPI's default Windows torch wheel bundles CUDA (~2–2.5 GB); first run then silently downloads `t5-small` (~242 MB) + distilbert (~260 MB). Fix: `[nlp]` extra split, lazy imports, print model name + size before `from_pretrained`, honor `HF_HUB_OFFLINE=1`, document the CPU-only index-url.
7. **Import-time transformers crash** (`summarizer.py:12` top-level import) — base install breaks when the CLI merely imports the pipeline; `--help` must never import torch. Fix: lazy import inside `_ensure_model()` + friendly "install `chat-analyzer-pro[nlp]`" hint.
8. **Packaging landmines + the `analyze` command name** — misnamed `_init_.py` breaks `pip install`; console scripts fail when the target isn't importable; `analyze` already collides with existing PyPI packages; wheels silently omit non-.py files (templates) unless declared. Fix in Phase 1; consider a collision-resistant name + always-working `python -m chat_analyzer`.
9. **HTML report encoding/escaping/path** — cp1252 mojibake, unescaped `<script>` from messages (injection vector in shared reports), cwd-relative output the user can't find, matplotlib `Agg` backend needed headless. Fix in Phase 5: `encoding='utf-8'` + `<meta charset>` + escape everything + report next to input + `matplotlib.use('Agg')` before import.
10. **tz-naive/aware datetime mixing** — live bug in the codebase (Telegram aware-UTC vs WhatsApp naive). Fix at the parser boundary: normalize to naive UTC; schema test asserts `df['datetime'].dt.tz is None`.

## Implications for Roadmap

Seven phases, dependency-ordered. This merges ARCHITECTURE.md's 7-step build order with PITFALLS.md's P1–P7 phase mapping (they align 1:1). Phase references from PITFALLS.md in parentheses.

### Phase 1: Package surgery + CLI skeleton (P1)
**Rationale:** Nothing is installable today — misnamed `_init_.py` markers, `src` as a top-level namespace, no pyproject. Every later phase needs `pip install -e .` + a working console script.
**Delivers:** `src/chat_analyzer/` package (git mv + clean `__init__.py` markers), 3 import-site fixes, `pyproject.toml` (hatchling, `requires-python >=3.11`, pruned dependency list, `[nlp]`/`[dev]` extras, entry point), Typer skeleton with instant `--help`, `__main__.py`, console encoding bootstrap.
**Addresses:** foundation for all table-stakes features; `--help` works.
**Avoids:** Pitfalls 5 (encoding), 8 (import-time crash review rule), 10 (packaging landmines + command-name decision), 12 (Streamlit-era dependency baggage — `pip freeze` must show no streamlit/plotly/seaborn).
**Decisions to make in planning:** command name (`analyze` collides — candidates: `chat-analyzer`, `cpro` + documented `python -m chat_analyzer` fallback); exact extras split metadata.

### Phase 2: Parser hardening (P2)
**Rationale:** Data correctness is decided here, before the pipeline and renderers build on top. The current parser fabricates timestamps and misparses regional variants — shipping that would ship wrong insights with a clean exit code. Cheapest to fix now.
**Delivers:** sample-based date-format detection, strict parsing with skip-count warnings (never `datetime.now()`), structural system/media classification with localized placeholders, Telegram 3-shape support + recursive `text` walk + drop counters, naive-UTC normalization at the parser boundary, `messages_to_dataframe()` moved into `ingest/ingestion.py`, parser-level diagnostics (needed by Phase 6 friendly errors).
**Addresses:** correctness foundation for every stats feature.
**Avoids:** Pitfalls 1, 2, 3, 4, 9. Verification: US-12h / EU-24h / iOS-bracket / 4-digit-year / FR-DE media fixtures; single-chat vs full-export Telegram JSON.

### Phase 3: Pipeline core + AnalysisResults contract (ARCH step 3)
**Rationale:** Architecture's core decision — one `run_pipeline`, one contract, two renderers. Depends on Phase 1 (package) + Phase 2 (correct parsers).
**Delivers:** `cli/pipeline.py` (staged, modular — progress hooks and `--light`/`--quiet` can attach per stage), `contracts.py` (`AnalysisResults` TypedDict), `adapters.py`, lazy summarizer import, one-command `analyze chat.txt` end-to-end to a dict, exit codes.
**Addresses:** table stakes — one-command pipeline, summary + per-participant stats, timeline + activity metrics, top words/emojis, sentiment (VADER), relationship health data — all via direct calls to existing core modules (never re-implemented).
**Avoids:** Pitfall 8 (lazy imports), anti-pattern 3 (re-implementing analysis — the old app's sin).

### Phase 4: Terminal output + inline charts (P3)
**Rationale:** The pivot's stated differentiator ("full insights stay in-terminal"). Renderer is an independent consumer of `AnalysisResults`, so it builds cleanly on Phase 3.
**Delivers:** `cli/render.py` — rich KPI panel + top-10 tables + plotext timeline/hour charts, ASCII-first with chart `try/except` degrade + `isatty()` gating + `--no-charts`, rich `Progress`/`Status` narration of every stage.
**Addresses:** terminal-first output split, progress indication (table stakes).
**Avoids:** Pitfall 6 (chart failures on Windows/pipes — degrade to text, never crash), Pitfall 13 (silent long waits — narrate every stage, print parsed-message count immediately).

### Phase 5: HTML report (P4)
**Rationale:** The shareable half of the duality; reuses the same `AnalysisResults` + existing `ChatVisualizer` figures via the base64 path. Zero new analysis work.
**Delivers:** `cli/report_html.py` (jinja2 template with autoescape), single self-contained file, KPI cards, hour×day heatmap, wordcloud, network graph, sentiment/emotion timelines, quote blocks; `matplotlib.use('Agg')`; report written next to the input with absolute path printed; auto-open (semantics tied to `--quiet`).
**Addresses:** self-contained HTML report (table stakes); wordcloud/heatmap/network (HTML-only items).
**Avoids:** Pitfall 11 (encoding/escaping/path), anti-pattern 4 (logging/print leakage from core).

### Phase 6: NLP extras gate + error-handling UX (P5 + P6)
**Rationale:** Heavy features must never affect the base install; friendly errors need Phase 2's parser diagnostics and are the non-technical user's lifeline. Last functional phase because `[nlp]` is a pure add-on over the finished pipeline.
**Delivers:** `[nlp]` extra end-to-end — `--light` fast path (VADER + rule-based emotion), 6-class emotion, summarization, model-download notice (name + size before `from_pretrained`), degrade-not-crash hints; `cli/errors.py` — file-not-found / wrong-format / empty / 0-messages each with remediation + export instructions; `--quiet`/`--output` semantics; exit codes 2 (usage) / 1 (analysis) / 0.
**Addresses:** differentiators — emotion, summarization, `--light`, relationship-health headline panel; friendly errors + basic flags.
**Avoids:** Pitfall 7 (install/model-download UX), Pitfall 8 (enforcement — fresh venv without `[nlp]`: help works, AI features print guidance).

### Phase 7: CI + real tests + cleanup (P7)
**Rationale:** Everything prior ships on trust until tests exercise the production modules in clean environments. The current `tests/` are self-contained mock reimplementations (CONCERNS.md) — they verify nothing.
**Delivers:** tests rewired to import `chat_analyzer.*` (production parsers, not mocks), fixture-based parser/HTML/encoding tests, clean-venv `pip install .` smoke test on Windows + Linux CI, Windows console (`chcp 1252`) + redirection tests, README quickstart, delete `app/` + `deployment/` (incl. the `exec()` module-fetcher).
**Addresses:** quality gate for everything; privacy story baked into README/help.
**Avoids:** Pitfall 10 verification (clean-venv console script), the "Looks Done But Isn't" checklist in PITFALLS.md.

### Phase Ordering Rationale
- **Package → parser → pipeline → renderers → extras → polish/CI** is a strict dependency chain: nothing installs before Phase 1; renderers need the contract before they render; NLP is an add-on over a finished pipeline.
- **Parser hardening precedes the pipeline** because data correctness is the foundation — renderers must never display fabricated timestamps (Pitfall 1), and parser diagnostics built in Phase 2 are consumed by friendly errors in Phase 6.
- **Both renderers come after one contract** — FEATURES.md's single most important constraint: analysis runs once, both renderers read the same dict. This keeps the terminal/HTML duality cheap.
- **UX polish lands before CI** so CI verifies the final behavior (exit codes, error messages, redirection safety), not a half-finished CLI.

### Research Flags
Phases likely needing deeper research/spikes during planning:
- **Phase 4:** plotext's maintenance status + Windows Terminal distortion — small spike: verify plotext output on a real Windows console at 80/40 cols; fallback is rich-only ASCII bars for v1 (PITFALLS.md's own suggestion). Decision, not open research.
- **Phase 6:** HF model download UX — verify model sizes/cache behavior with a cold `~/.cache/huggingface` on slow network; `--light` vs full default semantics. Model sizes in research are approximate (MEDIUM confidence).
- **Phase 1:** command-name collision research — check current PyPI/`which analyze` conflicts before committing to the name. Small, targeted.

Phases with standard patterns (skip research-phase):
- **Phase 2:** community patterns are well-documented (whatsapp-chat-analyzer's locale-hint ParserConfig; Telegram's official import-export schema); fixtures are enumerable.
- **Phase 3 / 5:** facade/adapters/TypedDict contracts and base64-embedded HTML are textbook patterns; packaging docs (PyPA src-layout) are authoritative.
- **Phase 7:** standard pytest + GitHub Actions matrix (windows-latest + ubuntu-latest) practices.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All versions verified against PyPI JSON API + devguide.python.org on 2026-07-31 (source: STACK.md) |
| Features | HIGH (ecosystem) / LOW (CLI-UX conventions) | Competitor features from primary sources (whatsapp-wrapped, whatstk); non-technical-user CLI conventions flagged LOW for validation (source: FEATURES.md) |
| Architecture | HIGH | Verified against actual `src/` code lines + PyPA official docs (source: ARCHITECTURE.md) |
| Pitfalls | MEDIUM-HIGH | Grounded in specific code lines + official docs; HF model sizes approximate; plotext maintenance risk is forward-looking (source: PITFALLS.md) |

**Overall confidence:** HIGH — the four research files cross-corroborate on phase structure and the repackaging-first conclusion.

### Gaps to Address
- **jinja2 vs stdlib template conflict:** STACK.md (HIGH) recommends jinja2 for autoescape; ARCHITECTURE.md proposes stdlib `string.Template` to avoid a dep. **Recommendation: jinja2** — chat content is untrusted user input and the report has many sections; `html.escape()` everywhere is more error-prone than template-level autoescape. Resolve in Phase 5 planning.
- **Python floor conflict:** ARCHITECTURE.md's example says `>=3.9`; STACK.md's verified data says `>=3.11`. **Adopt `>=3.11`** (pandas 3.0.5 / matplotlib 3.11.1 / networkx 3.6.1 require it) and update PROJECT.md's obsolete "3.8+" constraint during Phase 1.
- **Command name:** `analyze` collides with existing PyPI tools and may not be on Windows Scripts PATH. Decide in Phase 1; renaming later breaks README and user habits.
- **iOS `.zip` scope:** PROJECT.md scopes v1 to `.txt`/`.json`, but ingestion already handles zip — likely cheap to pull into v1. Decide in planning (FEATURES.md P2 flag).
- **plotext dependency risk:** unmaintained-author risk + Windows rendering issues — the Phase 4 spike should either pin it with an ASCII-fallback budget or go rich-only bars for v1.
- **`--quiet` vs auto-open semantics:** FEATURES.md recommends quiet = suppress stdout chatter but still write/open the report. Confirm in Phase 6 planning.
- **`[nlp]` vs `[ai]` extra naming:** ARCHITECTURE/STACK say `[nlp]`; PITFALLS examples say `[ai]`. Use `[nlp]` for consistency; update PITFALLS examples when writing the pyproject.
- **Chat-size cap value** (~100 MB warn/refuse) and WhatsApp locale override flag name — decide during Phase 2 planning.

## Sources

### Primary (HIGH confidence)
- PyPI JSON API — typer 0.27.0 metadata (click-free deps, `>=3.10`) and version/release-date checks for the full stack (rich, plotext, jinja2, hatchling, torch, pandas 3.0.5, numpy, matplotlib 3.11.1, networkx, transformers, pytest, ruff, mypy)
- devguide.python.org/versions — CPython lifecycle: 3.8 EOL 2024-10, 3.9 EOL 2025-10, 3.10 EOL 2026-10, 3.11 EOL 2027-10
- packaging.python.org (src-layout vs flat-layout), setuptools package discovery, entry-points spec
- core.telegram.org/import-export — official Telegram export schema (Chat object shapes, `text` as String/Array, service messages, >32-bit ids)
- Repo-internal verification: `.planning/codebase/` (STACK/ARCHITECTURE/STRUCTURE/CONCERNS), `src/` line-level inspection (whatsapp_parser.py, telegram_parser.py, summarizer.py:12/59, relationship_health.py:800, visualization.py)
- GitHub issue sources for Windows encoding crashes (caveman#152, AIPass#296, rich#2882/#3437) and the `sys.stdout.reconfigure` fix

### Secondary (MEDIUM confidence)
- whatsapp-wrapped (Duelion) GitHub README — closest competitive analog (flags, privacy framing, HTML formats)
- whatsapp-chat-analyzer (PyPI) — community parser patterns: locale hint for M/D vs D/M, multilingual media classification, BOM handling
- whatstk, huzefa10/whatsapp-chat-analyser, whatsapp-reality, gauravmeena0708, MendasD, JBoixCampos, Anish62027 — competitor feature landscape and anti-feature evidence
- whatswquiz.com + chatanalyzer.syncori.net — WhatsApp export format quirks (regional dates = "biggest source of parser bugs"; iOS/Android structure; 40k export cap)
- PyTorch forums + lerobot docs — Windows default CUDA wheel (~2–2.5 GB), CPU-only `--index-url`, per-platform index-urls inexpressible in pyproject
- plotext README (3×2 mosaic markers "not available in windows"; author maintenance notice) + HN report of Windows Terminal distortion
- Context7 library docs (Typer, Rich, Plotext) — API patterns, MEDIUM-HIGH

### Tertiary (LOW confidence)
- CLI UX conventions for non-technical users (help text, exit codes, remediation-in-errors) — flagged for validation during Phase 6 planning
- HF model download sizes (t5-small ~242 MB, distilbert ~260 MB) — approximate, verify in the Phase 6 spike

---
*Research completed: 2026-07-31*
*Ready for roadmap: yes*
