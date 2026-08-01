# Phase 2: One-Command Terminal Insights — Research

**Researched:** 2026-08-01
**Domain:** Python CLI pipeline — strict chat-export parsing (WhatsApp .txt / Telegram .json) → DataFrame → EDA + VADER sentiment → single-file HTML report card + rich terminal narration
**Confidence:** HIGH (all codebase claims verified by direct file inspection + live runtime experiments; versions verified against the installed environment; see per-claim tags)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**CLI Invocation (no flags — keep it simple)**
- **D-01:** Primary flow: `python -m chat_analyzer` → terminal prompts "Enter path to chat export" → user gives location → tool analyzes → writes HTML report card. The Phase 1 interactive prompt remains the default no-arg UX.
- **D-02:** `chat-analyzer <chat_file>` positional argument ALSO supported (CLI-02) — direct path runs without prompting. No-arg still prompts.
- **D-03:** NO CLI flags ship in this phase (not even `--no-charts`, `--date-format`, `--output`). `--help`/`--version` only. Keep it dead simple. Phase 3's `--output` (OUT-04) and Phase 4's `--with-nlp` come later.
- **D-04:** The HTML report is the **deliverable**; the terminal is the entry point + progress + pointer to the report.

**Progress & Error UX**
- **D-05:** Stage narration via rich Status/spinners: "Parsing chat…", "Computing insights…", "Writing report…". Parsed-message count surfaced immediately after parsing (CLI-03, success criterion 4). Windows-safe ASCII spinners.
- **D-06:** On bad path or unparseable file: friendly message + loop back to re-prompt (matches Phase 1's file-not-found loop). Full friendly-error-with-export-instructions is CLI-04 in Phase 4.

**Terminal Output**
- **D-07:** Terminal shows a compact summary panel AFTER analysis: total messages, participant count, date range. Plus the absolute report path. NO sentiment in terminal, no tables/panels of insights, no charts — insights live in the HTML report.

**HTML Report Card (the deliverable)**
- **D-08:** Written next to the input file as `<chat_name>_report.html`; absolute path printed after writing.
- **D-09:** Auto-opens in the default browser after generation (pulls forward v2 CLI-08). On failure to open, degrade gracefully (still print the path).
- **D-10:** Single self-contained HTML file: embedded CSS + JS tabs (no external libs, no CDN), charts base64-embedded as PNG data URIs, opens offline by double-click. `<meta charset="utf-8">`, `html.escape()` on all chat-derived content.
- **D-11:** Tabbed sections, each opening with a **narrative insight lead-in sentence** ("Alice initiated 65% of conversations…", "Most messages land on weekends…"), backed by charts/tables. Suggested tabs: Overview, Participants, Conversation Flow, Words & Emojis, Sentiment.
- **D-12:** Report includes matplotlib charts rendered to base64 PNG: messages-per-day line, hourly activity (heatmap or bar), per-participant bar, sentiment timeline. Reuses existing `ChatVisualizer` (already produces these figures). Set `matplotlib.use('Agg')` before any matplotlib import.
- **D-13:** Full depth of insights: EDA (volume, participants, date range, busiest day/week/hour, response time avg) + top words/emojis + VADER sentiment breakdown. Relationship health, emotion, summary, network EXCLUDED (Phase 4).
- **D-14:** Report filename sanitized (strip path separators, control chars, leading dots) even when derived from input name.

**Parser Hardening (Pitfalls 1-4, 9)**
- **D-15:** Strict parse: NEVER fabricate timestamps (`datetime.now()` fallback removed entirely). Lines failing to parse are counted in a `skipped_lines` counter and surfaced to the user.
- **D-16:** Skip surfacing: a single line — "Skipped N lines that couldn't be parsed" — in terminal narration + a note in the report. No per-line examples.
- **D-17:** Common WhatsApp formats only: existing multi-format attempts cover US 12h, EU 24h, iOS bracket, Android dash, 2/4-digit year, optional seconds. NO M/D-vs-D/M disambiguation heuristics, NO locale sniffing, NO override flags (user's focus is conversation insights, not date-format perfection).
- **D-18:** System messages (timestamp header but no sender — e.g. "Messages and calls are end-to-end encrypted", "X added Y") classified as `type="system"` and EXCLUDED from stats, counted in a separate counter. Never appended to the previous message.
- **D-19:** Telegram JSON: support both top-level shapes (bare Chat with `messages` for single-chat export; `chats.list[]` for full export), recursively join string + entity-dict `text` parts, filter `type="service"` messages, drop malformed via the shared skip counter (no silent `except: continue`).
- **D-20:** tz→naive UTC normalization at the parser boundary (Pitfall 9): Telegram tz-aware datetimes converted to naive UTC; both parsers produce naive-UTC `datetime` column. Schema test asserts `df['datetime'].dt.tz is None`.

**Requirement Re-mapping (to reconcile during planning)**
- **Pulled into Phase 2 from Phase 3:** OUT-03 (single-file HTML report), OUT-04 (`--output` path — note: flag deferred, default-path behavior ships now), OUT-05 (`--no-report` opt-out — report is the deliverable; confirm semantics in planning).
- **Dropped from Phase 2:** OUT-02 (inline plotext terminal charts) — plotext never ships; charts exist only in the HTML report. Update REQUIREMENTS.md.
- **Pulled forward from v2:** CLI-08 (auto-open report).
- **Stays Phase 4:** ANAL-07 (relationship health) as labeled; CLI-04 (friendly errors); ANAL-06/08/09 (emotion/summary/network, `[nlp]`-gated).

### the agent's Discretion
- Exact rich Status/panel styling and ASCII-safe symbols
- Report template structure and CSS design details (within the tabbed + narrative-lead-in decision)
- Which exact ChatVisualizer methods to reuse for the 4 chart types
- How the `messages_to_dataframe`/pipeline orchestration is structured (`cli/pipeline.py`, `adapters.py`, `contracts.py` per research) — reuse existing modules, no rewrite
- How OUT-05 (`--no-report`) semantics resolve in planning given the report is the deliverable

### Deferred Ideas (OUT OF SCOPE)

- Relationship health section in the report (initiator ratio, response lag, dominance, health score) — Phase 4 (ANAL-07, `[nlp]`-labeled)
- Emotion classification, conversation summarization, network graph — Phase 4 (`[nlp]`-gated ANAL-06/08/09)
- Friendly errors with WhatsApp/Telegram export instructions — Phase 4 (CLI-04)
- `--output` flag (OUT-04) — later phase (default-path behavior ships now)
- `--no-report` opt-out semantics — confirm in planning (report is the deliverable)
- Deep WhatsApp date-format disambiguation (M/D vs D/M majority vote, locale sniffing, `--date-format` override) — explicitly declined by user; keep common formats only
- plotext inline terminal charts (OUT-02) — dropped entirely; charts live only in the HTML report
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CLI-02 | `analyze <chat_file>` runs the full pipeline automatically | D-02 positional arg on `cli/main.py` `main()`; `run_pipeline()` in `cli/pipeline.py` (Pattern 2 below). Note: command name is `chat-analyzer` (Phase 1 D-01). |
| CLI-03 | Progress indicator while pipeline runs | rich `Status(..., spinner='line')` per stage (D-05); parsed-count surfaced immediately after parse (Pitfall 13 → verified rich Status API below). |
| ANAL-01 | Summary stats (volume, participants, date range, counts) | `ChatEDA.generate_comprehensive_summary()` + `analyze_message_volume()` (reused, unchanged) via adapter; date range from canonical df min/max. |
| ANAL-02 | Per-participant stats (messages, avg length, response behavior) | `df.groupby('sender')` message count + avg `message_length`; response time avg from `ChatEDA.analyze_conversation_dynamics()['avg_response_time']`. |
| ANAL-03 | Timeline/activity trends (messages per day/week/hour, busiest times) | `ChatEDA.analyze_message_volume()` (`daily_messages`, `hourly_activity`, `time_period_counts`, `sender_counts`); `peak_hour` from summary. |
| ANAL-04 | Top words and emojis with frequency | `ChatEDA.analyze_content()` (`word_frequency`, `emoji_frequency` — Counter objects). |
| ANAL-05 | VADER sentiment breakdown (per-message and per-participant) | `sentiment.add_sentiment_analysis()` + `get_sentiment_summary()` (VADER path always available; base install has vaderSentiment, no textblob/transformers → `consensus_sentiment` degrades to VADER). |
| OUT-01 | Terminal tables/panels — **REVERSED by D-04/D-07** | Insights live in the HTML report; terminal shows compact summary panel (volume/participants/date range) + path only. Must update REQUIREMENTS.md/ROADMAP.md (like OUT-02). |
| OUT-02 | Inline plotext terminal charts — **DROPPED** | plotext never ships (D-55); verified `plotext` imported nowhere in `src/` [VERIFIED: codebase grep]. Remove from pyproject deps. |
| (OUT-03 pulled fwd) | Self-contained single-file HTML report | `cli/report_html.py` jinja2 template, charts base64-embedded (verified Agg→PNG→data-URI pipeline works, see Code Examples). |
| (OUT-04 pulled fwd) | Output path — default-path behavior only, no flag | `<sanitized_input_stem>_report.html` next to input (D-08/D-14). |
| (OUT-05 pulled fwd) | `--no-report` — semantics to resolve in planning | Report IS the deliverable (D-04); no flag ships this phase (D-03). |
| (CLI-08 pulled fwd) | Auto-open report | `webbrowser.open("file://" + resolve())` in try/except, degrade to path print (D-09). |

</phase_requirements>

## Summary

Phase 2 builds the full "one command → report card" pipeline: `chat-analyzer <file>` (or the Phase 1 interactive prompt) parses a real WhatsApp `.txt` or Telegram `.json` export **strictly** (no fabricated timestamps, no silent drops, system messages classified out, tz→naive UTC), runs the existing analysis core (ChatEDA + VADER sentiment — reused unchanged), and renders a **self-contained tabbed HTML report card** with matplotlib charts embedded as base64 PNG data URIs and narrative insight lead-ins. The terminal is deliberately thin: stage narration with ASCII spinners, an immediate parsed-message count, a compact summary panel (volume + participants + date range), a skip-count line, and the absolute report path, which auto-opens in the default browser.

The two dominant facts that shape this phase: (1) **both live parser bugs were demonstrated in a runtime experiment** — an unparseable WhatsApp date is silently stamped `datetime.now()` (2026-08-01 in the test run), and the "Messages and calls are end-to-end encrypted." line is appended to the previous message's body (`'hi\nMessages and calls are end-to-end encrypted.'`); (2) **the analysis core is 100% reusable** — ChatEDA, `add_sentiment_analysis`/`get_sentiment_summary`, and 4 of `ChatVisualizer`'s 12 methods map exactly onto the required EDA, sentiment, and chart needs, with zero analysis-logic rewrites. The work is a thin `cli/` layer (pipeline + contracts + adapters + render + report_html), surgical parser hardening, one new core function (`messages_to_dataframe` in `ingest/ingestion.py`), and a small pyproject dependency change (add `jinja2`, remove `plotext`).

**Primary recommendation:** Route the pipeline's `.txt`/`.json` path through the **hardened `parser/*.py` modules** (that is where `datetime.now()` lives and where D-15..D-20 target), feed their row dicts through the new `messages_to_dataframe()` canonical builder, and consume the existing analysis modules unchanged. Do NOT fix the 39 pre-existing legacy test failures — that is Phase 4 QUAL-02 scope.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Chat-export parsing (strict, skip counts) | Core library (`parser/*.py`) | — | D-15..D-20 name these files; the `datetime.now()` fabrication only exists here. Pipeline consumes their output. |
| Dicts → canonical DataFrame (tz→naive UTC) | Core library (`ingest/ingestion.py` `messages_to_dataframe`) | — | Research Anti-Pattern 5: single source for dicts→df, never a second copy in `cli/`. |
| EDA stats / sentiment / per-participant metrics | Core library (`analysis/eda.py`, `analysis/sentiment.py`) | — | Reused unchanged; CLI adapts dicts, never re-implements (Anti-Pattern 3). |
| Matplotlib charts | Core library (`utils/visualization.py`) | `cli/` sets `Agg` + encodes PNG→base64 | ChatVisualizer already produces the 4 figure types; Agg backend + base64 is presentation. |
| Pipeline orchestration | `cli/pipeline.py` | — | Owns ordering, progress hooks, parse-report threading; no business logic (research Pattern 1). |
| Normalization to `AnalysisResults` | `cli/adapters.py` | — | Only place that knows each module's internal dict shape (research Pattern 2). |
| Terminal narration/summary | `cli/render.py` | — | D-05/D-07: thin — Status stages, ASCII panel, skip line, path. |
| HTML report card | `cli/report_html.py` | — | D-10..D-13: single-file jinja2 template, escaped content, base64 charts. |
| CLI arg parsing / validation / exit codes | `cli/main.py` | — | D-02/D-03/D-06: positional arg, no flags, friendly errors, encoding bootstrap (already present). |

## Standard Stack

### Core

| Library | Version (verified in env) | Purpose | Why Standard |
|---------|---------------------------|---------|--------------|
| typer | 0.27.0 | CLI framework | Phase 1 locked; 0.26+ dropped click, ships rich as hard dep. **Verified: `Typer.__init__` has NO `version` parameter → `--version` needs a manual eager callback.** [VERIFIED: runtime signature inspection + PyPI metadata] |
| rich | 14.3.3 | Terminal narration, Status spinners, panels | Hard dep of typer 0.27. Verified `console.status(..., spinner="line")` (ASCII `-\|/`) and `Panel(box=box.ASCII)` produce pure-ASCII output on cp1252-hostile consoles. [VERIFIED: runtime experiment] |
| jinja2 | 3.1.6 | HTML report templating | STATE.md locked decision: autoescape chosen over stdlib templates (chat content is untrusted). **In env but NOT in pyproject → must be added to `[project.dependencies]`.** [VERIFIED: pip show + pyproject.toml read]. Caution: `Environment(autoescape=...)` defaults to **False** in plain jinja2 — must set `autoescape=True` explicitly. [CITED: jinja2 docs — https://jinja.palletsprojects.com/en/3.1.x/api/] |
| pandas | 3.0.2 | DataFrame core | Phase 1 locked `>=2.0`. **Env runs pandas 3.0.x** — canonical builder and parser code must be 3.x-safe (lowercase freq aliases `'6h'` not `'6H'`; the uncommitted `tests/test_analysis.py` change is exactly this compat fix). [VERIFIED: runtime] |
| matplotlib | 3.10.8 | Charts → PNG → base64 | ChatVisualizer's proven `plt.savefig(io.BytesIO())` path; Agg backend verified working headless. [VERIFIED: runtime experiment — 25 KB data URI produced] |
| vaderSentiment | 3.3.2 | VADER sentiment (ANAL-05) | Reused via `sentiment.py`; zero deps, always available in base install. |
| webbrowser | stdlib | Auto-open report (D-09) | No dependency; `webbrowser.open()` wrapped in try/except per D-09. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| seaborn | 0.13.2 | Styled heatmap (hourly activity) | Used inside ChatVisualizer `plot_activity_heatmap`; already in stack. |
| numpy | 1.26.4 | Numeric (EDA internals) | Already in stack; floor `>=1.24`. |
| hatchling | (build backend) | Build backend | Phase 1 locked; non-`.py` template files inside the package are included in wheels by default [ASSUMED — verified pattern avoided by using an inline template constant, see Patterns]. |
| pytest | 9.0.2 | New phase tests | Follow `tests/test_phase1_smoke.py` plain-function style; legacy unittest files untouched (Phase 4 QUAL-02). |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| jinja2 + inline template string constant | jinja2 + separate `.j2` file loaded via `importlib.resources` | File-based is more maintainable but reintroduces Pitfall 10 package-data risk (wheel silently omitting non-`.py` files). Inline constant is zero-risk for a single report template; recommended. |
| `webbrowser.open()` (auto-open) | Manual "press Enter to open" | D-09 mandates auto-open with graceful degradation. |
| rich `box.ASCII` panel | rich default rounded box | Default box uses `┌─┐` glyphs — fine after the CLI's utf-8 reconfigure on modern Windows Terminal, but ASCII box is the Pitfall 5 "ASCII-first" recommendation for legacy CMD. Cosmetic (D-05 discretion). |
| `spinner='line'` | rich default `dots` spinner (braille `⠋⠙⠹…`) | Braille frames are non-ASCII; D-05 mandates Windows-safe ASCII spinners. `'line'` (`-\|/`) and `'simpleDots'` verified ASCII-clean in rich 14.3.3. |

**Installation (pyproject changes only):**
```toml
[project]
dependencies = [
    # ...existing Phase 1 list...
    "jinja2>=3.1",        # ADD — HTML report (autoescape)
    # "plotext>=5.3",     # REMOVE — OUT-02 dropped; verified imported nowhere in src/
]
```

**Version verification (run at research time, 2026-08-01):**
```bash
python -c "from importlib.metadata import version; [print(p, version(p)) for p in ['rich','jinja2','typer','matplotlib','pandas','vaderSentiment']]"
# rich 14.3.3 / jinja2 3.1.6 / typer 0.27.0 / matplotlib 3.10.8 / pandas 3.0.2 / vaderSentiment 3.3.2
```

## Package Legitimacy Audit

> Only **one** new manifest entry is introduced by this phase: `jinja2`. All other libraries (typer, rich, pandas, matplotlib, seaborn, vaderSentiment) were verified and human-approved in Phase 1's gate. slopcheck installed and executed on 2026-08-01.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| jinja2 | PyPI | ~18 yrs (2.x since 2008) | Very high (universal templating lib) | github.com/pallets/jinja | [OK] | Approved — add `jinja2>=3.1` to `[project.dependencies]` |
| plotext | PyPI | ~7 yrs | Low-moderate | github.com/piccolomo/plotext | (not re-scanned) | REMOVE from deps — OUT-02 dropped; `plotext` imported nowhere in `src/` [VERIFIED: codebase grep] |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none
**Manifest note:** `reportlab>=4.0.0` remains in pyproject from Phase 1 (imported by the deferred-but-importable `reporting/pdf_report.py`, which the Phase 1 smoke test requires to stay importable). Trimming it is out of Phase 2 scope.

## Architecture Patterns

### System Architecture Diagram

```
User runs: chat-analyzer <chat_file>   OR   python -m chat_analyzer  (prompt)
    │
    ▼
cli/main.py (typer) — encoding bootstrap (utf-8, errors=replace) · positional arg · --version callback
    │  validates path exists → friendly error + re-prompt (interactive) / exit 1 (positional)
    ▼
cli/pipeline.py  run_pipeline(path) → AnalysisResults          [the ONLY orchestration]
    │
    ├─ Stage "Parsing chat…" (rich Status, spinner='line')
    │    ├─ .txt  → WhatsAppParser (HARDENED): strict dates · skip counter · system classification
    │    ├─ .json → parse_telegram_chat (HARDENED): bare Chat + chats.list[] · recursive text join
    │    │                                                  · service filter · tz→naive UTC · skip counter
    │    └─ row dicts (datetime/sender/message/…) + ParseReport(parsed, skipped, system)
    ├─ messages_to_dataframe(rows)  [NEW in ingest/ingestion.py] → canonical df
    │    datetime · timestamp(alias) · date · hour · sender · message · message_length · source · uid
    │    (tz→naive UTC enforced; pandas 3.x-safe)
    ├─ Stage "Computing insights…"
    │    ├─ ChatEDA(df) → volume/dynamics/content + comprehensive summary        [REUSED]
    │    ├─ add_sentiment_analysis(df) + get_sentiment_summary(df)              [REUSED, VADER]
    │    └─ ChatVisualizer → 4 figures → base64 PNG data URIs  (Agg set BEFORE any matplotlib import)
    └─ adapters.adapt(...) → AnalysisResults (TypedDict, cli/contracts.py)
    │
    ▼
cli/render.py (terminal)  ── parsed count · skip line · ASCII summary panel (volume/participants/dates) · absolute path
cli/report_html.py (report) ── jinja2 autoescape template · 5 tabbed sections · narrative lead-ins · base64 charts · utf-8 file
    │
    ▼
webbrowser.open(file://…)  (try/except → still prints path)         Exit 0 / 1
```

### Recommended Project Structure

```
src/chat_analyzer/
├── cli/                          # NEW module files; existing main.py + __init__.py extended
│   ├── __init__.py               # unchanged (re-exports app)
│   ├── __main__.py               # unchanged (python -m chat_analyzer)
│   ├── main.py                   # MODIFIED: positional arg, --version callback, pipeline wiring
│   ├── pipeline.py               # NEW: run_pipeline(path) → AnalysisResults; Agg bootstrap; parse threading
│   ├── contracts.py              # NEW: AnalysisResults TypedDict + ParseReport dataclass
│   ├── adapters.py               # NEW: module dicts → AnalysisResults (+ narrative insight builders)
│   ├── render.py                 # NEW: rich Status stages, ASCII panel, skip line, path print
│   └── report_html.py            # NEW: jinja2 template (inline constant), fig→data-URI, sanitize, auto-open
├── parser/
│   ├── whatsapp_parser.py        # MODIFIED: strict dates (no now()), system classification, counters
│   └── telegram_parser.py        # MODIFIED: both shapes, recursive text join, service filter, tz, counters
├── ingest/
│   └── ingestion.py              # MODIFIED: + messages_to_dataframe(); normalize_message tz-safe ISO handling
├── analysis/
│   ├── eda.py                    # UNCHANGED (reused)
│   └── sentiment.py              # UNCHANGED (reused; emoji prints handled CLI-side, see Pitfall P-5)
└── utils/
    └── visualization.py          # MODIFIED (surgical): logging.basicConfig:19 → getLogger+NullHandler only
pyproject.toml                    # MODIFIED: + jinja2, − plotext
```

### Pattern 1: Thin CLI over a library core (Facade + Pipeline)

**What:** `cli/pipeline.py` is the only orchestration. Parsing, analysis, charting are core-library calls; the CLI never re-implements logic (research Anti-Pattern 3).
**When to use:** Always — this is the phase's core architectural decision.

```python
# cli/pipeline.py — recommended skeleton (exact API is planner discretion per CONTEXT)
import io, base64
from pathlib import Path
from typing import Any

def run_pipeline(path: Path, console) -> dict:   # -> AnalysisResults
    import matplotlib
    matplotlib.use("Agg")                          # BEFORE any matplotlib import (D-12, Pitfall 11)
    from chat_analyzer.parser.whatsapp_parser import WhatsAppParser
    from chat_analyzer.parser.telegram_parser import parse_telegram_chat_with_report

    with console.status("Parsing chat...", spinner="line"):
        if path.suffix.lower() == ".txt":
            df, report = _parse_whatsapp_rows(path)            # hardened parser → rows + counts
        elif path.suffix.lower() == ".json":
            df, report = _parse_telegram_rows(path)            # hardened parser → rows + counts
        else:
            raise UnsupportedFormat(path)
    # ... messages_to_dataframe → ChatEDA → sentiment → charts → adapters.adapt(...)
```

### Pattern 2: Canonical results contract via adapters

**What:** Each module returns its own shape (`ChatEDA.generate_comprehensive_summary` → nested dict; `get_sentiment_summary` → its own dict). `cli/adapters.py` maps them into one `AnalysisResults` TypedDict consumed by both `render.py` and `report_html.py`.
**When to use:** Two renderers (terminal + HTML) showing the same data without modifying core modules.

```python
# cli/contracts.py
from dataclasses import dataclass, field
from typing import TypedDict, Dict, List, Optional

@dataclass
class ParseReport:
    total_lines: int = 0
    parsed_messages: int = 0
    skipped_lines: int = 0
    system_messages: int = 0

class AnalysisResults(TypedDict):
    source: str                      # "whatsapp" | "telegram"
    parse: Dict[str, int]            # parsed_messages, skipped_lines, system_messages (D-15/16/18)
    stats: Dict[str, Any]            # volume, participants, date range, busiest day/hour, avg response, media
    participants: Dict[str, Any]     # per-sender: messages, avg_length, share %
    content: Dict[str, Any]          # top_words, top_emojis (ANAL-04)
    sentiment: Dict[str, Any]        # VADER breakdown (ANAL-05)
    charts: Dict[str, str]           # "timeline"|"activity"|"participants"|"sentiment" → base64 PNG data URI
    insights: List[str]              # narrative lead-in sentences (D-11)
    report_path: str                 # absolute path (D-08)
```

### Pattern 3: Strict parse + skip counter (never fabricate)

**What:** Every line is classified: `message` (regex + date parse OK) | `system` (timestamp header, no sender — D-18) | `continuation` (no header, previous message exists) | `skipped` (regex matched but date unparseable — D-15, or orphan). `datetime.now()` is deleted. Counters live on a `ParseReport` the pipeline surfaces (D-16).
**When to use:** All parsing in this phase. Verified live: the current parser fabricates `datetime.now()` for an invalid-month line and appends the encryption notice to the previous message — both must end.

### Pattern 4: Single-file HTML report via data URIs

**What:** `report_html.py` renders `AnalysisResults` with a jinja2 `Environment(autoescape=True)` into one self-contained HTML file; charts are `data:image/png;base64,...` URIs; CSS + tab JS inline; `<!DOCTYPE html>` + `<meta charset="utf-8">`; written `encoding="utf-8"`; filename sanitized (D-14).
**When to use:** The "shareable report card" deliverable. Verified: Agg→PNG→base64 data URI pipeline works in this environment.

### Anti-Patterns to Avoid

- **Re-implementing analysis in the CLI** (research Anti-Pattern 3): never copy EDA/sentiment math into `cli/` — call `ChatEDA`, `add_sentiment_analysis` directly. Degradation happens in features, never parallel implementations.
- **Second dict→df builder inside `cli/`** (Anti-Pattern 5): `messages_to_dataframe` is the single source, in `ingest/ingestion.py`.
- **`logging.basicConfig`/`print()` leaking into CLI output** (Anti-Pattern 4): neutralize `visualization.py:19`; capture core emoji `print()`s during analysis (see Pitfall P-5).
- **Eager matplotlib/pandas imports at CLI startup** (Anti-Pattern 2): `main.py` stays light; heavy imports happen inside `run_pipeline()` after `matplotlib.use("Agg")`. `--help`/`--version` stay instant.
- **Unescaped chat content in HTML** (CONCERNS.md `unsafe_allow_html` bug must not survive the pivot): jinja2 `autoescape=True` + `html.escape()` on any raw string interpolation.
- **Ambiguous-date "cleverness"** (D-17): no M/D-vs-D/M majority voting, no locale sniffing, no override flags. `%m/%d`-first order is the accepted common-formats behavior; ambiguity is documented, not solved.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Terminal narration/spinners/panels | ANSI escape strings | rich `Console.status()`, `Panel(box=box.ASCII)` | Windows encoding + piped-mode correctness (Pitfalls 5/6); rich already a hard typer dep |
| HTML templating + escaping | f-string HTML | jinja2 `Environment(autoescape=True)` | Chat content is untrusted input; f-strings are an injection/rendering hazard (STACK research; CONCERNS.md unsafe_allow_html legacy) |
| DataFrame build/normalization | Copy of the deleted app's converter in `cli/` | `ingest/ingestion.py: messages_to_dataframe()` | Third copy of schema logic + tz bug propagation (Anti-Pattern 5, CONCERNS.md) |
| Charts | New plot code | `ChatVisualizer.plot_message_timeline/plot_activity_heatmap/plot_user_activity/plot_sentiment_timeline` | All 4 required chart types already exist (D-12); only PNG→base64 encoding is new |
| EDA stats / sentiment | Re-derived metrics | `ChatEDA` + `add_sentiment_analysis`/`get_sentiment_summary` | Reuse, no rewrite (AGENTS.md); module dicts already produced |
| `--version` | Hardcoded string | `importlib.metadata.version("chat-analyzer-pro")` | Local, instant, never hits the network (STACK research) |
| Report auto-open | Subprocess `start`/`cmd` hacks | `webbrowser.open("file://" + str(path.resolve()))` | Cross-platform stdlib; wrap in try/except per D-09 |

**Key insight:** This phase's engineering risk is entirely in *wiring and hardening*, not in building analysis or rendering from scratch. The 85% of the value that already exists (`ChatEDA`, VADER, `ChatVisualizer`, the interactive CLI shell) must be reused verbatim; the new code is a thin orchestrator plus two surgical parser fixes.

## Common Pitfalls

### Pitfall 1: `datetime.now()` fabrication (research Pitfall 1 — VERIFIED LIVE)
**What goes wrong:** `whatsapp_parser.py:61,63,77,79` stamp any unparseable date with the current time. Demonstrated: line `25/13/26, 9:30 AM - Alice: hello` parsed as `2026-08-01 13:34:55` (today). Date range, hourly activity, response times become garbage — invisibly.
**Why it happens:** The `for...else: datetime.now()` + bare `except` pattern treats parse failure as recoverable.
**How to avoid:** Delete all four fallbacks. A regex-matched line whose date fails **every** format → `skipped_lines += 1`, return no message, never append as continuation. Surfaced per D-16.
**Warning signs:** date-range start == today for an old chat; parsed count << raw line count with no warning. Verification test: unknown-format fixture asserts no timestamp equals "now".

### Pitfall 2: System messages appended to the previous message (research Pitfall 3 — VERIFIED LIVE)
**What goes wrong:** Demonstrated: `parse_file` on `["...Alice: hi", "Messages and calls are end-to-end encrypted.", "...Bob: yo"]` yields row 0 body `'hi\nMessages and calls are end-to-end encrypted.'`.
**Why it happens:** A line that fails the message regex falls into the "continuation" branch.
**How to avoid (D-18):** Classify structurally — a line matching the timestamp header but with no `sender: ` part → `type="system"`, counted in `system_messages`, **never** appended. Bare non-message lines that are not continuations → skipped (counted). Media placeholders stay English-only for v1 (Pitfall 3 localized-media note; out of D-17/18 scope).

### Pitfall 3: Telegram shape drift + silent drops (research Pitfall 4)
**What goes wrong:** Current `telegram_parser.py:25` only does `data.get('messages')` (bare Chat works; full-export `chats.list[]` returns nothing). `text` lists drop dict parts without `'text'` (`:44-46`). `except: continue` (`:35-36`) silently drops malformed messages. `date` with `Z`/offset becomes tz-aware (`:34`), naive otherwise → Pitfall 9 mixing. Service messages (`type="service"`, no `from`) aren't classified.
**How to avoid (D-19/D-20):** Both shapes + friendly "not a chat export" error; recursive text join (str parts + dict `text` keys; media-fields-with-empty-text → `<Media omitted>`); `service` → system counter; malformed → `skipped_lines += 1`; date helper: `dt = datetime.fromisoformat(s.replace("Z", "+00:00"))` then `dt.astimezone(timezone.utc).replace(tzinfo=None)` for aware, pass-through for naive (Telegram exports are UTC per official schema).

### Pitfall 4: tz-naive/aware mixing (research Pitfall 9)
**What goes wrong:** WhatsApp parser → naive; Telegram parser → aware with `Z`. Any `sort_values('datetime')` or `(a - b)` raises `TypeError` or silently skews hours.
**How to avoid (D-20):** Single normalization contract at the parser boundary → naive UTC. `messages_to_dataframe` defensively re-checks (`tzinfo is not None` → convert). Schema test asserts `df['datetime'].dt.tz is None` for both parsers.

### Pitfall 5: Core `print()`/`logging.basicConfig` pollution during narration
**What goes wrong:** `sentiment.py` prints emoji lines at import (textblob/transformers missing) and inside `initialize_analyzers()`/`add_sentiment_analysis()` ("🚀 Initializing…", "✅ VADER analyzer loaded", "🔍 Analyzing sentiment for N messages…"). A bare `print()` while rich `Status` is active garbles the spinner line. `visualization.py:19` calls `logging.basicConfig` at import, hijacking log config (research Anti-Pattern 4).
**How to avoid:** (a) `visualization.py:19` → `logger = logging.getLogger(__name__); logger.addHandler(logging.NullHandler())` (matches `ingestion.py:30-31`) — surgical, CONTEXT-mandated. (b) In `pipeline.py`, wrap the analysis stage in `contextlib.redirect_stdout(io.StringIO())` (optionally log captured text) — CLI-side, zero core rewrite. The CLI's utf-8 reconfigure (`main.py:28-32`) already makes stray emoji prints non-crashing; capture keeps narration clean. `warnings.filterwarnings('ignore')` at `sentiment.py:7`/`visualization.py:23` are global but harmless for a one-shot CLI — leave.

### Pitfall 6: HTML report encoding/escaping/path (research Pitfall 11)
**What goes wrong:** Platform-default `open(path, 'w')` → cp1252 mojibake; missing `<meta charset="utf-8">`; unescaped `<script>` in a message breaks a shared report; report written to cwd surprises users.
**How to avoid (D-10/D-14):** `open(..., 'w', encoding='utf-8')`; `<!DOCTYPE html>` + `<meta charset="utf-8">`; jinja2 autoescape + `html.escape()` for anything injected raw; report next to input (`input.parent / f"{sanitize(stem)}_report.html"`); sanitize strips path separators/control chars/leading dots with a fallback name.

### Pitfall 7: Headless/backend crash on charts (research Pitfall 11 tail)
**What goes wrong:** Importing `matplotlib.pyplot` without a backend on a headless box raises `TclError` or spawns GUI windows.
**How to avoid (D-12):** `matplotlib.use("Agg")` as the first line of `run_pipeline()` — before `eda.py`/`sentiment.py`/`visualization.py` are imported (all three `import matplotlib.pyplot` at module top). Keep `main.py` matplotlib-free so `--help`/`--version` stay instant. Verified: Agg→`savefig(io.BytesIO())`→base64 works here.

### Pitfall 8: pandas 3.x incompatibilities in new code
**What goes wrong:** The env runs pandas **3.0.2** (STACK research warned to test core against 3.x). Uppercase `freq='H'` aliases are gone — the uncommitted `tests/test_analysis.py` diff (`'6H'`→`'6h'`) is exactly this. `ChatEDA.prepare_data` (`eda.py:23-24`) and `ChatVisualizer` already call `pd.to_datetime` defensively.
**How to avoid:** Write new code with lowercase freq strings and explicit `pd.to_datetime(..., errors=...)`; never assume 2.x behaviors. Do NOT "fix" the legacy test failures — Phase 4 QUAL-02.

### Pitfall 9: No progress feedback / silent long waits (research Pitfall 13)
**What goes wrong:** VADER over 40k messages + chart rendering takes seconds; no output → user Ctrl+C's.
**How to avoid (D-05, CLI-03):** Narrate every stage; print the parsed-message count **immediately after parsing** ("Parsed 12,483 messages from 3 participants (2 system, 37 skipped)") before analysis runs — it proves the file was understood and gives scale context.

## Code Examples

Verified patterns from official sources + runtime experiments in this repo:

### WhatsApp strict date parse + system classification + counters
```python
# Source: research Pitfalls 1/3 + verified current code at parser/whatsapp_parser.py:48-79
# Replace the for...else datetime.now() blocks (lines 61,63,77,79) with:
#   DATE_FORMATS = ["%m/%d/%y %I:%M %p", "%d/%m/%y %I:%M %p", "%m/%d/%Y %I:%M %p", "%d/%m/%Y %I:%M %p",
#                   "%m/%d/%y %I:%M:%S %p", "%d/%m/%y %I:%M:%S %p",   # 12h
#                   "%m/%d/%y %H:%M", "%d/%m/%y %H:%M", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M",
#                   "%m/%d/%y %H:%M:%S", "%d/%m/%y %H:%M:%S"]          # 24h  (D-17: no M/D-vs-D/M heuristics)
def _parse_datetime_strict(self, datetime_str: str):
    for fmt in self.DATE_FORMATS:
        try:
            return datetime.strptime(datetime_str, fmt)
        except ValueError:
            continue
    self.skipped_lines += 1          # D-15: never datetime.now(); counted, not silent
    return None

# System-message classification (D-18): header regex WITHOUT the "sender: " part.
# e.g. r"^(\d{1,2}/\d{1,2}/\d{2,4}),?\s(\d{1,2}:\d{2}(?::\d{2})?)\s?([AaPp][Mm])?\s?-\s(.+)$"
# A match here (no "<sender>: " group) → self.system_messages += 1, tagged type="system",
# NEVER appended to the previous message (verified bug: 'hi\nMessages and calls are end-to-end encrypted.')
```

### Telegram both shapes + recursive text join + tz→naive UTC
```python
# Source: research Pitfall 4 + core.telegram.org/import-export [CITED]
from datetime import datetime, timezone

def _load_messages(data: dict) -> list:
    if isinstance(data.get("messages"), list):          # bare Chat (single-chat export)
        return data["messages"]
    if isinstance(data.get("chats"), list):             # full export: chats.list[i].messages
        out = []
        for chat in data["chats"]:
            if isinstance(chat, dict) and isinstance(chat.get("messages"), list):
                out.extend(chat["messages"])
        return out
    raise ValueError("Not a Telegram chat export (no 'messages' or 'chats' key)")

def _join_text(parts) -> str:
    if isinstance(parts, str):
        return parts
    chunks = []
    for part in parts:                                  # str OR entity dict {"type": ..., "text": ...}
        if isinstance(part, str):
            chunks.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            chunks.append(part["text"])
    return "".join(chunks)

def _to_naive_utc(date_str: str) -> datetime:
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))   # handles Z and +HH:MM
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt  # D-20
```
Per-message loop: `type == "service"` → `report.system_messages += 1` (D-18/19); `type != "message"` → `skipped_lines += 1`; date parse failure → `skipped_lines += 1` + continue — **never** bare `except: continue`.

### messages_to_dataframe (NEW in ingest/ingestion.py)
```python
# Source: research Anti-Pattern 5 + verified normalize_message schema (ingestion.py:323-380)
# + verified live bug: telegram dicts carry date="2025-09-15T09:45:00", time="" (full ISO in date)
def messages_to_dataframe(messages: list[dict]) -> pd.DataFrame:
    rows = []
    for m in messages:
        if m.get("datetime"):                       # parser-path dicts
            dt = _to_naive_utc(str(m["datetime"]))
        elif "T" in str(m.get("date", "")):         # ingestion-path telegram: full ISO in date field
            dt = _to_naive_utc(str(m["date"]))
        elif m.get("date") and m.get("time"):
            dt = pd.to_datetime(f"{m['date']} {m['time']}")          # whatsapp "YYYY-MM-DD HH:MM"
        else:
            continue                                # unparseable → caller's skip accounting
        rows.append({
            "datetime": dt, "timestamp": dt,        # timestamp alias → ChatVisualizer (verified: it requires 'timestamp')
            "date": dt.date(), "hour": dt.hour,
            "sender": m.get("author") or m.get("sender") or m.get("from") or "unknown",
            "message": m.get("text") or m.get("message") or "",
            "message_length": len(m.get("text") or m.get("message") or ""),
            "source": m.get("source") or m.get("source_hint") or "unknown",
            "uid": m.get("uid") or m.get("id") or str(uuid.uuid4()),
        })
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df
```

### Agg bootstrap + chart → base64 data URI (D-12)
```python
# cli/pipeline.py — MUST run before importing eda/sentiment/visualization
import matplotlib
matplotlib.use("Agg")
from chat_analyzer.utils.visualization import ChatVisualizer   # safe after use("Agg")

def fig_to_data_uri(fig) -> str:          # verified: produced 25 KB data URI in this env
    import io, base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)                        # avoid figure leak across charts
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")

viz = ChatVisualizer()
charts = {
    "timeline":    fig_to_data_uri(viz.plot_message_timeline(df, resample_freq="D")),
    "activity":    fig_to_data_uri(viz.plot_activity_heatmap(df)),                # heatmap or hourly bar (D-12)
    "participants":fig_to_data_uri(viz.plot_user_activity(df, top_n=10)),
    "sentiment":   fig_to_data_uri(viz.plot_sentiment_timeline(df_sent, sentiment_score_col="vader_compound")),
}
```

### jinja2 autoescape report (D-10) — explicit autoescape is required
```python
# cli/report_html.py — Environment defaults to autoescape=False; set it explicitly
from jinja2 import Environment, select_autoescape
TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{{ title }}</title><style>...</style></head>
<body>
  <h1>{{ title }}</h1>
  {% for tab in tabs %}<button class="tab" onclick="showTab('{{ tab.id }}')">{{ tab.label }}</button>{% endfor %}
  <div class="panel" id="tab-overview">
    <p class="insight">{{ insights[0] }}</p>
    <img src="{{ charts['timeline'] }}" alt="Messages per day">
    ...
  </div>
  <script>function showTab(id){...}</script>
</body></html>"""
env = Environment(autoescape=select_autoescape(["html", "xml"]))
html_out = env.from_string(TEMPLATE).render(title=title, tabs=tabs, insights=insights, charts=charts)
with open(report_path, "w", encoding="utf-8") as fh:      # Pitfall 11: explicit utf-8
    fh.write(html_out)
```

### CLI positional arg + --version (D-02/D-03) — typer 0.27 has no `version` param [VERIFIED]
```python
# cli/main.py — extend, don't rewrite (Phase 1 encoding bootstrap stays at top of main())
import typer
from pathlib import Path
from typing import Optional

def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version
        typer.echo(f"chat-analyzer {version('chat-analyzer-pro')}")   # local, no network
        raise typer.Exit()

@app.command()
def main(
    chat_file: Optional[Path] = typer.Argument(None, help="Path to WhatsApp .txt or Telegram .json export"),
    version: Optional[bool] = typer.Option(None, "--version", is_eager=True,
                                           callback=_version_callback, help="Show version and exit"),
) -> None:
    # 1) encoding bootstrap (existing lines 28-32)
    # 2) if chat_file is None → existing interactive prompt loop (D-01)
    # 3) else → validate is_file() → run_pipeline → render → report → auto-open (D-02/D-06/D-09)
```
Verified current behavior: `python -m chat_analyzer --version` fails today ("No such option") — this closes that gap.

### Terminal narration (D-05/D-07/D-16) — verified ASCII-safe
```python
from rich.console import Console
from rich.panel import Panel
from rich import box
console = Console()
with console.status("Parsing chat...", spinner="line"):     # 'line' = ASCII -\|/  [VERIFIED]
    ...                                                     # stage work
console.print(f"[OK] Parsed {n} messages from {p} participants")
if skipped:
    console.print(f"[WARN] Skipped {skipped} lines that couldn't be parsed")   # D-16, single line
console.print(Panel(
    f"Total messages: {n}\nParticipants: {p}\nDate range: {start} to {end}",
    title="Summary", box=box.ASCII))                        # box.ASCII = pure +-|   [VERIFIED]
console.print(f"Report: {report_path}")                     # absolute path (D-08)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `datetime.now()` fallback on WhatsApp parse failure | Strict parse + `skipped_lines` counter surfaced to user | This phase (D-15) | Date-range/hour stats become trustworthy; silent data corruption ends |
| System messages appended to previous message | Structural `type="system"` classification, excluded from stats, counted | This phase (D-18) | No more "hi\nMessages and calls are end-to-end encrypted." bodies |
| Telegram: `data.get('messages')` only, silent `except: continue` | Both shapes + recursive text join + service filter + counted skips | This phase (D-19) | Full exports and entity-array texts parse; drops are visible |
| Telegram tz-aware vs WhatsApp naive datetime mixing | Both parsers → naive UTC at the boundary | This phase (D-20) | No more `TypeError` on compare; Telegram hours consistent with WhatsApp |
| plotext inline terminal charts | Charts exist only in the HTML report (base64 PNG) | This phase (OUT-02 dropped) | One visualization path; no terminal chart failure modes (Pitfall 6 moot) |
| Terminal = full insights output (OUT-01) | Terminal = progress + pointer; HTML report = insights (D-04/D-07) | This phase | Two-output drift eliminated; single report contract |

**Deprecated/outdated:**
- `datetime.now()` at `whatsapp_parser.py:61,63,77,79` — removed this phase (verified: fabricates today's date for invalid input).
- `plotext>=5.3` in pyproject — remove; imported nowhere in `src/` [VERIFIED].
- The `for...else: timestamp = datetime.now()` + bare-`except` pattern — replaced by strict `_parse_datetime_strict` returning None + counter.
- `tests/test_analysis.py` `freq='6H'` — pre-existing uncommitted change (pandas 3.x compat); leave untouched, Phase 4 QUAL-02.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | hatchling wheel target includes non-`.py` files inside the package by default | HTML report | LOW — sidestepped by using an inline template constant; if a separate `.j2` file is chosen instead, must verify wheel contents |
| A2 | `webbrowser.open()` reliably opens `file://` URLs on Windows | Auto-open | LOW — D-09 mandates try/except degradation to path print regardless |
| A3 | pandas 3.x uppercase freq aliases (`'H'`) are deprecated/removed | Parser/builder | MEDIUM — env is pandas 3.0.2 and the pre-existing test change confirms; new code uses lowercase/ISO only, so impact is limited to *old* code (Phase 4) |
| A4 | `Messages and calls are end-to-end encrypted.` is reliably line 2 of WhatsApp exports | Parser | LOW — D-18 structural classification catches it regardless of position |
| A5 | rich `Panel` default (rounded) box renders acceptably on modern Windows Terminal after utf-8 reconfigure | Terminal | LOW — ASCII box recommended as the safe default (Pitfall 5), cosmetics are D-05 discretion |
| A6 | `get_sentiment_summary`'s `consensus_sentiment` key is present when only VADER is available | Adapters | LOW — verified logic: `sentiment_cols` includes `vader_sentiment` → `consensus_sentiment` = VADER labels in base install |

## Open Questions

1. **Pipeline data path: parser-direct vs `process_uploaded_file` delegation.** The CONTEXT names `process_uploaded_file` as "pipeline entry point" AND names `parser/*.py` as the hardening targets; only the latter contains the `datetime.now()` fabrication. Recommended resolution: pipeline calls the hardened parser modules directly for `.txt`/`.json` (rows + `ParseReport`), keeping `process_uploaded_file` for other formats/back-compat. Planner must pick one and keep it consistent.
   - What we know: both paths exist; ingestion's `parse_whatsapp_text`/`parse_json_chat` have their own (un-hardened) behavior and the telegram dict path carries a date-field bug (`date` = full ISO, `time` = "").
   - What's unclear: whether `process_uploaded_file` should also delegate `.txt`/`.json` to the hardened parsers (nice single entry) or stay untouched.
   - Recommendation: parser-direct in `run_pipeline`; `messages_to_dataframe` normalizes either dict source.
2. **OUT-05 (`--no-report`) semantics.** Report is the deliverable (D-04) and no flags ship (D-03). Recommendation: OUT-05 resolves as "not applicable in Phase 2 — report always generated"; leave REQUIREMENTS.md note for Phase 3/4.
3. **Parser API for skip/system counts.** Recommendation: `ParseReport` dataclass populated during parse, returned by new `*_with_report` entry points while old `parse_file`/`parse_telegram_chat` signatures are preserved (QUAL-01 non-breaking). Exact shape is planner discretion.
4. **Sentiment print capture.** Recommendation: `contextlib.redirect_stdout` around the analysis stage in `pipeline.py`. Confirm no test depends on `add_sentiment_analysis` printing.
5. **--version output format.** Recommendation: `chat-analyzer 0.1.0` via `importlib.metadata.version("chat-analyzer-pro")`. Trivial, but worth pinning in the plan.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | runtime | ✓ | 3.11.8 (floor `>=3.11` met) | — |
| pandas | core | ✓ | 3.0.2 (3.x active — code must be 3.x-safe) | — |
| matplotlib | charts | ✓ | 3.10.8 (Agg verified headless) | — |
| rich | narration | ✓ | 14.3.3 | — |
| typer | CLI | ✓ | 0.27.0 (no click; no `version` param) | manual `--version` callback |
| jinja2 | HTML report | ✓ | 3.1.6 (**in env; NOT yet in pyproject — must add**) | — |
| vaderSentiment | sentiment | ✓ | 3.3.2 | — |
| pytest | new tests | ✓ | 9.0.2 | — |
| plotext | — | ✓ installed | 5.3.2 | REMOVE from manifest (unused, OUT-02 dropped) |
| Default browser (auto-open) | D-09 | ✓ (Windows) [ASSUMED] | — | try/except → path print |
| `chat-analyzer` console script | CLI-02 | ✓ (Phase 1) | — | `python -m chat_analyzer` |

**Missing dependencies with no fallback:** none.

## Validation Architecture

> **SKIPPED** — `.planning/config.json` sets `workflow.nyquist_validation: false` explicitly. Per the research protocol, this section is omitted when the flag is explicitly false.

## Security Domain

> `security_enforcement` is absent from `.planning/config.json` → treated as enabled.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (local CLI, no accounts) |
| V3 Session Management | no | — (one-shot process, no sessions) |
| V4 Access Control | no | — (local file access only) |
| V5 Input Validation | **yes** | jinja2 `Environment(autoescape=True)` + `html.escape()` on all chat-derived content (D-10); strict parse rejects malformed lines instead of fabricating |
| V6 Cryptography | no | — (no secrets handled) |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| HTML injection via chat content in a *shared* report (the old app's `unsafe_allow_html` bug — CONCERNS.md) | Tampering | jinja2 autoescape (explicitly enabled — defaults off) + `html.escape()`; verification: fixture message containing `<script>`/`<3` renders inert |
| Filename injection via malicious input name | Tampering | D-14 sanitization (strip path separators, control chars, leading dots; fallback name); report path is derived, never user-supplied in Phase 2 |
| Mojibake/exfiltration-adjacent encoding failures | (availability) | `open(..., encoding='utf-8')` + `<meta charset="utf-8">` (Pitfall 11) |
| `datetime.now()` timestamp fabrication → wrong-data "insights" | (integrity) | Strict parse + skip counter (D-15) — a skipped line is honest; a fake timestamp is corrupt data |
| Emoji/unicode crash on cp1252 consoles (breaks the tool's own error output) | (availability) | `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` at CLI entry (Pitfall 5, already in Phase 1 `main.py`) |
| Local file reading of arbitrary user-supplied paths | (permission) | In scope by design (user's own files); no remote code, no `exec()` (Phase 1 QUAL-04 smoke test enforces absence) |

## Sources

### Primary (HIGH confidence)
- [VERIFIED: codebase inspection] `src/chat_analyzer/parser/whatsapp_parser.py` (datetime.now() at 61/63/77/79; message regex; `_add_features`; continuation logic) — read in full.
- [VERIFIED: codebase inspection] `src/chat_analyzer/parser/telegram_parser.py` (shape handling, text join, except:continue, tz) — read in full.
- [VERIFIED: codebase inspection] `src/chat_analyzer/ingest/ingestion.py` (`process_uploaded_file:399`, `normalize_message:323`, `parse_whatsapp_text`, `parse_json_chat`, `try_parse_datetime`) — read in full; telegram dict date-field bug observed live.
- [VERIFIED: codebase inspection] `src/chat_analyzer/analysis/eda.py` (ChatEDA methods), `src/chat_analyzer/analysis/sentiment.py` (`add_sentiment_analysis`, `get_sentiment_summary`, VADER path), `src/chat_analyzer/utils/visualization.py` (ChatVisualizer 12 methods; `timestamp` column requirement; `logging.basicConfig:19`) — read in full.
- [VERIFIED: codebase inspection] `src/chat_analyzer/cli/main.py` (encoding bootstrap, prompt loop), `cli/__init__.py`, `__main__.py`, `pyproject.toml` (deps list, scripts entry), `tests/test_phase1_smoke.py` (Phase 1 test patterns).
- [VERIFIED: runtime experiment] Parser bug demos (datetime.now() fabrication; system-line append), `process_uploaded_file`/`parse_telegram_chat` output shapes, typer 0.27 signature (no `version` param), rich ASCII spinner list + `Panel(box=box.ASCII)` ASCII output, Agg→base64 data URI, pytest baseline (39 failed / 72 passed; distribution reporting 15, end_to_end 11, parser 8, analysis 5), pandas 3.0.2.
- [VERIFIED: runtime] Env versions via `importlib.metadata`: rich 14.3.3, jinja2 3.1.6, typer 0.27.0, matplotlib 3.10.8, pandas 3.0.2, plotext 5.3.2, pytest 9.0.2.
- [VERIFIED: tool] slopcheck `install jinja2` → [OK]; `pip index versions jinja2` → 3.1.6 current.
- [CITED: official docs] Telegram export schema — https://core.telegram.org/import-export (bare Chat vs `chats.list`; `text` String-or-Array; service messages; `id` > 32 bits).
- [CITED: official docs] jinja2 autoescape defaults off — https://jinja.palletsprojects.com/en/3.1.x/api/ ; typer callbacks/options — https://typer.tiangolo.com/.
- [CITED: project research] `.planning/research/PITFALLS.md` (1-6, 9, 11-13), `.planning/research/ARCHITECTURE.md` (pipeline/adapters/contracts; Anti-Patterns 2/4/5), `.planning/research/STACK.md` (jinja2, matplotlib→base64), `.planning/codebase/CONCERNS.md` (tz bug, unsafe_allow_html legacy), `.planning/codebase/TESTING.md` (tests never import src), `.planning/STATE.md`, `.planning/ROADMAP.md`.

### Secondary (MEDIUM confidence)
- WhatsApp export format quirks (regional formats, system events, encryption notice line 2, localized media placeholders) — whatsquiz.com blog + chatanalyzer.syncori.net, both cited in PITFALLS.md. [CITED via PITFALLS.md]

### Tertiary (LOW confidence)
- hatchling default inclusion of non-`.py` package files in wheels — training knowledge, not verified this session (A1; avoided by inline template constant).

## Metadata

**Confidence breakdown:**
- Standard stack: **HIGH** — versions verified in env; only new dep is jinja2 (slopcheck [OK], installed 3.1.6).
- Architecture: **HIGH** — pipeline/adapters/contracts pattern grounded in research ARCHITECTURE + CONTEXT integration points; parser bugs demonstrated live.
- Pitfalls: **HIGH** — both headline parser bugs reproduced in runtime experiments; remaining items cited from project PITFALLS.md.
- Security: **MEDIUM** — mitigation controls are standard/verified, but no penetration-style testing performed (appropriate for a local CLI phase).

**Research date:** 2026-08-01
**Valid until:** 2026-08-08 (fast-moving: pandas 3.x, typer 0.27 ecosystem; parser/export formats stable)

---

## Reused vs New vs Modified (per module)

| Module | Status | What changes |
|--------|--------|--------------|
| `analysis/eda.py` — ChatEDA | **REUSED** (unchanged) | All four required methods (`generate_comprehensive_summary`, `analyze_message_volume`, `analyze_conversation_dynamics`, `analyze_content`) used as-is |
| `analysis/sentiment.py` — VADER | **REUSED** (unchanged) | `add_sentiment_analysis` + `get_sentiment_summary` called directly; base install → `consensus_sentiment` = VADER. Emoji prints handled CLI-side (redirect_stdout), not by rewriting the module |
| `utils/visualization.py` — ChatVisualizer | **MODIFIED** (surgical, 1 line) | `logging.basicConfig(level=logging.INFO)` at line 19 → `getLogger(__name__)` + `NullHandler` (CONTEXT-mandated, research Anti-Pattern 4). All 12 plot methods untouched; 4 reused |
| `parser/whatsapp_parser.py` | **MODIFIED** | Remove `datetime.now()` at 61/63/77/79 (strict parse + `skipped_lines`); system-message classification (D-18, never append); expose `ParseReport` counters. `parse_line`/`parse_file` df-returning behavior preserved for QUAL-01 |
| `parser/telegram_parser.py` | **MODIFIED** | Both top-level shapes (D-19); recursive text join (str + entity dicts); `type="service"` → system counter; malformed → skip counter (no `except: continue`); `sender = from or actor or "Unknown"`; tz→naive UTC (D-20) |
| `ingest/ingestion.py` | **MODIFIED** (additive) | **NEW** `messages_to_dataframe()` (Anti-Pattern 5, tz→naive UTC, `timestamp` alias); `normalize_message` gains tz-safe ISO handling for the telegram date-field bug. `process_uploaded_file`/`normalize_message` otherwise unchanged |
| `cli/main.py` | **MODIFIED** | Positional `chat_file` arg (D-02); `--version` eager callback (D-03, typer 0.27 has no built-in); pipeline + render + report wiring; interactive prompt loop preserved (D-01/D-06); encoding bootstrap preserved |
| `cli/pipeline.py` | **NEW** | `run_pipeline(path) → AnalysisResults`; `matplotlib.use("Agg")` bootstrap; parse threading (rows + `ParseReport`); analysis orchestration; chart → base64 |
| `cli/contracts.py` | **NEW** | `AnalysisResults` TypedDict + `ParseReport` dataclass |
| `cli/adapters.py` | **NEW** | Module dicts → `AnalysisResults`; narrative insight sentence builders (D-11) |
| `cli/render.py` | **NEW** | rich Status stages (ASCII spinner), parsed count, skip line (D-16), ASCII summary panel (D-07), absolute path (D-08) |
| `cli/report_html.py` | **NEW** | jinja2 autoescape template (inline constant), 5 tabs, lead-ins, base64 charts, filename sanitize (D-14), utf-8 write, auto-open (D-09) |
| `cli/__init__.py`, `cli/__main__.py`, `chat_analyzer/__init__.py`, `chat_analyzer/__main__.py` | **REUSED** (unchanged) | Entry plumbing from Phase 1 |
| `pyproject.toml` | **MODIFIED** | Add `jinja2>=3.1`; remove `plotext>=5.3` (OUT-02 dropped, verified unused) |
| `reporting/*`, `analysis/emotion.py`, `analysis/relationship_health.py`, `analysis/summarizer.py`, `analysis/network_graph.py` | **REUSED** (untouched, not imported by Phase 2 pipeline) | Phase 4 `[nlp]` scope |
| `data/sample_chats/*` | **REUSED** (unchanged) | `whatsapp_sample.txt` (27 msgs) + `telegram_sample.json` (bare-Chat shape, 5 msgs) for smoke tests; **new fixtures needed** for system messages, chats.list shape, entity-array text, service messages, Z-suffix dates (per Pitfalls 1-4 verification) |
| `tests/` (legacy 4 files) | **UNTOUCHED** | See note below |

## Pre-existing Test State — DO NOT FIX THIS PHASE

- **39 legacy test failures** across `tests/test_reporting.py` (15), `tests/test_end_to_end.py` (11), `tests/test_parser.py` (8), `tests/test_analysis.py` (5) were confirmed at research time (`pytest tests -q` → `39 failed, 72 passed`). These are the self-contained legacy suite that never imports `chat_analyzer.*` (documented in `.planning/codebase/TESTING.md`) and are **Phase 4 QUAL-02 scope** ("Tests pass for the new CLI"). This phase must NOT attempt to fix them; new Phase 2 tests are additive (follow `tests/test_phase1_smoke.py`'s plain-pytest style and must exercise the real `chat_analyzer.*` modules per AGENTS.md).
- **`tests/test_analysis.py` carries a pre-existing uncommitted change** (`freq='6H'` → `freq='6h'` at two sites — a pandas 3.x compatibility fix). It predates this phase, is unrelated to Phase 2 work, and must remain untouched (do not commit it, do not revert it, do not expand it). `git status` at research time shows only this one modified file.
