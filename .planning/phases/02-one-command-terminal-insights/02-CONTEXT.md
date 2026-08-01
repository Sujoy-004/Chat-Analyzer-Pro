# Phase 2: One-Command Terminal Insights - Context

**Gathered:** 2026-08-01
**Status:** Ready for planning

> **Scope note:** Discussion reshaped the phase. The HTML report card is now the
> primary deliverable (previously Phase 3 OUT-03/04/05). Phases 2+3 effectively
> merge into "one command → report card." OUT-02 (inline plotext terminal charts)
> is dropped. Auto-open (v2 CLI-08) is pulled forward. Relationship health stays
> Phase 4. The planner must reconcile ROADMAP.md/REQUIREMENTS.md with these
> decisions before planning.

<domain>
## Phase Boundary

`chat-analyzer <chat_file>` (or `python -m chat_analyzer` with interactive prompt) parses a real WhatsApp `.txt` or Telegram `.json` export correctly and produces a **self-contained, decorated HTML "report card"** with tabbed sections, narrative insight lead-ins, charts (matplotlib → base64 PNG), top words/emojis, and a VADER sentiment breakdown. Terminal shows stage narration, a compact summary panel (volume + participants + date range), skip counts, and the absolute report path. The report auto-opens in the default browser. Parsing is strict (no fabricated timestamps, skip counter, system-message classification, Telegram both-shape support, tz→naive UTC). No heavy NLP (emotion/summary/network) and no relationship health in this phase.

</domain>

<decisions>
## Implementation Decisions

### CLI Invocation (no flags — keep it simple)
- **D-01:** Primary flow: `python -m chat_analyzer` → terminal prompts "Enter path to chat export" → user gives location → tool analyzes → writes HTML report card. The Phase 1 interactive prompt remains the default no-arg UX.
- **D-02:** `chat-analyzer <chat_file>` positional argument ALSO supported (CLI-02) — direct path runs without prompting. No-arg still prompts.
- **D-03:** NO CLI flags ship in this phase (not even `--no-charts`, `--date-format`, `--output`). `--help`/`--version` only. Keep it dead simple. Phase 3's `--output` (OUT-04) and Phase 4's `--with-nlp` come later.
- **D-04:** The HTML report is the **deliverable**; the terminal is the entry point + progress + pointer to the report.

### Progress & Error UX
- **D-05:** Stage narration via rich Status/spinners: "Parsing chat…", "Computing insights…", "Writing report…". Parsed-message count surfaced immediately after parsing (CLI-03, success criterion 4). Windows-safe ASCII spinners.
- **D-06:** On bad path or unparseable file: friendly message + loop back to re-prompt (matches Phase 1's file-not-found loop). Full friendly-error-with-export-instructions is CLI-04 in Phase 4.

### Terminal Output
- **D-07:** Terminal shows a compact summary panel AFTER analysis: total messages, participant count, date range. Plus the absolute report path. NO sentiment in terminal, no tables/panels of insights, no charts — insights live in the HTML report.

### HTML Report Card (the deliverable)
- **D-08:** Written next to the input file as `<chat_name>_report.html`; absolute path printed after writing.
- **D-09:** Auto-opens in the default browser after generation (pulls forward v2 CLI-08). On failure to open, degrade gracefully (still print the path).
- **D-10:** Single self-contained HTML file: embedded CSS + JS tabs (no external libs, no CDN), charts base64-embedded as PNG data URIs, opens offline by double-click. `<meta charset="utf-8">`, `html.escape()` on all chat-derived content.
- **D-11:** Tabbed sections, each opening with a **narrative insight lead-in sentence** ("Alice initiated 65% of conversations…", "Most messages land on weekends…"), backed by charts/tables. Suggested tabs: Overview, Participants, Conversation Flow, Words & Emojis, Sentiment.
- **D-12:** Report includes matplotlib charts rendered to base64 PNG: messages-per-day line, hourly activity (heatmap or bar), per-participant bar, sentiment timeline. Reuses existing `ChatVisualizer` (already produces these figures). Set `matplotlib.use('Agg')` before any matplotlib import.
- **D-13:** Full depth of insights: EDA (volume, participants, date range, busiest day/week/hour, response time avg) + top words/emojis + VADER sentiment breakdown. Relationship health, emotion, summary, network EXCLUDED (Phase 4).
- **D-14:** Report filename sanitized (strip path separators, control chars, leading dots) even when derived from input name.

### Parser Hardening (Pitfalls 1-4, 9)
- **D-15:** Strict parse: NEVER fabricate timestamps (`datetime.now()` fallback removed entirely). Lines failing to parse are counted in a `skipped_lines` counter and surfaced to the user.
- **D-16:** Skip surfacing: a single line — "Skipped N lines that couldn't be parsed" — in terminal narration + a note in the report. No per-line examples.
- **D-17:** Common WhatsApp formats only: existing multi-format attempts cover US 12h, EU 24h, iOS bracket, Android dash, 2/4-digit year, optional seconds. NO M/D-vs-D/M disambiguation heuristics, NO locale sniffing, NO override flags (user's focus is conversation insights, not date-format perfection).
- **D-18:** System messages (timestamp header but no sender — e.g. "Messages and calls are end-to-end encrypted", "X added Y") classified as `type="system"` and EXCLUDED from stats, counted in a separate counter. Never appended to the previous message.
- **D-19:** Telegram JSON: support both top-level shapes (bare Chat with `messages` for single-chat export; `chats.list[]` for full export), recursively join string + entity-dict `text` parts, filter `type="service"` messages, drop malformed via the shared skip counter (no silent `except: continue`).
- **D-20:** tz→naive UTC normalization at the parser boundary (Pitfall 9): Telegram tz-aware datetimes converted to naive UTC; both parsers produce naive-UTC `datetime` column. Schema test asserts `df['datetime'].dt.tz is None`.

### Requirement Re-mapping (to reconcile during planning)
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

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope & requirements
- `.planning/ROADMAP.md` §Phase 2 — phase goal, 5 success criteria, requirement mapping (NOTE: §Phase 3 requirements OUT-03/04/05 are pulled forward per this CONTEXT; planner must reconcile)
- `.planning/REQUIREMENTS.md` — CLI-02, CLI-03, ANAL-01..05, OUT-01 (terminal tables/panels), OUT-03/04/05 (pulled forward), OUT-02 (to be dropped); ANAL-07 stays Phase 4; CLI-08 (auto-open) pulled from v2
- `.planning/PROJECT.md` — project context, core value, key decisions (D-01 chat-analyzer name, Phase 1 validations)
- `.planning/STATE.md` — Phase 1 complete; parser hardening locked into Phase 2 (no fabricated timestamps, strict parse + skip counts, tz→naive UTC)

### Research (from new-project — authoritative for parser/pipeline pitfalls)
- `.planning/research/PITFALLS.md` — Pitfall 1 (no `datetime.now()` fallback), Pitfall 2 (WhatsApp regional date formats — user chose "common formats only"), Pitfall 3 (system messages + localized media placeholders), Pitfall 4 (Telegram JSON shape drift), Pitfall 5 (Windows cp1252 encoding — already handled in Phase 1 CLI), Pitfall 6 (terminal charts — moot, dropped), Pitfall 9 (tz-naive/aware mixing)
- `.planning/research/ARCHITECTURE.md` — `cli/pipeline.py` + `adapters.py` + `contracts.py` + `render.py` pattern; AnalysisResults TypedDict; Anti-Pattern 2 (lazy heavy imports); Anti-Pattern 4 (`logging.basicConfig`/`print()` leaking from core); Anti-Pattern 5 (`messages_to_dataframe` moved from app)
- `.planning/research/STACK.md` — rich for terminal rendering, matplotlib → base64 PNG for HTML (report-only; plotext no longer needed), jinja2 autoescape for HTML
- `.planning/research/SUMMARY.md` — consolidated recommendations

### Codebase map
- `.planning/codebase/ARCHITECTURE.md` — component responsibilities, normalized message contract, analysis results dict shapes (for adapters)
- `.planning/codebase/STACK.md` — dependency landscape (pre-pivot)
- `.planning/codebase/CONCERNS.md` — `exec()`/`unsafe_allow_html` web-app concerns must NOT carry into the CLI; known tz bug
- `.planning/codebase/CONVENTIONS.md` — code style to preserve

### Project instruction file
- `AGENTS.md` — project conventions (Python >=3.11, lean base, reuse analysis modules, no web-app-only code)

### Prior phase context
- `.planning/phases/01-package-foundation/01-CONTEXT.md` — D-01..D-11 carry forward (command name, interactive prompt, distribution, package scope)

No external specs — requirements fully captured in decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/chat_analyzer/ingest/ingestion.py` `process_uploaded_file` (line 399) + `normalize_message` (line 323) — the ingestion path already accepts file paths and returns normalized message dicts; pipeline entry point
- `src/chat_analyzer/analysis/eda.py` `ChatEDA` — `analyze_message_volume`, `analyze_conversation_dynamics`, `analyze_content`, `generate_comprehensive_summary` cover volume/participants/dates/top words/emojis
- `src/chat_analyzer/analysis/sentiment.py` `add_sentiment_analysis` + `get_sentiment_summary` (VADER, always available) — sentiment breakdown
- `src/chat_analyzer/utils/visualization.py` `ChatVisualizer` — 12 matplotlib plot methods already produce the chart types needed (timeline, activity, per-participant, sentiment)
- `src/chat_analyzer/cli/main.py` — Phase 1 CLI: encoding bootstrap (cp1252), interactive prompt loop, degrade-not-crash convention (BLE001 noqa) — extend, don't rewrite
- `data/sample_chats/whatsapp_sample.txt`, `data/sample_chats/telegram_sample.json` — fixtures for smoke tests and manual runs

### Established Patterns
- Function-style modules taking/returning DataFrames or dicts; DataFrame columns `datetime, sender, message` are the de-facto interface
- Optional-dependency gates via try/except ImportError + `*_AVAILABLE` flags (matches `[nlp]` lazy-import convention)
- `logging.basicConfig` at import time in `relationship_health.py`/`visualization.py` — should be neutralized in the CLI phase (research Anti-Pattern 4)
- matplotlib `plt.savefig(...)` path in ChatVisualizer — proven path for PNG → base64

### Integration Points
- New `cli/pipeline.py` orchestrates: `process_uploaded_file` → dataframe builder → EDA + sentiment → adapters → AnalysisResults → HTML renderer
- `messages_to_dataframe`/df builder — single source for dicts → df (research Anti-Pattern 5); must normalize tz→naive UTC
- `matplotlib.use('Agg')` set before any matplotlib import so headless/report runs work
- Parser fixes land in `src/chat_analyzer/parser/whatsapp_parser.py` (remove `datetime.now()` at lines 61,63,77,79; system-message classification) and `src/chat_analyzer/parser/telegram_parser.py` (both shapes, recursive text join, service filter, skip counter, tz fix)

</code_context>

<specifics>
## Specific Ideas

- The user's mental model: run `python -m chat_analyzer` → terminal asks for the exported file → tool analyzes → a beautiful, decorated HTML report card appears (auto-opened) showing where the conversation is going/leading. The focus is INSIGHTS and conversation flow, not parser engineering perfection.
- "No flags. I just want it simple."
- The report is a "report card" — well-decorated, inferred insights, tabbed, narrative lead-ins.
- Relationship health explicitly deferred ("Defer health to Phase 4") even though it's cheap — user respects the requirement mapping.

</specifics>

<deferred>
## Deferred Ideas

- Relationship health section in the report (initiator ratio, response lag, dominance, health score) — Phase 4 (ANAL-07, `[nlp]`-labeled)
- Emotion classification, conversation summarization, network graph — Phase 4 (`[nlp]`-gated ANAL-06/08/09)
- Friendly errors with WhatsApp/Telegram export instructions — Phase 4 (CLI-04)
- `--output` flag (OUT-04) — later phase (default-path behavior ships now)
- `--no-report` opt-out semantics — confirm in planning (report is the deliverable)
- Deep WhatsApp date-format disambiguation (M/D vs D/M majority vote, locale sniffing, `--date-format` override) — explicitly declined by user; keep common formats only
- plotext inline terminal charts (OUT-02) — dropped entirely; charts live only in the HTML report

</deferred>

---

*Phase: 2-One-Command Terminal Insights*
*Context gathered: 2026-08-01*
