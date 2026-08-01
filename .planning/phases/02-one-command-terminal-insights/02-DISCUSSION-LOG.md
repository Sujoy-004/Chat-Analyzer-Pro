# Phase 2: One-Command Terminal Insights - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-01
**Phase:** 2-One-Command Terminal Insights
**Areas discussed:** CLI invocation model, Terminal output layout, Inline chart behavior, Parser hardening UX

---

## CLI Invocation Model

| Option | Description | Selected |
|--------|-------------|----------|
| Positional arg + keep prompt | Direct run OR interactive no-arg prompt | ✓ |
| Positional arg only | No prompt fallback | |
| Keep prompt only | No positional arg | |

**User's choice:** Positional arg + keep prompt
**Notes:** Phase 1 D-03 prompt stays as the no-arg UX.

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal (help/version only) | No flags this phase | |
| Add --no-charts | Escape hatch for piped terminals | |
| Full set now | --no-charts, --date-format, --locale, --output | |

**User's choice (free-text):** "look no flag. i just want it simple. user will run the 'python -m <something>' then terminal will ask for the exported file location. after user gives the location, the project will analyze the chat and provide user the report card in HTML file (well decorated, inferred insights, where the conversation is flowing etc)"
**Notes:** No CLI flags at all. HTML report card is the deliverable. Terminal = prompt + progress + pointer.

| Option | Description | Selected |
|--------|-------------|----------|
| HTML report is the deliverable | Pull Phase 3 HTML forward; merge 2+3 | ✓ |
| Keep terminal-first, HTML in Phase 3 | Preserve roadmap split | |

**User's choice:** HTML report is the deliverable
**Notes:** Phases 2+3 merge into "one command → report card."

| Option | Description | Selected |
|--------|-------------|----------|
| Stage narration | rich Status spinners per stage | ✓ |
| Minimal plain lines | Few plain text lines | |
| Progress bars | tqdm over iterations | |

**User's choice:** Stage narration
**Notes:** Parsed-message count surfaced right after parsing (CLI-03, SC4).

| Option | Description | Selected |
|--------|-------------|----------|
| Re-prompt on error | Friendly message + loop back | ✓ |
| Error + exit 1 | Stop on typo | |
| Mixed | Re-prompt path, exit on parse | |

**User's choice:** Re-prompt on error

---

## Terminal Output Layout

| Option | Description | Selected |
|--------|-------------|----------|
| Terminal = conduit | Narration + path only | |
| Compact summary + report path | Summary panel + pointer to HTML | ✓ |
| Full terminal + HTML | Doubles rendering work | |

**User's choice:** Compact summary + report path

| Option | Description | Selected |
|--------|-------------|----------|
| Volume + participants + dates + sentiment | Includes sentiment | |
| Volume + participants + dates | No sentiment in terminal | ✓ |
| Just parsed counts | Plain line only | |

**User's choice:** Volume + participants + dates

| Option | Description | Selected |
|--------|-------------|----------|
| Single-page report card | One decorated page | |
| Tabbed multi-section | Tabs/navigation | |
| Narrative-style report | Insight sentences + charts | |

**User's choice (free-text):** "2+3" → then confirmed Tabs + narrative lead-ins

| Option | Description | Selected |
|--------|-------------|----------|
| Tabs + narrative lead-ins | Tabbed sections with insight sentences | ✓ |
| Narrative-first, no tabs | Story scroll | |
| Tabs, minimal narrative | Plain tables | |

**User's choice:** Tabs + narrative lead-ins

| Option | Description | Selected |
|--------|-------------|----------|
| Next to input + print path | Path printed, no open | |
| Next to input + auto-open | Opens in browser | ✓ |
| Prompt to open | Interactive ask | |

**User's choice:** Next to input + auto-open
**Notes:** Pulls forward v2 CLI-08.

| Option | Description | Selected |
|--------|-------------|----------|
| Moderate depth | Top 15 words, top 10 emojis, etc. | |
| Light depth | Top 5 words, minimal | |
| Full depth | Full EDA + sentiment detail | ✓ |

**User's choice:** Full depth

| Option | Description | Selected |
|--------|-------------|----------|
| Include health now | Cheap, base deps only | |
| Defer health to Phase 4 | Keep ANAL-07 labeled | ✓ |

**User's choice:** Defer health to Phase 4

| Option | Description | Selected |
|--------|-------------|----------|
| Charts in report | matplotlib → base64 PNG | ✓ |
| Tables + narrative only | No charts | |

**User's choice:** Charts in report

| Option | Description | Selected |
|--------|-------------|----------|
| Embedded CSS/JS tabs | Single-file, offline | ✓ |
| Scroll layout, no tabs | Simplest | |
| Multi-page | Complicates single-file | |

**User's choice:** Embedded CSS/JS tabs

---

## Inline Chart Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| No terminal charts | All visuals in HTML report | ✓ |
| Terminal + HTML charts | plotext plus report | |

**User's choice:** No terminal charts
**Notes:** Avoids Pitfall 6 entirely.

| Option | Description | Selected |
|--------|-------------|----------|
| Drop OUT-02, plotext not used | Retire the requirement | ✓ |
| Keep OUT-02 best-effort | Nice-to-have terminal chart | |

**User's choice:** Drop OUT-02, plotext not used
**Notes:** plotext never ships; REQUIREMENTS.md must be updated.

---

## Parser Hardening UX

| Option | Description | Selected |
|--------|-------------|----------|
| Strict parse + skip counter | Honest skip, never now() | ✓ |
| Hard fail on unknown format | One bad line kills run | |
| Keep fallback + warn | Contradicts locked decision | |

**User's choice:** Strict parse + skip counter

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-detect from sample | Format sniffing + majority vote | |
| Fixed format order | Current, no fallback | |
| Auto-detect + override flag | Adds --date-format | |

**User's choice (free-text):** "dont bother our aim to analyse what and where the conversation is going/leading to. understand?"
**Notes:** User redirected — focus on insights, not date-format perfection.

| Option | Description | Selected |
|--------|-------------|----------|
| Common formats only | No M/D-D/M heuristics | ✓ |
| Full format detection | Pitfall 2 best practice | |
| Minimal change | No now(), add skips | |

**User's choice:** Common formats only

| Option | Description | Selected |
|--------|-------------|----------|
| Single skip count | One line in narration + report | ✓ |
| Count + sample lines | Show skipped snippets | |
| Silent | Contradicts SC5 | |

**User's choice:** Single skip count

| Option | Description | Selected |
|--------|-------------|----------|
| Classify + exclude | System type, excluded from stats | ✓ |
| Keep appending | Corrupts content | |
| Classify + count in report | Show system count | |

**User's choice:** Classify + exclude

| Option | Description | Selected |
|--------|-------------|----------|
| Both shapes + entity text + tz fix | Full Telegram hardening | ✓ |
| Minimal Telegram change | Bare Chat only | |
| Defer Telegram hardening | WhatsApp only | |

**User's choice:** Both shapes + entity text + tz fix

---

## the agent's Discretion

- Exact rich Status/panel styling and ASCII-safe symbols
- Report template structure and CSS design (within tabbed + narrative-lead-in decision)
- Which ChatVisualizer methods to reuse for the 4 chart types
- Pipeline/adapter/contract structure (per research ARCHITECTURE.md)
- How OUT-05 (`--no-report`) semantics resolve given the report is the deliverable

## Deferred Ideas

- Relationship health section — Phase 4 (ANAL-07)
- Emotion/summary/network — Phase 4 (`[nlp]`-gated)
- Friendly export-instruction errors — Phase 4 (CLI-04)
- `--output` flag (OUT-04) — later phase
- Deep WhatsApp date-format disambiguation — declined by user
- plotext terminal charts (OUT-02) — dropped entirely
