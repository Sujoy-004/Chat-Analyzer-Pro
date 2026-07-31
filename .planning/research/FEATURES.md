# Feature Research

**Domain:** pip-installable chat-analysis CLI (WhatsApp `.txt` / Telegram `.json` → terminal insights + HTML report)
**Researched:** 2026-07-31
**Confidence:** HIGH (verified against ecosystem) / LOW (where flagged)

## Feature Landscape

**Positioning context.** The target user is *not* a developer. They exported a WhatsApp chat and want to run one command. The entire feature surface must serve "export → run `analyze chat.txt` → read insights → share the HTML report," with zero configuration. The analysis core already exists in `src/` (validated in PROJECT.md) — so complexity ratings below reflect **CLI-wrapping cost**, not analysis cost. The differentiators are nearly free because the engine is already built.

### Table Stakes (Users Expect These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| One-command pipeline (`analyze <file>`) | The core value: "one command turns a raw chat export into insights." No subcommands, no config, no interactive prompts on first run | LOW | Entry point in `pyproject.toml`. Default run = everything, exit 0. **Highest-priority feature in the entire product.** |
| Summary statistics | Every chat analyzer in the ecosystem leads with: total messages, participants, conversation span (first→last date), total words, media count, links count | LOW | Exists: `ChatEDA.analyze_message_volume`, `analyze_content`, `WhatsAppParser.get_basic_stats`. Render as rich table + KPI panel. |
| Per-participant stats | Message count + % contribution + ranking. Users immediately ask "who messages the most?" | LOW | Exists: `analyze_conversation_dynamics`. Terminal: top-10 table (truncate for large groups). |
| Timeline trends (daily/monthly message volume) | "How has our conversation evolved?" is a top-3 expected insight in every comparable tool | LOW | Exists: `ChatEDA.analyze_message_volume`; terminal line/bar chart via plotext; HTML line chart via existing matplotlib timeline. |
| Activity patterns (hour-of-day, day-of-week) | "When are we most active?" — expected; the classic weekday×hour heatmap is the crowd-pleaser | LOW (metrics) / MED (heatmap rendering) | Metrics exist (`_categorize_time`). Terminal: hour histogram + top-3 hours text. Heatmap is HTML-only (see split below). |
| Word frequency (top words) | Top-20 words after stopword removal is expected; wordcloud is the shareable centerpiece | LOW | Exists: `analyze_content` + `preprocess_text`. Terminal: top-15 word list. Wordcloud HTML-only. |
| Emoji breakdown | Top emojis is an expected "fun" stat across all comparable tools | LOW | Exists: `extract_emojis`, `plot_emoji_distribution`. Terminal: top-10 emoji list (unicode renders on modern terminals; fall back to `:name:` text on legacy consoles). |
| Sentiment breakdown (pos/neg/neutral) + over time | Expected in every comparable tool; sentiment distribution is table stakes, per-participant and over-time are expected in better ones | LOW | Exists: VADER + HF, `plot_sentiment_distribution`, `plot_sentiment_timeline`. VADER is the ecosystem baseline; keep HF optional (see `--light`). |
| Terminal output with inline charts | The stated differentiator of the pivot (PROJECT.md): "full insights stay in-terminal." Terminal is the *first* read; HTML is the shareable report | MED | plotext (bar/line/histogram/date, subplots, zero deps) + rich (tables/panels/status). Define a terminal-first section set (below). |
| Self-contained HTML report | Ecosystem standard: one `.html` file with everything inlined (charts, tables, CSS) — shareable via email/Slack, works offline, no viewer-side deps | MED | Reuse `src/utils/visualization.py` matplotlib figures → base64 PNG embedded in a template. **Do not require JavaScript** (whatsapp-wrapped needs Plotly/JS; our matplotlib approach is simpler and mobile-safe). |
| Friendly error handling + export instructions | The #1 real-world failure for this user: wrong file, wrong format, iOS zip, locale date formats. Errors must tell them what happened AND how to fix it (incl. how to export from WhatsApp/Telegram) | MED | Distinguish: file-not-found, not-a-chat-file, wrong-format (expected WhatsApp `DD/MM/YYYY, HH:MM - Name:`), locale-variant date formats, empty chat, encoding issues (UTF-8/emoji). Print remediation with rich panel, exit code 1. |
| Progress indication during analysis | Heavy NLP (torch, HF emotion ~450MB model, summarization) makes runs take seconds-to-minutes. No feedback = "is it hung?" | LOW | rich `status` spinners for indeterminate stages (model load) + `progress` bar for per-message emotion classification. Include first-run model-download notice. |
| Basic flags: `--help`, `--output`, `--quiet` | Minimum CLI hygiene; `--quiet` matches ecosystem (whatsapp-wrapped has it) and keeps terminal clean for scripting | LOW | `--output` sets HTML report directory; default `./`. `--quiet` = suppress non-essential terminal chatter (still write HTML). |
| Handles both WhatsApp `.txt` and Telegram `.json` | Stated scope (PROJECT.md). Telegram support already puts us ahead of most WhatsApp-only tools | LOW | Parsers exist. Auto-detect by extension + content sniff, don't make the user declare format. |

### Differentiators (Competitive Advantage)

The analysis engine already exists — **these are nearly free to expose and are the reason anyone picks this tool over the dozens of WhatsApp analyzers.**

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Emotion classification (6 classes) | Almost nobody in the ecosystem goes beyond positive/negative/neutral. "Anger spike in March" is a wow moment | LOW (exists: `EmotionAnalyzer`, `get_emotion_summary`, `get_emotion_timeline`) | HF model + rule-based fallback. HTML: distribution + timeline. Terminal: summary table. Gate behind `--light` flag. |
| Relationship health score | The signature feature of the original app (`relationship_health.py`, 1175-line orchestrator). Turns a stats dump into "what's actually going on in this relationship" | LOW (exists) | Headline score panel in terminal (rich) + full component breakdown (starters/initiators/response/dominance) in HTML. **Include in v1 — it is the hook.** |
| Conversation summarization | Rare in the ecosystem (only topic-modeling tools like MendasD/whatsapp-analyzer touch it). "What was this conversation about?" is a genuinely different answer | MED (exists: `summarizer.py`; slowest stage) | Terminal: 3-5 bullet takeaways. HTML: longer section. Runs last; `--light` skips it. |
| Network graph analysis (group chats) | Who-replies-to-whom, key participants, subgroups — near-unique in CLI form; strong for group-chat users | LOW (exists: `network_graph.py`) | HTML-only rendering (matplotlib). Terminal: top-3 "connectors" text summary. |
| Terminal + HTML duality | Ecosystem is split: HTML-report tools (whatsapp-wrapped) or web-app tools (most others). Both "instant insights in terminal" AND "shareable report" in one command is unique | MED | Requires the terminal-first/HTML-only chart split to be deliberate (see below). |
| Privacy / local processing as the explicit story | Ecosystem leaders (whatsapp-wrapped "100% Private", MendasD "no message content ever leaves your machine") all lead with this — it's table stakes for *credibility* but a differentiator for *conversion*. A CLI with no accounts/no hosting makes the claim structurally true | LOW | Bake into README, `--help`, and report footer: "Analyzed 100% on your device. Nothing was uploaded." |
| Telegram `.json` support | Almost the entire ecosystem is WhatsApp-only (whatstk, whatsapp-wrapped, etc.). Telegram users have zero good options | LOW | Parser exists. |
| `--light` fast path | First-run UX: VADER-only sentiment + rule-based emotion, no HF download, no summarization. Answers "I want my report NOW" | LOW (both fallbacks exist in `sentiment.py`/`emotion.py`) | Default = full pipeline (validated decision in PROJECT.md). `--light` = fast. |
| `--json` machine-readable output | Cheap scriptable surface; useful for power users and future tooling; trivial with rich dict results | LOW | Prints analysis summary as JSON to stdout; suppresses charts. P2. |
| Auto-open the HTML report on completion | "Report saved to `./chat_analysis_report.html` — opening in your browser" (`webbrowser.open`). Zero-config delight | LOW | Honor `--quiet` (don't auto-open if quiet? Or still open — decide in phase). |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Interactive TUI mode (Textual) | "Wouldn't it be cool to explore in the terminal?" | Builds a second UI, doubles state management, most of the exploration value is already in the HTML report. Pure CLI is a stated constraint (PROJECT.md: "GUI of any kind" out of scope) | Keep output-only CLI; the HTML report is the exploration surface |
| Cloud upload / account sync / "analyze online" | Users of the old Streamlit app expect web | Structurally contradicts the privacy story (Core Value: "no accounts, no hosting"). Irreversible trust damage | Never offer; advertise local-only instead |
| PDF report | Ecosystem tools offer it (Anish62027, gauravmeena0708) | Deferred in PROJECT.md for good reason: HTML covers sharing; PDF adds reportlab/weasyprint weight + Windows font hell. Nobody shares a PDF from a chat report | Ship self-contained HTML; revisit only if explicitly asked |
| Excel export | JBoixCampos offers it; "give me the data" | Data-dump request from a tool whose value is *interpretation*. Every table in HTML is already copy-pasteable | HTML tables; `--json` for real data needs |
| Telemetry / feedback collection | "How do we know if users like it?" (Anish62027 literally ships a feedback form) | A privacy-first local tool that phones home is a betrayal of the Core Value. Kills the selling point | Issue tracker + GitHub stars; optionally `--no-telemetry`-style opt-in later (probably never) |
| Runtime code download + `exec()` (carried over from Streamlit app) | The web app fetched modules from GitHub to stay thin | Remote code execution in a tool that handles people's most intimate data = critical vulnerability (already flagged in `.planning/codebase/CONCERNS.md`) | Ship everything in the pip package; HF model weights are the only legitimate download (cached, standard `transformers` behavior) |
| OCR / PDF / image ingestion passthrough | `src/ingest/ingestion.py` supports it (web-app legacy) | Chat analysis is about *conversation text*; OCR of screenshots is a different product. Adds tesseract/poppler system deps that break `pip install` on Windows | CLI accepts `.txt`, `.json`, and (iOS-export) `.zip` only; everything else → friendly error |
| "Wrapped" style slideshow / video output | whatsapp-wrapped's aesthetic | The terminal is our identity; a Plotly slideshow adds JS-heavy HTML and Playwright tooling for zero extra insight | Clean static HTML report with inline charts |
| Live / real-time chat monitoring | "Analyze my chat as messages come in" | Requires account access (Telethon) — huge scope, privacy contradiction, no export needed | Batch analysis only (v1); never track live |

### Terminal-First vs HTML-Only Output Split (design decision)

Terminal output is a **curated executive summary**, not the full report:

| Insight | Terminal (plotext/rich) | HTML (matplotlib base64) |
|---------|------------------------|--------------------------|
| Summary KPIs | rich panel (messages, participants, span, words, media, links) | KPI cards |
| Per-participant ranking | rich table, top 10 | full table |
| Timeline | plotext bar/line (daily or monthly) | matplotlib line chart |
| Hour-of-day activity | plotext histogram + top-3 hours | heatmap (day × hour) |
| Top words | rich list, top 15 | wordcloud + top-20 table |
| Emoji | rich list, top 10 | bar chart + top-20 table |
| Sentiment | split bar + VADER pie-as-bars; sentiment-over-time line (plotext) | distribution + timeline + per-participant |
| Emotion | summary table (6 classes) | distribution + timeline |
| Relationship health | score panel + verdict | full component breakdown + trend |
| Network graph | top-3 "connectors" text | full graph viz + metrics |
| Summarization | 3-5 bullet takeaways | longer narrative section |
| Top positive/negative messages | — | quote blocks with sentiment score |

Rationale: wordclouds, network graphs, and heatmaps degrade to noise in ASCII; the terminal's job is *orientation*, the HTML's job is *exploration and sharing*.

## Feature Dependencies

```
analyze <file> (CLI entry)
    ├──requires──> format auto-detection (txt/json/zip)
    │                    ├──requires──> WhatsApp parser
    │                    └──requires──> Telegram parser
    ├──requires──> analysis pipeline (EDA → sentiment → emotion → health → network → summary)
    ├──requires──> terminal renderer (rich tables/panels + plotext charts)
    ├──requires──> HTML report renderer (matplotlib figures → base64 embed)
    └──requires──> exit-code + error-handling layer

Progress indication ──enhances──> analysis pipeline (staged reporting)
--light ──enhances──> emotion/sentiment (uses rule-based + VADER-only paths)
--quiet ──conflicts──> auto-open HTML (decide: quiet still writes report, does not auto-open)
--output ──requires──> report writer parameterization
--user / --from / --to filters ──requires──> df-filter before analysis (pandas slice; LOW)
--json ──requires──> analysis pipeline to return serializable dicts (already the norm in src/)
```

### Dependency Notes

- **One-command pipeline requires staged analysis + staged rendering**: the pipeline must be modular so progress reporting and `--light`/`--quiet` can hook each stage. Do not build the CLI as one monolithic function.
- **Terminal and HTML renderers are independent consumers of the same analysis dicts**: this keeps the duality cheap — analysis runs once, both renderers read the result. This is the single most important architectural constraint for the features (see ARCHITECTURE.md).
- **`--light` enhances emotion/sentiment**: both already have non-HF fallbacks (`_rule_based_emotion`, VADER-only config) — the flag selects the path.
- **Error-handling layer requires parser-level diagnostics**: WhatsApp date-locale failures must surface which line/format broke, or the friendly-error feature is impossible.
- **`--user`/`--from`/`--to` conflict with nothing**: filters apply before analysis, so they reuse the entire pipeline. Defer to v1.x only to keep the flag surface minimal.
- **`--quiet` conflicts with auto-open**: define semantics in the phase (recommend: `--quiet` suppresses terminal output but still writes and opens the report — "quiet" is about stdout chatter, not the deliverable).

## MVP Definition

### Launch With (v1)

Ruthless cut: the analysis engine exists, so v1 is an *exposure* problem. These validate the pivot:

- [ ] **One-command pipeline** — `analyze chat.txt` parses → analyzes → prints terminal report → writes HTML → prints path. Exit codes 0/1.
- [ ] **Summary + per-participant stats** — KPIs, ranking table.
- [ ] **Timeline + hour-activity** — terminal charts + HTML line/heatmap.
- [ ] **Top words + top emojis** — terminal lists + HTML wordcloud/bar.
- [ ] **Sentiment breakdown** (distribution + over time) — VADER path always works.
- [ ] **Emotion breakdown** — 6-class summary (HF with `--light` fallback).
- [ ] **Relationship health score** — the signature hook; headline panel in terminal, breakdown in HTML.
- [ ] **Terminal inline charts** — plotext bar/line/histogram for the terminal-first set.
- [ ] **Self-contained HTML report** — matplotlib base64, no JS dependency, works offline.
- [ ] **Friendly errors + export instructions** — the non-technical user's lifeline.
- [ ] **Progress indication** — rich status/progress; first-run model notice.
- [ ] **`--help`, `--output`, `--quiet`**.

Launch-with summary: everything that makes the Core Value true ("one command → real insights, locally, fast, shareable") and nothing that adds surface.

### Add After Validation (v1.x)

- [ ] **`--light` fast path** — trigger: first-run feedback shows HF download frustration.
- [ ] **`--user` per-participant filter** — trigger: group-chat users ask "what about just me?" (needs flag-surface review).
- [ ] **`--from` / `--to` date-range filter** — trigger: same user feedback.
- [ ] **`--json` output** — trigger: any power-user/automation demand.
- [ ] **iOS `.zip` export support** — trigger: share of users exporting from iPhone ("Without Media" produces a zip). Decision flag: PROJECT.md scopes v1 to `.txt`/`.json`, but `src/ingest` already handles zip — likely cheap to pull into v1. Decide during planning.
- [ ] **`--name` custom report title** (whatsapp-wrapped parity) — trigger: cosmetic polish pass.

### Future Consideration (v2+)

- [ ] Additional platforms (Instagram/Discord/Signal) — requires new parsers; out of scope per PROJECT.md.
- [ ] PDF report — only if explicitly asked; HTML covers sharing.
- [ ] Anonymize/redact mode — interesting privacy differentiator (strip names before report); real scope, defer.
- [ ] Multi-language CLI (es/en) — JBoixCampos has it; defer until an international user asks.
- [ ] Weekly digest / Telegram bot (`src/reporting/weekly_digest.py`) — deferred per PROJECT.md; it's a scheduler + credentials problem, not a chat-analysis problem.
- [ ] Compare-two-chats mode (MendasD's `compare` subcommand) — novel but v2 at best.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| One-command pipeline | HIGH | LOW (entry point + orchestration) | P1 |
| Summary + per-participant stats | HIGH | LOW (exists) | P1 |
| Timeline + activity | HIGH | LOW (exists) | P1 |
| Top words + emojis | MED | LOW (exists) | P1 |
| Sentiment (VADER + HF) | HIGH | LOW (exists) | P1 |
| Emotion breakdown | HIGH | LOW-MED (exists; model load time) | P1 |
| Relationship health score | HIGH | LOW (exists; orchestrator ready) | P1 |
| Terminal inline charts | HIGH | MED (plotext wrappers per section) | P1 |
| HTML report (self-contained) | HIGH | MED (template + base64 pipeline) | P1 |
| Friendly errors + export help | HIGH | MED (parser diagnostics + copy) | P1 |
| Progress indication | HIGH | LOW (rich) | P1 |
| `--help` / `--output` / `--quiet` | MED | LOW | P1 |
| Conversation summarization | MED | MED (slowest; runs last) | P2 |
| Network graph (groups) | MED | LOW (exists; HTML-only) | P2 |
| `--light` fast path | MED | LOW (fallbacks exist) | P2 |
| iOS `.zip` input | MED | LOW (ingestion exists) | P2 (decide vs P1 in planning) |
| `--user` / `--from` / `--to` filters | MED | LOW-MED (df filter + flags) | P2 |
| `--json` output | LOW-MED | LOW | P2 |
| `--name` / `--year` filters | LOW | LOW | P3 |
| PDF report | LOW | HIGH | P3 (anti-feature until asked) |
| TUI / live monitoring / cloud | — | HIGH | Never (anti-features) |

## Competitor Feature Analysis

| Feature | whatsapp-wrapped (CLI, closest analog) | whatstk (library + CSV CLI) | Web-app analyzers (huzefa10, JBoixCampos, Anish62027, gauravmeena0708) | Our Approach |
|---------|----------------------------------------|------------------------------|----------------------------------------------------------------------|--------------|
| CLI entry | `whatsapp-wrapped chat.zip` | `whatstk-to-csv in out.csv` | none (Streamlit) | `analyze chat.txt` full pipeline |
| Terminal output | none (HTML only) | CSV conversion only | none | rich tables + plotext charts |
| HTML report | interactive Plotly (JS required); `--static` pre-render optional | none | n/a | self-contained static, base64 matplotlib, no JS |
| Sentiment | not offered | not offered | pos/neg/neutral + over time (Anish, gaurav) | VADER + optional HF |
| Emotion (6-class) | not offered | not offered | emotion trends (gaurav: 5 classes) | 6-class HF + rule-based fallback |
| Summarization | not offered | not offered | not offered | conversation summarizer |
| Relationship health | not offered | not offered | not offered | flagship scoring pipeline |
| Network graph | not offered | not offered | not offered (some group stats) | interaction network + subgroups |
| Telegram | not offered | not offered | not offered | native `.json` parser |
| Privacy framing | "100% Private — all processing stays on your device" | n/a | JBoixCampos: "complete privacy" | local-only + no-account + no-hosting; stated in help/README/footer |
| Errors/export help | README section only | docs | n/a (web) | in-terminal remediation messages |
| Progress | Colab-style cells | n/a | n/a | rich progress + first-run model notice |
| Export formats | HTML (interactive/static) | CSV | HTML index + per-user (gaurav), Excel (JBoixCampos), PDF (Anish) | HTML only (v1) + `--json` later |
| Flags | `--name --year --static --output --quiet --help` | positional in/out | n/a | `--help --output --quiet` (+ `--light`, filters later) |

**Gap we fill:** no existing tool combines *terminal-first insights*, *self-contained HTML*, *sentiment + emotion + summarization + relationship health + network graph*, *WhatsApp + Telegram*, and *local-only processing* in one command. whatsapp-wrapped is closest on positioning but is HTML-only and stats-only; whatstk is a library, not an analysis tool. Every other serious analyzer is a web app.

## Sources

- **whatsapp-wrapped** (Duelion) — GitHub README: https://github.com/Duelion/whatsapp-wrapped — HIGH (primary source, closest analog; flags, privacy framing, HTML formats)
- **whatstk** 0.8.1 (lucasrodes) — GitHub + PyPI: https://github.com/lucasrodes/whatstk, https://pypi.org/project/whatstk/ — HIGH (ecosystem reference; CLI = CSV conversion only; WhatsApp-only)
- **huzefa10/whatsapp-chat-analyser** — GitHub: https://github.com/huzefa10/whatsapp-chat-analyser — MED (web search, feature list corroborated by multiple tools)
- **whatsapp-reality** (PyPI) — https://pypi.org/project/whatsapp-reality/ — MED (feature list: stats/timeline/emoji/sentiment/reply-time)
- **gauravmeena0708/whatsapp-analyzer + whatsapp-groupchat-analyzer** (PyPI) — web search — MED (index.html + per-user HTML reports, emotion trends, POS/behavioral insights)
- **MendasD/whatsapp-analyzer** — GitHub — MED (single 2026 source; local-only Click CLI, topics, privacy framing)
- **JBoixCampos/whatsapp-chat-analyzer** — GitHub — MED (web search; Excel export, response times, multilingual)
- **Anish62027/Whatsapp_Chat_Analyzer** — GitHub — MED (web search; PDF/Excel export, feedback form — used as anti-feature evidence)
- **plotext** 5.3.2 — PyPI: https://pypi.org/project/plotext/ — HIGH (terminal chart capabilities verified; zero deps; saves to colored HTML)
- **rich** 15.0.0 — PyPI: https://pypi.org/project/rich/ — HIGH (tables, progress bars, status spinners, panels verified; legacy Windows console = 16-color/emoji limits)
- **HTML-report ecosystem** (tessera-report, datainpane, folio, report-creator, vizblend) — PyPI/GitHub — MED (self-contained single-file HTML is the confirmed ecosystem standard; base64/inline-SVG embedding)
- **Existing codebase** `src/` + `.planning/codebase/STRUCTURE.md` + `.planning/PROJECT.md` — HIGH (all "exists" claims verified by direct read)
- **CLI UX conventions for non-technical users** — training data, general conventions (help text, exit codes, remediation in errors) — LOW (flag for validation during plan phase)

---
*Feature research for: Chat-Analyzer-Pro CLI pivot*
*Researched: 2026-07-31*
