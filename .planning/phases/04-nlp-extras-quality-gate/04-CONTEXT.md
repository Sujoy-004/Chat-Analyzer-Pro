# Phase 4: NLP Extras & Quality Gate - Context

**Gathered:** 2026-08-03
**Status:** Ready for planning

> **Scope note:** This discussion reshaped the Phase 3 leftovers. OUT-04 (`--output`)
> and OUT-05 (`--no-report`) both resolve as **no flag** — the project ships flag-free
> for its non-technical target user. The report is ALWAYS generated and saved to the
> current working directory. The planner must reconcile ROADMAP.md/REQUIREMENTS.md
> with these decisions (both OUT-04/OUT-05 remain "Pending" in REQUIREMENTS.md and
> are re-mapped here).

<domain>
## Phase Boundary

The full v1 feature set ships: 6-class emotion classification (ANAL-06), relationship health (ANAL-07), conversation summarization (ANAL-08), and network graph analysis (ANAL-09) are wired into the pipeline and the HTML report card. NLP is **always integrated** — no `--with-nlp` flag. A silent availability check decides whether to prompt the user about downloading models; heavy deps (torch, transformers) stay gated behind the `[nlp]` extra + lazy imports. Friendly, actionable errors with WhatsApp/Telegram export instructions and a correct exit code (CLI-04). Tests exercise the real `chat_analyzer.*` modules (QUAL-02). README is quickstart-first so a friend can follow it (QUAL-03). Phase 3 leftovers (OUT-04/05) resolve as no-flag: the report always generates to the current working directory, auto-opens in the browser, and is the deliverable.

</domain>

<decisions>
## Implementation Decisions

### NLP Integration Model (the "always integrated" flow)
- **D-01:** NLP is always part of the product intent — no `--with-nlp` flag. The analysis pipeline always *prepares* for NLP; whether heavy models actually run depends on a silent availability check (D-02).
- **D-02:** Silent availability check at startup: verify (a) transformers/torch importable AND (b) the emotion model cached in `~/.cache/huggingface`. If present, use NLP silently — **no prompting whatsoever**.
- **D-03:** If the full 3GB torch build is already on the system, use it — no prompting.
- **D-04:** If NLP is missing, present a 3-option menu to the interactive user:
  1. Download full torch (~3GB) — best quality
  2. Download CPU-only torch + model (~0.6GB total) — "always integrated" recommended default
  3. No download — run basic analysis (lower quality)
  The menu is ONLY shown when NLP is missing (interactive runs only). Fresh-clone friend experience: clone → run → answer 3-option question once → full analysis.
- **D-05:** "Download" means the tool installs the `[nlp]` extra at runtime: pip installs torch (CPU-only via `--index-url https://download.pytorch.org/whl/cpu` for the 0.6GB path, or default full torch for the 3GB path) + transformers, then downloads the emotion model weights. Announce model name + size before downloading (ROADMAP success criterion 2). Model downloads via transformers are announced with name and size before they start.
- **D-06:** Non-interactive/positional runs (`chat-analyzer chat.txt`) never prompt: use NLP if available, else run basic analysis silently and print a single hint line (e.g., `pip install chat-analyzer-pro[nlp]` for richer insights).

### Relationship Health (not gated)
- **D-07:** Relationship health (ANAL-07) is ALWAYS available — it is cheap pandas/numpy/matplotlib code (no torch/transformers), so gating it behind `[nlp]` adds friction with zero benefit. REQUIREMENTS.md labels ANAL-07 as `[nlp]`; this decision overrides the label. The planner must update REQUIREMENTS.md traceability accordingly.

### Report & Flag Surface (Phase 3 leftovers resolved)
- **D-08:** NO CLI flags ship in Phase 4 (consistent with Phase 2 D-03). OUT-04 (`--output`) and OUT-05 (`--no-report`) both resolve as **not applicable — no flag**. The report is the deliverable and is always generated.
- **D-09:** Report is saved to the **current working directory** (where the user runs the command), named after the chat file (`<chat_name>_report.html`). This replaces Phase 2 D-08's "next to the input file". For a friend who clones the repo and runs from it, the report appears right in the cloned repo.
- **D-10:** The report auto-opens in the default browser after generation (keeps Phase 2 D-09). On failure, degrade gracefully and still print the absolute path.

### Report Tabs for NLP Insights
- **D-11:** New tabs per insight in the HTML report: Emotion, Relationship Health, Conversation Summary, Network. Follows the existing Phase 2 tabbed-report-card pattern (Overview, Participants, Conversation Flow, Words & Emojis, Sentiment). Each tab opens with a narrative insight lead-in sentence. Charts rendered matplotlib → base64 PNG (existing `ChatVisualizer`), jinja2 autoescape on all chat-derived content.

### Progress UX
- **D-12:** Real-time progress bar / loading screen in the terminal during the run (stage narration + progress). User's explicit request: "analysis done and shown in terminal, real time progress bar like loading screen, then HTML auto-opens."

### Friendly Errors (CLI-04)
- **D-13:** Exit code 1 for any failure, but each failure type (missing file, wrong format, empty chat, unparseable lines) gets a distinct friendly message plus matching export instructions.
- **D-14:** Export instructions are inline in the message: what went wrong, why (1 line), and the exact how-to-export steps for the relevant app (WhatsApp: Settings→Chats→Export chat; Telegram: desktop-app export). Not a README pointer, not auto-open.
- **D-15:** Interactive no-arg runs re-prompt on bad file (keeps Phase 2 D-06 loop). Positional runs exit with the friendly message.

### Testing (QUAL-02)
- **D-16:** Rewire the legacy analysis tests (tests/test_analysis.py etc.) to import and call the REAL `chat_analyzer.*` modules with small fixture DataFrames, instead of the current duplicated-logic copies. Keep unittest (the existing framework) — no new test framework.
- **D-17:** Heavy model load (transformers/torch pipelines, model download) is mocked with unittest.mock in tests so the suite is fast and offline-safe; the mocked functions are exercised through the real pipeline/adapters on small fixture DataFrames. NLP tests do not require the `[nlp]` extra to be installed to run.

### README (QUAL-03)
- **D-18:** README is quickstart-first: (1) one-line "what this is", (2) export instructions for WhatsApp/Telegram, (3) install, (4) the single command, (5) what the NLP download question means. A friend never reads past step 2 before trying it.
- **D-19:** README presents the NLP download options **neutrally** (all three with sizes, no recommendation): full 3GB, CPU-only 0.6GB, or none.

### the agent's Discretion
- Exact structure of the 3-option download menu UI and its rich rendering
- Which specific HuggingFace emotion model (must be 6-class and reasonably small); model name + size announced before download
- How the runtime pip install of `[nlp]` extra is implemented/guarded (subprocess pip, error handling when offline/no pip)
- Rich progress-bar styling and stage labels
- Report tab/CSS details for the 4 new tabs (within the tabbed + narrative-lead-in decision)
- Which existing `ChatVisualizer` methods to reuse for emotion/health/network charts
- Exact friendly-error copy and export-instruction wording
- Exact test-file organization for the rewired real-module tests

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope & requirements
- `.planning/ROADMAP.md` §Phase 4 — phase goal, 7 success criteria, requirement mapping (NOTE: OUT-04/OUT-05 resolve as no-flag per this CONTEXT D-08; ANAL-07 is always-on per D-07 — reconcile REQUIREMENTS.md traceability)
- `.planning/REQUIREMENTS.md` — ANAL-06/07/08/09, CLI-04, QUAL-02, QUAL-03, OUT-04, OUT-05 (flag resolutions per D-07/D-08)
- `.planning/PROJECT.md` — project context, core value, key decisions (always-integrated NLP intent, lean base install PKG-02/03)
- `.planning/STATE.md` — Phase 3 absorbed into Phase 4; OUT-04/05 folded in; Phase 1/2 decisions locked

### Research (from new-project — authoritative for pitfalls/architecture)
- `.planning/research/STACK.md` — rich for terminal progress, matplotlib → base64 PNG for HTML, jinja2 autoescape, transformers<6 pin, CPU-only torch index
- `.planning/research/ARCHITECTURE.md` — `cli/pipeline.py` + `adapters.py` + `contracts.py` + `render.py` pattern; AnalysisResults TypedDict; Anti-Pattern 2 (lazy heavy imports — critical for the always-integrated NLP flow); Anti-Pattern 4 (`logging.basicConfig`/`print()` leaking)
- `.planning/research/SUMMARY.md` — consolidated recommendations
- `.planning/research/PITFALLS.md` — Pitfall 6 (terminal charts moot), Pitfall 2 (WhatsApp regional date formats), Windows cp1252 (already handled)

### Codebase map
- `.planning/codebase/ARCHITECTURE.md` — component responsibilities, analysis results dict shapes (for wiring NLP into adapters/report)
- `.planning/codebase/STACK.md` — dependency landscape; torch/transformers were deployment-only before the pivot — now behind `[nlp]` extra
- `.planning/codebase/TESTING.md` — legacy tests duplicate logic instead of importing real modules (QUAL-02 rewiring target); unittest conventions, `run_*_tests()` aggregators
- `.planning/codebase/CONVENTIONS.md` — optional-dependency `_AVAILABLE` flags + try/except ImportError pattern (the lazy-load convention NLP must follow); double-quote style; Google docstrings
- `.planning/codebase/CONCERNS.md` — `exec()`/`unsafe_allow_html` must NOT carry into the CLI

### Project instruction file
- `AGENTS.md` — project conventions (Python >=3.11, lean base, reuse analysis modules, no web-app-only code)

### Prior phase context
- `.planning/phases/02-one-command-terminal-insights/02-CONTEXT.md` — D-01..D-20 carry forward (report is the deliverable D-04; terminal = entry+progress+pointer D-07; no flags D-03; tabbed report + narrative lead-ins D-11; auto-open D-09)
- `.planning/phases/01-package-foundation/01-CONTEXT.md` — D-01..D-11 carry forward (command name, distribution, `[nlp]` extra structure)

No external specs — requirements fully captured in decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/chat_analyzer/analysis/emotion.py` `EmotionAnalyzer` + `_initialize_model` — 6-class classification, transformers lazy-import already structured (module-level `_emotion_analyzer`/`_emotion_model_loaded` singletons, CONVENTIONS.md:24)
- `src/chat_analyzer/analysis/relationship_health.py` `analyze_relationship_health` (line 1071) — pandas-only, already composed from identify_conversation_starters → initiator_ratio → response_patterns → dominance → health_score; NO torch import (verified)
- `src/chat_analyzer/analysis/summarizer.py` `ConversationSummarizer` — transformers lazy-import (line 12)
- `src/chat_analyzer/analysis/network_graph.py` — networkx (base dep) + matplotlib; no torch
- `src/chat_analyzer/utils/visualization.py` `ChatVisualizer` — 12 matplotlib plot methods (reuse for emotion/health/network charts)
- `src/chat_analyzer/cli/report_html.py` — Phase 2 single-file report renderer; add tabs following the same pattern
- `src/chat_analyzer/cli/pipeline.py` + `adapters.py` + `contracts.py` — wiring point for new AnalysisResults sections
- `src/chat_analyzer/cli/main.py` — typer app, `--help`/`--version` only; the interactive prompt loop (D-06 re-prompt) lives here
- `src/chat_analyzer/analysis/eda.py` `ChatEDA` — basic-analysis path already delivers volume/participants/words/emojis

### Established Patterns
- Optional-dependency gates via try/except ImportError + `*_AVAILABLE` flags (CONVENTIONS.md:62-99) — the NLP availability check (D-02) must follow this
- Lazy-initialized module singletons `_analyzer`/`_loaded` (CONVENTIONS.md:24) — matches always-integrated lazy NLP
- Function-style modules returning dicts/DataFrames; `datetime, sender, message` DataFrame contract
- Report: jinja2 autoescape, matplotlib → base64 PNG, tabbed with narrative lead-ins (Phase 2 D-10/D-11/D-12)
- Return-safe-defaults instead of raising (CONVENTIONS.md:75-88) — the basic-analysis fallback path

### Integration Points
- Pipeline: `process_uploaded_file` → dataframe → EDA + sentiment (+ NLP sections when available) → adapters → AnalysisResults → render
- New: availability check gate before the NLP stage; 3-option download menu in the no-arg interactive path
- New: 4 report tabs in `cli/report_html.py`
- `pyproject.toml` `[nlp]` extra currently `["torch>=2.0", "transformers>=4.30,<6"]` — CPU-only torch option needs the `--index-url` install path (runtime), not a pyproject change
- Tests: `tests/test_analysis.py` (rewire to real modules), new coverage for CLI-04 errors, pipeline-with-mocked-NLP

</code_context>

<specifics>
## Specific Ideas

- User's mental model: "I run the command in terminal → analysis is done and shown in terminal with a real-time progress bar/loading screen → HTML automatically opens in browser."
- "NLP should always be integrated. Because why else is this project for?" — NLP is the point of the tool, not an add-on.
- Fresh friend: clone → run → if NLP missing, answer ONE 3-option question (3GB / 0.6GB / none) → full or basic analysis. Never asked twice on the same machine.
- "No flags. Keep it simple." — explicit, repeated; the friend should never learn a flag.
- Report location: "stored in this repo (the cloned repo)" — current working directory, named after the chat file.

</specifics>

<deferred>
## Deferred Ideas

- Per-feature NLP flags (`--emotion`, `--health`, etc.) — explicitly rejected; keep one silent/flag-free model
- Subset selection of NLP features — rejected; all-or-nothing via the availability check
- `--output` path flag (OUT-04) — rejected as a flag; report always to current working directory (D-08/D-09). If a real need emerges post-v1, revisit as a v2 flag
- `--no-report` opt-out (OUT-05) — dropped entirely; report is the deliverable (D-08)
- In-tool auto-open of README on error — rejected; inline export steps suffice
- Switch to pytest — rejected; keep unittest
- Real-model inference in tests — rejected; mock the heavy model, test the real pipeline (D-17)

</deferred>

---

*Phase: 4-NLP Extras & Quality Gate*
*Context gathered: 2026-08-03*
