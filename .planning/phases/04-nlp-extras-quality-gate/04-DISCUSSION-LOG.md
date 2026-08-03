# Phase 4: NLP Extras & Quality Gate - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-03
**Phase:** 4-NLP Extras & Quality Gate
**Areas discussed:** NLP gate surface, Model & download UX, --no-report & --output, Friendly errors (CLI-04), Testing (QUAL-02), README (QUAL-03)

---

## NLP Gate Surface

| Option | Description | Selected |
|--------|-------------|----------|
| --with-nlp gates 3, health always-on | One --with-nlp flag unlocks emotion/summarization/network; relationship health always-on (cheap pandas-only) | |
| One flag gates all 4 | All four behind --with-nlp, matching REQUIREMENTS ANAL-07 [nlp] label | |
| Per-feature flags | --emotion / --health / --summary / --network | |

**User's choice:** Free-text: "NLP should always be integrated... the project should automatically check in secret if those NLPs are already downloaded... if yes, use those without informing the user." Reshaped to: **no --with-nlp flag at all**; silent availability check + optional download question.
**Notes:** User rejected flags entirely and repeatedly. NLP is the point of the project, not an add-on.

## Model & Download UX

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-install CPU-only torch + model (~0.6GB) | pip --index-url cpu + model | |
| Print the pip command, tool downloads model only | User runs pip install themselves | |
| Auto-install default torch (~3GB) | In-tool pip, heaviest | |
| No download question at all | NLP only if already present | |
| Auto-install 3GB if present, else 3-option menu | Use existing 3GB silently; else offer 3GB / 0.6GB / none | ✓ |

**User's choice:** "If those 3GB is already present in the system, use that, or else provide the options: 1. download 3GB, 2. download 0.5-0.6GB, 3. no download, accept relatively low quality result."
**Notes:** Reported real sizes to the user (torch CUDA ~2.5-3GB, CPU-only ~200-250MB, transformers ~15MB, model ~260-330MB). Silent check = deps importable + model cached.

## --no-report & --output

| Option | Description | Selected |
|--------|-------------|----------|
| Full terminal output | --no-report shows full insights in terminal tables/panels | |
| Summary panel only | --no-report shows compact summary + narration | |
| Drop --no-report entirely | Report always generated | ✓ |
| File or directory accepted | --output flexible | |
| File path only | --output must end .html | |
| Directory only | --output always a directory | |

**User's choice:** "I don't want any flags. Keep it simple, the output HTML will be stored in this repo (the cloned repo)."
**Notes:** Both OUT-04 and OUT-05 resolve as no-flag. Report saved to current working directory (D-09), named after the chat file.

## Friendly Errors (CLI-04)

| Option | Description | Selected |
|--------|-------------|----------|
| Exit 1, per-type message | Exit 1 + specific friendly message + export steps | ✓ |
| Distinct exit codes | Per-failure codes for scripts | |
| Always exit 0 | Never looks crashed | |
| Inline message + steps | What/wrong-why + how-to-export steps | ✓ |
| Point to README | Message points to README section | |
| Auto-open README on error | Browser opens README export section | |
| Re-prompt loop on no-arg | Loop back to "Enter path to chat export:" | ✓ |
| Exit immediately always | Any error exits | |
| Loop interactively, exit positional | Combines both | |

**User's choice:** Exit 1 + per-type message; inline export steps; re-prompt loop on no-arg.
**Notes:** Keeps Phase 2 D-06 loop behavior for interactive runs.

## Testing (QUAL-02)

| Option | Description | Selected |
|--------|-------------|----------|
| Rewire legacy tests to real modules | Import real chat_analyzer modules, mock only transformers/torch | ✓ |
| Add new real-module tests alongside | Old duplicated tests stay | |
| Replace legacy tests wholesale | Delete duplicated-logic tests | |
| Keep unittest | Existing framework, no new deps | ✓ |
| Switch to pytest | Richer fixtures, new dev dep | |
| Mock heavy model, test real pipeline | unittest.mock transformers, real adapters/pipeline | ✓ |
| Skip NLP tests when deps missing | skipIf not importable | |
| Real model inference | Slow, most realistic | |

**User's choice:** Rewire legacy tests to real modules; keep unittest; mock heavy model + test real pipeline.

## README (QUAL-03)

| Option | Description | Selected |
|--------|-------------|----------|
| Quickstart-first README | What/export/install/command/download-meaning | ✓ |
| Add a Quickstart section | Less rework | |
| NLP-download explainer focus | Sizes + 3-option flow | |
| Recommend CPU-only (0.6GB) | Default recommendation | |
| Neutral 3-option listing | All three with sizes, no recommendation | ✓ |
| Recommend full 3GB | Best-quality framing | |

**User's choice:** Quickstart-first README; neutral 3-option download listing.

---

## the agent's Discretion

- Download-menu UI structure and rich rendering
- Specific HuggingFace emotion model (6-class, reasonably small) + announced name/size
- Runtime pip install implementation/guards for the [nlp] extra
- Progress-bar styling and stage labels
- New report tab / CSS details
- ChatVisualizer methods reused for new charts
- Friendly-error and export-instruction copy
- Rewired test-file organization

## Deferred Ideas

- Per-feature NLP flags and subset selection — rejected
- OUT-04 --output flag — no-flag per user; v2 candidate if real need
- OUT-05 --no-report — dropped; report is the deliverable
- Auto-open README on error — rejected
- pytest — rejected
- Real-model inference in tests — rejected
