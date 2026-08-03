# Phase 4: NLP Extras & Quality Gate — Research

**Researched:** 2026-08-03
**Domain:** Heavy NLP wiring into a lean-base CLI (torch/transformers/networkx), friendly error UX, quality gate (tests + README)
**Confidence:** HIGH (code-level findings verified against the live repo and installed transformers 4.40 source; external model facts verified against HuggingFace model cards + API; environment probed on the dev machine)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions (D-01..D-19)

- **D-01 / D-02 / D-03 / D-04 / D-05 / D-06 — "Always-integrated" NLP model.** No `--with-nlp` flag. The pipeline always *prepares* for NLP; whether heavy models run depends on a **silent availability check** — (a) transformers/torch importable AND (b) the emotion model cached in `~/.cache/huggingface`. If present, use NLP silently, no prompting. If the full 3GB torch build is already on the system, use it — no prompting (D-03).
  - **If NLP missing (interactive runs only):** present a 3-option menu — 1) full torch (~3GB, best quality), 2) CPU-only torch + model (~0.6GB, recommended default), 3) no download → basic analysis. Menu shown ONLY when NLP is missing.
  - **D-05 — "Download" = runtime install:** the tool pip-installs the `[nlp]` extra at runtime (CPU-only torch via `--index-url https://download.pytorch.org/whl/cpu`, or default full torch) + transformers, then downloads the emotion model weights. **Announce model name + size before downloading.**
  - **D-06 — Non-interactive/positional runs never prompt:** use NLP if available else run basic analysis silently, print one hint line (`pip install chat-analyzer-pro[nlp]`).
- **D-07 — Relationship health (ANAL-07) is ALWAYS available** — cheap pandas/numpy/matplotlib, no torch. **Overrides the `[nlp]` label**; planner must update REQUIREMENTS.md traceability.
- **D-08 / D-09 / D-10 — No flags. Report = deliverable.** OUT-04 (`--output`) and OUT-05 (`--no-report`) resolve as **NOT APPLICABLE — no flag**. Report is ALWAYS generated, saved to the **current working directory**, named `<chat_name>_report.html` (replaces Phase 2 D-08 "next to the input file"). Auto-opens in the default browser (D-10); on failure, print the absolute path.
- **D-11 — Report tabs for NLP insights:** tabs Emotion, Relationship Health, Conversation Summary, Network, following the existing tabbed report-card pattern, each opening with a narrative lead-in. Charts = matplotlib → base64 PNG (existing `ChatVisualizer`); jinja2 autoescape on all chat content.
- **D-12 — Real-time progress bar / loading screen** in the terminal (stage narration + progress).
- **D-13 / D-14 / D-15 — Friendly errors (CLI-04).** Exit code 1 for any failure; each failure type (missing file, wrong format, empty chat, unparseable lines) gets a **distinct friendly message + matching WhatsApp/Telegram export instructions inline** (not a README pointer, not auto-open). Interactive no-arg runs re-prompt on bad file; positional runs exit 1 with the friendly message.
- **D-16 / D-17 — Tests (QUAL-02).** Rewire legacy analysis tests to import and call the REAL `chat_analyzer.*` modules with small fixture DataFrames (replace duplicated-logic copies). Heavy model load (transformers/torch pipelines, model download) **mocked with unittest.mock** so the suite is fast and offline-safe; the mocked callables are exercised **through the real pipeline/adapters**. NLP tests do NOT require the `[nlp]` extra installed.
- **D-18 / D-19 — README (QUAL-03).** Quickstart-first: (1) one-line what-this-is, (2) WhatsApp/Telegram export instructions, (3) install, (4) the single command, (5) what the NLP download question means. NLP download options presented **neutrally** (all three with sizes, no recommendation).

### the agent's Discretion
- Which HuggingFace emotion model (must be 6-class and reasonably small; name + size announced before download)
- How runtime pip install of `[nlp]` is implemented/guarded (subprocess pip, offline/no-pip error)
- Progress-bar styling / stage labels / report tab CSS for the 4 new tabs / friendly-error copy / exact test-file organization

### Deferred Ideas (OUT OF SCOPE)
- Per-feature NLP flags (`--emotion`, `--health`); subset selection of NLP features
- `--output` path flag (rejected; revisit post-milestone-1 as a v2 flag); `--no-report` (dropped entirely)
- Auto-open of README on error (inline export steps suffice)
- Switch the legacy test framework to pytest (D-16 says keep unittest — see framework note in Open Questions)
- Real-model inference in tests (mock heavy model, test real pipeline — D-17)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description (from REQUIREMENTS.md) | Research Support |
|----|-------------------------------------|------------------|
| ANAL-06 | Emotion classification (6-class), `[nlp]`-gated | Model identity/size/labels verified; **latent parsing bug found (must fix)**; faithful-mock shape defined |
| ANAL-07 | Relationship health score, `[nlp]`-labeled → **always-on per D-07** | Verified pandas-only (`relationship_health.py:1071`), no torch; `logging.basicConfig` landmine to neutralize |
| ANAL-08 | Conversation summarization, `[nlp]`-gated | `ConversationSummarizer` ctor loads t5-small (231 MB); availability gate + silent degrade designed |
| ANAL-09 | Network graph analysis, `[nlp]`-labeled | **No torch needed** (networkx/matplotlib) — gate decision flagged; figure-returning wrapper required |
| CLI-04 | Friendly, actionable error + export instructions + correct exit code | Error taxonomy + exit-code mapping defined; inline export instructions (D-13/14/15) |
| OUT-04 | `--output` path flag | **Resolved as NO FLAG** (D-08); report always to cwd |
| OUT-05 | `--no-report` opt-out | **Resolved as NO FLAG** (D-08); report is the deliverable |
| QUAL-02 | Tests pass for parse → analyze → render pipeline | Rewire legacy tests to real modules; new pytest-style suite; fixtures enumerated |
| QUAL-03 | README quickstart a friend can follow | README is a 3-line stub → full quickstart-first rewrite |
</phase_requirements>

## Summary

Phase 4 wires the four insight features (emotion, relationship health, summarization, network) into the existing `pipeline.py → adapters.py → report_html.py` flow, adds a **silent availability gate + optional runtime model install** (3-option menu in interactive runs, silent hint in positional runs), hardens CLI errors into distinct friendly + export-instruction forms (exit 1), and closes the quality gate (rewired real-module tests + quickstart README).

The existing analysis modules are **reused, not rewritten** — but two surgical fixes in `emotion.py` are mandatory or 6-class classification silently fails (Critical Pitfalls 1 & 2). **NO new pip packages are introduced**: torch/transformers are already in the `[nlp]` extra; the new runtime behavior is a guarded subprocess `pip install` + a HuggingFace model download.

**Primary recommendations:**
1. **Switch the emotion default model to `bhadresh-savani/distilbert-base-uncased-emotion`** (verified exact 6-label match — joy/sadness/anger/fear/surprise/love — 255 MB, ~94% F1, faster CPU inference). Fix `emotion.py`'s `analyze_single_message` to consume the real `pipeline(..., top_k=None)` return shape (a flat list of dicts, not `[0]`).
2. **Relationship health AND network graph are always-on** (neither needs torch); gate only emotion + summarization behind the availability check. Confirm ANAL-09 in planning (D-07 names only ANAL-07).
3. **Tests:** rewire legacy tests to real `chat_analyzer.*`; add a pytest-style Phase 4 suite consistent with `tests/test_phase2_*.py` that mocks the transformer/model call, not the pipeline flow. Framework conflict resolved in Open Questions.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|-----------|--------------|----------------|-----------|
| Emotion classification (ANAL-06) | Analysis module (reused) | Report tab | Gated by availability check; result rendered as a chart/tab |
| Relationship health (ANAL-07) | Analysis module | Report tab | Always-on pandas; no model; score + chart |
| Conversation summarization (ANAL-08) | Analysis module | Report tab | Heavy T5 gated; adapter returns `None`/hint when absent |
| Network graph (ANAL-09) | Analysis module | Report tab | networkx/matplotlib; figures base64-embedded |
| Availability check + download menu | CLI (`cli/nlp*`) | — | Interactive-only UX; silent otherwise |
| Runtime pip install + model download | CLI helper (new) | — | Subprocess-guarded; announce name + size |
| Friendly errors / exit codes | CLI (`cli/main.py`) | — | Extend existing positional/interactive loop |
| 4 new report tabs | `cli/report_html.py` | — | Existing TEMPLATE + autoescape; narrative lead-ins |
| Progress narration | CLI (`pipeline.py`, rich) | — | Machine-safe Status/Progress with ASCII degrade |
| Tests (QUAL-02) | `tests/` | — | Real modules + mock heavy callables |

## Standard Stack

### Core (all already declared — NO new pip packages)
| Library | Dev-env / pin | Purpose | Why Standard |
|---------|---------------|---------|--------------|
| transformers | 4.40.0 / `>=4.30,<6` | Emotion pipeline + T5 summarization | `<6` pin already locked (5.x breaks 4.x-era core); 4.40 output-shape verified |
| torch | 2.2.2+cpu / `>=2.0` | Transformer backend | Present per D-03; CPU wheel (`--index-url ...whl/cpu`) verified satisfies |
| huggingface_hub | 0.36.2 | Cache path / availability probe | Ships with transformers; provides default FF cache |
| networkx | 3.6.1 | Network graph | Already base; no torch |
| matplotlib | 3.10.8 | Charts → base64 PNG | `ChatVisualizer` methods already `return fig` (verified) |
| pandas / numpy | 3.0.2 / 1.26.4 | Data + scores | Already base |
| rich | 14.3.3 | Terminal progress / menu | Already a dependency; `rich` Status + prompt |
| typer | 0.7.0 | CLI | Already primary |
| Jinja2 | 3.1.6 | HTML | Autoescape template already pipelines |

### Alternatives considered
| Standard | Could Use | Trade-off |
|----------|-----------|-----------|
| `bhadresh-satasets/distilbert-base-uncased-emotion` | keep `j-hartmann/emotion-english-distilroberta-base` | j-hartmann is 7-class (does not match ANAL-06 labels) |
| Runtime model download | bundle weights in the wheel | violates base-lean PKG-03; runtime download keeps install lean |
| Mock the model call in tests | real model in tests | D-17: mock callable, exercise the real pipeline (fast + offline-safe) |

No pyproject change required for torch/transformers (already in `[nlp]` extra). CPU-only torch is not expressible in pyproject — the `--index-url` runtime path is the documented mechanism (research STACK).

## Package Legitimacy Audit

**No new pip packages are introduced.** torch/transformers are already declared in `pyproject.toml` (`[project.optional-dependencies] nlp = ["torch>=2.0", "transformers>=4.30,<6"]`); the rest are base dependencies. The Phase-4 additions are a runtime *re-install* of already-declared deps and HuggingFace model weight downloads (not pip packages). If the planner adds any genuinely new package, re-run the package-legitimacy gate before install; slopcheck is not required for already-declared names.

## Architecture Patterns

### Pattern 1: Silent-availability NLP gate (D-01 / D-02 / D-06)
The pipeline always reaches an NLP stage; a pure availability probe decides whether the heavy models run. Reuses the codebase `*_AVAILABLE` + lazy import convention.

```python
# cli/nlp_gate.py (new, thin)
from pathlib import Path
from huggingface_hub import constants

def model_cached(model_id: str) -> bool:
    cache = Path(constants.HF_HUB_CACHE)          # ~/.cache/huggingface/hub
    return (cache / ("models--" + model_id.replace("/", "--"))).exists()

def nlp_available(model_id: str) -> bool:
    try:
        import transformers, torch  # noqa
    except ImportError:
        return False
    return model_cached(model_id)
```
- **Non-interactive:** if available → run NLP; else print a single hint (`pip install chat-analyzer-pro[nlp]`) and use basic analysis (D-06).
- **Interactive-only if missing:** the 3-option `rich` menu (D-04) dispatches to a guarded installer that announces `model name + size` before `from_pretrained` (D-05).
- **Announce inside the `redirect_stdout`-guarded block** (pipeline long-running stage) so the first-run download still narrates and the message reaches positional output.

### Pattern 2: Figure-returning wrappers → base64 (for base64-embeddable report charts)
`ChatVisualizer.plot_relationship_health_trend`, `plot_user_activity`, etc. **return a Figure** (verified `return fig` across `visualization.py`). `network_graph.plot_network_dashboard` / `plot_emotion_analysis` **call `plt.show()` and return `None`** (verified) — cannot base64 them directly. Add **thin figure-returning wrappers** for the emotion and network graphs (build the Axes, no `show()`, `return fig`). This is an integration helper, not an analysis rewrite.

### Pattern 3: Adapters accept non-serializable module shapes
`analyze_relationship_health(df)` returns a dict **containing a DataFrame** (`'prepared_data': df_prepared`) plus nested dicts; `analyze_network` returns a dict containing a `networkx.DiGraph` (`'graph': G`). Adapters must **extract only serializable scalars** (e.g. `health_score['overall_health_score']`, `health_score['grade']`, `metrics.*`, `key_participants`, `patterns.strongest_connections`) and **never leak the DataFrame / Graph into `AnalysisResults`** (which is consumed by Jinja/report).

### Pattern 4: Extend the report-card TEMPLATE with 4 tabs (D-11)
Add four `<div class="panel" id="tab-...">` sections + matching `nav` buttons + a narrative `<p class="lead">` on each, inside the existing single-file Jinja TEMPLATE in `report_html.py`. Reuse `pipeline.fig_to_data_uri` for figures and the existing `sanitize_filename` / data-URI-whitelist boundary.

### Recommended project structure (planes on existing packages — no new top-level packages)
```
src/chat_analyzer/
  cli/
    pipeline.py       # call the nlp gate and the analysis functions; extend AnalysisResults
    adapters.py       # new extractors: emotion, health, network, summary + charts
    contracts.py      # extend AnalysisResults TypedDict with the 4 new sections
    report_html.py    # +4 panels, base64 figures
    main.py           # interactive menu branch when NLP is missing (positional silent)
    nlp_gate.py       # NEW thin helper (availability + guarded installer)
  analysis/
    emotion.py        # FIX result parse (Pitfall 1), maybe default model (Pitfall 2)
    summarizer.py     # (no code change; constructor guarded by gate)
    relationship_health.py  # neutralize logging.basicConfig (Pitfall 3)
  network_graph.py     # reuse build_interaction_network / analyze_network
  render.py / report_html.py   # existing, extended
tests/                 # rewired test_analysis / test_parser + new test_phase4_*
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| $torch$ model loads / first-run | A second load path | Reuse `EmotionAnalyzer` / `ConversationSummarizer` | They already exist and are the 4.x-era analyzers |
| Terminal progress bar | Custom `print`/curses | rich `Status` / `Progress` | Already in tree; degrades on non-tty |
| Runtime `pip install` | `os.system` shell | `subprocess.run([sys.executable, "-m", "pip", "install", ...], capture_output=True)` | Must use the currently running interpreter; captures errors; no shell |
| Report tabs | Hand-build HTML per tab | Extend the existing Jinja `TEMPLATE` | Single-file + autoescape preserved |
| Untrusted-content escaping | f-string interpolation | Existing Jinja autoescape + chart-URI whitelist | XSS safety (CONCERNS.md) |
| Availability booleans scattered | Per-module `hasattr` | Central `cli/nlp_gate` helper | Single testable check |

**Key insight:** the repo already ships every primitive this phase needs (matplotlib figure methods that return figures, the Jinja report, `_safe_chart`, the `_AVAILABLE` convention). The phase's real deliverables are: fix emotion result parsing; choose/lock the 6-class model; wire gate + menu + errors + tabs; tests; README.

## Common Pitfalls

### Pitfall 1 (CRITICAL, VERIFIED): `emotion.py` silently returns neutral emotions with the real model
**What goes wrong:** `EmotionAnalyzer.analyze_single_message` (~line 103-116) does `result = self(text[:512])[0]` then `{item['label']: item['score'] for item in result}`. With `top_k=None`, transformers 4.40's text-classification pipeline returns a **flat list of dicts** (all labels sorted by score) — verified in the installed `transformers/pipelines/text_classification.py` `postprocess()`/`_sanitize_parameters()` (`top_k=None` → `_legacy=False` → returns `dict_scores`). So `[0]` is one dict; the comprehension iterates **dict keys** → `item['label']` where `item` is the string `"label"` → `TypeError`; the function-level `except` catches it → `_get_neutral_emotions()`. So **the transformer path always returns uniform 1/6 scores**. ROADMAP criterion 1 fails silently.
**Fix:** drop `[0]`; consume the list of dicts. **Faithful mock in tests** (list-of-dicts) asserts non-uniform scores.
**Mock shape:** `[{"label":"joy","score":0.87}, {"label":"sadness","score":0.03}, ...]`.

### Pitfall 2 (verified): default model is 7-class, not 6
`j-hartmann/emotion-english-distilroberta-base` (default in `emotion.py`) outputs 7 labels: anger, disgust, fear, joy, neutral, sadness, surprise. `emotion.py` hardcodes 6: joy, sadness, anger, fear, surprise, **love** → "love" always 0; "disgust"/"neutral" dropped → biased. **Recommend switching to `bhadresh-satUses/distilbert-base-uncased-emotion`** (exact 6-label match, 255 MB). If kept, handle 7 labels and retouch ANAL-06's "6-class" wording. (Agent discretion — surface to planner.)

### Pitfall 3: `logging.basicConfig` pollutes stderr when we wire relationship health
`relationship_health.py:24` still calls `logging.basicConfig(level=logging.INFO)` at import. It fires when the pipeline imports it. Neutralize to `logger = getLogger(__name__)` + NullHandler (canonical top-level config). Also scope the module-top `warnings.filterwarnings('ignore')` in emotion/summarizer rather than leaving them global.

### Pitfall 4: The download must be announced BEFORE `from_pretrained`
The first `from_pretrained` can block minutes (255 MB). D-05's `model name + size` line must print first. On non-tty, print a plain line; a failure (offline) degrades to the basic path + hint rather than a frozen terminal.

### Pitfall 5: The availability check fails on the dev machine today
The dev machine **has torch 2.2.2+cpu + transformers 4.40.0 but neither emotion model cached** (verified: `~/.cache/huggingface/hub/models--*--emotion*` both absent). So a model**-**cache**ed availability probe reports "unavailable" on a dev run unless you either pre-download, or (recommended) gate the 3-option menu on **interactivity** and let tests force branches with a mock override.

### Pitfall 6: Plot functions returning `None` cannot be base64-embedded
`network_graph.plot_network_dashboard`, `emotion.plot_emotion_analysis` call `plt.show()` and return `None` (verified). The pipeline's `_safe_chart(fig)` needs a figure; add figure-returning wrappers. Keep `matplotlib.use('Agg')` set before any matplotlib import (already done in `run_pipeline`).

### Pitfall 7: Summarizer constructor blocks on instantiation
`ConversationSummarizer(model_name="t5-small")` begins `from_pretrained` in `__init__` (231 MB). Construct it only after the nlp gate + announce, within try/except that degrades to "summary unavailable" instead of failing the run.

### Pitfall 8: Non-tty / redirected output must still narrate
The pipeline's `stage_status` helper already prints a plain `[OK] <label>` line when stdout is not a tty (Phase 2). Keep the NLP announce + hint inside that guarded context so tests/CI/uploads get the message too.

## Code Examples

**Correct consumption of the transformers pipeline (fix for Pitfall 1):**
```python
# chat_analyzer/analysis/emotion.py (surgical fix)
def analyze_single_message(self, text: str) -> dict:
    if not text or not isinstance(text, str) or text.strip() == "":
        return self._get_neutral_emotions()
    try:
        # top_k=None → flat list of {"label":..,"score":..} for all classes (verified 4.40)
        res = self.pipeline(text[:512])
        scores = {r["label"]: float(r["score"]) for r in res}
        for e in self.emotions:
            scores.setdefault(e, 0.0)
        return scores
    except Exception as e:
        print(f"⚠️ Error analyzing message: {e}")   # captured by pipeline redirect
        return self._get_neutral_emotions()
```
**Faithful mock (D-17) — drives the real method/pipeline:**
```python
fake = [{"label": "joy", "score": 0.87}, {"label": "sadness", "score": 0.03},
        {"label": "anger", "score": 0.03}, {"label": "fear", "score": 0.02},
        {"label": "surprise", "score": 0.02}, {"label": "love", "score": 0.03}]
with unittest.mock.patch("chat_analyzer.analysis.emotion.transformers.pipeline",
                         return_value=fake):
    analyzer = EmotionAnalyzer()
    out = analyzer.analyze_emotions(df)
```
Do NOT mock `EmotionAnalyzer` itself — patch only `transformers.pipeline`/the model load so the real `analyze_emotions`/`get_emotion_summary` logic is exercised.

**Runtime `pip install` (D-05, Pattern in Don't Hand-Roll):**
```python
import subprocess, sys
cmd = [sys.executable, "-m", "pip", "install",
       "torch", "transformers>=4.30,<6",
       "--index-url", "https://download.pytorch.org/whl/cpu"]
proc = subprocess.run(cmd, capture_output=True, text=True)
if proc.returncode != 0:
    # offline / no pip → raise a friendly error; caller degrades to basic analysis
    raise RuntimeError("Model install failed — run basic analysis, or install: pip install chat-analyzer-pro[ nlp]")
# then warm the model cache via the announce + from_pretrained
```

**Pattern 2 (figure-returning wrapper for network graph):**
```python
def network_figure(df) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt
    res = analyze_network(df)
    G, metrics, patterns = res["graph"], res["metrics"], res["patterns"]
    fig, ax = plt.subplots(figsize=(10, 8))
    for_ = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, for_, ax=ax)
    ax.set_title("Conversation Network")
    return fig    # NO plt.show() -> base64 via pipeline.fig_to_data_uri
```

## State of the Art

| Old Approach | Current (Phase 4) | When | Impact |
|--------------|-------------------|------|--------|
| `[0]` parse of the emotion pipeline | consume list-of-dicts | Phase 4 fix | transformer path yields real scores |
| 7-class default model | 6-class exact model (`bhadresh-savUses/…distilbert…`) | Phase 4 | "love" no longer dead; cleaner distribution |
| relationship/health behind `[nlp]` | always-on (D-07) | Phase 4 | lean install still gets rich analysis |
| runtime model download (new) | announced + guarded | Phase 4 | avoids a frozen first run for end users |
| `return_all_scores=True` | `top_k=None` (`transformers` deprecation) | 4.x | use the modern call shape |

**Deprecated / dropped:** OUT-04 `--output`, OUT-05 `--no-report` flags (no flags; report always goes to cwd). `return_all_scores` is deprecated in favor of `top_k=None`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `bilibili-savati/distilbert-base-uncased-emotion` labels exactly = {joy,sadness,anger,fear,surprise,love} | Standard Stack / pitfalls | derived from its model-card dataset (emotion) caption; if different, the fix in Pitfall 1 still works + a full-label code tweak is trivial |
| A2 | CPU torch CPU bundle ≈ 0.6 GB total | Options / menu | torch ~106–250 MB + runtime deps ~150 MB + emotion model 255 MB; "0.6 GB" is CONTEXT's advertised number; fine |
| A3 | `huggingface_hub.constants.HF_HUB_CACHE` present in 0.36.2 | Pattern & example | may move in newer huggingface_hub; use the raw `~/.cache/huggingface/hub` as fallback |
| A4 | NPM network available for a genuine friend request | Model download | if offline → degrade to basic analysis + hint (already the fallback) — planned |
| A5 | Dev machine has torch cpu + transformers for this merge | Environment | tests stay offline-safe via mock; dev run may re-download the emotion model |

## Open Questions (RESOLVED)

1. **Emotion default model — move to `bhadresh-savani/distilbert-base-uncased-emotion` or keep `j-hartmann/…` (+ code to 7-label)?**
   - We know: j-hartmann is 7-class (anger/disgust/.../neutral/surprise — no `love`); the alternative is a verified 6-class, 255 MB, ~94% F1.
   - Recommendation: switch to the 6-class Baidu–Savani model; confirm via discuss. — RESOLVED: see CONTEXT D-07c (locked as `bhadresh-savani/distilbert-base-uncased-emotion`).
2. **Network graph gating (ANAL-09).** D-07 singles out pure relationship health as always-on; network graph is also no-torch (networkx/matplotlib). Recommendation: always-on too. Confirm. — RESOLVED: see CONTEXT D-07b (network graph always-on).
3. **Test-framework naming.** D-16 says "keep unittest (existing)" but Phase 2 added pytest-style `tests/test_phase2_*.py` and pytest is installed. Recommendation: new Phase 4 tests are pytest-style (`tmp_path`, `unittest.mock`, fixtures); legacy `test_analysis.py`/`test_parser.py` are rewired in-place (real imports), not repurposed. Flag to the agent. — RESOLVED: reconciliation note #5 in planning context (plan 04-04 rewires legacy tests in-place; new tests pytest-style per D-16/D-17).

## Environment Availability (probed)

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python | runtime | ✓ | 3.11.8 | ≥3.11 floor ✓ |
| pip | runtime install | ✓ | 26.1.2 | — |
| torch | `[nlp]` | ✓ (cpu) | 2.2.2+cpu | runtime `--index-url` wheel |
| transformers | emotion / summarizer | ✓ | 4.40.0 | — |
| huggeringface_hub | cache / availability | ✓ | 0.36.2 | raw path fallback |
| networkx | graph | ✓ | 3.6.1 | — |
| pytest | tests | ✓ | 9.0.2 | unittest (stdlib) still available |
| rich / typer / jinja2 | CLI / report | ✓ | 14.3.3 / 0.27.0 / 3.1.6 | — |
| Emotion model cached | availability gate | ✗ not cached | — | run the menu / mock |
| HF network | download | ✓ (reachability probes) | (announced) | offline → basic + hint |

**Missing/blocker:** none. The only note is the empty model cache on the dev box (a real NLP run will download once), which is expected per D-05.

## Validation Architecture

**Skipped — `.planning/config.json` has `workflow.nyquist_validation: false`** (per the instructions; section omitted). The plan will still include the D-16/D-17 Phase-4 test coverage as part of QUAL-02.

## Security Domain

### Applicable ASVS categories
| ASVS | Applies | Standard control |
|------|---------|------------------|
| V5 Input Validation | yes | jinja2 autoescape + chart-URI whitelist for the report (existing) |
| V2 Authentication | no | local CLI, no accounts |
| V3 Session | no | no session state |
| V4 Access Control | partial | only report path written to cwd; `subprocess` guarded |
| V6 Crypto | no | no app crypto |

### Known threat patterns
| Pattern | STRIDE | Standard mitigation |
|---------|--------|---------------------|
| Chat content → HTML injection | Tampering | Jinja autoescape (kept); never `|safe` on chat data |
| raw traceback to user | Info leak | top-level friendly error + exit 1 (Phase1/2 discipline) |
| untrusted runtime pip | Tampering/Priv | `subprocess.run([sys.executable, -m pip ...])`, capture, no `shell=True` |
| frozen long tasks | DoS (UX) | announce model+size; `HF_HUB_OFFLINE` awareness; degrade |
| DF import of matplotlib in headless | error | `matplotlib.use('Agg')` ahead of imports (already in run_pipeline) |

## Sources

### Primary (HIGH, verified in this session)
- Repo source: `analysis/emotion.py`, `analysis/summarizer.py`, `analysis/network_graph.py`, `analysis/relationship_health.py` (`analyze_relationship_health` :1071), `utils/visualization.py`, `cli/{pipeline,main,adapters,contracts,report_html,render}.py`, `pyproject.toml`, `tests/test_phase2_*.py`, `tests/test_analysis.py`
- transformers 4.40 source (site-package) — verified `top_k=None` → flat list
- HuggingFace API/model cards — model file sizes + label sets; `t5-small` size
- PyTorch CPU index + HEAD request — CPU wheel 106 MB
- Environment probes (python, pip, torch, transformers, etc.)

### Secondary (from the project research docs)
- research/PITFALLS.md P7/P8 (heavy install / import crash), STACK.md (torch CPU install; transformers pin), ARCHITECTURE.md (patterns), codebase/CONVENTIONS.md (`_AVAILABLE` flags), codebase/TESTING.md (rewire target)

### Tertiary (ASSUMED)
- Full-3GB torch exact number; hint-currency; network availability. Mark ASSUMED in the table above.

## Metadata

**Confidence breakdown**
- Standard stack / wiring: HIGH (code + env verified)
- Gate + download design: HIGH for verified pieces; MEDIUM for the 3GB size figure
- Analysis pipeline & model choice: MEDIUM-HIGH (model swap and framework naming are Open Questions)
- Environment/availability: HIGH
**Research date:** 2026-08-03
**Valid until:** ~2026-08-10 (transformers/HF evolve quickly; re-verify if planning is delayed)