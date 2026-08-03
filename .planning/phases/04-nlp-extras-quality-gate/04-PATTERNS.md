# Phase 4: NLP Extras & Quality Gate — Pattern Map

**Mapped:** 2026-08-03
**Files analyzed:** 12 (10 code modified/created + 2 test sets) + 1 README
**Analogs found:** 12 / 12 code files (README = doc-only, no code analog)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/chat_analyzer/cli/nlp_gate.py` (NEW) | utility | request-response (availability probe) | `cli/pipeline.py` thin helpers + `analysis/sentiment.py` `_AVAILABLE` | partial (pattern composite — RESEARCH Pattern 1) |
| `src/chat_analyzer/analysis/emotion.py` (MOD) | analyzer | transform | itself (`analyze_single_message` + `_initialize_model`) | exact (surgical fix in place) |
| `src/chat_analyzer/cli/pipeline.py` (MOD) | controller/orchestrator | pipeline/transform | itself (`stage_status`, `_safe_chart`, `run_pipeline`) | exact (extend in place) |
| `src/chat_analyzer/cli/adapters.py` (MOD) | adapter/transform | transform (dict → contract) | itself (`adapt` defensive `.get()` pattern) | exact (add extractor functions) |
| `src/chat_analyzer/cli/contracts.py` (MOD) | contract/model | contract | itself (AnalysisResults TypedDict) | exact (extend keys) |
| `src/chat_analyzer/cli/report_html.py` (MOD) | component/renderer | file-I/O | itself (TEMPLATE + `write_report`) | exact (+4 tabs, cwd location) |
| `src/chat_analyzer/cli/main.py` (MOD) | controller (CLI entry) | request-response (interactive) | itself (typer app + re-prompt loop) | exact (menu branch) |
| `src/chat_analyzer/analysis/relationship_health.py` (MOD) | analyzer | transform | `utils/visualization.py:18-20` (NullHandler) | role-match (logging fix) |
| `src/chat_analyzer/analysis/network_graph.py` (MOD) | analyzer + chart | transform + file-I/O (figure) | `utils/visualization.py` `plot_relationship_health_trend` (return fig) | role-match (wrapper) |
| `tests/test_analysis.py` (MOD) | test | test | `tests/test_phase2_pipeline.py` | role-match (rewire target) |
| `tests/test_phase4_*.py` (NEW) | test | test | `tests/test_phase2_pipeline.py` + `test_phase2_report.py` + `test_phase2_cli.py` | role-match |
| `README.md` (MOD) | doc | doc | no code analog — D-18/D-19 content spec | n/a |
| `.planning/REQUIREMENTS.md` (MOD, non-code) | doc | doc | no analog — traceability reconciliation (D-07/D-08) | n/a |

**Files NOT modified:** `pyproject.toml` (torch/transformers already in `[nlp]` extra, pyproject.toml:27 — CPU-only torch is a runtime `--index-url` path, not a pyproject change); `analysis/summarizer.py` (constructor-guarded, no code change per RESEARCH).

---

## Pattern Assignments

### `src/chat_analyzer/cli/nlp_gate.py` (NEW — utility, request-response)

**Analog:** `src/chat_analyzer/cli/pipeline.py` module-level thin helpers + `src/chat_analyzer/analysis/sentiment.py` optional-dep gate. No single exact analog exists — compose the two.

**Module shape** (follow `pipeline.py` lines 1-27: future-annotations import, stdlib first, module logger):
```python
# pipeline.py:16-27
from __future__ import annotations

import base64
import contextlib
import io
import logging
from pathlib import Path

from chat_analyzer.cli.contracts import AnalysisResults, ParseReport
from chat_analyzer.ingest.ingestion import messages_to_dataframe

logger = logging.getLogger(__name__)
```

**Availability check** (RESEARCH.md Pattern 1, lines 112-127 — reuses the codebase `try/except ImportError` gate from `sentiment.py:24-29`):
```python
# sentiment.py:24-29 (the _AVAILABLE gate convention — CONVENTIONS.md:65-74)
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Install with: pip install transformers torch")
```
```python
# RESEARCH.md Pattern 1 (lines 113-127) — nlp_gate.py target shape
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
Note: RESEARCH.md A3 — `huggingface_hub.constants.HF_HUB_CACHE` may move; fall back to raw `~/.cache/huggingface/hub`.

**Guarded runtime installer** (RESEARCH.md Don't-Hand-Roll row 3, lines 233-243 — never `os.system`, no `shell=True`):
```python
import subprocess, sys
cmd = [sys.executable, "-m", "pip", "install",
       "torch", "transformers>=4.30,<6",
       "--index-url", "https://download.pytorch.org/whl/cpu"]
proc = subprocess.run(cmd, capture_output=True, text=True)
if proc.returncode != 0:
    raise RuntimeError("Model install failed — run basic analysis, or install: pip install chat-analyzer-pro[ nlp]")
```
**Model constant:** `bhadresh-savani/distilbert-base-uncased-emotion` (CONTEXT D-07c — locked; ignore RESEARCH.md's typo'd spellings). Must announce name + size (~255 MB) BEFORE `from_pretrained` (Pitfall 4). Silent-degrade contract: no exceptions escape to the user for offline/no-pip — caller falls back to basic analysis + one hint line (D-06).

---

### `src/chat_analyzer/analysis/emotion.py` (MOD — surgical fix + model swap)

**Analog:** itself. Two surgical edits, no rewrite.

**Fix 1 — `analyze_single_message` parse bug (RESEARCH Pitfall 1, CRITICAL).** Current broken code at `emotion.py:103-123`:
```python
# emotion.py:106 (BROKEN — [0] indexes the flat list; iterating dict keys → TypeError → neutral fallback)
result = self.pipeline(text[:512])[0]  # Limit to 512 chars for efficiency
emotion_scores = {item['label']: item['score'] for item in result}
```
Replace with (RESEARCH.md lines 204-219 — drop `[0]`; `top_k=None` in transformers 4.40 returns a flat list of dicts):
```python
# RESEARCH.md Code Examples (surgical fix)
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

**Fix 2 — default model swap (Pitfall 2).** `emotion.py:39`:
```python
def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):  # 7-class — wrong
```
→ default `model_name: str = "bhadresh-savani/distilbert-base-uncased-emotion"` (6-class, exact label match to `self.emotions` at line 48: joy/sadness/anger/fear/surprise/love).

**Fix 3 (optional, RESEARCH Pitfall 3):** scope `warnings.filterwarnings('ignore')` at `emotion.py:26` — match the canonical `logger = getLogger(__name__)` + NullHandler shape (see relationship_health section).

**Keep:** module-level singletons `emotion.py:29-30`:
```python
_emotion_analyzer = None
_emotion_model_loaded = False
```
**Keep:** lazy `from transformers import pipeline` inside `_initialize_model` (`emotion.py:60-79`), `top_k=None`, `device=-1`. **Keep:** `_get_neutral_emotions` (line 125-127) as the degrade path.

---

### `src/chat_analyzer/cli/pipeline.py` (MOD — call nlp gate + extend AnalysisResults)

**Analog:** itself. Extend `run_pipeline` (lines 64-157), reusing the existing stage/guard helpers verbatim.

**Stage narration helper to reuse** (`pipeline.py:50-61` — keep NLP announce/hint inside this guarded context per Pitfall 8):
```python
def stage_status(console, label: str):
    if console.is_terminal:
        return console.status(label, spinner="line")
    console.print(f"[OK] {label}...")
    return contextlib.nullcontext()
```

**Chart-degrade + base64 helpers to reuse for new NLP charts** (`pipeline.py:30-47`):
```python
def fig_to_data_uri(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)  # never leak figures between runs
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")

def _safe_chart(fig) -> str:
    try:
        return fig_to_data_uri(fig)
    except Exception:
        logger.exception("chart encoding failed; substituting empty string")
        return ""
```

**Lazy heavy imports + stdout capture** (extend the existing block `pipeline.py:109-151` — this is where the gate call + emotion/summary stages slot in):
```python
with stage_status(console, "Computing insights"):
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        from chat_analyzer.analysis import sentiment as _sentiment
        _sentiment.TRANSFORMERS_AVAILABLE = False   # pin VADER path (keep)

        from chat_analyzer.analysis.eda import ChatEDA
        eda = ChatEDA(df)
        ...
        charts = { ... "sentiment": _safe_chart(...) }
    if captured.getvalue():
        logger.debug("Captured analysis-stage output:\n%s", captured.getvalue())
```
The **availability gate must run BEFORE any heavy import** (D-02 silent check; never prompt). Then `adapt(...)` call at `pipeline.py:153-156` gains the new extracted args. `matplotlib.use("Agg")` first at `pipeline.py:66-68` — keep; new figure wrappers must not call `plt.show()`.

---

### `src/chat_analyzer/cli/adapters.py` (MOD — new extractors)

**Analog:** itself. The module docstring (lines 1-8) states the contract: "the ONLY place that knows each module's internal dict shape... Every access is a defensive `.get()`: an empty edge-case dict must never KeyError here."

**Signature pattern to extend** (`adapters.py:19-29`):
```python
def adapt(
    source,
    parse: ParseReport,
    df,
    summary,
    volume,
    dynamics,
    content,
    sentiment,
    charts,
) -> AnalysisResults:
```
New params join here (emotion_summary, health_summary, network_summary, summary_text, new chart URIs). Build new dict blocks following the existing defensive style (`adapters.py:89-100`):
```python
sent_dist = sentiment.get("sentiment_distribution") or {}
vader = (sentiment.get("average_scores") or {}).get("vader_compound") or {}
sentiment_block: dict = {
    "distribution": {str(k): int(v) for k, v in sent_dist.items()},
    "avg_compound": vader.get("mean"),
    ...
}
```

**Non-serializable extraction rule (RESEARCH Pattern 3, lines 135-136):** `analyze_relationship_health(df)` returns dicts containing a **DataFrame** (`'prepared_data'`, `relationship_health.py:1113`) and `analyze_network` returns a **networkx.DiGraph** (`'graph'`, `network_graph.py:500`). Extract ONLY serializable scalars — `health_score['overall_health_score']`, `health_score['grade']`, `metrics.*`, `key_participants`, `patterns.strongest_connections`, `subgroups` — never leak DataFrame/Graph into `AnalysisResults` (Jinja-consumed).

**Narrative lead-ins** — extend `build_insights` (`adapters.py:120-165`), same rule: values from stats, never the string `"None"`; capped `return insights[:7]` → raise cap to ~11 for 4 new tabs.

**Import style** (`adapters.py:10-16`):
```python
from __future__ import annotations
from collections import Counter
import pandas as pd
from chat_analyzer.cli.contracts import AnalysisResults, ParseReport
```

---

### `src/chat_analyzer/cli/contracts.py` (MOD — extend AnalysisResults)

**Analog:** itself. Extend the TypedDict (`contracts.py:24-39`) — pattern:
```python
class AnalysisResults(TypedDict):
    source: str
    parse: dict[str, int]
    stats: dict[str, Any]
    participants: dict[str, Any]
    content: dict[str, Any]
    sentiment: dict[str, Any]
    charts: dict[str, str]
    insights: list[str]
    report_path: str
```
Add 4 keys: `emotion: dict[str, Any]`, `health: dict[str, Any]`, `network: dict[str, Any]`, `summary: dict[str, Any]` (names at planner discretion). Keep `from dataclasses import dataclass` + `ParseReport` (lines 9-21) untouched. Docstring updates per file convention (lines 24-29).

---

### `src/chat_analyzer/cli/report_html.py` (MOD — +4 tabs, cwd location)

**Analog:** itself. Two edits:

**Edit 1 — report location (D-09: cwd instead of next-to-input).** Current line 191:
```python
report_path = input_path.parent / f"{stem}_report.html"
```
→ `report_path = Path.cwd() / f"{stem}_report.html"`. Docstring line 8 ("next to the input") and `write_report` docstring line 163 must be updated too. `test_phase2_cli.py:112-127` asserts the old location — planner must reconcile that test (report lands in cwd = tmp_path when CLI runs there).

**Edit 2 — 4 new tab panels.** Extend `TEMPLATE` (lines 28-148) exactly per the existing pattern: nav buttons (lines 62-68), panels with `<p class="lead">` narrative (e.g. lines 70-82), `{% if charts.X %}<img class="chart" src="{{ charts.X }}">{% endif %}` for charts, `<table>` for scalar data. New tab ids: `emotion`, `health`, `summary`, `network`. The `showTab` JS (lines 134-145) is id-agnostic — no change. **Keep autoescape + chart-URI whitelist boundary** (lines 164-169):
```python
charts = {
    name: (uri if uri.startswith(_CHART_PREFIX) else "")
    for name, uri in results["charts"].items()
}
env = Environment(autoescape=select_autoescape(["html", "xml"]))
```
**Keep** `sanitize_filename` (151-159), utf-8 write (192), `open_report` degrade (196-202).

---

### `src/chat_analyzer/cli/main.py` (MOD — interactive NLP menu branch)

**Analog:** itself. The typer app + re-prompt loop is the insertion point.

**Friendly-error discipline to extend (CLI-04, D-13/D-14)** — existing `main.py:81-98`:
```python
if chat_file is not None:
    if not chat_file.is_file():
        typer.echo(f"File not found: {chat_file}", err=True)
        raise typer.Exit(code=1)
    ...
    try:
        _analyze_path(chat_file)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    raise typer.Exit(code=0)
```
Each failure type gains a distinct message + inline export steps (WhatsApp: Settings→Chats→Export chat; Telegram: desktop-app export). Exit code stays 1.

**Interactive re-prompt loop** (`main.py:101-118`) — the 3-option NLP menu (D-04) slots in **after** the file path validates, only when the availability check reports NLP missing. Menu via rich `prompt`/`Console` (rich already a dependency; `Console()` instantiated in `_analyze_path` at line 38-42):
```python
while True:
    path = Path(typer.prompt("Enter path to chat export").strip().strip('"').strip("'"))
    ...
    try:
        _analyze_path(path)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        continue
    raise typer.Exit(code=0)
```
Options per D-04: 1) full torch ~3GB, 2) CPU-only torch + model ~0.6GB (recommended default), 3) no download → basic. Menu only when interactive AND NLP missing (D-05, Pitfall 5). Positional runs never prompt (D-06) — the existing positional branch (81-99) stays.

**Keep:** lazy imports inside `_analyze_path` (lines 36-43, Anti-Pattern 2), Windows utf-8 reconfigure (73-79), `--version` eager callback (28-33).

---

### `src/chat_analyzer/analysis/relationship_health.py` (MOD — neutralize logging.basicConfig)

**Analog:** `src/chat_analyzer/utils/visualization.py:18-20` (the canonical NullHandler shape — RESEARCH Pitfall 3).

**Current landmine** (`relationship_health.py:23-25`):
```python
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```
**Fix to** (`visualization.py:18-20` — exact copy target):
```python
# Configure logging (Anti-Pattern 4: never hijack global log config at import)
logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)
```
**Do NOT touch** `analyze_relationship_health` (`relationship_health.py:1071-1128`) — verified pandas-only, no torch. Its return-dict shape (`health_score['overall_health_score']`, `health_score['grade']`, `prepared_data` DataFrame) is consumed by the new adapter (Pattern 3 extraction). Keep `import logging` at line 21.

---

### `src/chat_analyzer/analysis/network_graph.py` (MOD — figure-returning wrapper)

**Analog:** `ChatVisualizer.plot_relationship_health_trend` (`visualization.py:502-556`) — the figure-returning convention to match (`return fig`, no `plt.show()`). Current `plot_network_dashboard` (`network_graph.py:362-470`) and `plot_network_graph` (`266-359`) both end in `plt.show()` and return `None` — cannot base64-embed (Pitfall 6).

**Add a thin wrapper** (RESEARCH Pattern 2, lines 245-256 — build Axes, no `show()`, `return fig`):
```python
def network_figure(df) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt
    res = analyze_network(df)
    G, metrics, patterns = res["graph"], res["metrics"], res["patterns"]
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos, ax=ax)
    ax.set_title("Conversation Network")
    return fig    # NO plt.show() -> base64 via pipeline.fig_to_data_uri
```
Wrapper lives in `network_graph.py` (or `adapters.py` at planner discretion) — reuse `build_interaction_network`/`analyze_network` (lines 23-60, 473-505), do not rewrite analysis. Same treatment for an emotion figure (reuse `emotion.py` `plot_emotion_analysis` chart parts minus `plt.show()` at line 447, or `ChatVisualizer` bar/pie methods). All wrappers must run under the `matplotlib.use("Agg")` set in `run_pipeline` (pipeline.py:66-68).

---

### `tests/test_analysis.py` (MOD — rewire to real modules)

**Analog:** `tests/test_phase2_pipeline.py` — the pytest-style real-module pattern (D-16). Current `test_analysis.py` duplicates logic (e.g., lines 91-110 mock scores by hand instead of calling real functions). Rewire each `Test*` class to import and call real `chat_analyzer.*` modules on small fixture DataFrames. Reference the fixture-DataFrame pattern from `test_phase2_pipeline.py:97-113`:
```python
df = messages_to_dataframe(
    [{"datetime": "2025-09-15T09:45:00", "sender": "A", "message": "just me here"}]
)
eda = ChatEDA(df)
summary = eda.generate_comprehensive_summary()
...
results = adapt("whatsapp", ParseReport(source="whatsapp", parsed_messages=1),
                df, summary, volume, dynamics, content, sent, {})
joined = "\n".join(results["insights"])
assert "None" not in joined
```
Keep the `_AVAILABLE` pin convention where a module may lazily load HF (`test_phase2_pipeline.py:85,95`):
```python
from chat_analyzer.analysis import sentiment as _sentiment
_sentiment.TRANSFORMERS_AVAILABLE = False
```
Keep the unittest framework per D-16 (test_analysis.py currently unittest-based, lines 6-9) — but note RESEARCH Open Question 3: new `test_phase4_*` files may be pytest-style like Phase 2. Do not repurpose `run_analysis_tests()` aggregators (lines 381-397) — rewire in place.

---

### `tests/test_phase4_*.py` (NEW — pipeline-with-mocked-NLP, CLI-04 errors, report tabs)

**Analog composite:** `test_phase2_pipeline.py` (in-process pipeline), `test_phase2_report.py` (crafted AnalysisResults dict), `test_phase2_cli.py` (subprocess + BROWSER=__none__).

**Faithful model mock (D-17, RESEARCH lines 220-230) — patch `transformers.pipeline`, NOT the analyzer:**
```python
fake = [{"label": "joy", "score": 0.87}, {"label": "sadness", "score": 0.03},
        {"label": "anger", "score": 0.03}, {"label": "fear", "score": 0.02},
        {"label": "surprise", "score": 0.02}, {"label": "love", "score": 0.03}]
with unittest.mock.patch("chat_analyzer.analysis.emotion.transformers.pipeline",
                         return_value=fake):
    analyzer = EmotionAnalyzer()
    out = analyzer.analyze_emotions(df)
assert not (out["emotion_joy"] == out["emotion_sadness"]).all()   # non-uniform scores
```
Also mock the availability gate + installer (patch `chat_analyzer.cli.nlp_gate.nlp_available`/`model_cached`) so suites run fast and offline-safe WITHOUT `[nlp]` installed (D-17). New `test_phase4_cli.py` subprocess tests copy the `_run` helper (`test_phase2_cli.py:47-64` — `env["BROWSER"] = "__none__"`, utf-8, `timeout=300`) and the `_copy_sample` helper (67-70).

**Report-card assertion pattern to extend** (`test_phase2_report.py:21-70` crafted `_results()` dict + `test_tabs_and_insights` at 168-172 — add the 4 new tab ids; `test_phase2_cli.py:130-140` count `data:image/png;base64,` ≥ 4 → now ≥ 8).

---

## Shared Patterns

### Lazy heavy imports / optional-dep gates (D-02, CONVENTIONS.md:65-74)
**Source:** `src/chat_analyzer/analysis/sentiment.py:24-29` (the `_AVAILABLE` flag convention); `src/chat_analyzer/analysis/emotion.py:60-79` (lazy `from transformers import pipeline` inside a guarded `_initialize_*`).
**Apply to:** `nlp_gate.py` (gate), `pipeline.py` (emotion/summary stages), new adapters. Heavy model construction happens ONLY after the gate passes, inside try/except that degrades (Pitfall 7: `ConversationSummarizer` ctor downloads t5-small 231 MB — construct only after announce, degrade to "summary unavailable").

### Module-level singletons (CONVENTIONS.md:24)
**Source:** `emotion.py:29-30` (`_emotion_analyzer = None`, `_emotion_model_loaded = False`); `sentiment.py:39-40`.
**Apply to:** emotion path unchanged; any new lazy-loaded summary/emotion analyzer object in pipeline must reuse these singletons — never a second load path (Don't Hand-Roll row 1).

### Chart → base64 PNG (D-11)
**Source:** `pipeline.py:30-38` (`fig_to_data_uri`) + `pipeline.py:41-47` (`_safe_chart` degrade).
**Apply to:** all 4 new report charts. Figures must come from figure-returning methods (`visualization.py` returns `fig` everywhere; new wrappers for `network_graph.py` and `emotion.py` plots).

### Error handling: friendly message + exit 1, never traceback (CLI-04)
**Source:** `main.py:81-99` (positional: `typer.echo(str(exc), err=True)`; `raise typer.Exit(code=1) from None`) + `main.py:101-118` (interactive: echo + `continue`).
**Apply to:** all new failure paths — missing file, wrong format, empty chat, unparseable lines each get a distinct message + inline WhatsApp/Telegram export steps (D-14). The nlp_gate installer failure degrades to basic + hint, never a frozen terminal (Pitfall 4).

### Progress narration (D-12, Pitfall 8)
**Source:** `pipeline.py:50-61` (`stage_status`: rich Status on tty, plain `[OK] <label>` otherwise).
**Apply to:** every new long-running stage (model download announce, emotion, summary) — announce + narration must reach piped/CI output too.

### Jinja2 autoescape + chart-URI whitelist (XSS boundary)
**Source:** `report_html.py:164-172` (`_CHART_PREFIX` filter + `Environment(autoescape=select_autoescape(["html", "xml"]))`).
**Apply to:** all 4 new tabs. Never `|safe` chat-derived content (CONCERNS.md); only internally-generated `data:image/png;base64,` URIs reach the template.

### Logging: NullHandler, never `logging.basicConfig` at import (Pitfall 3)
**Source:** `visualization.py:18-20`.
**Apply to:** `relationship_health.py:24` (fix), `emotion.py` / `summarizer.py` `warnings.filterwarnings('ignore')` scoping (optional).

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `README.md` | doc | doc | 3-line stub (README.md:1-3); rewrite per CONTEXT D-18/D-19 content spec (quickstart-first: one-liner → export steps → install → single command → NLP download question; neutral 3-option presentation) |
| `.planning/REQUIREMENTS.md` | doc | doc | Traceability reconciliation only: ANAL-07/ANAL-09 → always-on (D-07/D-07b), OUT-04/OUT-05 → no-flag (D-08) |

## Metadata

**Analog search scope:** `src/chat_analyzer/cli/` (pipeline, adapters, contracts, report_html, render, main, `__init__`), `src/chat_analyzer/analysis/` (emotion, sentiment, summarizer, relationship_health, network_graph, eda, `__init__`), `src/chat_analyzer/utils/visualization.py`, `tests/` (all 11 files), `pyproject.toml`, `README.md`, `.planning/codebase/CONVENTIONS.md`
**Files scanned:** 26
**Pattern extraction date:** 2026-08-03

### Planner reconciliation notes
1. **`report_html.py:191` location change** (input parent → cwd) breaks `test_phase2_cli.py:112-127` ("report next to input") and `test_phase2_report.py:122-125` — update those assertions in Phase 4.
2. **`adapters.adapt` signature grows** — the direct-call tests at `test_phase2_pipeline.py:109-113` must pass the new args (defaults acceptable).
3. **`analysis/__init__.py` has a stale broken re-export** (CONVENTIONS.md:165) — new code should import from real modules (`chat_analyzer.analysis.emotion` etc.), not via `analysis/__init__` re-exports.
4. **Model name typo variance in RESEARCH.md** (`bhadresh-savUses`, `bilibili-savati`) — the locked name is CONTEXT D-07c: `bhadresh-savani/distilbert-base-uncased-emotion`.
5. **`test_phase2_pipeline.py:42` asserts `set(results["charts"]) == {timeline, activity, participants, sentiment}`** — extend for the 4 new chart keys.
