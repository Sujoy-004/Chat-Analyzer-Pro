"""Phase 4 NLP-gate tests (ANAL-06/ANAL-08, D-02/D-06/D-07c/D-17).

Exercises the REAL chat_analyzer.cli.pipeline.run_pipeline + adapters with
the heavy model callables mocked (D-17 — faithful list-of-dicts shape from
RESEARCH Pitfall 1). The gate is forced both ways so the suite is fast and
offline-safe without the [nlp] extra (RESEARCH Pitfall 5: the dev machine
has transformers but no cached emotion model, so tests must force branches).

Test A (gate ON): the real EmotionAnalyzer consumes the flat list-of-dicts
that transformers 4.40's pipeline returns with top_k=None and produces
NON-uniform scores — the regression trap for the old `[0]` parse bug
(Pitfall 1). The real ConversationSummarizer produces a summary text. The
charts gain an "emotion" base64 URI.

Test B (gate OFF): silent basic run — emotion/summary are None, no error,
no prompt (D-02/D-06).

Test C (report): the HTML report carries id="tab-emotion" and
id="tab-summary" with real content when the gate is ON and the
pip-install unavailable note when it is OFF.
"""

import contextlib
import io
import shutil
import unittest.mock
from pathlib import Path

import pytest
from rich.console import Console

from chat_analyzer.cli.pipeline import run_pipeline

try:
    import transformers
except ImportError:  # [nlp] extra not installed — gate-ON tests skip (D-17)
    transformers = None

DATA = Path(__file__).resolve().parents[1] / "data" / "sample_chats"

# Faithful 6-label shape (D-17 / RESEARCH Pitfall 1): exactly what
# transformers 4.40's text-classification pipeline returns with top_k=None.
FAITHFUL_SCORES = [
    {"label": "joy", "score": 0.87},
    {"label": "sadness", "score": 0.03},
    {"label": "anger", "score": 0.03},
    {"label": "fear", "score": 0.02},
    {"label": "surprise", "score": 0.02},
    {"label": "love", "score": 0.03},
]


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False)


def _fake_emotion_classifier(text):
    """Content-varied classifier: same faithful shape, dominant label varies
    per message so the distribution is non-uniform (the buggy [0] parse
    yields uniform 1/6 scores for every message and cannot pass Test A)."""
    lowered = str(text).lower()
    if "❤️" in lowered or "love" in lowered:
        return [
            {"label": "love", "score": 0.8},
            {"label": "joy", "score": 0.1},
            {"label": "sadness", "score": 0.02},
            {"label": "anger", "score": 0.02},
            {"label": "fear", "score": 0.02},
            {"label": "surprise", "score": 0.04},
        ]
    if "sorry" in lowered or "sad" in lowered or "miss" in lowered:
        return [
            {"label": "sadness", "score": 0.8},
            {"label": "joy", "score": 0.1},
            {"label": "anger", "score": 0.02},
            {"label": "fear", "score": 0.03},
            {"label": "surprise", "score": 0.02},
            {"label": "love", "score": 0.03},
        ]
    return FAITHFUL_SCORES


def _fake_summarizer(text, **kwargs):
    """T5 summarizer callable mock — returns one summary text."""
    return [{"summary_text": "A test summary."}]


class _FakeT5Model:
    """Stand-in for T5ForConditionalGeneration mocking the direct
    generate() path the summarizer uses (no pipeline() abstraction)."""

    def generate(self, **kwargs):
        return [[0]]


class _FakeT5Tokenizer:
    """Stand-in for T5Tokenizer — callable and decodable, mirroring how
    ConversationSummarizer builds inputs and decodes outputs."""

    def __call__(self, text, **kwargs):
        return {"input_ids": [[0]]}

    def decode(self, output, skip_special_tokens=True):
        return "A test summary."


def _fake_pipeline_factory(task, *args, **kwargs):
    """transformers.pipeline factory mock: task-aware so the emotion
    classifier and the T5 summarizer each get a fitting callable."""
    if task == "summarization":
        return _fake_summarizer
    return _fake_emotion_classifier


@contextlib.contextmanager
def _mocked_nlp(gate_on: bool = True):
    """Force the gate and mock the heavy model callables (D-17).

    Patching the REAL transformers.pipeline attribute is what emotion.py and
    summarizer.py import inside their lazy initializers, so the real
    EmotionAnalyzer/ConversationSummarizer logic runs end-to-end. The gate
    attribute is patched on the nlp_gate module so run_pipeline sees it.
    """
    from chat_analyzer.cli import nlp_gate

    if transformers is None and gate_on:
        pytest.skip("transformers not installed — model-load mock needs the [nlp] extra (D-17)")

    with unittest.mock.patch.object(nlp_gate, "nlp_available", lambda *a, **k: gate_on):
        if not gate_on:
            yield
            return
        with (
            unittest.mock.patch("transformers.pipeline", side_effect=_fake_pipeline_factory),
            unittest.mock.patch.object(
                transformers.T5Tokenizer, "from_pretrained", return_value=_FakeT5Tokenizer()
            ),
            unittest.mock.patch.object(
                transformers.T5ForConditionalGeneration, "from_pretrained", return_value=_FakeT5Model()
            ),
        ):
            yield


@pytest.fixture(autouse=True)
def _reset_emotion_singletons():
    """Each test gets a fresh EmotionAnalyzer load — the module-level
    singleton would otherwise short-circuit _initialize_model and reuse a
    stale pipeline from a previous test's patch context."""
    import chat_analyzer.analysis.emotion as emotion_mod

    emotion_mod._emotion_analyzer = None
    emotion_mod._emotion_model_loaded = False
    yield
    emotion_mod._emotion_analyzer = None
    emotion_mod._emotion_model_loaded = False


def test_emotion_summary_with_mocked_nlp():
    """Test A (gate ON): emotion + summary render from the real modules."""
    with _mocked_nlp(gate_on=True):
        results = run_pipeline(DATA / "whatsapp_sample.txt", _console())

    emotion = results["emotion"]
    assert emotion is not None
    dist = emotion["distribution"]
    assert dist, "emotion distribution must be present"
    assert len(set(dist.values())) >= 2, "emotion scores must not be uniform"
    assert emotion["dominant"] in dist

    avg = emotion["average_scores"]
    assert len(set(avg.values())) >= 2, "average emotion scores must not be uniform"

    summary = results["summary"]
    assert summary is not None
    assert summary["text"] and summary["text"].strip()

    assert "emotion" in results["charts"]
    assert results["charts"]["emotion"].startswith("data:image/png;base64,")


def test_basic_run_without_nlp():
    """Test B (gate OFF): silent basic run — emotion/summary None (D-02/D-06)."""
    with _mocked_nlp(gate_on=False):
        results = run_pipeline(DATA / "whatsapp_sample.txt", _console())

    assert results["emotion"] is None
    assert results["summary"] is None
    assert results["stats"]["total_messages"] == 27
    assert results["participants"]
    assert results["sentiment"]["distribution"]


def test_report_contains_emotion_and_summary_tabs(tmp_path, monkeypatch):
    """Test C: the HTML report carries the emotion + summary tabs — real
    content with the gate ON, the pip-install unavailable note OFF."""
    from chat_analyzer.cli.report_html import write_report

    src = tmp_path / "whatsapp_sample.txt"
    shutil.copyfile(DATA / "whatsapp_sample.txt", src)
    monkeypatch.chdir(tmp_path)

    with _mocked_nlp(gate_on=True):
        results = run_pipeline(DATA / "whatsapp_sample.txt", _console())
    report = write_report(results, src)
    html = report.read_text(encoding="utf-8")
    assert 'id="tab-emotion"' in html
    assert 'id="tab-summary"' in html
    assert "A test summary." in html

    with _mocked_nlp(gate_on=False):
        basic = run_pipeline(DATA / "whatsapp_sample.txt", _console())
    basic_report = write_report(basic, src)
    basic_html = basic_report.read_text(encoding="utf-8")
    assert 'id="tab-emotion"' in basic_html
    assert "pip install chat-analyzer-pro[nlp]" in basic_html
