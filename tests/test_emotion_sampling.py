"""Option C — sampled emotion inference for large chats (fast, no real model).

Proves the deterministic stratified-sample path in EmotionAnalyzer and the
pipeline gate that drives it. All model callables are mocked (D-17) exactly
like tests/test_perf_parity_emotion.py: the module-level ``_emotion_analyzer``
singleton and ``_emotion_model_loaded`` flag are patched with a batch-faithful
classifier, so the REAL EmotionAnalyzer logic runs without torch inference.

Covered behavior (plan Option C):

- a. Below/at the cap -> EXACT path: output equals the sequential reference
       and no `emotion_scored` column / attrs are added.
- b. Above the cap -> sampled: `emotion_scored` column exists, exactly
       `cap` rows are scored, everything else is neutral 1/6, and the
       attrs carry correct scored/total/cap.
- c. Determinism: two runs yield identical sampled rows + identical summary.
- d. Sampled-vs-exact tolerance: on a large deterministic fixture the sampled
       summary (means + distribution shares) is within a generous tolerance
       of the exact summary — the sample is representative, not garbage.
- e. Stratification: with heavily imbalanced senders, every sender with > 0
       scorable messages appears in the sample at least once.
- f. env parsing: emotion_sample_cap() for absent/empty/0/off/45k/garbage.
- g. pipeline gating: non-tty console auto-samples when over the cap; a tty
       prompts and honors y/N (y -> sampled, N/empty -> exact).
- h. sampling disabled via env (cap None) -> exact even when huge.

These are FAST tests (no pytest.mark.slow) — the pipeline runs reuse the
test_phase4_nlp mocking pattern so no model loads or downloads.
"""

import contextlib
import io
import os
import unittest.mock
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from io import StringIO

# Headless-first (Pitfall 7): importing chat_analyzer.analysis.emotion pulls
# matplotlib; pin Agg BEFORE any pyplot import so figures are headless-safe.
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest
from rich.console import Console

from chat_analyzer.analysis import emotion as _emotion_module
from chat_analyzer.analysis.emotion import EmotionAnalyzer
from chat_analyzer.cli import nlp_gate
from chat_analyzer.cli.pipeline import run_pipeline
from chat_analyzer.ingest.ingestion import messages_to_dataframe

try:
    import transformers
except ImportError:  # [nlp] extra not installed — pipeline mocks skip (D-17)
    transformers = None

EMOTIONS = ("joy", "sadness", "anger", "fear", "surprise", "love")
EMOTION_COLS = [f"emotion_{e}" for e in EMOTIONS]

JOY_SCORES = [
    {"label": "joy", "score": 0.87},
    {"label": "sadness", "score": 0.03},
    {"label": "anger", "score": 0.03},
    {"label": "fear", "score": 0.02},
    {"label": "surprise", "score": 0.02},
    {"label": "love", "score": 0.03},
]
LOVE_SCORES = [
    {"label": "love", "score": 0.8},
    {"label": "joy", "score": 0.1},
    {"label": "sadness", "score": 0.02},
    {"label": "anger", "score": 0.02},
    {"label": "fear", "score": 0.02},
    {"label": "surprise", "score": 0.04},
]
SAD_SCORES = [
    {"label": "sadness", "score": 0.8},
    {"label": "joy", "score": 0.1},
    {"label": "anger", "score": 0.02},
    {"label": "fear", "score": 0.03},
    {"label": "surprise", "score": 0.02},
    {"label": "love", "score": 0.03},
]


def _per_text(text):
    lowered = str(text).lower()
    if "love" in lowered:
        return LOVE_SCORES
    if "sad" in lowered or "sorry" in lowered:
        return SAD_SCORES
    return JOY_SCORES


def _classifier(texts, **kwargs):
    """Batch-faithful: a list input gets a list of per-message results; a bare
    string gets a single result (so analyze_single_message works unchanged)."""
    if isinstance(texts, str):
        return _per_text(texts)
    return [_per_text(t) for t in texts]


def _make_analyzer(classifier=_classifier):
    """Real EmotionAnalyzer with the transformers pipeline mocked (D-17),
    mirroring tests/test_perf_parity_emotion.py: patch the module-level model
    cache so _initialize_model short-circuits to the fake pipeline."""
    with (
        redirect_stdout(StringIO()),
        unittest.mock.patch.object(_emotion_module, "_emotion_analyzer", classifier),
        unittest.mock.patch.object(_emotion_module, "_emotion_model_loaded", True),
    ):
        return EmotionAnalyzer()


def _small_fixture_df():
    """Mix of scorable and skip-ruled messages (2 scorable, 4 skipped)."""
    return messages_to_dataframe([
        {"datetime": "2024-01-01T09:00:00", "sender": "Alice", "message": "I love this!"},
        {"datetime": "2024-01-01T09:01:00", "sender": "Bob", "message": ""},
        {"datetime": "2024-01-01T09:02:00", "sender": "Alice", "message": "   "},
        {"datetime": "2024-01-01T09:03:00", "sender": "Bob", "message": "<Media omitted>"},
        {"datetime": "2024-01-01T09:04:00", "sender": "Alice", "message": "Hi"},
        {"datetime": "2024-01-01T09:05:00", "sender": "Bob", "message": "just checking in"},
    ])


def _big_fixture_df(n=240):
    """Deterministic large frame: alternating senders, monotonically
    increasing timestamps, and a stable 1:2:1 love/joy/sad content mix that
    any proportional (sender, time) sample preserves."""
    rows = []
    base = datetime(2024, 1, 1, 9, 0, 0)  # noqa: DTZ001 - naive fixture base; no tz semantics needed
    for i in range(n):
        sender = "Alice" if i % 2 == 0 else "Bob"
        kind = i % 4
        if kind == 0:
            msg = f"I love this message number {i}"
        elif kind == 1:
            msg = f"what a great day it is, message {i}"
        elif kind == 2:
            msg = f"I am sorry about that, message {i}"
        else:
            msg = f"just a regular check-in message {i}"
        rows.append({
            "datetime": base + timedelta(minutes=i),
            "sender": sender,
            "message": msg,
        })
    return messages_to_dataframe(rows)


def _sequential_reference(analyzer, df):
    """The pre-refactor per-message algorithm, driven through the real public
    analyze_single_message API so the skip rules and scoring are bit-identical."""
    seq = df.copy()
    for col in EMOTION_COLS:
        seq[col] = 0.0
    for idx, row in df.iterrows():
        scores = analyzer.analyze_single_message(row["message"])
        for emotion, score in scores.items():
            seq.at[idx, f"emotion_{emotion}"] = score
    seq["dominant_emotion"] = seq[EMOTION_COLS].idxmax(axis=1).str.replace("emotion_", "")
    seq["emotion_confidence"] = seq[EMOTION_COLS].max(axis=1)
    return seq


# --- a. below/at cap -> exact ---------------------------------------------


def test_below_and_at_cap_stay_exact():
    analyzer = _make_analyzer()
    df = _small_fixture_df()
    with redirect_stdout(StringIO()):
        at_cap = analyzer.analyze_emotions(df, sample_cap=2)
        below_cap = analyzer.analyze_emotions(df, sample_cap=100)

    expected = _sequential_reference(analyzer, df)
    cols = EMOTION_COLS + ["dominant_emotion", "emotion_confidence"]
    import pandas as pd

    pd.testing.assert_frame_equal(at_cap[cols], expected[cols])
    pd.testing.assert_frame_equal(below_cap[cols], expected[cols])
    assert "emotion_scored" not in at_cap.columns
    assert "emotion_scored" not in below_cap.columns
    assert "emotion_sample" not in at_cap.attrs
    assert "emotion_sample" not in below_cap.attrs


# --- b. above cap -> sampled ----------------------------------------------


def test_above_cap_scores_exactly_cap_rows():
    analyzer = _make_analyzer()
    df = _big_fixture_df(240)
    cap = 50
    with redirect_stdout(StringIO()):
        out = analyzer.analyze_emotions(df, sample_cap=cap)

    assert "emotion_scored" in out.columns
    scored = out["emotion_scored"]
    assert int(scored.sum()) == cap, "exactly cap rows must be model-scored"

    neutral = out.loc[~scored, EMOTION_COLS]
    assert (neutral == 1 / 6).all(axis=None), "unselected rows keep neutral 1/6"

    attrs = out.attrs["emotion_sample"]
    assert attrs["scored"] == cap
    assert attrs["total"] == len(df)
    assert attrs["cap"] == cap
    assert attrs["sampled"] is True

    # dominant_emotion / emotion_confidence still computed over the full frame
    assert out["dominant_emotion"].notna().all()
    assert out["emotion_confidence"].notna().all()


# --- c. determinism --------------------------------------------------------


def test_deterministic_sample_and_summary():
    analyzer = _make_analyzer()
    df = _big_fixture_df(240)
    with redirect_stdout(StringIO()):
        out1 = analyzer.analyze_emotions(df, sample_cap=50)
        out2 = analyzer.analyze_emotions(df, sample_cap=50)

    assert out1["emotion_scored"].to_numpy().tolist() == (
        out2["emotion_scored"].to_numpy().tolist()
    ), "two runs must pick the same sample rows (random_state=42)"
    s1 = analyzer.get_emotion_summary(out1[out1["emotion_scored"]])
    s2 = analyzer.get_emotion_summary(out2[out2["emotion_scored"]])
    assert s1 == s2, "two runs must produce the identical sampled summary"


# --- d. sampled-vs-exact tolerance ----------------------------------------


def test_sampled_summary_close_to_exact():
    analyzer = _make_analyzer()
    df = _big_fixture_df(240)
    with redirect_stdout(StringIO()):
        exact = analyzer.analyze_emotions(df, sample_cap=None)
        sampled = analyzer.analyze_emotions(df, sample_cap=60)

    exact_sum = analyzer.get_emotion_summary(exact)
    sampled_sum = analyzer.get_emotion_summary(sampled[sampled["emotion_scored"]])

    for emotion in EMOTIONS:
        diff = abs(
            exact_sum["average_emotion_scores"][emotion]
            - sampled_sum["average_emotion_scores"][emotion]
        )
        assert diff < 0.15, f"emotion {emotion} mean abs diff {diff:.3f} >= 0.15"

    exact_dist = exact_sum["emotion_distribution"]
    sampled_dist = sampled_sum["emotion_distribution"]
    exact_total = max(sum(exact_dist.values()), 1)
    sampled_total = max(sum(sampled_dist.values()), 1)
    for emotion in EMOTIONS:
        share_diff = abs(
            exact_dist.get(emotion, 0) / exact_total
            - sampled_dist.get(emotion, 0) / sampled_total
        )
        assert share_diff < 0.15, (
            f"emotion {emotion} distribution share diff {share_diff:.3f} >= 0.15"
        )


# --- e. stratification: every sender represented ----------------------------


def test_imbalanced_senders_all_represented():
    rows = []
    base = datetime(2024, 1, 1, 9, 0, 0)  # noqa: DTZ001 - naive fixture base; no tz semantics needed
    for i in range(100):
        sender = "Alice" if i < 90 else "Bob"
        rows.append({
            "datetime": base + timedelta(minutes=i),
            "sender": sender,
            "message": f"message number {i} with enough words",
        })
    df = messages_to_dataframe(rows)
    analyzer = _make_analyzer()
    with redirect_stdout(StringIO()):
        out = analyzer.analyze_emotions(df, sample_cap=10)

    sampled_senders = set(out.loc[out["emotion_scored"], "sender"])
    assert {"Alice", "Bob"} <= sampled_senders, (
        "every sender with > 0 scorable messages must appear in the sample"
    )
    assert int(out["emotion_scored"].sum()) == 10


# --- f. env parsing --------------------------------------------------------


def test_emotion_sample_cap_env_parsing(monkeypatch):
    env = "CHAT_ANALYZER_EMOTION_SAMPLE"
    monkeypatch.delenv(env, raising=False)
    assert nlp_gate.emotion_sample_cap() == 50000  # absent -> default

    monkeypatch.setenv(env, "")
    assert nlp_gate.emotion_sample_cap() == 50000  # empty -> default

    for off in ("0", "off", "OFF", "false", "FALSE"):
        monkeypatch.setenv(env, off)
        assert nlp_gate.emotion_sample_cap() is None, f"{off!r} must disable"

    monkeypatch.setenv(env, "45000")
    assert nlp_gate.emotion_sample_cap() == 45000

    monkeypatch.setenv(env, "garbage")
    assert nlp_gate.emotion_sample_cap() == 50000  # garbage -> default, no crash

    monkeypatch.setenv(env, "-5")
    assert nlp_gate.emotion_sample_cap() == 50000  # non-positive -> default


# --- pipeline mocks (g/h) --------------------------------------------------


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


@contextlib.contextmanager
def _mocked_models():
    """Patch the heavy model callables (D-17): the emotion singletons plus the
    T5 summarizer's direct from_pretrained path, so run_pipeline runs the real
    modules fast and offline. Mirrors test_phase4_nlp.py::_mocked_nlp."""
    if transformers is None:
        pytest.skip("transformers not installed — model-load mocks need the [nlp] extra (D-17)")
    with (
        unittest.mock.patch.object(_emotion_module, "_emotion_analyzer", _classifier),
        unittest.mock.patch.object(_emotion_module, "_emotion_model_loaded", True),
        unittest.mock.patch("transformers.pipeline", return_value=_classifier),
        unittest.mock.patch.object(
            transformers.T5Tokenizer, "from_pretrained", return_value=_FakeT5Tokenizer()
        ),
        unittest.mock.patch.object(
            transformers.T5ForConditionalGeneration,
            "from_pretrained",
            return_value=_FakeT5Model(),
        ),
    ):
        yield


def _big_whatsapp_file(tmp_path, n=40):
    """Generate a 40-message WhatsApp-style export (all messages scorable)."""
    lines = []
    for i in range(n):
        sender = "Alice" if i % 3 == 0 else "Bob"
        msg = f"I love this one {i}" if i % 4 == 0 else f"regular message {i} here"
        hour = 9 + (i // 60) % 3
        lines.append(f"12/25/23, {hour}:{i % 60:02d} AM - {sender}: {msg}")
    path = tmp_path / "big.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _non_tty_console():
    # force_terminal=False pins is_terminal to False (a read-only rich
    # property), which is what the plan's "monkeypatch console.is_terminal
    # -> False" achieves for piped/CI/test runs.
    return Console(file=io.StringIO(), force_terminal=False)


# --- g. pipeline gating ----------------------------------------------------


def test_pipeline_auto_samples_off_tty(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_ANALYZER_EMOTION_SAMPLE", "10")
    path = _big_whatsapp_file(tmp_path)
    console = _non_tty_console()
    assert console.is_terminal is False

    with _mocked_models():
        results = run_pipeline(path, console, nlp_enabled=True)

    sample = results["emotion"]["sample"]
    assert sample is not None, "non-tty over-cap run must AUTO-SAMPLE"
    assert sample["sampled"] is True
    assert sample["scored"] == 10
    assert sample["total"] == 40
    assert sample["cap"] == 10


def test_pipeline_tty_prompt_honors_choice(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_ANALYZER_EMOTION_SAMPLE", "10")
    path = _big_whatsapp_file(tmp_path)

    # 'y' -> sampled
    console = Console(file=io.StringIO(), force_terminal=True)
    monkeypatch.setattr(console, "input", lambda prompt="": "y")
    with _mocked_models():
        results = run_pipeline(path, console, nlp_enabled=True)
    assert results["emotion"]["sample"] is not None
    assert results["emotion"]["sample"]["sampled"] is True

    # 'N' -> exact (no sample label)
    console = Console(file=io.StringIO(), force_terminal=True)
    monkeypatch.setattr(console, "input", lambda prompt="": "N")
    with _mocked_models():
        results = run_pipeline(path, console, nlp_enabled=True)
    assert results["emotion"]["sample"] is None

    # empty input (default NO) -> exact
    console = Console(file=io.StringIO(), force_terminal=True)
    monkeypatch.setattr(console, "input", lambda prompt="": "")
    with _mocked_models():
        results = run_pipeline(path, console, nlp_enabled=True)
    assert results["emotion"]["sample"] is None


# --- h. sampling disabled via env ------------------------------------------


def test_sampling_disabled_via_env_stays_exact(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_ANALYZER_EMOTION_SAMPLE", "0")
    assert nlp_gate.emotion_sample_cap() is None

    path = _big_whatsapp_file(tmp_path)
    console = _non_tty_console()
    with _mocked_models():
        results = run_pipeline(path, console, nlp_enabled=True)
    assert results["emotion"]["sample"] is None, (
        "cap None (sampling disabled) must stay exact even when huge"
    )

    # analyzer level: sample_cap=None on a large frame -> exact, no extras
    analyzer = _make_analyzer()
    df = _big_fixture_df(240)
    with redirect_stdout(StringIO()):
        out = analyzer.analyze_emotions(df, sample_cap=None)
    assert "emotion_scored" not in out.columns
    assert "emotion_sample" not in out.attrs