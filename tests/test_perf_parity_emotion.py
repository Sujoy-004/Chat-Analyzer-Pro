"""Emotion batch-inference parity tests (pure performance refactor).

Proves `analyze_emotions`' batched path (which now honors `batch_size`)
produces IDENTICAL scores to the old sequential per-message path:

- test_batch_matches_sequential: batched == sequential (real single-message
  public API) on a fixture mixing scorable / empty / whitespace / media /
  short messages.
- test_batch_size_honored: the pipeline is called ONCE PER BATCH (a list +
  batch_size kwarg), never per-message.
- test_rule_based_fallback_identical: with pipeline=None the loop path stays
  identical to today's rule-based scoring.
- test_transformers5_nested_batch: the 4.x-vs-5.x shape normalization must
  also work per-item in a batch.

The mocked classifiers are batch-faithful: they accept a list of texts (plus
the batch_size/top_k kwargs the batched path passes) and return a per-message
list — exactly what a real transformers text-classification pipeline with
top_k=None returns for a list input. They also accept a bare string so the
sequential reference calls exercise `analyze_single_message` unchanged.

These are FAST tests (no pytest.mark.slow).
"""

import os
import unittest.mock
from contextlib import redirect_stdout
from io import StringIO

# Headless-first (Pitfall 7): importing chat_analyzer.analysis.emotion pulls
# matplotlib; pin Agg BEFORE any pyplot import so figures are headless-safe.
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from chat_analyzer.analysis.emotion import EmotionAnalyzer
from chat_analyzer.ingest.ingestion import messages_to_dataframe

EMOTIONS = ('joy', 'sadness', 'anger', 'fear', 'surprise', 'love')
EMOTION_COLS = [f'emotion_{e}' for e in EMOTIONS]

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


def _per_text(text):
    return LOVE_SCORES if "love" in str(text).lower() else JOY_SCORES


def _classifier(texts, **kwargs):
    """Batch-faithful: a list input gets a list of per-message results; a bare
    string gets a single result (so analyze_single_message works unchanged)."""
    if isinstance(texts, str):
        return _per_text(texts)
    return [_per_text(t) for t in texts]


def _nested_classifier(texts, **kwargs):
    """transformers-5.x shape: each per-message result is [[{label,score},...]]."""

    def _nested(text):
        return [list(LOVE_SCORES)] if "love" in str(text).lower() else [list(JOY_SCORES)]

    if isinstance(texts, str):
        return _nested(texts)
    return [_nested(t) for t in texts]


def _make_analyzer(classifier):
    """Real EmotionAnalyzer with the transformers pipeline mocked (D-17),
    mirroring tests/test_analysis.py: patch the module-level model cache so
    _initialize_model short-circuits to the fake pipeline and never touches
    the optional [nlp] dependency."""
    from chat_analyzer.analysis import emotion as _emotion_module

    with (
        redirect_stdout(StringIO()),
        unittest.mock.patch.object(_emotion_module, "_emotion_analyzer", classifier),
        unittest.mock.patch.object(_emotion_module, "_emotion_model_loaded", True),
    ):
        return EmotionAnalyzer()


def _fixture_df() -> pd.DataFrame:
    """Mix of scorable and skip-ruled messages (2 scorable, 4 skipped)."""
    return messages_to_dataframe([
        {"datetime": "2024-01-01T09:00:00", "sender": "Alice", "message": "I love this!"},
        {"datetime": "2024-01-01T09:01:00", "sender": "Bob", "message": ""},
        {"datetime": "2024-01-01T09:02:00", "sender": "Alice", "message": "   "},
        {"datetime": "2024-01-01T09:03:00", "sender": "Bob", "message": "<Media omitted>"},
        {"datetime": "2024-01-01T09:04:00", "sender": "Alice", "message": "Hi"},
        {"datetime": "2024-01-01T09:05:00", "sender": "Bob", "message": "just checking in"},
    ])


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


def test_batch_matches_sequential():
    analyzer = _make_analyzer(_classifier)
    with redirect_stdout(StringIO()):
        batched = analyzer.analyze_emotions(_fixture_df())

    expected = _sequential_reference(analyzer, _fixture_df())
    cols = EMOTION_COLS + ["dominant_emotion", "emotion_confidence"]
    pd.testing.assert_frame_equal(batched[cols], expected[cols])


def test_batch_size_honored():
    calls = []

    def spy(texts, **kwargs):
        calls.append((texts, kwargs))
        return _classifier(texts)

    analyzer = _make_analyzer(spy)
    with redirect_stdout(StringIO()):
        analyzer.analyze_emotions(_fixture_df(), batch_size=32)

    assert calls, "pipeline was never called"
    assert all(isinstance(c[0], list) for c in calls), (
        "pipeline called per-message instead of batched"
    )
    assert all(c[1].get("batch_size") == 32 for c in calls), (
        "batch_size kwarg not honored"
    )


def test_rule_based_fallback_identical():
    analyzer = _make_analyzer(_classifier)
    analyzer.pipeline = None  # force the rule-based branch of analyze_emotions
    assert analyzer.pipeline is None

    with redirect_stdout(StringIO()):
        batched = analyzer.analyze_emotions(_fixture_df())

    expected = _sequential_reference(analyzer, _fixture_df())
    cols = EMOTION_COLS + ["dominant_emotion", "emotion_confidence"]
    pd.testing.assert_frame_equal(batched[cols], expected[cols])


def test_transformers5_nested_batch():
    analyzer = _make_analyzer(_nested_classifier)
    with redirect_stdout(StringIO()):
        df_emo = analyzer.analyze_emotions(_fixture_df())

    assert df_emo["dominant_emotion"].nunique() > 1, (
        "nested 5.x shape must unwrap per item in a batch, not degrade to 1/6"
    )
    assert df_emo[EMOTION_COLS].max().max() > 0.5


def test_small_batch_size_still_honored():
    """batch_size smaller than the scorable count is honored (multiple calls)."""
    calls = []

    def spy(texts, **kwargs):
        calls.append((list(texts), kwargs))
        return _classifier(texts)

    analyzer = _make_analyzer(spy)
    with redirect_stdout(StringIO()):
        analyzer.analyze_emotions(_fixture_df(), batch_size=1)

    assert len(calls) == 2, f"expected 2 single-item batch calls, got {len(calls)}"
    assert all(len(c[0]) == 1 for c in calls)


def test_batch_parse_failure_degrades_atomically():
    """H1 regression: a per-item parse failure in the MIDDLE of a batch must
    not leak already-yielded scores into `scored` (that shifted scores onto
    the wrong rows). The whole chunk must fall back per-message, aligned."""
    def flaky(texts, **kwargs):
        if isinstance(texts, str):
            return _per_text(texts)
        out = []
        for t in texts:
            if "bad" in str(t).lower():
                out.append([{"label": "joy"}])  # unparseable: missing 'score'
            else:
                out.append(_per_text(t))
        return out

    df = messages_to_dataframe([
        {"datetime": "2024-01-01T09:00:00", "sender": "A", "message": "I love this!"},
        {"datetime": "2024-01-01T09:01:00", "sender": "B", "message": "this is bad"},
        {"datetime": "2024-01-01T09:02:00", "sender": "A", "message": "just checking in"},
        {"datetime": "2024-01-01T09:03:00", "sender": "B", "message": ""},
        {"datetime": "2024-01-01T09:04:00", "sender": "A", "message": "  "},
        {"datetime": "2024-01-01T09:05:00", "sender": "B", "message": "<Media omitted>"},
    ])
    analyzer = _make_analyzer(flaky)
    with redirect_stdout(StringIO()):
        batched = analyzer.analyze_emotions(df)

    expected = _sequential_reference(analyzer, df)
    cols = EMOTION_COLS + ["dominant_emotion", "emotion_confidence"]
    pd.testing.assert_frame_equal(batched[cols], expected[cols])


def test_transformers5_whole_batch_wrapped():
    """M1: transformers 5.x can wrap the WHOLE batch one level deeper than 4.x
    ([[res0, res1, ...]] instead of [res0, res1, ...]). The alignment check must
    unwrap it so batching is not abandoned to per-message on 5.x."""

    def wrapped(texts, **kwargs):
        if isinstance(texts, str):
            return _nested_classifier(texts)
        return [[_nested_classifier(t) for t in texts]]

    analyzer = _make_analyzer(wrapped)
    with redirect_stdout(StringIO()):
        df_emo = analyzer.analyze_emotions(_fixture_df())

    assert df_emo["dominant_emotion"].nunique() > 1, (
        "whole-batch 5.x wrap must be unwrapped and batch-scored, not degraded"
    )
    assert df_emo[EMOTION_COLS].max().max() > 0.5
