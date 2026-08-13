"""
Perf-parity tests for the dedupe + parallel VADER path in sentiment.py.

Proves that _score_vader_parallel (deterministic dedupe, sequential below the
threshold, multiprocess above it) produces vader_* column values bit-identical
to the original per-message ``apply(analyze_vader)`` path. The fixture mixes
positive/negative/neutral messages, empty strings, whitespace, the
"<Media omitted>" marker, a non-string value, NaN, and DUPLICATED messages so
dedupe provably collapses scoring calls.

Test A exercises the default sequential dedupe path (and counts polarity_scores
calls to prove dedupe collapses 12 rows -> 6 scoring calls). Test B forces the
ProcessPoolExecutor path (threshold=1) and asserts the same parity — on Windows
the workers spawn fresh interpreters, which is exactly why the top-level
helpers (_vader_score_batch, _score_vader_parallel) are importable.
"""

import io
import os
from contextlib import redirect_stdout

# Headless-first (same guarantee as test_analysis.py): sentiment.py imports
# matplotlib.pyplot at module import — pin Agg BEFORE any pyplot import so the
# spawn workers inherit a working headless backend too.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

# sentiment.py prints optional-dependency notices at import time (e.g. "⚠️
# TextBlob not available…"); on a cp1252 console the emoji raises
# UnicodeEncodeError. Suppress import-time stdout (D-17 pattern).
with redirect_stdout(io.StringIO()):
    from chat_analyzer.analysis import sentiment as _sentiment
    from chat_analyzer.ingest.ingestion import messages_to_dataframe

# Pin the VADER-only path BEFORE any analysis runs: never let
# initialize_analyzers() build the HF pipeline (would download the cardiffnlp
# model — D-17 / T-04-15 pattern, same as test_analysis.py).
_sentiment.TRANSFORMERS_AVAILABLE = False


def _build_fixture() -> pd.DataFrame:
    """Mixed-message DataFrame with duplicates so dedupe provably collapses rows.

    Duplicated messages appear multiple times; empty string, whitespace,
    "<Media omitted>", a non-string value and NaN cover the coercion/neutral
    edge cases. The int/NaN values are injected after messages_to_dataframe
    (that builder requires len()-able message values) — both the original
    apply(analyze_vader) path and _score_vader_parallel handle them identically
    (str(non-string) coercion / pd.isna short-circuit).
    """
    messages = [
        {'datetime': '2023-12-01T09:00:00', 'sender': 'Alice', 'message': 'I love this!'},
        {'datetime': '2023-12-01T09:01:00', 'sender': 'Alice', 'message': 'I love this!'},
        {'datetime': '2023-12-01T09:02:00', 'sender': 'Bob', 'message': 'I hate this'},
        {'datetime': '2023-12-01T09:03:00', 'sender': 'Bob', 'message': 'I hate this'},
        {'datetime': '2023-12-01T09:04:00', 'sender': 'Carol', 'message': 'The weather is cloudy'},
        {'datetime': '2023-12-01T09:05:00', 'sender': 'Carol', 'message': ''},
        {'datetime': '2023-12-01T09:06:00', 'sender': 'Dave', 'message': '   '},
        {'datetime': '2023-12-01T09:07:00', 'sender': 'Dave', 'message': '<Media omitted>'},
        {'datetime': '2023-12-01T09:08:00', 'sender': 'Eve', 'message': 'Great job!'},
        {'datetime': '2023-12-01T09:09:00', 'sender': 'Eve', 'message': 'Great job!'},
        {'datetime': '2023-12-01T09:10:00', 'sender': 'Frank', 'message': 'placeholder'},
        {'datetime': '2023-12-01T09:11:00', 'sender': 'Frank', 'message': 'placeholder'},
    ]
    df = messages_to_dataframe(messages)
    # messages_to_dataframe builds a pandas str-dtype column (rejects non-str
    # values); cast to object so int/NaN injection is possible. Both analysis
    # paths handle object values identically (str() coercion / pd.isna
    # short-circuit).
    df = df.astype({'message': object})
    df.loc[10, 'message'] = 123           # non-string value (str-coerced identically)
    df.loc[11, 'message'] = float('nan')  # NaN (pd.isna short-circuit -> neutral)
    return df


def _reference_vader_columns(df: pd.DataFrame) -> pd.DataFrame:
    """vader_* columns exactly as the ORIGINAL per-message path produced them."""
    with redirect_stdout(io.StringIO()):
        _sentiment.initialize_analyzers()
    scores = df['message'].apply(_sentiment.analyze_vader)
    ref = pd.DataFrame({
        'vader_compound': [r['compound'] for r in scores],
        'vader_pos': [r['pos'] for r in scores],
        'vader_neu': [r['neu'] for r in scores],
        'vader_neg': [r['neg'] for r in scores],
    })
    ref['vader_sentiment'] = ref['vader_compound'].apply(_sentiment.categorize_sentiment)
    return ref


def _assert_vader_parity(actual: pd.DataFrame, ref: pd.DataFrame) -> None:
    """vader_* values must match the reference with row order preserved."""
    assert len(actual) == len(ref)
    for col in ('vader_compound', 'vader_pos', 'vader_neu', 'vader_neg'):
        np.testing.assert_allclose(
            actual[col].to_numpy(dtype=float),
            ref[col].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-12,
            err_msg=f'{col} differs between parallel/dedupe path and per-message apply',
        )
    assert actual['vader_sentiment'].tolist() == ref['vader_sentiment'].tolist()


def test_default_dedupe_path_matches_apply_reference(monkeypatch):
    """Sequential dedupe path: values identical AND scoring calls provably collapse.

    12 rows contain 6 unique scorable strings (dups + empty/whitespace/NaN
    collapse); a per-row apply would have made 12 polarity_scores calls.
    """
    df = _build_fixture()
    ref = _reference_vader_columns(df)

    calls = []
    original = _sentiment.SentimentIntensityAnalyzer.polarity_scores

    def _counting(self, text):
        calls.append(text)
        return original(self, text)

    # Class-level patch: add_sentiment_analysis re-initializes the analyzer
    # (fresh instance), so the count must follow the class, not one instance.
    monkeypatch.setattr(_sentiment.SentimentIntensityAnalyzer, 'polarity_scores', _counting)
    with redirect_stdout(io.StringIO()):
        actual = _sentiment.add_sentiment_analysis(df)

    assert len(calls) == 6, (
        f'dedupe collapsed scoring to {len(calls)} calls, expected 6 '
        f'(unique scorable strings) < {len(df)} rows'
    )
    assert len(calls) < len(df), 'dedupe must reduce scoring calls below row count'
    _assert_vader_parity(actual, ref)


def test_parallel_path_matches_apply_reference(monkeypatch):
    """ProcessPoolExecutor path (threshold=1): values identical to the reference.

    On Windows the workers spawn fresh interpreters; the top-level helpers make
    the parallel path importable and the spawn-child print gating keeps module
    import silent in the workers.
    """
    df = _build_fixture()
    ref = _reference_vader_columns(df)

    monkeypatch.setattr(_sentiment, '_VADER_PARALLEL_THRESHOLD', 1)
    with redirect_stdout(io.StringIO()):
        actual = _sentiment.add_sentiment_analysis(df)

    _assert_vader_parity(actual, ref)
