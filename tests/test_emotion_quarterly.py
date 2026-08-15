"""Quarterly emotion aggregation + timeline chart spec coverage.

Covers the pure ``get_emotion_quarterly`` module function (emotion.py), the
``EmotionAnalyzer.get_emotion_quarterly`` method wrapper that delegates to it,
and ``build_emotion_timeline_spec`` (chart_json.py) which consumes the
``emotion_summary["quarterly"]`` contract:

- a. Per-quarter MEAN of the six emotion_* columns, oldest quarter first,
       scores rounded to 6 decimals, entries
       ``[{"quarter": "2026Q1", "scores": {joy: ..., ...}}]``.
- b. Degenerate frames: empty -> [], None -> [], missing ``datetime`` -> [].
- c. Sampled mode: when the frame carries an ``emotion_scored`` bool column,
       only the SCORED rows are aggregated (report labels and quarterly
       values agree); all-unscored -> [].
- d. Malformed input never raises: unparseable datetimes, missing emotion
       columns, non-boolean ``emotion_scored`` -> [] (defensive contract).
- e. The analyzer method wrapper returns exactly what the module function
       returns (including the degenerate cases).
- f. build_emotion_timeline_spec: None when quarterly data is absent or all
       entries are malformed; with data, xAxis.data = quarter strings, one
       line series per emotion in the fixed six-label order, yAxis min 0
       max 1, NaN scores dropped to None, and the whole spec JSON-safe
       (numpy floats coerced, no NaN leaks).

These are FAST tests (no pytest.mark.slow) and never touch a real model
(D-17): the analyzer used by the wrapper test is built with the module-level
singleton patched, mirroring tests/test_perf_parity_emotion.py. Frames are
built directly with pandas (no ingestion); they keep the message-shape
(datetime, sender, message, emotion_* columns).
"""

import json
import os
import unittest.mock
from contextlib import redirect_stdout
from io import StringIO

# Headless-first (Pitfall 7): importing chat_analyzer.analysis.emotion pulls
# matplotlib; pin Agg BEFORE any pyplot import so figures are headless-safe.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from chat_analyzer.analysis import emotion as _emotion_module
from chat_analyzer.analysis.emotion import (
    EMOTION_LABELS,
    EmotionAnalyzer,
    get_emotion_quarterly,
)
from chat_analyzer.cli.chart_json import build_emotion_timeline_spec


def _scores(joy, sadness, anger, fear, surprise, love):
    """Six emotion_* column values, keyed the way analyze_emotions fills them."""
    return {
        "emotion_joy": joy,
        "emotion_sadness": sadness,
        "emotion_anger": anger,
        "emotion_fear": fear,
        "emotion_surprise": surprise,
        "emotion_love": love,
    }


def _tl_scores(values):
    """Timeline-style score dict keyed by bare emotion label (no prefix)."""
    return dict(zip(EMOTION_LABELS, values))


def _dummy_classifier(texts, **kwargs):
    """Stand-in pipeline; get_emotion_quarterly never calls it."""
    return []


def _make_analyzer():
    """Real EmotionAnalyzer with the transformers pipeline mocked (D-17),
    mirroring tests/test_perf_parity_emotion.py: patch the module-level model
    cache so _initialize_model short-circuits to the fake pipeline."""
    with (
        redirect_stdout(StringIO()),
        unittest.mock.patch.object(_emotion_module, "_emotion_analyzer", _dummy_classifier),
        unittest.mock.patch.object(_emotion_module, "_emotion_model_loaded", True),
    ):
        return EmotionAnalyzer()


def _two_quarter_frame():
    """2025Q4 (2 rows) + 2026Q1 (1 row). The NEWER quarter's row comes FIRST
    on purpose: the contract is oldest-quarter-first regardless of row order.
    All score values are exact binary fractions so mean == literal floats."""
    return pd.DataFrame([
        {"datetime": "2026-02-14T12:00:00", "sender": "Alice", "message": "m2",
         **_scores(0.7, 0.1, 0.1, 0.05, 0.03, 0.02)},
        {"datetime": "2025-10-05T09:00:00", "sender": "Alice", "message": "m0",
         **_scores(0.25, 0.125, 0.5, 0.25, 0.0625, 0.5)},
        {"datetime": "2025-12-20T18:30:00", "sender": "Bob", "message": "m1",
         **_scores(0.75, 0.375, 0.5, 0.125, 0.1875, 0.25)},
    ])


# --- a. per-quarter means ---------------------------------------------------


def test_quarterly_means_oldest_first_rounded():
    result = get_emotion_quarterly(_two_quarter_frame())

    assert [entry["quarter"] for entry in result] == ["2025Q4", "2026Q1"], (
        "quarters must be oldest first even when the newer row leads the frame"
    )
    assert list(result[0]["scores"].keys()) == list(EMOTION_LABELS), (
        "score keys must follow the fixed six-label order"
    )

    # 2025Q4 means over two rows: (0.25+0.75)/2, (0.125+0.375)/2, ...
    assert result[0]["scores"] == {
        "joy": 0.5,
        "sadness": 0.25,
        "anger": 0.5,
        "fear": 0.1875,
        "surprise": 0.125,
        "love": 0.375,
    }
    # 2026Q1 is a single row -> mean == its own scores
    assert result[1]["scores"] == {
        "joy": 0.7,
        "sadness": 0.1,
        "anger": 0.1,
        "fear": 0.05,
        "surprise": 0.03,
        "love": 0.02,
    }


def test_quarterly_rounds_to_six_decimals():
    df = pd.DataFrame([
        {"datetime": "2026-01-05T09:00:00", **_scores(1 / 3, 2 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3)},
        {"datetime": "2026-01-06T09:00:00", **_scores(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
    ])
    result = get_emotion_quarterly(df)

    assert len(result) == 1
    assert result[0]["quarter"] == "2026Q1"
    assert result[0]["scores"]["joy"] == 0.166667, (
        "mean(1/3, 0) must be rounded to 6 decimals"
    )
    assert result[0]["scores"]["sadness"] == 0.333333


# --- b. degenerate frames ---------------------------------------------------


def test_quarterly_empty_frame_returns_empty_list():
    assert get_emotion_quarterly(pd.DataFrame()) == []


def test_quarterly_none_returns_empty_list():
    assert get_emotion_quarterly(None) == []


def test_quarterly_missing_datetime_returns_empty_list():
    df = pd.DataFrame([
        {"sender": "Alice", "message": "hi", **_scores(0.1, 0.1, 0.1, 0.1, 0.1, 0.1)},
    ])
    assert get_emotion_quarterly(df) == []


# --- c. sampled mode --------------------------------------------------------


def test_quarterly_sampled_mode_aggregates_only_scored_rows():
    df = pd.DataFrame([
        {"datetime": "2026-01-05T09:00:00", "emotion_scored": True,
         **_scores(0.1, 0.1, 0.1, 0.1, 0.1, 0.1)},
        {"datetime": "2026-01-06T09:00:00", "emotion_scored": True,
         **_scores(0.2, 0.2, 0.2, 0.2, 0.2, 0.2)},
        # unscored row with a huge score would skew the mean if it leaked in
        {"datetime": "2026-01-07T09:00:00", "emotion_scored": False,
         **_scores(0.9, 0.9, 0.9, 0.9, 0.9, 0.9)},
    ])
    result = get_emotion_quarterly(df)

    assert result[0]["scores"]["joy"] == 0.15, (
        "only SCORED rows may be aggregated (0.9 unscored row excluded)"
    )
    assert result[0]["scores"]["sadness"] == 0.15
    # equivalence with an explicit filter — the canonical reference
    assert result == get_emotion_quarterly(df[df["emotion_scored"]])


def test_quarterly_sampled_all_unscored_returns_empty():
    df = pd.DataFrame([
        {"datetime": "2026-01-05T09:00:00", "emotion_scored": False,
         **_scores(0.1, 0.1, 0.1, 0.1, 0.1, 0.1)},
    ])
    assert get_emotion_quarterly(df) == []


# --- d. malformed input never raises ---------------------------------------


def test_quarterly_never_raises_on_malformed_input():
    # datetimes that cannot be parsed -> []
    bad_datetimes = pd.DataFrame([
        {"datetime": "not-a-date", **_scores(0.1, 0.1, 0.1, 0.1, 0.1, 0.1)},
        {"datetime": None, **_scores(0.2, 0.2, 0.2, 0.2, 0.2, 0.2)},
    ])
    assert get_emotion_quarterly(bad_datetimes) == []

    # emotion_* columns missing -> []
    no_emotion_cols = pd.DataFrame({
        "datetime": ["2026-01-01T00:00:00"],
        "sender": ["Alice"],
        "message": ["hi"],
    })
    assert get_emotion_quarterly(no_emotion_cols) == []

    # emotion_scored present but not boolean -> []
    bad_scored = pd.DataFrame([
        {"datetime": "2026-01-05T09:00:00", "emotion_scored": "yes",
         **_scores(0.1, 0.1, 0.1, 0.1, 0.1, 0.1)},
    ])
    assert get_emotion_quarterly(bad_scored) == []


# --- e. analyzer method wrapper --------------------------------------------


def test_analyzer_wrapper_delegates_to_module_function():
    analyzer = _make_analyzer()
    df = _two_quarter_frame()

    with redirect_stdout(StringIO()):
        via_method = analyzer.get_emotion_quarterly(df)
    assert via_method == get_emotion_quarterly(df)
    assert via_method[0]["quarter"] == "2025Q4"

    with redirect_stdout(StringIO()):
        assert analyzer.get_emotion_quarterly(None) == []
        assert analyzer.get_emotion_quarterly(pd.DataFrame()) == []


# --- f. build_emotion_timeline_spec ----------------------------------------


def test_timeline_spec_none_when_no_quarterly_data():
    assert build_emotion_timeline_spec(None) is None
    assert build_emotion_timeline_spec({}) is None
    assert build_emotion_timeline_spec({"distribution": {"joy": 1}}) is None
    assert build_emotion_timeline_spec({"quarterly": None}) is None
    assert build_emotion_timeline_spec({"quarterly": []}) is None
    # every entry malformed -> no quarters collected -> None
    assert build_emotion_timeline_spec({"quarterly": ["garbage", 42]}) is None


def test_timeline_spec_with_data():
    summary = {"quarterly": [
        {"quarter": "2025Q4", "scores": _tl_scores((0.5, 0.25, 0.5, 0.1875, 0.125, 0.375))},
        {"quarter": "2026Q1", "scores": _tl_scores((0.7, 0.1, 0.1, 0.05, 0.03, 0.02))},
    ]}
    spec = build_emotion_timeline_spec(summary)

    assert spec is not None
    assert spec["xAxis"]["type"] == "category"
    assert spec["xAxis"]["data"] == ["2025Q4", "2026Q1"]
    assert spec["yAxis"]["min"] == 0
    assert spec["yAxis"]["max"] == 1

    assert [s["name"] for s in spec["series"]] == list(EMOTION_LABELS), (
        "one line series per emotion, in the fixed six-label order"
    )
    assert all(s["type"] == "line" for s in spec["series"])
    assert spec["series"][0]["data"] == [0.5, 0.7]  # joy
    assert spec["series"][5]["data"] == [0.375, 0.02]  # love
    assert spec["tooltip"]["trigger"] == "axis"

    # the whole spec is JSON-serializable end to end
    assert json.loads(json.dumps(spec)) == spec


def test_timeline_spec_skips_malformed_entries():
    summary = {"quarterly": [
        {"quarter": "2026Q1", "scores": _tl_scores((0.1, 0.1, 0.1, 0.1, 0.1, 0.1))},
        "not-a-dict",
        42,
        {"quarter": None, "scores": _tl_scores((0.9, 0.9, 0.9, 0.9, 0.9, 0.9))},
        {"quarter": "2026Q2", "scores": "nope"},
        {"quarter": "2026Q3"},
        {"quarter": "2026Q4", "scores": _tl_scores((0.2, 0.2, 0.2, 0.2, 0.2, 0.2))},
    ]}
    spec = build_emotion_timeline_spec(summary)

    assert spec["xAxis"]["data"] == ["2026Q1", "2026Q4"], (
        "malformed entries must be skipped, valid ones kept in order"
    )
    assert len(spec["series"]) == 6
    assert spec["series"][0]["data"] == [0.1, 0.2]


def test_timeline_spec_nan_scores_become_none():
    summary = {"quarterly": [
        {"quarter": "2026Q1",
         "scores": {"joy": float("nan"), **{e: 0.2 for e in EMOTION_LABELS[1:]}}},
    ]}
    spec = build_emotion_timeline_spec(summary)

    assert spec["series"][0]["data"] == [None], (
        "NaN scores must be dropped to None, never leak into JSON"
    )
    assert spec["series"][1]["data"] == [0.2]
    json.dumps(spec)  # must not raise


def test_timeline_spec_json_safe_with_numpy_values():
    summary = {"quarterly": [
        {"quarter": "2026Q1", "scores": _tl_scores(np.float64([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]))},
        {"quarter": "2026Q2", "scores": _tl_scores(np.float64([0.4, 0.1, 0.2, 0.1, 0.1, 0.1]))},
    ]}
    spec = build_emotion_timeline_spec(summary)

    assert all(isinstance(v, float) for v in spec["series"][0]["data"]), (
        "numpy scalars must be coerced to plain floats (JSON-safe)"
    )
    assert json.loads(json.dumps(spec)) == spec