"""Sliding-window rolling-health rewrite: bit-identical parity tests.

`calculate_rolling_health_score` was rewritten from the pre-grouped per-date
concatenation into an incremental sliding-window algorithm. The rewrite MUST be
bit-identical to the old output. These tests pin that contract:

- `_reference_rolling_health_score` (below) is the ORIGINAL boolean-mask
  implementation, captured verbatim from tests/test_perf_parity.py:456-498 and
  held here as the parity gold standard. Copying it locally avoids a circular
  dependency on test_perf_parity (which imports the same module under test).
- Every parity assertion compares the COMPLETE DataFrame via
  `pd.testing.assert_frame_equal` with defaults — dtypes, column names, order
  and values must all match exactly. No approximate score comparison.

NaT contract: the boolean-mask reference CRASHES on a NaT datetime row
(`sorted(df['date'].unique())` raises `TypeError: Cannot compare NaT with
datetime.date object`, confirmed on pandas 3.0). The rewrite treats NaT
datetimes as inert (unparseable, mirroring ingestion COR-02) — no crash, and
output identical to the NaT-free projection.

All tests are FAST-suite (no pytest.mark.slow).
"""

import inspect
import os
from datetime import timedelta

# Headless-first (Pitfall 7): relationship_health imports matplotlib.pyplot at
# module import; pin Agg BEFORE any pyplot import so figures are headless-safe.
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd
import pytest

import chat_analyzer.analysis.relationship_health as rel_health
from chat_analyzer.analysis.relationship_health import (
    analyze_response_patterns,
    calculate_dominance_scores,
    calculate_initiator_ratio,
    calculate_relationship_health_score,
    calculate_rolling_health_score,
    identify_conversation_starters,
    logger,
)
from chat_analyzer.ingest.ingestion import messages_to_dataframe

# ============================================================================
# Reference implementation: the ORIGINAL pre-optimization rolling health score,
# captured verbatim from tests/test_perf_parity.py (boolean-mask windowing).
# This is the parity gold standard the incremental rewrite must match exactly.
# ============================================================================

def _reference_rolling_health_score(
    df: pd.DataFrame, window_days: int = 7, min_messages: int = 10
) -> pd.DataFrame:
    """The ORIGINAL boolean-mask rolling health implementation, captured
    verbatim before the pre-grouped-date optimization."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')

    # Group by date
    df['date'] = df['datetime'].dt.date
    dates = sorted(df['date'].unique())

    health_scores = []

    for i, current_date in enumerate(dates):
        window_start = current_date - timedelta(days=window_days)
        window_df = df[(df['date'] >= window_start) & (df['date'] <= current_date)]

        if len(window_df) < min_messages:
            continue

        try:
            # Calculate metrics for this window
            window_df = identify_conversation_starters(window_df.reset_index(drop=True))
            initiator_metrics = calculate_initiator_ratio(window_df)
            response_metrics = analyze_response_patterns(window_df)
            dominance_metrics = calculate_dominance_scores(window_df)
            health_score = calculate_relationship_health_score(
                initiator_metrics, response_metrics, dominance_metrics
            )

            health_scores.append({
                'date': current_date,
                'health_score': health_score['overall_health_score'],
                'grade': health_score['grade'],
                'message_count': len(window_df)
            })
        except Exception as e:  # noqa: BLE001 - a failing window is skipped, never allowed to crash the series
            logger.warning(f"Failed to calculate health score for {current_date}: {e!s}")
            continue

    return pd.DataFrame(health_scores)


# ============================================================================
# Fixtures
# ============================================================================

def _rolling_fixture() -> pd.DataFrame:
    """32 messages across 8 distinct dates, 2 senders (>=10 msgs per 7-day window)."""
    start = pd.Timestamp('2024-02-01 09:00:00')
    messages = []
    for d in range(8):
        for h in range(4):
            sender = 'Alice' if (d + h) % 2 == 0 else 'Bob'
            messages.append({
                'datetime': start + pd.Timedelta(days=d, hours=h),
                'sender': sender,
                'message': f'day {d} message {h}',
            })
    return messages_to_dataframe(messages)


def _rolling_parity_fixture() -> pd.DataFrame:
    """80 messages across 40 consecutive daily dates, 2 senders alternating per
    day — every 7-day window holds >= 14 messages (dense above min_messages=10)."""
    start = pd.Timestamp('2024-04-01 09:00:00')
    messages = []
    for d in range(40):
        for h in range(2):
            sender = 'Alice' if (d + h) % 2 == 0 else 'Bob'
            messages.append({
                'datetime': start + pd.Timedelta(days=d, hours=h),
                'sender': sender,
                'message': f'day {d} message {h}',
            })
    return messages_to_dataframe(messages)


def _sparse_rolling_fixture() -> pd.DataFrame:
    """30 consecutive dates: days 0-27 hold 1 message each (every window of pure
    sparse days is below min_messages=10 so those dates are filtered), and the
    final 2 days hold 8 messages each (their windows pass min_messages=10)."""
    start = pd.Timestamp('2024-05-01 09:00:00')
    messages = []
    for d in range(28):
        sender = 'Alice' if d % 2 == 0 else 'Bob'
        messages.append({
            'datetime': start + pd.Timedelta(days=d),
            'sender': sender,
            'message': f'sparse {d}',
        })
    for d in (28, 29):
        for h in range(8):
            sender = 'Alice' if (d + h) % 2 == 0 else 'Bob'
            messages.append({
                'datetime': start + pd.Timedelta(days=d, hours=h),
                'sender': sender,
                'message': f'dense {d}-{h}',
            })
    return messages_to_dataframe(messages)


def _single_date_fixture(n_msgs: int) -> pd.DataFrame:
    """All n_msgs on ONE date; used for the single-date boundary test."""
    dt = pd.Timestamp('2024-08-01 09:00:00')
    messages = [
        {
            'datetime': dt + pd.Timedelta(minutes=5 * i),
            'sender': 'Alice' if i % 2 == 0 else 'Bob',
            'message': f'm{i}',
        }
        for i in range(n_msgs)
    ]
    return messages_to_dataframe(messages)


def _nat_rolling_fixture() -> pd.DataFrame:
    """Raw DataFrame (NOT via messages_to_dataframe — that builder drops NaT
    datetimes per COR-02) with 8 dense dates plus one NaT-datetime orphan row."""
    base = pd.Timestamp('2024-06-01 08:00:00')
    rows = []
    for d in range(8):
        for h in range(2):
            rows.append({
                'datetime': base + pd.Timedelta(days=d, hours=h),
                'sender': 'Alice' if (d + h) % 2 == 0 else 'Bob',
                'message': f'd{d}h{h}',
            })
    rows.append({
        'datetime': pd.NaT,
        'sender': 'Carol',
        'message': 'orphan row with no parseable datetime',
    })
    df = pd.DataFrame(rows)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def _large_rolling_fixture(n_dates: int = 210) -> pd.DataFrame:
    """n_dates consecutive dates x 2 msgs (dense, >=10 msgs/window). Sized so the
    parallel branch (threshold monkeypatched to 1) is genuinely exercised."""
    start = pd.Timestamp('2024-01-01 09:00:00')
    messages = []
    for d in range(n_dates):
        for h in range(2):
            sender = 'Alice' if (d + h) % 2 == 0 else 'Bob'
            messages.append({
                'datetime': start + pd.Timedelta(days=d, hours=h),
                'sender': sender,
                'message': f'd{d}h{h}',
            })
    return messages_to_dataframe(messages)


def _three_sender_fixture() -> pd.DataFrame:
    """12 days x 3 messages (one per sender per day), 3 participants rotating:
    only the day's first sender (Alice at 09:00) is ever a conversation starter."""
    base = pd.Timestamp('2024-07-01 09:00:00')
    senders = ['Alice', 'Bob', 'Carol']
    messages = []
    for d in range(12):
        for k, sender in enumerate(senders):
            messages.append({
                'datetime': base + pd.Timedelta(days=d, hours=9 + k),
                'sender': sender,
                'message': f'd{d}-{sender}',
            })
    return messages_to_dataframe(messages)


def _single_sided_fixture() -> pd.DataFrame:
    """Exercises one-sided balance branches and the `len(...) >= 2` guards.

    Days 0-9: Carol opens each day (starter, never replies), Alice initiates at
    09:00 (starter), Bob replies within 5 minutes (responder, never initiates).
    Days 10-19: empty gap. Days 20-27: Alice only (2/day) so any window inside
    the tail is a single-sender window — value_counts has len 1 and the
    `len(...) >= 2` guards in dominance/initiator metrics fire.
    """
    base = pd.Timestamp('2024-09-01 00:01:00')
    messages = []
    for d in range(10):
        messages.append({
            'datetime': base + pd.Timedelta(days=d, hours=0),
            'sender': 'Carol',
            'message': f'd{d} open',
        })
        messages.append({
            'datetime': base + pd.Timedelta(days=d, hours=9),
            'sender': 'Alice',
            'message': f'd{d} alice',
        })
        messages.append({
            'datetime': base + pd.Timedelta(days=d, hours=9, minutes=5),
            'sender': 'Bob',
            'message': f'd{d} bob',
        })
    for d in range(20, 28):
        for h in (0, 1):
            messages.append({
                'datetime': base + pd.Timedelta(days=d, hours=h),
                'sender': 'Alice',
                'message': f'tail {d}-{h}',
            })
    return messages_to_dataframe(messages)


def _mixed_density_fixture() -> pd.DataFrame:
    """25 dates with wildly varying per-day counts (0-18) and empty gaps, so
    sliding windows gain and lose days asymmetrically across the run."""
    start = pd.Timestamp('2024-10-01 08:00:00')
    per_day = [3, 0, 6, 15, 1, 12, 0, 2, 0, 9, 4, 18, 0, 1, 7, 0, 10, 2, 0, 5, 0, 8, 0, 1, 11]
    messages = []
    for d, count in enumerate(per_day):
        for j in range(count):
            sender = ['Alice', 'Bob', 'Carol'][(d + j) % 3]
            messages.append({
                'datetime': start + pd.Timedelta(days=d, hours=j),
                'sender': sender,
                'message': f'd{d}-{j}',
            })
    return messages_to_dataframe(messages)


# ============================================================================
# Parity tests
# ============================================================================

def test_rolling_health_sliding_matches_reference():
    """Sliding-window rewrite is bit-identical to the boolean-mask reference
    across window sizes and thresholds on both existing fixtures."""
    fixtures = {
        '40-date-parity': _rolling_parity_fixture(),
        '8-date': _rolling_fixture(),
    }
    for label, df in fixtures.items():
        for window_days in (7, 14):
            for min_messages in (4, 10):
                reference = _reference_rolling_health_score(df, window_days, min_messages)
                result = calculate_rolling_health_score(df, window_days, min_messages)

                assert not reference.empty, (
                    f'{label} wd={window_days} mm={min_messages}: fixture must score a window'
                )
                pd.testing.assert_frame_equal(reference, result)


def test_rolling_health_min_messages_filtering():
    """Dates whose window falls below min_messages are absent; dense-day windows
    survive. Sparse fixture: only the 2 dense tail days score at mm=10."""
    df = _sparse_rolling_fixture()
    total_dates = df['date'].nunique()

    for window_days, min_messages in ((7, 10), (7, 4)):
        reference = _reference_rolling_health_score(df, window_days, min_messages)
        result = calculate_rolling_health_score(df, window_days, min_messages)

        pd.testing.assert_frame_equal(reference, result)
        assert not result.empty, 'dense tail must score'
        assert result['message_count'].ge(min_messages).all(), (
            f'every scored window must hold >= {min_messages} messages'
        )
        assert len(result) <= total_dates

    at_10 = calculate_rolling_health_score(df, 7, 10)
    at_4 = calculate_rolling_health_score(df, 7, 4)

    assert len(at_10) < len(at_4), 'stricter threshold must filter strictly more dates'
    assert len(at_10) == 2, 'mm=10: only the two dense days score'
    assert any(c != at_10['message_count'].iloc[0] for c in at_10['message_count'])

    expected_dates = set(at_4['date']) | set(at_10['date'])
    assert all(set(_reference_rolling_health_score(df, 7, mm)['date']) <= expected_dates
               for mm in (4, 10))


def test_rolling_health_empty_input():
    """Empty (schema-bearing) frame: both reference and rewrite return a 0-row,
    0-column DataFrame (pd.DataFrame([])) — assert exact equality and emptiness."""
    full = _rolling_fixture()
    empty = full.iloc[:0]

    reference = _reference_rolling_health_score(empty)
    result = calculate_rolling_health_score(empty)

    assert len(reference) == 0
    assert reference.columns.tolist() == []
    pd.testing.assert_frame_equal(reference, result)
    assert len(result) == 0
    assert result.columns.tolist() == []


def test_rolling_health_single_date():
    """Single-date input: >= min_messages -> exactly one row; too few -> zero rows."""
    for n_msgs in (12, 16):
        df = _single_date_fixture(n_msgs)
        reference = _reference_rolling_health_score(df, 7, 10)
        result = calculate_rolling_health_score(df, 7, 10)

        pd.testing.assert_frame_equal(reference, result)
        assert len(result) == 1, f'{n_msgs} msgs on one date must yield exactly one row'
        assert result['message_count'].iloc[0] == n_msgs

    for n_msgs in (3, 5, 9):
        df = _single_date_fixture(n_msgs)
        reference = _reference_rolling_health_score(df, 7, 10)
        result = calculate_rolling_health_score(df, 7, 10)

        pd.testing.assert_frame_equal(reference, result)
        assert len(result) == 0, f'{n_msgs} msgs must be below the 10-message threshold'


def test_rolling_health_nat_datetime():
    """NaT datetime row must be inert. Documented behavior: the boolean-mask
    reference RAISES TypeError on a NaT row (sorted(df['date'].unique()) cannot
    order NaT against datetime.date — confirmed on pandas 3.0). The rewrite must
    not crash and must produce exactly the reference's NaT-free result (NaT rows
    never satisfy any window mask, so dropping them changes nothing)."""
    nat_df = _nat_rolling_fixture()

    with pytest.raises(TypeError):
        _reference_rolling_health_score(nat_df, 7, 4)

    clean = nat_df[nat_df['datetime'].notna()].reset_index(drop=True)
    reference = _reference_rolling_health_score(clean, 7, 4)
    assert not reference.empty, 'dense NaT-free portion must score'

    result = calculate_rolling_health_score(nat_df, 7, 4)
    pd.testing.assert_frame_equal(reference, result)

    assert result['health_score'].between(0, 1).all(), 'scores must stay in [0, 1]'
    assert result['message_count'].ge(4).all()

    starters = identify_conversation_starters(nat_df.reset_index(drop=True).copy())
    nat_idx = starters.index[starters['datetime'].isna()][0]
    assert bool(starters.loc[nat_idx, 'is_conversation_starter']) is True, (
        'documented deviation: NaT rows are marked as starters'
    )


def test_rolling_health_sequential_matches_parallel_sliding(monkeypatch):
    """The ProcessPoolExecutor branch (threshold forced to 1) recomputes windows
    with the unmodified reference worker and must be bit-identical to the
    sequential incremental path (threshold forced huge). Sized so the parallel
    path is genuinely exercised."""
    df = _large_rolling_fixture()

    monkeypatch.setattr(rel_health, '_ROLLING_PARALLEL_MIN_DATES', 10**9)
    sequential = calculate_rolling_health_score(df, 7, 10)

    monkeypatch.setattr(rel_health, '_ROLLING_PARALLEL_MIN_DATES', 1)
    parallel = calculate_rolling_health_score(df, 7, 10)

    assert not sequential.empty, 'large fixture must produce scored windows'
    pd.testing.assert_frame_equal(sequential, parallel)


def test_rolling_health_multiple_participants():
    """3 senders alternating, plus a never-initiates / never-replies / single-
    sender-tail fixture — exercises the `len(...) >= 2` guards. Exact parity."""
    fixtures = {
        'three-sender': _three_sender_fixture(),
        'single-sided': _single_sided_fixture(),
    }
    for label, df in fixtures.items():
        for window_days, min_messages in ((7, 4), (7, 10)):
            reference = _reference_rolling_health_score(df, window_days, min_messages)
            result = calculate_rolling_health_score(df, window_days, min_messages)

            assert not reference.empty, f'{label} wd={window_days} mm={min_messages} must score'
            pd.testing.assert_frame_equal(reference, result)


def test_rolling_health_mixed_density():
    """Varying per-day counts and empty gaps make windows gain/lose days
    asymmetrically; the sliding rewrite must still match the reference exactly."""
    df = _mixed_density_fixture()

    for window_days, min_messages in ((7, 4), (7, 10)):
        reference = _reference_rolling_health_score(df, window_days, min_messages)
        result = calculate_rolling_health_score(df, window_days, min_messages)

        assert not reference.empty, f'wd={window_days} mm={min_messages} must score'
        pd.testing.assert_frame_equal(reference, result)

    counts = calculate_rolling_health_score(df, 7, 4)['message_count'].tolist()
    assert len(set(counts)) > 5, 'window sizes must genuinely vary across the fixture'


def test_rolling_health_public_contract():
    """Public signature and module surface are unchanged by the rewrite."""
    params = list(inspect.signature(calculate_rolling_health_score).parameters.items())
    assert [name for name, _ in params] == ['df', 'window_days', 'min_messages']

    df_param, window_param, min_messages_param = params
    assert df_param[1].default is inspect.Parameter.empty
    assert window_param[1].default == 7
    assert min_messages_param[1].default == 10

    # The incremental internals must be contained — no leaking helper names.
    assert not hasattr(rel_health, 'rolling_health_step')
    assert not hasattr(rel_health, '_rolling_health_step')


def test_rolling_health_dtype_and_order_parity():
    """check_exact=True frame equality plus explicit dtype / column-order /
    row-order checks — the rewrite must be bit-identical, not merely close."""
    df = _rolling_fixture()
    reference = _reference_rolling_health_score(df, 7, 10)
    result = calculate_rolling_health_score(df, 7, 10)

    pd.testing.assert_frame_equal(reference, result, check_exact=True)

    assert result.dtypes.equals(reference.dtypes)
    assert result.columns.tolist() == reference.columns.tolist()
    assert list(result['date']) == list(reference['date'])
    assert list(result['health_score']) == list(reference['health_score'])
    assert list(result['grade']) == list(reference['grade'])
    assert list(result['message_count']) == list(reference['message_count'])