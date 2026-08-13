"""Performance-refactor parity tests for ChatEDA.conversation dynamics.

Proves a pure-performance change is behavior-identical:

- analyze_conversation_dynamics: the per-row `df_sorted.iloc[i]` Python loop
  (response-time extraction on sender switches) is replaced with vectorized
  pandas (`shift()` sender comparison + `diff()` datetimes). Every returned
  value must be bit-identical to the pre-refactor loop.
- generate_comprehensive_summary: the three analyze_* calls may now be passed
  in as precomputed results (default None = compute internally), so the CLI
  pipeline can avoid computing each analysis twice. Passing precomputed
  results must produce an exactly-equal summary dict to the no-arg call.

Fixtures cover: alternating A/B, same-sender runs, day gaps, all-same-sender,
single message, and three senders — all with valid datetimes (no NaT).

These are FAST tests (no pytest.mark.slow).
"""

import os

# Headless-first (Pitfall 7): eda.py imports matplotlib.pyplot at module
# import; pin Agg BEFORE any pyplot import so figure creation is headless-safe.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from chat_analyzer.analysis.eda import ChatEDA
from chat_analyzer.ingest.ingestion import messages_to_dataframe

# ============================================================================
# Reference implementation: the FULL ORIGINAL analyze_conversation_dynamics
# loop, captured verbatim BEFORE the vectorization refactor.
# ============================================================================


def _reference_dynamics(df: pd.DataFrame) -> dict:
    """The ORIGINAL pre-vectorization .iloc loop, captured verbatim.

    Operates on the prepared ChatEDA frame (self.df) exactly as the original
    method did: sort by datetime, then walk rows 1..n-1 collecting the
    minute-difference whenever the sender changes.
    """
    df_sorted = df.sort_values('datetime').reset_index(drop=True)
    response_times = []

    for i in range(1, len(df_sorted)):
        if df_sorted.iloc[i]['sender'] != df_sorted.iloc[i - 1]['sender']:
            time_diff = (
                df_sorted.iloc[i]['datetime']
                - df_sorted.iloc[i - 1]['datetime']
            ).total_seconds() / 60
            response_times.append(time_diff)

    return {
        'response_times': response_times,
        'avg_response_time': np.mean(response_times) if response_times else None,
        'balance_ratio': min(df['sender'].value_counts()) / max(df['sender'].value_counts())
    }


# ============================================================================
# Fixture builders (1-hour spaced timestamps, valid datetimes only)
# ============================================================================

_START = pd.Timestamp('2024-01-01 09:00:00')


def _build_df(senders: list[str]) -> pd.DataFrame:
    """Build a fixture df with 1-hour spaced messages (no NaT datetimes)."""
    messages = [
        {
            'datetime': _START + pd.Timedelta(hours=i),
            'sender': sender,
            'message': f'message {i} from {sender}',
        }
        for i, sender in enumerate(senders)
    ]
    return messages_to_dataframe(messages)


def _build_day_gap_df() -> pd.DataFrame:
    """Fixture with a 24-hour day gap (offsets 2 -> 26) across a sender switch."""
    hour_offsets = [0, 1, 2, 26, 27]
    senders = ['A', 'B', 'A', 'B', 'A']
    messages = [
        {
            'datetime': _START + pd.Timedelta(hours=h),
            'sender': sender,
            'message': f'message at hour {h}',
        }
        for h, sender in zip(hour_offsets, senders)
    ]
    return messages_to_dataframe(messages)


# sender sequences covering: alternating A/B, same-sender runs, day gaps,
# all-same-sender, single message, three senders.
FIXTURE_BUILDERS = {
    'alternating': lambda: _build_df(['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']),
    'same_sender_runs': lambda: _build_df(['A', 'A', 'A', 'B', 'B', 'A', 'B', 'B', 'B']),
    'day_gaps': _build_day_gap_df,
    'all_same_sender': lambda: _build_df(['A', 'A', 'A', 'A', 'A']),
    'single_message': lambda: _build_df(['A']),
    'three_senders': lambda: _build_df(['A', 'B', 'C', 'A', 'B', 'C', 'C', 'B', 'A']),
}


# ============================================================================
# Parity tests
# ============================================================================

def test_dynamics_vectorized_matches_reference():
    """Vectorized response-time extraction is identical to the original loop."""
    for name, builder in FIXTURE_BUILDERS.items():
        eda = ChatEDA(builder())

        reference = _reference_dynamics(eda.df)
        actual = eda.analyze_conversation_dynamics()

        assert actual['response_times'] == reference['response_times'], (
            f'{name}: response_times differs'
        )
        assert actual['avg_response_time'] == reference['avg_response_time'], (
            f'{name}: avg_response_time differs'
        )
        assert actual['balance_ratio'] == reference['balance_ratio'], (
            f'{name}: balance_ratio differs'
        )


def test_single_message_avg_is_none():
    """Single-message chat: empty response_times, None avg (both paths agree)."""
    eda = ChatEDA(_build_df(['A']))
    result = eda.analyze_conversation_dynamics()

    assert result['response_times'] == []
    assert result['avg_response_time'] is None


def test_day_gap_switch_has_large_response_time():
    """A sender switch across a 24-hour gap is kept (original loop keeps it too)."""
    eda = ChatEDA(_build_day_gap_df())
    result = eda.analyze_conversation_dynamics()

    assert 24 * 60 in result['response_times']
    assert result['avg_response_time'] == (60 + 60 + 24 * 60 + 60) / 4


def test_summary_with_precomputed_equals_default():
    """Precomputed analyze_* results produce an exactly-equal summary dict."""
    df = _build_df(['A', 'B', 'A', 'B', 'C', 'A', 'B', 'C'])
    eda = ChatEDA(df)

    default_summary = eda.generate_comprehensive_summary()

    volume = eda.analyze_message_volume()
    dynamics = eda.analyze_conversation_dynamics()
    content = eda.analyze_content()
    precomputed_summary = eda.generate_comprehensive_summary(
        volume_analysis=volume,
        dynamics_analysis=dynamics,
        content_analysis=content,
    )

    assert precomputed_summary == default_summary
