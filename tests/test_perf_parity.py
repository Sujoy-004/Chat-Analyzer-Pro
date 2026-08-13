"""Performance-refactor parity tests for the relationship-health module.

Proves two pure-performance changes are behavior-identical:

- calculate_dominance_scores: the two `df.loc[i, ...]` Python loops (conversation
  endings + burst run-length encoding) are replaced with vectorized pandas.
  Every returned value must be bit-identical to the pre-refactor loops.
- analyze_relationship_health: the four DEAD gamification calls
  (friendship_index/streaks/milestones/emoji_personality) are gated behind
  include_gamification, while rolling_health is independently gated behind
  include_rolling_health. With both defaults True, existing callers are
  bit-identical; with include_gamification=False the four dead keys are absent
  and rolling_health is still computed.

These are FAST tests (no pytest.mark.slow).
"""

import math
import os
from datetime import timedelta

# Headless-first (Pitfall 7): relationship_health imports matplotlib.pyplot
# at module import; pin Agg BEFORE any pyplot import so figures are headless-safe.
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from chat_analyzer.analysis.relationship_health import (
    analyze_relationship_health,
    analyze_response_patterns,
    calculate_dominance_scores,
    calculate_initiator_ratio,
    calculate_relationship_health_score,
    calculate_rolling_health_score,
    identify_conversation_starters,
    logger,
)
from chat_analyzer.cli.adapters import _build_health_block
from chat_analyzer.ingest.ingestion import messages_to_dataframe

# ============================================================================
# Reference implementation: the FULL ORIGINAL calculate_dominance_scores,
# captured verbatim BEFORE the vectorization refactor. Only the tail is
# changed to also hand back the loop-modified df_copy (for column parity).
# ============================================================================

def _reference_dominance_scores(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Calculate conversation dominance patterns.
    
    Args:
        df: DataFrame with conversation data
        
    Returns:
        Dictionary with dominance analysis metrics
    """
    total_messages = len(df)
    
    # 1. Message count dominance
    message_counts = df['sender'].value_counts()
    if len(message_counts) >= 2:
        message_dominance_score = 1 - abs(message_counts.values[0] - message_counts.values[1]) / total_messages
    else:
        message_dominance_score = 0.0
    
    # 2. Message length dominance
    if 'message_length' in df.columns:
        length_distribution = df.groupby('sender')['message_length'].sum()
        total_chars = df['message_length'].sum()
        if len(length_distribution) >= 2 and total_chars > 0:
            length_dominance_score = 1 - abs(length_distribution.values[0] - length_distribution.values[1]) / total_chars
        else:
            length_dominance_score = 1.0
        
        avg_lengths = df.groupby('sender')['message_length'].mean()
    else:
        length_dominance_score = 1.0
        length_distribution = pd.Series()
        avg_lengths = pd.Series()
    
    # 3. Conversation control patterns
    conversation_endings = []
    for i in range(len(df)):
        # If next message is a conversation starter or we're at the end
        if i == len(df) - 1 or (i < len(df) - 1 and df.loc[i+1, 'is_conversation_starter']):
            conversation_endings.append(df.loc[i, 'sender'])
    
    ending_counts = pd.Series(conversation_endings).value_counts()
    if len(ending_counts) >= 2:
        control_balance = 1 - abs(ending_counts.values[0] - ending_counts.values[1]) / len(conversation_endings)
    else:
        control_balance = 0.0
    
    # 4. Message burst patterns
    df_copy = df.copy()
    df_copy['is_burst'] = False
    df_copy['burst_length'] = 1
    
    current_sender = None
    burst_length = 0
    
    for i in range(len(df_copy)):
        if df_copy.loc[i, 'sender'] == current_sender:
            burst_length += 1
            df_copy.loc[i, 'burst_length'] = burst_length
            if burst_length > 1:
                df_copy.loc[i, 'is_burst'] = True
                if i > 0 and df_copy.loc[i-1, 'burst_length'] == 1:
                    df_copy.loc[i-1, 'is_burst'] = True
        else:
            current_sender = df_copy.loc[i, 'sender']
            burst_length = 1
    
    burst_stats = df_copy.groupby('sender').agg({
        'is_burst': 'sum',
        'burst_length': ['max', 'mean']
    })
    
    # Composite dominance score
    composite_dominance = (message_dominance_score + length_dominance_score + control_balance) / 3
    
    # Interpretation
    if composite_dominance >= 0.9:
        interpretation = "Excellent balance - very equal participation"
    elif composite_dominance >= 0.8:
        interpretation = "Good balance - minor differences in participation"
    elif composite_dominance >= 0.7:
        interpretation = "Moderate balance - some dominance patterns visible"
    else:
        interpretation = "Imbalanced - clear dominance by one participant"
    
    results = {
        'message_count_balance': message_dominance_score,
        'message_length_balance': length_dominance_score,
        'conversation_control_balance': control_balance,
        'composite_dominance_score': composite_dominance,
        'interpretation': interpretation,
        'message_distribution': message_counts.to_dict(),
        'length_distribution': length_distribution.to_dict() if not length_distribution.empty else {},
        'avg_message_lengths': avg_lengths.to_dict() if not avg_lengths.empty else {},
        'conversation_enders': ending_counts.to_dict(),
        'burst_stats': burst_stats.to_dict() if not burst_stats.empty else {}
    }
    
    return results, df_copy


# ============================================================================
# Fixture builders
# ============================================================================

def _build_df(senders: list[str]) -> pd.DataFrame:
    """Build a fixture df (1-hour spaced messages) with conversation starters."""
    start = pd.Timestamp('2024-01-01 09:00:00')
    messages = [
        {
            'datetime': start + pd.Timedelta(hours=i),
            'sender': sender,
            'message': f'message {i} from {sender}',
        }
        for i, sender in enumerate(senders)
    ]
    df = messages_to_dataframe(messages)
    return identify_conversation_starters(df)


DOMINANCE_FIXTURES = {
    'alternating': ['Alice', 'Bob', 'Alice', 'Bob', 'Alice', 'Bob', 'Alice', 'Bob'],
    # 1-message run (Bob) immediately followed by a >=2 run (Carol, Carol) —
    # exercises the original :238-241 quirk (first row of a >=2 run marked).
    'quirk_one_then_two': ['Alice', 'Bob', 'Carol', 'Carol'],
    'runs_of_three': ['Alice', 'Alice', 'Alice', 'Bob', 'Alice', 'Bob', 'Bob', 'Bob'],
    'single_message': ['Alice'],
    'all_same_sender': ['Alice', 'Alice', 'Alice', 'Alice', 'Alice'],
    'mixed': ['Alice', 'Alice', 'Bob', 'Carol', 'Carol', 'Carol', 'Dana'],
}


# ============================================================================
# CHANGE B parity tests
# ============================================================================

def test_health_vectorized_matches_original():
    """Vectorized dominance scores are bit-identical to the original loops."""
    for name, senders in DOMINANCE_FIXTURES.items():
        df = _build_df(senders)

        reference_results, reference_copy = _reference_dominance_scores(df)
        actual_results = calculate_dominance_scores(df)

        for key in (
            'message_count_balance',
            'message_length_balance',
            'conversation_control_balance',
            'composite_dominance_score',
            'conversation_enders',
            'burst_stats',
        ):
            assert actual_results[key] == reference_results[key], (
                f'{name}: key {key!r} differs'
            )

        # Rebuild the burst columns the vectorized way and assert exact parity
        # with the original loop-modified frame.
        vector = df.copy()
        sender = vector['sender']
        run_id = (sender != sender.shift()).cumsum()
        vector['burst_length'] = sender.groupby(run_id).cumcount() + 1
        run_size = sender.groupby(run_id).transform('size')
        vector['is_burst'] = run_size >= 2

        assert vector['is_burst'].equals(reference_copy['is_burst']), (
            f'{name}: is_burst column differs'
        )
        assert vector['burst_length'].equals(reference_copy['burst_length']), (
            f'{name}: burst_length column differs'
        )


# ============================================================================
# CHANGE C / C2 contract tests
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


def test_gamification_skip_keeps_contract():
    """include_gamification=False drops the dead keys but keeps the core contract."""
    df = _rolling_fixture()
    res = analyze_relationship_health(df, include_gamification=False)

    for key in ('health_score', 'initiator_analysis', 'response_analysis', 'dominance_analysis'):
        assert key in res, key

    rolling = res['rolling_health']
    assert isinstance(rolling, pd.DataFrame)

    for dead_key in ('friendship_index', 'streaks', 'milestones', 'emoji_personality'):
        assert dead_key not in res, dead_key

    block = _build_health_block(res)
    assert isinstance(block['overall_score'], float)
    assert isinstance(block['grade'], str)


def test_default_keeps_all_keys():
    """Default call keeps every gamification + rolling key (bit-identical contract)."""
    df = _rolling_fixture()
    res = analyze_relationship_health(df)

    for key in ('friendship_index', 'streaks', 'milestones', 'rolling_health', 'emoji_personality'):
        assert key in res, key


def test_rolling_present_when_gamification_off():
    """rolling_health is still computed when gamification is skipped."""
    df = _rolling_fixture()
    res = analyze_relationship_health(df, include_gamification=False)

    rolling = res['rolling_health']
    assert isinstance(rolling, pd.DataFrame)
    assert not rolling.empty
    assert 'health_score' in rolling.columns


# ============================================================================
# identify_conversation_starters + analyze_response_patterns parity
# (vectorized so the per-window rolling_health loop stops being O(n) Python)
# ============================================================================

def _reference_identify_conversation_starters(
    df: pd.DataFrame, gap_threshold_minutes: int = 60
) -> pd.DataFrame:
    """The ORIGINAL pre-vectorization loop, captured verbatim."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    df['time_diff'] = df['datetime'].diff()
    df['time_diff_minutes'] = df['time_diff'].dt.total_seconds() / 60
    df['prev_sender'] = df['sender'].shift(1)
    df['is_conversation_starter'] = False

    df.loc[0, 'is_conversation_starter'] = True

    for i in range(1, len(df)):
        current_time = df.loc[i, 'datetime']
        prev_time = df.loc[i-1, 'datetime']
        time_gap_minutes = (current_time - prev_time).total_seconds() / 60
        different_day = current_time.date() != prev_time.date()

        if time_gap_minutes > gap_threshold_minutes or different_day:
            df.loc[i, 'is_conversation_starter'] = True

    return df


def _reference_response_patterns(df: pd.DataFrame) -> dict:
    """The ORIGINAL pre-vectorization iterrows loop + downstream stats."""
    response_df = df[df['is_conversation_starter'] == False].copy()

    response_analysis = []
    for i, row in response_df.iterrows():
        prev_sender = row['prev_sender']
        current_sender = row['sender']
        response_time = row['time_diff_minutes']

        if prev_sender != current_sender and pd.notna(response_time):
            response_analysis.append({
                'responder': current_sender,
                'responded_to': prev_sender,
                'response_time_minutes': response_time,
                'datetime': row['datetime']
            })

    if not response_analysis:
        return {'error': 'No valid responses found'}

    response_analysis_df = pd.DataFrame(response_analysis)

    response_stats = response_analysis_df.groupby('responder')['response_time_minutes'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)

    response_pairs = response_analysis_df.groupby(['responded_to', 'responder']).agg({
        'response_time_minutes': ['count', 'mean']
    }).round(2)

    overall_avg_response = response_analysis_df['response_time_minutes'].mean()

    if len(response_stats) >= 2:
        avg_times = response_stats['mean'].values
        response_balance = abs(avg_times[0] - avg_times[1])
        responsiveness_score = max(0, 1 - (overall_avg_response / 120))
        balance_score = max(0, 1 - (response_balance / 60))
    else:
        response_balance = 0
        responsiveness_score = max(0, 1 - (overall_avg_response / 120))
        balance_score = 1.0

    return {
        'response_stats': response_stats.to_dict(),
        'response_pairs': response_pairs.to_dict(),
        'overall_avg_response_minutes': overall_avg_response,
        'response_time_difference': response_balance,
        'responsiveness_score': responsiveness_score,
        'response_balance_score': balance_score,
        'total_responses_analyzed': len(response_analysis_df)
    }


def _starter_fixture() -> pd.DataFrame:
    """Same-day small gap, same-day >threshold gap, day-change, same-sender
    follow-ups and a day change — exercises every starter/response branch."""
    base = pd.Timestamp('2024-03-01 08:00:00')
    timestamps = [
        base,                                          # Alice (starter)
        base + pd.Timedelta(minutes=10),               # Bob responds
        base + pd.Timedelta(minutes=20),               # Bob again (excluded)
        base + pd.Timedelta(minutes=90),               # Bob gap>60 -> starter
        base + pd.Timedelta(minutes=95),               # Alice responds
        pd.Timestamp('2024-03-02 09:00:00'),           # Alice day change -> starter
        pd.Timestamp('2024-03-02 09:30:00'),           # Bob responds
    ]
    senders = ['Alice', 'Bob', 'Bob', 'Bob', 'Alice', 'Alice', 'Bob']
    messages = [
        {'datetime': t, 'sender': s, 'message': 'hi'}
        for t, s in zip(timestamps, senders)
    ]
    return messages_to_dataframe(messages)


def test_identify_vectorized_matches_original():
    """Vectorized conversation-starter detection is identical to the loop."""
    df = _starter_fixture()
    for threshold in (10, 60, 120):
        ref = _reference_identify_conversation_starters(df, threshold)
        act = identify_conversation_starters(df, threshold)
        for col in ('datetime', 'time_diff', 'time_diff_minutes', 'prev_sender', 'is_conversation_starter'):
            assert ref[col].equals(act[col]), f'threshold={threshold}: {col} differs'


def _response_equal(a, b) -> bool:
    """Deep equality that treats NaN as equal — response_stats carries NaN std
    values, and Python dict `==` would otherwise report nan != nan."""

    def _eq(x, y) -> bool:
        try:
            if isinstance(x, float) and isinstance(y, float):
                return x == y or (math.isnan(x) and math.isnan(y))
        except TypeError:
            pass
        return x == y

    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_response_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_response_equal(x, y) for x, y in zip(a, b))
    return _eq(a, b)


def test_response_patterns_vectorized_matches_original():
    """Vectorized response-pattern extraction is identical to the iterrows loop."""
    df = identify_conversation_starters(_starter_fixture())
    assert _response_equal(
        _reference_response_patterns(df), analyze_response_patterns(df)
    )


def test_single_row_identify():
    """Single-message chat: row 0 starter, all diffs NaN — identical to loop."""
    df = messages_to_dataframe([
        {'datetime': '2024-03-01 09:00:00', 'sender': 'Alice', 'message': 'hi'},
    ])
    ref = _reference_identify_conversation_starters(df)
    act = identify_conversation_starters(df)
    for col in ('is_conversation_starter', 'time_diff_minutes', 'prev_sender'):
        assert ref[col].equals(act[col]), f'{col} differs'


def test_nat_datetime_rows_marked_starter():
    """M2: NaT datetime rows are robustly marked as starters. The original loop
    CRASHED on them (NaT.date() -> AttributeError); the vectorized detector
    deliberately treats them as a change. This documents the deviation."""
    df = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-03-01 09:00:00', 'NaT', '2024-03-02 09:00:00']),
        'sender': ['Alice', 'Bob', 'Alice'],
        'message': ['hi', 'hey', 'yo'],
    })
    out = identify_conversation_starters(df)
    nat_idx = out.index[out['datetime'].isna()][0]
    assert bool(out.loc[nat_idx, 'is_conversation_starter']) is True


# ============================================================================
# calculate_rolling_health_score parity: per-date windowing built by
# concatenating pre-grouped date frames instead of full-df boolean masks.
# The window row SET per date is provably identical (df is datetime-sorted;
# groupby preserves within-group order; concat in ascending date order == the
# boolean-filtered slice), so health_score/grade/message_count must match.
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


def test_rolling_health_pre_grouped_matches_original():
    """Pre-grouped-date windowing is bit-identical to the boolean-mask loop."""
    df = _rolling_parity_fixture()
    for window_days, min_messages in ((7, 10), (14, 10), (7, 4)):
        reference = _reference_rolling_health_score(df, window_days, min_messages)
        new = calculate_rolling_health_score(df, window_days, min_messages)

        assert not reference.empty, 'fixture must produce at least one scored window'
        for col in ('date', 'health_score', 'grade', 'message_count'):
            assert reference[col].equals(new[col]), (
                f'window_days={window_days} min_messages={min_messages}: {col} differs'
            )
        assert reference.equals(new), (
            f'window_days={window_days} min_messages={min_messages}: full frame differs'
        )
