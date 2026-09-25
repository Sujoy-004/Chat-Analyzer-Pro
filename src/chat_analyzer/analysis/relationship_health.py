"""
Relationship Health Metrics Module - UPDATED with Day 14 Gamification

This module analyzes relationship health through conversation patterns including:
- Initiator ratio (who starts conversations)
- Response lag analysis (response times and patterns)  
- Dominance scores (message count, length, conversation control)
- Overall relationship health scoring
- Rolling health score tracking (Day 9)
- Friendship Index & Gamification (Day 14)
- Integration with visualization.py (Day 13)

"""

import logging
from datetime import datetime, timedelta
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logging (Anti-Pattern 4: never hijack global log config at import)
logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def identify_conversation_starters(df: pd.DataFrame, gap_threshold_minutes: int = 60) -> pd.DataFrame:
    """
    Identify conversation starters based on time gaps and date changes.
    
    Args:
        df: DataFrame with 'datetime' and 'sender' columns
        gap_threshold_minutes: Minutes gap to consider as new conversation start
        
    Returns:
        DataFrame with added 'is_conversation_starter' and 'time_diff_minutes' columns
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Calculate time differences
    df['time_diff'] = df['datetime'].diff()
    df['time_diff_minutes'] = df['time_diff'].dt.total_seconds() / 60
    df['prev_sender'] = df['sender'].shift(1)
    
    # Mark conversation starters based on gaps or date changes (vectorized).
    # A gap above the threshold OR a calendar-day change starts a new
    # conversation. Row 0 is always a starter: its NaN gap compares False,
    # but its date-change comparison against the shifted NaT is True.
    different_day = df['datetime'].dt.normalize() != df['datetime'].dt.normalize().shift()
    gap_ok = df['time_diff_minutes'] > gap_threshold_minutes
    df['is_conversation_starter'] = (gap_ok | different_day).fillna(False)
    
    # First message is always a conversation starter
    df.loc[0, 'is_conversation_starter'] = True
    
    return df


def calculate_initiator_ratio(df: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate who initiates conversations more often.
    
    Args:
        df: DataFrame with conversation starter information
        
    Returns:
        Dictionary with initiator statistics and balance scores
    """
    initiator_counts = df[df['is_conversation_starter']==True]['sender'].value_counts()
    total_conversations = df['is_conversation_starter'].sum()
    
    if total_conversations == 0:
        return {'error': 'No conversation starters found'}
    
    # Calculate ratios
    ratios = {}
    for sender, count in initiator_counts.items():
        ratios[f'{sender}_initiation_ratio'] = count / total_conversations
    
    # Balance score (1.0 = perfectly balanced)
    if len(initiator_counts) >= 2:
        balance_score = 1 - abs(initiator_counts.values[0] - initiator_counts.values[1]) / total_conversations
    else:
        balance_score = 0.0  # Only one person initiates
    
    # Interpretation
    if balance_score >= 0.8:
        interpretation = "Excellent balance - both participants initiate conversations equally"
    elif balance_score >= 0.6:
        interpretation = "Good balance - slight preference but healthy"
    elif balance_score >= 0.4:
        interpretation = "Moderate imbalance - one person initiates more often"
    else:
        interpretation = "High imbalance - one person dominates conversation initiation"
    
    return {
        'initiator_counts': initiator_counts.to_dict(),
        'total_conversations': total_conversations,
        'balance_score': balance_score,
        'interpretation': interpretation,
        **ratios
    }


def analyze_response_patterns(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze response lag patterns and responsiveness.
    
    Args:
        df: DataFrame with conversation data
        
    Returns:
        Dictionary with response analysis metrics
    """
    # Get valid responses (excluding conversation starters and same-sender continuations)
    response_df = df[df['is_conversation_starter'] == False].copy()
    
    # Valid response: different sender with valid response time (vectorized).
    # Preserves the exact row order and field values of the old iterrows loop.
    valid = (
        (response_df['prev_sender'] != response_df['sender'])
        & response_df['time_diff_minutes'].notna()
    )
    response_analysis = response_df[valid].rename(columns={
        'sender': 'responder',
        'prev_sender': 'responded_to',
        'time_diff_minutes': 'response_time_minutes',
    })[['responder', 'responded_to', 'response_time_minutes', 'datetime']].to_dict('records')
    
    if not response_analysis:
        return {'error': 'No valid responses found'}
    
    response_analysis_df = pd.DataFrame(response_analysis)
    
    # Calculate statistics
    response_stats = response_analysis_df.groupby('responder')['response_time_minutes'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)
    
    # Calculate response patterns (who responds to whom)
    response_pairs = response_analysis_df.groupby(['responded_to', 'responder']).agg({
        'response_time_minutes': ['count', 'mean']
    }).round(2)
    
    # Overall metrics
    overall_avg_response = response_analysis_df['response_time_minutes'].mean()
    
    # Balance scores
    if len(response_stats) >= 2:
        avg_times = response_stats['mean'].values
        response_balance = abs(avg_times[0] - avg_times[1])
        responsiveness_score = max(0, 1 - (overall_avg_response / 120))  # 120min = very slow
        balance_score = max(0, 1 - (response_balance / 60))  # 60min difference = imbalanced
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


def calculate_dominance_scores(df: pd.DataFrame) -> dict[str, Any]:
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
    next_starter = df['is_conversation_starter'].shift(-1, fill_value=False)
    is_ending = next_starter | (np.arange(len(df)) == len(df) - 1)
    conversation_endings = df.loc[is_ending, 'sender'].tolist()
    
    ending_counts = pd.Series(conversation_endings).value_counts()
    if len(ending_counts) >= 2:
        control_balance = 1 - abs(ending_counts.values[0] - ending_counts.values[1]) / len(conversation_endings)
    else:
        control_balance = 0.0
    
    # 4. Message burst patterns
    df_copy = df.copy()
    sender = df_copy['sender']
    run_id = (sender != sender.shift()).cumsum()
    df_copy['burst_length'] = sender.groupby(run_id).cumcount() + 1
    run_size = sender.groupby(run_id).transform('size')
    df_copy['is_burst'] = run_size >= 2
    
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
    
    return {
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


def calculate_relationship_health_score(
    initiator_metrics: dict[str, Any],
    response_metrics: dict[str, Any], 
    dominance_metrics: dict[str, Any],
    weights: dict[str, float] | None = None
) -> dict[str, Any]:
    """
    Calculate overall relationship health score from component metrics.
    
    Args:
        initiator_metrics: Results from calculate_initiator_ratio()
        response_metrics: Results from analyze_response_patterns() 
        dominance_metrics: Results from calculate_dominance_scores()
        weights: Custom weights for components (default: balanced weighting)
        
    Returns:
        Dictionary with overall health score and assessment
    """
    if weights is None:
        weights = {
            'initiation': 0.25,      # 25% - who starts conversations
            'responsiveness': 0.35,   # 35% - how they respond (most important)
            'balance': 0.20,         # 20% - response time balance  
            'dominance': 0.20        # 20% - conversation control
        }
    
    # Extract component scores (handle potential errors)
    initiation_score = initiator_metrics.get('balance_score', 0)
    responsiveness_score = response_metrics.get('responsiveness_score', 0)
    response_balance_score = response_metrics.get('response_balance_score', 1)
    dominance_score = dominance_metrics.get('composite_dominance_score', 0)
    
    # Calculate weighted overall score
    overall_score = (
        weights['initiation'] * initiation_score +
        weights['responsiveness'] * responsiveness_score +
        weights['balance'] * response_balance_score +
        weights['dominance'] * dominance_score
    )
    
    # Grade and interpretation
    if overall_score >= 0.90:
        grade = "EXCELLENT"
        description = "Highly balanced and healthy relationship with great communication patterns"
    elif overall_score >= 0.80:
        grade = "VERY GOOD"
        description = "Strong relationship with good communication balance and responsiveness"
    elif overall_score >= 0.70:
        grade = "GOOD"
        description = "Healthy relationship with minor areas for improvement"
    elif overall_score >= 0.60:
        grade = "FAIR"
        description = "Decent relationship but some imbalances in communication patterns"
    else:
        grade = "NEEDS IMPROVEMENT"
        description = "Significant imbalances that may affect relationship health"
    
    # Identify strengths and areas for improvement
    strengths = []
    areas_for_improvement = []
    
    if initiation_score >= 0.8:
        strengths.append("✅ Balanced conversation initiation")
    else:
        areas_for_improvement.append("⚠️ One person initiates more conversations")
    
    if responsiveness_score >= 0.8:
        strengths.append("✅ Both are very responsive")
    else:
        areas_for_improvement.append("⚠️ Slower response times")
    
    if response_balance_score >= 0.8:
        strengths.append("✅ Similar response time patterns")
    else:
        areas_for_improvement.append("⚠️ Significant difference in response speeds")
    
    if dominance_score >= 0.8:
        strengths.append("✅ Excellent participation balance")
    else:
        areas_for_improvement.append("⚠️ Some dominance in conversation patterns")
    
    return {
        'overall_health_score': overall_score,
        'grade': grade,
        'description': description,
        'component_scores': {
            'initiation_balance': initiation_score,
            'responsiveness': responsiveness_score,
            'response_balance': response_balance_score,
            'dominance_balance': dominance_score
        },
        'weights_used': weights,
        'strengths': strengths,
        'areas_for_improvement': areas_for_improvement
    }


# ============================================================================
# DAY 9: ROLLING HEALTH SCORE TRACKER
# ============================================================================

# Production always takes the incremental sliding-window path below: each window
# is scored in ~1ms from columns computed ONCE over the full frame, so a process
# pool adds only pickling/IPC cost. The multiprocessing branch is retained as
# the reference/verification path (it runs the unmodified _rolling_window_score
# recompute) and is reached only when tests force the threshold low — e.g.
# test_rolling_health_parallel_matches_sequential sets it to 1 to prove the
# sequential incremental output is identical to the parallel recompute.
_ROLLING_PARALLEL_MIN_DATES = 2**62

# Workers are modest (2 P-cores + 8 E-cores on the target): more churns CPU and
# memory without a faster fan-out.
_ROLLING_PARALLEL_WORKERS = 4


def _rolling_window_score(args) -> dict | None:
    """Score ONE rolling-health window (reference/worker entry for the forced
    parallel verification path). Full per-window recompute, bit-identical to the
    boolean-mask original — this is the parity gold standard `_rolling_window_fast`
    must match."""
    window_df, current_date, min_messages = args
    if len(window_df) < min_messages:
        return None
    try:
        window_df = identify_conversation_starters(window_df.reset_index(drop=True))
        initiator_metrics = calculate_initiator_ratio(window_df)
        response_metrics = analyze_response_patterns(window_df)
        dominance_metrics = calculate_dominance_scores(window_df)
        health_score = calculate_relationship_health_score(
            initiator_metrics, response_metrics, dominance_metrics
        )
        return {
            'date': current_date,
            'health_score': health_score['overall_health_score'],
            'grade': health_score['grade'],
            'message_count': len(window_df),
        }
    except Exception as e:  # noqa: BLE001 - a failing window is skipped, never allowed to crash the series
        return {'error': current_date, 'message': str(e)}


def _collect_health_scored(scored: list[dict | None]) -> pd.DataFrame:
    """Build the output DataFrame from scored window dicts, skipping below
    threshold (None) and error windows exactly like the original code."""
    health_scores = []
    for r in scored:
        if r is None:
            continue
        if 'error' in r:
            logger.warning(f"Failed to calculate health score for {r['error']}: {r['message']}")
            continue
        health_scores.append(r)
    return pd.DataFrame(health_scores)


def _date_row_spans(date_series: pd.Series) -> dict:
    """Map each distinct date to its trailing-exclusive [start, stop) row offsets
    within a date-sorted frame. Trailing NaT rows (unsortable, never inside any
    window) fall outside the last span."""
    spans: dict = {}
    limit = len(date_series)
    start = None
    prev = None
    for i in range(limit):
        d = date_series.iloc[i]
        if pd.isna(d):
            limit = i
            break
        if d != prev:
            if prev is not None:
                spans[prev] = (start, i)
            start = i
            prev = d
    if prev is not None:
        spans[prev] = (start, limit)
    return spans


def _rolling_window_fast(window_df: pd.DataFrame, current_date) -> dict | None:
    """Score ONE rolling-health window from columns already computed once over
    the full frame (see calculate_rolling_health_score). Mirrors
    _rolling_window_score's metric math bit-for-bit while skipping its redundant
    per-window identify_conversation_starters pass and the dead median/std/pair
    aggregation the original response block recomputed.

    The window frame carries is_conversation_starter, prev_sender and
    time_diff_minutes from the single full-frame identify pass. Those are
    WINDOW-INVARIANT: every window's first row is the first message of its oldest
    date, which the day-change rule already marks as a starter (the global row 0
    is force-marked), so the window-local reset could only re-force a flag that
    is already True; every non-starter row's diff is against an in-window
    predecessor and therefore identical to the full-frame value. Every live
    value below is produced by the same pandas routine over the same row set as
    the reference, so output is bit-identical.
    """
    try:
        n = len(window_df)
        starters = window_df['is_conversation_starter']

        # calculate_initiator_ratio -> balance_score
        starter_counts = window_df.loc[starters, 'sender'].value_counts()
        total_conversations = starters.sum()
        if total_conversations == 0:
            initiation_score = 0.0
        elif len(starter_counts) >= 2:
            initiation_score = 1 - abs(starter_counts.values[0] - starter_counts.values[1]) / total_conversations
        else:
            initiation_score = 0.0

        # analyze_response_patterns -> overall mean + rounded per-responder means
        valid_response = (
            (~starters)
            & (window_df['prev_sender'] != window_df['sender'])
            & window_df['time_diff_minutes'].notna()
        )
        response_times = window_df.loc[valid_response, 'time_diff_minutes']
        if len(response_times) == 0:
            responsiveness_score = 0.0
            response_balance_score = 1.0
        else:
            overall_avg_response = response_times.mean()
            response_means = (
                window_df.loc[valid_response, ['sender', 'time_diff_minutes']]
                .groupby('sender')['time_diff_minutes']
                .mean()
                .round(2)
            )
            if len(response_means) >= 2:
                avg_times = response_means.values  # alphabetical sender order (groupby sort=True)
                response_balance = abs(avg_times[0] - avg_times[1])
                responsiveness_score = max(0, 1 - (overall_avg_response / 120))
                response_balance_score = max(0, 1 - (response_balance / 60))
            else:
                response_balance = 0
                responsiveness_score = max(0, 1 - (overall_avg_response / 120))
                response_balance_score = 1.0

        # calculate_dominance_scores -> composite_dominance_score
        message_counts = window_df['sender'].value_counts()
        if len(message_counts) >= 2:
            message_dominance_score = 1 - abs(message_counts.values[0] - message_counts.values[1]) / n
        else:
            message_dominance_score = 0.0

        if 'message_length' in window_df.columns:
            length_distribution = window_df.groupby('sender')['message_length'].sum()
            total_chars = window_df['message_length'].sum()
            if len(length_distribution) >= 2 and total_chars > 0:
                length_dominance_score = 1 - abs(length_distribution.values[0] - length_distribution.values[1]) / total_chars
            else:
                length_dominance_score = 1.0
        else:
            length_dominance_score = 1.0

        is_ending = starters.shift(-1, fill_value=False) | (np.arange(n) == n - 1)
        conversation_endings = window_df.loc[is_ending, 'sender'].tolist()
        ending_counts = pd.Series(conversation_endings).value_counts()
        if len(ending_counts) >= 2:
            control_balance = 1 - abs(ending_counts.values[0] - ending_counts.values[1]) / len(conversation_endings)
        else:
            control_balance = 0.0

        composite_dominance_score = (message_dominance_score + length_dominance_score + control_balance) / 3

        # calculate_relationship_health_score with the default balanced weights
        overall_score = (
            0.25 * initiation_score
            + 0.35 * responsiveness_score
            + 0.20 * response_balance_score
            + 0.20 * composite_dominance_score
        )
        if overall_score >= 0.90:
            grade = "EXCELLENT"
        elif overall_score >= 0.80:
            grade = "VERY GOOD"
        elif overall_score >= 0.70:
            grade = "GOOD"
        elif overall_score >= 0.60:
            grade = "FAIR"
        else:
            grade = "NEEDS IMPROVEMENT"

        return {
            'date': current_date,
            'health_score': overall_score,
            'grade': grade,
            'message_count': n,
        }
    except Exception as e:  # noqa: BLE001 - a failing window is skipped, never allowed to crash the series
        return {'error': current_date, 'message': str(e)}


def calculate_rolling_health_score(
    df: pd.DataFrame,
    window_days: int = 7,
    min_messages: int = 10
) -> pd.DataFrame:
    """
    Calculate rolling relationship health score over time.
    
    Incremental sliding-window: one identify_conversation_starters pass over the
    full frame replaces the per-window passes (its starter/diff/prev columns are
    window-invariant), and each window is then a positional slice of that
    precomputed frame scored by _rolling_window_fast. The O(dates^2) concat
    window build, per-window re-sort/re-derivation, and IPC are gone.

    Args:
        df: Prepared DataFrame with conversation metrics
        window_days: Rolling window size in days
        min_messages: Minimum messages required for calculation
        
    Returns:
        DataFrame with date and rolling health scores
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    
    # Group by date (NaT datetimes are inert — they sort ito no window and only
    # crash the original sorted(date.unique()) call).
    df['date'] = df['datetime'].dt.date
    dates = sorted(d for d in df['date'].unique() if pd.notna(d))
    if not dates:
        return pd.DataFrame([])

    # Forced-reference parallel path (see _ROLLING_PARALLEL_MIN_DATES): reruns the
    # ORIGINAL per-window recompute in a process pool. Never taken at the default
    # threshold; exercised by the sequential == parallel parity tests.
    if len(dates) >= _ROLLING_PARALLEL_MIN_DATES:
        try:
            by_date = {d: g.reset_index(drop=True) for d, g in df.groupby('date')}
            window_args = []
            for current_date in dates:
                window_start = current_date - timedelta(days=window_days)
                window_df = pd.concat([by_date[d] for d in dates if window_start <= d <= current_date])
                window_args.append((window_df, current_date, min_messages))

            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            max_workers = min(_ROLLING_PARALLEL_WORKERS, mp.cpu_count() or 1)
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                scored = list(pool.map(_rolling_window_score, window_args, chunksize=16))
            return _collect_health_scored(scored)
        except Exception:
            logger.exception("parallel rolling-health failed; using sequential")

    # Incremental sequential path (production default).
    full = identify_conversation_starters(df.reset_index(drop=True))
    spans = _date_row_spans(full['date'])

    health_scores = []
    lo_idx = 0
    n_dates = len(dates)
    for current_date in dates:
        window_start = current_date - timedelta(days=window_days)
        while lo_idx < n_dates and dates[lo_idx] < window_start:
            lo_idx += 1
        first_date = dates[lo_idx] if lo_idx < n_dates else current_date
        lo, hi = spans[first_date][0], spans[current_date][1]
        if hi - lo < min_messages:
            continue
        r = _rolling_window_fast(full.iloc[lo:hi].reset_index(drop=True), current_date)
        if r is None:
            continue
        if 'error' in r:
            logger.warning(f"Failed to calculate health score for {r['error']}: {r['message']}")
            continue
        health_scores.append(r)

    return pd.DataFrame(health_scores)


# ============================================================================
# DAY 14: GAMIFICATION FEATURES - FRIENDSHIP INDEX & EXTRAS
# ============================================================================

def calculate_friendship_index(df: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate comprehensive Friendship Index combining multiple metrics.
    
    Args:
        df: Prepared DataFrame with all conversation data
        
    Returns:
        Dictionary with Friendship Index and detailed breakdown
    """
    df = df.copy()
    
    # Ensure we have prepared data
    if 'is_conversation_starter' not in df.columns:
        df = identify_conversation_starters(df)
    
    # Component 1: Communication Frequency (0-25 points)
    total_messages = len(df)
    days_span = (df['datetime'].max() - df['datetime'].min()).days + 1
    messages_per_day = total_messages / days_span if days_span > 0 else 0
    
    frequency_score = min(25, (messages_per_day / 50) * 25)  # Cap at 50 messages/day for max score
    
    # Component 2: Conversation Balance (0-25 points)
    initiator_metrics = calculate_initiator_ratio(df)
    balance_score = initiator_metrics.get('balance_score', 0) * 25
    
    # Component 3: Responsiveness (0-20 points)
    response_metrics = analyze_response_patterns(df)
    responsiveness_score = response_metrics.get('responsiveness_score', 0) * 20
    
    # Component 4: Engagement Quality (0-15 points)
    if 'message_length' in df.columns:
        avg_length = df['message_length'].mean()
        engagement_score = min(15, (avg_length / 100) * 15)  # Cap at 100 chars
    else:
        engagement_score = 0
    
    # Component 5: Consistency (0-15 points) - Streak-based
    streaks = detect_conversation_streaks(df)
    consistency_score = min(15, (streaks['longest_streak'] / 30) * 15)  # Cap at 30 days
    
    # Total Friendship Index (0-100)
    friendship_index = frequency_score + balance_score + responsiveness_score + engagement_score + consistency_score
    
    # Tier system
    if friendship_index >= 90:
        tier = "BEST FRIENDS 👑"
        description = "Exceptional friendship with outstanding communication"
    elif friendship_index >= 75:
        tier = "CLOSE FRIENDS 💎"
        description = "Strong friendship with great interaction patterns"
    elif friendship_index >= 60:
        tier = "GOOD FRIENDS ⭐"
        description = "Solid friendship with regular communication"
    elif friendship_index >= 45:
        tier = "FRIENDS 🙂"
        description = "Developing friendship with room to grow"
    else:
        tier = "ACQUAINTANCES 👋"
        description = "Early stage or infrequent communication"
    
    return {
        'friendship_index': round(friendship_index, 2),
        'tier': tier,
        'description': description,
        'breakdown': {
            'frequency': round(frequency_score, 2),
            'balance': round(balance_score, 2),
            'responsiveness': round(responsiveness_score, 2),
            'engagement': round(engagement_score, 2),
            'consistency': round(consistency_score, 2)
        },
        'metrics': {
            'messages_per_day': round(messages_per_day, 2),
            'total_days': days_span,
            'total_messages': total_messages
        }
    }


def detect_conversation_streaks(df: pd.DataFrame) -> dict[str, Any]:
    """
    Detect conversation streaks (consecutive days with messages).
    
    Args:
        df: DataFrame with datetime column
        
    Returns:
        Dictionary with streak statistics
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    
    # Get unique conversation days
    conversation_days = sorted(df['date'].unique())
    
    if not conversation_days:
        return {
            'current_streak': 0,
            'longest_streak': 0,
            'total_active_days': 0,
            'streak_history': []
        }
    
    # Calculate streaks
    streaks = []
    current_streak = 1
    longest_streak = 1
    
    for i in range(1, len(conversation_days)):
        days_diff = (conversation_days[i] - conversation_days[i-1]).days
        
        if days_diff == 1:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            if current_streak > 1:
                streaks.append(current_streak)
            current_streak = 1
    
    # Add final streak
    if current_streak > 1:
        streaks.append(current_streak)
    
    # Check if current streak is active (last message within 24 hours)
    last_message_date = conversation_days[-1]
    today = datetime.now().date()  # noqa: DTZ005 - streak "today" is local-calendar semantics, deliberately naive
    days_since_last = (today - last_message_date).days
    
    active_streak = current_streak if days_since_last <= 1 else 0
    
    return {
        'current_streak': active_streak,
        'longest_streak': longest_streak,
        'total_active_days': len(conversation_days),
        'streak_history': streaks,
        'days_since_last_message': days_since_last
    }


def analyze_emoji_personality(df: pd.DataFrame, message_col: str = 'message') -> dict[str, Any]:
    """
    Analyze emoji usage patterns to determine communication personality.
    
    Args:
        df: DataFrame with message column
        message_col: Name of message column
        
    Returns:
        Dictionary with emoji personality analysis per sender
    """
    import re
    
    emoji_pattern = re.compile("["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    emoji_categories = {
        'positive': ['😊', '😄', '😃', '😁', '🙂', '😍', '🥰', '😘', '❤️', '💕', '👍', '✨', '🎉'],
        'negative': ['😢', '😭', '😔', '😞', '😟', '😕', '☹️', '😠', '😡', '💔', '👎'],
        'neutral': ['😐', '😑', '😶', '🤔', '🙄', '😬'],
        'excited': ['🤩', '😆', '🎊', '🎈', '🔥', '💪', '🚀', '⚡'],
        'laughing': ['😂', '🤣', '😹', 'LOL']
    }
    
    personality_analysis = {}
    
    for sender in df['sender'].unique():
        sender_df = df[df['sender'] == sender]
        
        all_emojis = []
        for msg in sender_df[message_col].dropna():
            emojis = emoji_pattern.findall(str(msg))
            all_emojis.extend(emojis)
        
        if not all_emojis:
            personality_analysis[sender] = {
                'personality_type': 'Text-focused',
                'emoji_usage': 'Low',
                'top_emojis': [],
                'traits': ['Prefers words over emojis']
            }
            continue
        
        # Count emojis
        emoji_counts = pd.Series(all_emojis).value_counts()
        total_emojis = len(all_emojis)
        total_messages = len(sender_df)
        emoji_rate = total_emojis / total_messages
        
        # Categorize emojis
        category_scores = {cat: 0 for cat in emoji_categories}
        for emoji in all_emojis:
            for category, emoji_list in emoji_categories.items():
                if emoji in emoji_list:
                    category_scores[category] += 1
        
        # Determine personality
        dominant_category = max(category_scores, key=category_scores.get)
        
        personality_types = {
            'positive': ('Optimist 🌟', ['Cheerful', 'Upbeat', 'Encouraging']),
            'excited': ('Enthusiast 🚀', ['Energetic', 'Passionate', 'Dynamic']),
            'laughing': ('Comedian 😂', ['Humorous', 'Fun-loving', 'Light-hearted']),
            'negative': ('Emotional 💭', ['Expressive', 'Sensitive', 'Genuine']),
            'neutral': ('Balanced ⚖️', ['Moderate', 'Thoughtful', 'Reserved'])
        }
        
        personality_type, traits = personality_types.get(dominant_category, ('Expressive', ['Unique', 'Creative']))
        
        # Usage level
        if emoji_rate >= 1.5:
            usage_level = 'Very High'
        elif emoji_rate >= 0.8:
            usage_level = 'High'
        elif emoji_rate >= 0.3:
            usage_level = 'Moderate'
        else:
            usage_level = 'Low'
        
        personality_analysis[sender] = {
            'personality_type': personality_type,
            'emoji_usage': usage_level,
            'emoji_per_message': round(emoji_rate, 2),
            'total_emojis': total_emojis,
            'top_emojis': emoji_counts.head(5).to_dict(),
            'category_breakdown': category_scores,
            'traits': traits
        }
    
    return personality_analysis


def detect_milestones(df: pd.DataFrame) -> dict[str, Any]:
    """
    Detect conversation milestones and achievements.
    
    Args:
        df: DataFrame with conversation data
        
    Returns:
        Dictionary with milestone achievements
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    total_messages = len(df)
    days_span = (df['datetime'].max() - df['datetime'].min()).days + 1
    
    # Define milestones
    message_milestones = [100, 500, 1000, 5000, 10000, 25000, 50000]
    day_milestones = [7, 30, 100, 365, 730]
    
    achievements = []
    
    # Message count milestones
    for milestone in message_milestones:
        if total_messages >= milestone:
            achievements.append({
                'type': 'messages',
                'milestone': f'{milestone:,} Messages',
                'icon': '💬',
                'achieved': True,
                'date': df.iloc[milestone-1]['datetime'] if milestone <= total_messages else None
            })
    
    # Days active milestones
    for milestone in day_milestones:
        if days_span >= milestone:
            achievements.append({
                'type': 'duration',
                'milestone': f'{milestone} Days',
                'icon': '📅',
                'achieved': True,
                'date': df['datetime'].min() + timedelta(days=milestone)
            })
    
    # Special achievements
    streak_data = detect_conversation_streaks(df)
    if streak_data['longest_streak'] >= 7:
        achievements.append({
            'type': 'streak',
            'milestone': f'{streak_data["longest_streak"]}-Day Streak',
            'icon': '🔥',
            'achieved': True,
            'date': None
        })
    
    # Late night chatter (messages after 11 PM)
    late_night_msgs = df[df['datetime'].dt.hour >= 23]
    if len(late_night_msgs) > 50:
        achievements.append({
            'type': 'special',
            'milestone': 'Night Owl',
            'icon': '🦉',
            'achieved': True,
            'date': None
        })
    
    # Early bird (messages before 6 AM)
    early_msgs = df[df['datetime'].dt.hour < 6]
    if len(early_msgs) > 50:
        achievements.append({
            'type': 'special',
            'milestone': 'Early Bird',
            'icon': '🐦',
            'achieved': True,
            'date': None
        })
    
    # Weekend warriors
    df['day_of_week'] = df['datetime'].dt.dayofweek
    weekend_msgs = df[df['day_of_week'].isin([5, 6])]
    if len(weekend_msgs) / total_messages > 0.3:
        achievements.append({
            'type': 'special',
            'milestone': 'Weekend Warrior',
            'icon': '🎮',
            'achieved': True,
            'date': None
        })
    
    return {
        'total_achievements': len(achievements),
        'achievements': achievements,
        'progress': {
            'total_messages': total_messages,
            'days_active': days_span,
            'next_message_milestone': next((m for m in message_milestones if m > total_messages), None),
            'next_day_milestone': next((m for m in day_milestones if m > days_span), None)
        }
    }


# ============================================================================
# VISUALIZATION INTEGRATION (Day 13)
# ============================================================================

def plot_relationship_health_dashboard_enhanced(
    analysis_results: dict[str, Any],
    figsize: tuple[int, int] = (20, 14),
    use_viz_module: bool = True
) -> None:
    """
    Create enhanced relationship health visualization dashboard.
    Uses the new visualization.py module if available.
    
    Args:
        analysis_results: Results from analyze_relationship_health()
        figsize: Figure size tuple
        use_viz_module: Whether to use visualization.py ChatVisualizer
    """
    if use_viz_module:
        try:
            # Use prepared data from analysis
            df = analysis_results.get('prepared_data')
            if df is not None:
                # Create multi-panel dashboard
                fig = plt.figure(figsize=figsize)
                gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
                
                # Add timestamp column for visualization module
                if 'datetime' in df.columns:
                    df['timestamp'] = df['datetime']
                
                # 1. Health Score Timeline (if rolling data available)
                ax1 = fig.add_subplot(gs[0, :])
                rolling_health = calculate_rolling_health_score(df, window_days=7)
                if not rolling_health.empty:
                    rolling_health['date'] = pd.to_datetime(rolling_health['date'])
                    ax1.plot(rolling_health['date'], rolling_health['health_score'],
                            linewidth=3, marker='o', color='#667eea')
                    ax1.fill_between(rolling_health['date'], rolling_health['health_score'],
                                    alpha=0.3, color='#667eea')
                    ax1.set_title('Relationship Health Trend', fontsize=14, fontweight='bold')
                    ax1.set_ylabel('Health Score')
                    ax1.set_ylim(0, 1)
                    ax1.grid(True, alpha=0.3)
                    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
                
                # 2. Friendship Index Gauge
                ax2 = fig.add_subplot(gs[1, 0])
                friendship_data = calculate_friendship_index(df)
                _plot_friendship_gauge(ax2, friendship_data)
                
                # 3. Streak Visualization
                ax3 = fig.add_subplot(gs[1, 1])
                streak_data = detect_conversation_streaks(df)
                _plot_streak_display(ax3, streak_data)
                
                # 4. Emoji Personality
                ax4 = fig.add_subplot(gs[1, 2])
                if 'message' in df.columns:
                    emoji_data = analyze_emoji_personality(df)
                    _plot_emoji_personality(ax4, emoji_data)
                
                # 5-6. Component Scores
                ax5 = fig.add_subplot(gs[2, :2])
                _plot_health_components(ax5, analysis_results['health_score'])
                
                # 7. Achievements
                ax6 = fig.add_subplot(gs[2, 2])
                milestones = detect_milestones(df)
                _plot_achievements(ax6, milestones)
                
                fig.suptitle('Comprehensive Relationship Analysis Dashboard',
                           fontsize=16, fontweight='bold', y=0.995)
                
                plt.tight_layout()
                plt.show()
                return
        except ImportError:
            logger.warning("visualization.py not found, using fallback visualization")
    
    # Fallback to original dashboard
    _plot_original_dashboard(analysis_results, figsize)


def _plot_friendship_gauge(ax, friendship_data: dict[str, Any]) -> None:
    """Plot friendship index as a gauge."""
    score = friendship_data['friendship_index']
    
    # Create gauge
    colors = ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71', '#27AE60']
    ranges = [0, 45, 60, 75, 90, 100]
    
    for i in range(len(colors)):
        start = ranges[i] / 100 * np.pi
        end = ranges[i+1] / 100 * np.pi
        angles = np.linspace(start, end, 20)
        x = np.cos(angles)
        y = np.sin(angles)
        ax.fill_between(x, 0, y, color=colors[i], alpha=0.8)
    
    # Needle
    needle_angle = score / 100 * np.pi
    ax.plot([0, 0.8*np.cos(needle_angle)], [0, 0.8*np.sin(needle_angle)],
           'k-', linewidth=4)
    ax.plot(0.8*np.cos(needle_angle), 0.8*np.sin(needle_angle), 'ko', markersize=8)
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Friendship Index\n{friendship_data["tier"]}',
                fontweight='bold', fontsize=11)
    ax.text(0, -0.15, f'{score:.0f}/100', ha='center', fontsize=16, fontweight='bold')


def _plot_streak_display(ax, streak_data: dict[str, Any]) -> None:
    """Plot streak information."""
    ax.axis('off')
    
    current = streak_data['current_streak']
    longest = streak_data['longest_streak']
    
    info_text = f"🔥 Current Streak\n{current} days\n\n"
    info_text += f"🏆 Longest Streak\n{longest} days\n\n"
    info_text += f"📅 Active Days\n{streak_data['total_active_days']}"
    
    ax.text(0.5, 0.5, info_text, transform=ax.transAxes,
           fontsize=12, ha='center', va='center',
           bbox={'boxstyle': 'round', 'facecolor': '#fff3cd', 'alpha': 0.8})
    ax.set_title('Conversation Streaks', fontweight='bold', fontsize=11)


def _plot_emoji_personality(ax, emoji_data: dict[str, Any]) -> None:
    """Plot emoji personality analysis."""
    ax.axis('off')
    
    text_parts = []
    for sender, data in emoji_data.items():
        text_parts.append(f"{sender}:")
        text_parts.append(f"{data['personality_type']}")
        text_parts.append(f"Usage: {data['emoji_usage']}")
        text_parts.append("")
    
    text = '\n'.join(text_parts)
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
           fontsize=10, ha='center', va='center',
           bbox={'boxstyle': 'round', 'facecolor': '#e7f3ff', 'alpha': 0.8})
    ax.set_title('Emoji Personalities', fontweight='bold', fontsize=11)


def _plot_health_components(ax, health_score: dict[str, Any]) -> None:
    """Plot health score components as bar chart."""
    components = health_score['component_scores']
    names = ['Initiation', 'Responsiveness', 'Balance', 'Dominance']
    values = list(components.values())
    
    colors = ['#667eea' if v >= 0.7 else '#ffc107' if v >= 0.5 else '#f44336' for v in values]
    bars = ax.barh(names, values, color=colors, alpha=0.8)
    
    for bar, value in zip(bars, values):
        ax.text(value + 0.02, bar.get_y() + bar.get_height()/2,
               f'{value:.2f}', va='center', fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('Score')
    ax.set_title('Health Components', fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')


def _plot_achievements(ax, milestones: dict[str, Any]) -> None:
    """Plot achievement badges."""
    ax.axis('off')
    
    achievements = milestones['achievements']
    recent = achievements[-5:] if len(achievements) > 5 else achievements
    
    text = f"🏆 Achievements ({milestones['total_achievements']})\n\n"
    for ach in recent:
        text += f"{ach['icon']} {ach['milestone']}\n"
    
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
           fontsize=10, ha='center', va='center',
           bbox={'boxstyle': 'round', 'facecolor': '#d4edda', 'alpha': 0.8})
    ax.set_title('Latest Achievements', fontweight='bold', fontsize=11)


def _plot_original_dashboard(analysis_results: dict[str, Any], figsize: tuple[int, int]) -> None:
    """Original dashboard implementation (fallback)."""
    health_score = analysis_results['health_score']
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Relationship Health Dashboard', fontsize=16, fontweight='bold')
    
    # Health Score Gauge
    ax1 = axes[0, 0]
    overall_score = health_score['overall_health_score']
    
    colors = ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71', '#27AE60']
    ranges = [0.0, 0.4, 0.6, 0.8, 0.9, 1.0]
    
    for i in range(len(colors)):
        start_angle = ranges[i] * np.pi
        end_angle = ranges[i+1] * np.pi
        angles = np.linspace(start_angle, end_angle, 20)
        x = np.cos(angles)
        y = np.sin(angles)
        ax1.fill_between(x, 0, y, color=colors[i], alpha=0.8)
    
    needle_angle = overall_score * np.pi
    ax1.plot([0, 0.8*np.cos(needle_angle)], [0, 0.8*np.sin(needle_angle)], 'k-', linewidth=4)
    ax1.plot(0.8*np.cos(needle_angle), 0.8*np.sin(needle_angle), 'ko', markersize=8)
    
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(0, 1.1)
    ax1.set_aspect('equal')
    ax1.set_title(f'Health Score: {overall_score:.2f}\n({health_score["grade"]})', fontweight='bold')
    ax1.text(0, -0.2, f'{overall_score:.2f}', ha='center', fontsize=20, fontweight='bold')
    ax1.axis('off')
    
    # Component Scores
    ax2 = axes[0, 1]
    components = health_score['component_scores']
    categories = ['Initiation', 'Responsiveness', 'Balance', 'Dominance']
    values = list(components.values())
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax2.plot(angles, values, 'o-', linewidth=2, color='#3498DB')
    ax2.fill(angles, values, alpha=0.25, color='#3498DB')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.set_title('Component Breakdown', fontweight='bold')
    ax2.grid(True)
    
    # Message Distribution
    ax3 = axes[0, 2]
    if 'message_distribution' in analysis_results['dominance_analysis']:
        msg_dist = analysis_results['dominance_analysis']['message_distribution']
        ax3.pie(msg_dist.values(), labels=msg_dist.keys(), autopct='%1.1f%%',
               colors=['#FF6B6B', '#4ECDC4'], startangle=90)
        ax3.set_title('Message Distribution', fontweight='bold')
    
    # Response Times
    ax4 = axes[1, 0]
    if 'response_stats' in analysis_results['response_analysis']:
        response_stats = analysis_results['response_analysis']['response_stats']
        if 'mean' in response_stats:
            names = list(response_stats['mean'].keys())
            values = list(response_stats['mean'].values())
            ax4.bar(names, values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
            ax4.set_ylabel('Avg Response Time (min)')
            ax4.set_title('Response Speed', fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
    
    # Initiation
    ax5 = axes[1, 1]
    if 'initiator_counts' in analysis_results['initiator_analysis']:
        init_counts = analysis_results['initiator_analysis']['initiator_counts']
        ax5.bar(init_counts.keys(), init_counts.values(),
                      color=['#E74C3C', '#3498DB'], alpha=0.8)
        ax5.set_ylabel('Conversations Started')
        ax5.set_title('Initiation', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
    
    # Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    summary = f"Grade: {health_score['grade']}\n\n"
    summary += "Strengths:\n"
    for s in health_score['strengths'][:2]:
        summary += f"{s}\n"
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=9,
            verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'lightgray', 'alpha': 0.8})
    ax6.set_title('Summary', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_relationship_health(
    df: pd.DataFrame,
    gap_threshold_minutes: int = 60,
    include_gamification: bool = True,
    include_rolling_health: bool = True
) -> dict[str, Any]:
    """
    Complete relationship health analysis pipeline with gamification features.
    
    Args:
        df: DataFrame with columns: datetime, sender, message, (optional: message_length)
        gap_threshold_minutes: Time gap to consider new conversation start
        include_gamification: Whether to include Day 14 gamification features
        include_rolling_health: Whether to include the rolling health score series
        
    Returns:
        Complete relationship health analysis results with gamification
    """
    # Prepare data
    df_prepared = identify_conversation_starters(df, gap_threshold_minutes)
    
    # Calculate core metrics
    initiator_metrics = calculate_initiator_ratio(df_prepared)
    response_metrics = analyze_response_patterns(df_prepared)
    dominance_metrics = calculate_dominance_scores(df_prepared)
    
    # Calculate overall health score
    health_score = calculate_relationship_health_score(
        initiator_metrics, response_metrics, dominance_metrics
    )
    
    # Base results
    results = {
        'conversation_stats': {
            'total_messages': len(df),
            'unique_senders': df['sender'].nunique(),
            'date_range': f"{df_prepared['datetime'].min()} to {df_prepared['datetime'].max()}",
            'total_conversations': initiator_metrics.get('total_conversations', 0),
            'avg_response_time': response_metrics.get('overall_avg_response_minutes', None)
        },
        'initiator_analysis': initiator_metrics,
        'response_analysis': response_metrics,
        'dominance_analysis': dominance_metrics,
        'health_score': health_score,
        'prepared_data': df_prepared
    }
    
    # Add gamification features if requested
    if include_gamification:
        results['friendship_index'] = calculate_friendship_index(df_prepared)
        results['streaks'] = detect_conversation_streaks(df_prepared)
        results['milestones'] = detect_milestones(df_prepared)
        
        if 'message' in df_prepared.columns:
            results['emoji_personality'] = analyze_emoji_personality(df_prepared)
    
    if include_rolling_health:
        # Add rolling health score
        results['rolling_health'] = calculate_rolling_health_score(df_prepared)
    
    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """
    Example of how to use the enhanced relationship health analysis.
    """
    # Sample data
    sample_data = {
        'datetime': pd.date_range('2024-01-01 09:00:00', periods=100, freq='3h'),
        'sender': ['Alice', 'Bob'] * 50,
        'message': ['Hey!', 'Hi there!'] * 50,
        'message_length': np.random.randint(10, 150, 100)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Run complete analysis with gamification
    results = analyze_relationship_health(df, include_gamification=True)
    
    # Display results
    print("=== RELATIONSHIP HEALTH ANALYSIS ===")
    print(f"\nOverall Score: {results['health_score']['overall_health_score']:.2f}")
    print(f"Grade: {results['health_score']['grade']}")
    
    print("\n=== FRIENDSHIP INDEX ===")
    print(f"Score: {results['friendship_index']['friendship_index']:.2f}/100")
    print(f"Tier: {results['friendship_index']['tier']}")
    
    print("\n=== STREAKS ===")
    print(f"Current: {results['streaks']['current_streak']} days")
    print(f"Longest: {results['streaks']['longest_streak']} days")
    
    print("\n=== ACHIEVEMENTS ===")
    print(f"Total: {results['milestones']['total_achievements']}")
    
    # Create enhanced visualization
    plot_relationship_health_dashboard_enhanced(results)
    
    return results


if __name__ == "__main__":
    example_usage()
