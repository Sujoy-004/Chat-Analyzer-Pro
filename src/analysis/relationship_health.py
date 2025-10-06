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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    df['is_conversation_starter'] = False
    
    # First message is always a conversation starter
    df.loc[0, 'is_conversation_starter'] = True
    
    # Mark conversation starters based on gaps or date changes
    for i in range(1, len(df)):
        current_time = df.loc[i, 'datetime']
        prev_time = df.loc[i-1, 'datetime']
        time_gap_minutes = (current_time - prev_time).total_seconds() / 60
        different_day = current_time.date() != prev_time.date()
        
        if time_gap_minutes > gap_threshold_minutes or different_day:
            df.loc[i, 'is_conversation_starter'] = True
    
    return df


def calculate_initiator_ratio(df: pd.DataFrame) -> Dict[str, Any]:
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


def analyze_response_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze response lag patterns and responsiveness.
    
    Args:
        df: DataFrame with conversation data
        
    Returns:
        Dictionary with response analysis metrics
    """
    # Get valid responses (excluding conversation starters and same-sender continuations)
    response_df = df[df['is_conversation_starter'] == False].copy()
    
    response_analysis = []
    for i, row in response_df.iterrows():
        prev_sender = row['prev_sender']
        current_sender = row['sender']
        response_time = row['time_diff_minutes']
        
        # Valid response: different sender with valid response time
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


def calculate_dominance_scores(df: pd.DataFrame) -> Dict[str, Any]:
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
    initiator_metrics: Dict[str, Any],
    response_metrics: Dict[str, Any], 
    dominance_metrics: Dict[str, Any],
    weights: Dict[str, float] = None
) -> Dict[str, Any]:
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

def calculate_rolling_health_score(
    df: pd.DataFrame,
    window_days: int = 7,
    min_messages: int = 10
) -> pd.DataFrame:
    """
    Calculate rolling relationship health score over time.
    
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
        except Exception as e:
            logger.warning(f"Failed to calculate health score for {current_date}: {str(e)}")
            continue
    
    return pd.DataFrame(health_scores)


# ============================================================================
# DAY 14: GAMIFICATION FEATURES - FRIENDSHIP INDEX & EXTRAS
# ============================================================================

def calculate_friendship_index(df: pd.DataFrame) -> Dict[str, Any]:
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


def detect_conversation_streaks(df: pd.DataFrame) -> Dict[str, Any]:
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
    today = datetime.now().date()
    days_since_last = (today - last_message_date).days
    
    active_streak = current_streak if days_since_last <= 1 else 0
    
    return {
        'current_streak': active_streak,
        'longest_streak': longest_streak,
        'total_active_days': len(conversation_days),
        'streak_history': streaks,
        'days_since_last_message': days_since_last
    }


def analyze_emoji_personality(df: pd.DataFrame, message_col: str = 'message') -> Dict[str, Any]:
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
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
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


def detect_milestones(df: pd.DataFrame) -> Dict[str, Any]:
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
    analysis_results: Dict[str, Any],
    figsize: Tuple[int, int] = (20, 14),
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
            from src.utils.visualization import ChatVisualizer
            viz = ChatVisualizer(figsize=(12, 6))
            
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


def _plot_friendship_gauge(ax, friendship_data: Dict[str, Any]) -> None:
    """Plot friendship index as a gauge."""
    score = friendship_data['friendship_index']
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
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


def _plot_streak_display(ax, streak_data: Dict[str, Any]) -> None:
    """Plot streak information."""
    ax.axis('off')
    
    current = streak_data['current_streak']
    longest = streak_data['longest_streak']
    
    info_text = f"🔥 Current Streak\n{current} days\n\n"
    info_text += f"🏆 Longest Streak\n{longest} days\n\n"
    info_text += f"📅 Active Days\n{streak_data['total_active_days']}"
    
    ax.text(0.5, 0.5, info_text, transform=ax.transAxes,
           fontsize=12, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.8))
    ax.set_title('Conversation Streaks', fontweight='bold', fontsize=11)


def _plot_emoji_personality(ax, emoji_data: Dict[str, Any]) -> None:
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
           bbox=dict(boxstyle='round', facecolor='#e7f3ff', alpha=0.8))
    ax.set_title('Emoji Personalities', fontweight='bold', fontsize=11)


def _plot_health_components(ax, health_score: Dict[str, Any]) -> None:
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


def _plot_achievements(ax, milestones: Dict[str, Any]) -> None:
    """Plot achievement badges."""
    ax.axis('off')
    
    achievements = milestones['achievements']
    recent = achievements[-5:] if len(achievements) > 5 else achievements
    
    text = f"🏆 Achievements ({milestones['total_achievements']})\n\n"
    for ach in recent:
        text += f"{ach['icon']} {ach['milestone']}\n"
    
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
           fontsize=10, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='#d4edda', alpha=0.8))
    ax.set_title('Latest Achievements', fontweight='bold', fontsize=11)


def _plot_original_dashboard(analysis_results: Dict[str, Any], figsize: Tuple[int, int]) -> None:
    """Original dashboard implementation (fallback)."""
    health_score = analysis_results['health_score']
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Relationship Health Dashboard', fontsize=16, fontweight='bold')
    
    # Health Score Gauge
    ax1 = axes[0, 0]
    overall_score = health_score['overall_health_score']
    
    theta = np.linspace(0, np.pi, 100)
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
            bars = ax4.bar(names, values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
            ax4.set_ylabel('Avg Response Time (min)')
            ax4.set_title('Response Speed', fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
    
    # Initiation
    ax5 = axes[1, 1]
    if 'initiator_counts' in analysis_results['initiator_analysis']:
        init_counts = analysis_results['initiator_analysis']['initiator_counts']
        bars = ax5.bar(init_counts.keys(), init_counts.values(),
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
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax6.set_title('Summary', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_relationship_health(
    df: pd.DataFrame,
    gap_threshold_minutes: int = 60,
    include_gamification: bool = True
) -> Dict[str, Any]:
    """
    Complete relationship health analysis pipeline with gamification features.
    
    Args:
        df: DataFrame with columns: datetime, sender, message, (optional: message_length)
        gap_threshold_minutes: Time gap to consider new conversation start
        include_gamification: Whether to include Day 14 gamification features
        
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
    
    print(f"\n=== FRIENDSHIP INDEX ===")
    print(f"Score: {results['friendship_index']['friendship_index']:.2f}/100")
    print(f"Tier: {results['friendship_index']['tier']}")
    
    print(f"\n=== STREAKS ===")
    print(f"Current: {results['streaks']['current_streak']} days")
    print(f"Longest: {results['streaks']['longest_streak']} days")
    
    print(f"\n=== ACHIEVEMENTS ===")
    print(f"Total: {results['milestones']['total_achievements']}")
    
    # Create enhanced visualization
    plot_relationship_health_dashboard_enhanced(results)
    
    return results


if __name__ == "__main__":
    example_usage()
