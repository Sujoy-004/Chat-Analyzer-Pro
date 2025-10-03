"""
Relationship Health Metrics Module - Extended for Day 9

This module analyzes relationship health through conversation patterns including:
- Initiator ratio (who starts conversations)
- Response lag analysis (response times and patterns)  
- Dominance scores (message count, length, conversation control)
- Overall relationship health scoring
- **NEW DAY 9**: Rolling health score tracking over time windows
- **NEW DAY 9**: Trend analysis and forecasting

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any


# ============================================================================
# DAY 5 FUNCTIONS - Original Implementation
# ============================================================================

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
    for i in range(periods):
        forecast_x = len(rolling_health_df) + i
        forecast_score = slope * forecast_x + intercept
        
        # Clamp score between 0 and 100
        forecast_score = max(0, min(100, forecast_score))
        
        forecast_date = last_date + timedelta(days=i+1)
        
        forecast_data.append({
            'date': forecast_date,
            'health_score': forecast_score,
            'is_forecast': True
        })
    
    return pd.DataFrame(forecast_data)


def generate_health_insights(rolling_health_df: pd.DataFrame, df_prepared: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate actionable insights from rolling health analysis.
    
    **NEW DAY 9 FUNCTION**
    
    Args:
        rolling_health_df: DataFrame from calculate_rolling_health_scores()
        df_prepared: Original prepared data with conversation markers
        
    Returns:
        Dictionary with insights, recommendations, and alerts
    """
    insights = {
        'trend_analysis': {},
        'component_analysis': {},
        'recommendations': [],
        'alerts': []
    }
    
    # Trend Analysis
    trend_metrics = analyze_health_trend(rolling_health_df)
    insights['trend_analysis'] = trend_metrics
    
    # Component Analysis
    avg_initiation = rolling_health_df['initiation_balance'].mean()
    avg_responsiveness = rolling_health_df['responsiveness'].mean()
    avg_participation = rolling_health_df['participation_balance'].mean()
    
    insights['component_analysis'] = {
        'initiation_balance': avg_initiation,
        'responsiveness': avg_responsiveness,
        'participation_balance': avg_participation
    }
    
    # Identify weakest and strongest
    components = {
        'Conversation Initiation': avg_initiation,
        'Responsiveness': avg_responsiveness,
        'Participation Balance': avg_participation
    }
    insights['weakest_component'] = min(components, key=components.get)
    insights['strongest_component'] = max(components, key=components.get)
    
    # Generate Recommendations
    if avg_initiation < 0.7:
        insights['recommendations'].append({
            'priority': 'HIGH',
            'area': 'Conversation Initiation',
            'issue': f'Initiation balance is low ({avg_initiation:.2%})',
            'suggestion': 'Encourage both participants to start conversations more equally'
        })
    
    if avg_responsiveness < 0.6:
        insights['recommendations'].append({
            'priority': 'HIGH',
            'area': 'Responsiveness',
            'issue': f'Response times are slow ({avg_responsiveness:.2%})',
            'suggestion': 'Try to respond more promptly to messages'
        })
    
    if avg_participation < 0.7:
        insights['recommendations'].append({
            'priority': 'MEDIUM',
            'area': 'Participation',
            'issue': f'Message participation is imbalanced ({avg_participation:.2%})',
            'suggestion': 'Balance message volume between participants'
        })
    
    # Generate Alerts
    if trend_metrics.get('total_change', 0) < -5:
        insights['alerts'].append({
            'severity': 'WARNING',
            'message': f"Health score declined by {abs(trend_metrics['total_change']):.1f} points",
            'action': 'Review recent communication patterns'
        })
    
    if trend_metrics.get('volatility', 0) > 10:
        insights['alerts'].append({
            'severity': 'ATTENTION',
            'message': f"High score volatility detected ({trend_metrics['volatility']:.1f})",
            'action': 'Work on consistent communication habits'
        })
    
    return insights


def plot_rolling_health_dashboard(
    rolling_health_df: pd.DataFrame,
    trend_metrics: Dict[str, Any] = None,
    forecast_df: pd.DataFrame = None,
    figsize: Tuple[int, int] = (20, 12)
) -> None:
    """
    Create comprehensive rolling health score visualization.
    
    **NEW DAY 9 FUNCTION**
    
    Args:
        rolling_health_df: DataFrame from calculate_rolling_health_scores()
        trend_metrics: Optional trend analysis from analyze_health_trend()
        forecast_df: Optional forecast from forecast_health_scores()
        figsize: Figure size tuple
    """
    from scipy.stats import linregress
    
    if trend_metrics is None:
        trend_metrics = analyze_health_trend(rolling_health_df)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Main Health Score Trend
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(rolling_health_df['date'], rolling_health_df['health_score'], 
             marker='o', linewidth=3, markersize=12, color='#3498DB', label='Health Score')
    
    # Add trend line
    x_numeric = np.arange(len(rolling_health_df))
    trend_line = trend_metrics['slope'] * x_numeric + trend_metrics['intercept']
    ax1.plot(rolling_health_df['date'], trend_line, '--', linewidth=2, 
             color='#E74C3C', alpha=0.7, label=f'Trend (slope: {trend_metrics["slope"]:.2f}/day)')
    
    # Add forecast if provided
    if forecast_df is not None and len(forecast_df) > 0:
        ax1.plot(forecast_df['date'], forecast_df['health_score'], 
                 'o--', linewidth=2, markersize=10, color='#E74C3C', 
                 label='Forecast', alpha=0.7)
        ax1.axvspan(rolling_health_df['date'].iloc[-1], forecast_df['date'].iloc[-1], 
                    alpha=0.1, color='gray')
    
    # Threshold zones
    ax1.axhspan(85, 100, alpha=0.1, color='green', label='Excellent (85-100)')
    ax1.axhspan(70, 85, alpha=0.1, color='yellow', label='Good (70-85)')
    ax1.axhspan(55, 70, alpha=0.1, color='orange', label='Fair (55-70)')
    ax1.axhspan(0, 55, alpha=0.1, color='red', label='Needs Work (0-55)')
    
    ax1.set_title('Relationship Health Score Over Time', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Health Score', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Annotate points
    for idx, row in rolling_health_df.iterrows():
        ax1.annotate(f"{row['health_score']:.1f}", 
                    (row['date'], row['health_score']),
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=9, fontweight='bold')
    
    # 2. Component Scores Over Time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(rolling_health_df['date'], rolling_health_df['initiation_balance'] * 100, 
             marker='o', label='Initiation', linewidth=2, markersize=8)
    ax2.plot(rolling_health_df['date'], rolling_health_df['responsiveness'] * 100, 
             marker='s', label='Responsiveness', linewidth=2, markersize=8)
    ax2.plot(rolling_health_df['date'], rolling_health_df['participation_balance'] * 100, 
             marker='^', label='Participation', linewidth=2, markersize=8)
    ax2.set_title('Component Trends', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score (%)', fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Message Activity Over Time
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(rolling_health_df['date'], rolling_health_df['message_count'], 
            color='#9B59B6', alpha=0.7, edgecolor='black')
    ax3.set_title('Message Volume per Window', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Message Count', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)
    
    for idx, row in rolling_health_df.iterrows():
        ax3.text(row['date'], row['message_count'] + 0.3, 
                str(int(row['message_count'])), 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 4. Conversation Frequency
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.bar(rolling_health_df['date'], rolling_health_df['conversation_count'], 
            color='#E67E22', alpha=0.7, edgecolor='black')
    ax4.set_title('Conversations per Window', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Conversation Count', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(axis='x', rotation=45)
    
    for idx, row in rolling_health_df.iterrows():
        ax4.text(row['date'], row['conversation_count'] + 0.1, 
                str(int(row['conversation_count'])), 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 5. Health Score Distribution
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(rolling_health_df['health_score'], bins=5, color='#2ECC71', 
             alpha=0.7, edgecolor='black')
    ax5.axvline(trend_metrics['average_score'], color='red', 
               linestyle='--', linewidth=2, label=f"Mean: {trend_metrics['average_score']:.1f}")
    ax5.set_title('Health Score Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Health Score', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Score Change Analysis
    ax6 = fig.add_subplot(gs[2, 1])
    if len(rolling_health_df) > 1:
        score_changes = rolling_health_df['health_score'].diff().dropna()
        colors = ['#2ECC71' if x >= 0 else '#E74C3C' for x in score_changes]
        ax6.bar(range(len(score_changes)), score_changes, color=colors, alpha=0.7, edgecolor='black')
        ax6.axhline(0, color='black', linewidth=1)
        ax6.set_title('Day-to-Day Score Changes', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Time Period', fontsize=10)
        ax6.set_ylabel('Score Change', fontsize=10)
        ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Summary Statistics Box
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    # Background color based on average score
    if trend_metrics['average_score'] >= 85:
        bg_color = '#D5F4E6'
    elif trend_metrics['average_score'] >= 70:
        bg_color = '#FFF9C4'
    elif trend_metrics['average_score'] >= 55:
        bg_color = '#FFE0B2'
    else:
        bg_color = '#FFCDD2'
    
    summary_text = f"""ROLLING HEALTH SUMMARY

Average Score: {trend_metrics['average_score']:.1f}/100
Score Range: {trend_metrics['max_score'] - trend_metrics['min_score']:.1f} points
Volatility: {trend_metrics['volatility']:.1f}

Trend: {trend_metrics['trend_emoji']} {trend_metrics['trend_direction'].replace('_', ' ')}
Slope: {trend_metrics['slope']:.2f} points/day
R² Value: {trend_metrics['r_squared']:.3f}

Best Day: {rolling_health_df.loc[rolling_health_df['health_score'].idxmax(), 'date']}
Score: {trend_metrics['max_score']:.1f}

Lowest Day: {rolling_health_df.loc[rolling_health_df['health_score'].idxmin(), 'date']}
Score: {trend_metrics['min_score']:.1f}

Total Messages: {rolling_health_df['message_count'].sum():.0f}
Total Conversations: {rolling_health_df['conversation_count'].sum():.0f}
"""
    
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.8, pad=1))
    
    plt.suptitle('Rolling Relationship Health Score Tracker (Day 9)', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


# Example usage function
def example_usage():
    """
    Example of how to use the relationship health analysis functions.
    """
    # Sample data structure
    sample_data = {
        'datetime': pd.date_range('2023-12-25 09:30:00', periods=20, freq='30min'),
        'sender': ['Alice', 'Bob'] * 10,
        'message': ['Sample message'] * 20,
        'message_length': np.random.randint(10, 100, 20)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Run complete analysis (Day 5)
    results = analyze_relationship_health(df)
    
    # Display results
    print("=== RELATIONSHIP HEALTH ANALYSIS ===")
    print(f"Overall Score: {results['health_score']['overall_health_score']:.2f}")
    print(f"Grade: {results['health_score']['grade']}")
    print(f"Description: {results['health_score']['description']}")
    
    # Create visualization
    plot_relationship_health_dashboard(results)
    
    # Run rolling analysis (Day 9)
    rolling_scores = calculate_rolling_health_scores(df, window_days=1)
    trend_analysis = analyze_health_trend(rolling_scores)
    forecast = forecast_health_scores(rolling_scores, periods=2)
    insights = generate_health_insights(rolling_scores, results['prepared_data'])
    
    print("\n=== ROLLING HEALTH ANALYSIS ===")
    print(f"Trend: {trend_analysis['trend_direction']} ({trend_analysis['slope']:.2f}/day)")
    print(f"Average Score: {trend_analysis['average_score']:.1f}/100")
    
    # Create rolling visualization
    plot_rolling_health_dashboard(rolling_scores, trend_analysis, forecast)
    
    return results, rolling_scores, insights


if __name__ == "__main__":
    # Run example if script is executed directly
    example_usage() i in range(1, len(df)):
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
        strengths.append("Balanced conversation initiation")
    else:
        areas_for_improvement.append("One person initiates more conversations")
    
    if responsiveness_score >= 0.8:
        strengths.append("Both are very responsive")
    else:
        areas_for_improvement.append("Slower response times")
    
    if response_balance_score >= 0.8:
        strengths.append("Similar response time patterns")
    else:
        areas_for_improvement.append("Significant difference in response speeds")
    
    if dominance_score >= 0.8:
        strengths.append("Excellent participation balance")
    else:
        areas_for_improvement.append("Some dominance in conversation patterns")
    
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


def analyze_relationship_health(df: pd.DataFrame, gap_threshold_minutes: int = 60) -> Dict[str, Any]:
    """
    Complete relationship health analysis pipeline.
    
    Args:
        df: DataFrame with columns: datetime, sender, message, (optional: message_length)
        gap_threshold_minutes: Time gap to consider new conversation start
        
    Returns:
        Complete relationship health analysis results
    """
    # Prepare data
    df_prepared = identify_conversation_starters(df, gap_threshold_minutes)
    
    # Calculate all metrics
    initiator_metrics = calculate_initiator_ratio(df_prepared)
    response_metrics = analyze_response_patterns(df_prepared)
    dominance_metrics = calculate_dominance_scores(df_prepared)
    
    # Calculate overall health score
    health_score = calculate_relationship_health_score(
        initiator_metrics, response_metrics, dominance_metrics
    )
    
    # Compile complete results
    return {
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
        'prepared_data': df_prepared  # Include for further analysis/visualization
    }


def plot_relationship_health_dashboard(analysis_results: Dict[str, Any], figsize: Tuple[int, int] = (18, 12)) -> None:
    """
    Create comprehensive relationship health visualization dashboard.
    
    Args:
        analysis_results: Results from analyze_relationship_health()
        figsize: Figure size tuple
    """
    health_score = analysis_results['health_score']
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Relationship Health Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Health Score Gauge
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
    
    # Add needle
    needle_angle = overall_score * np.pi
    needle_x = [0, 0.8 * np.cos(needle_angle)]
    needle_y = [0, 0.8 * np.sin(needle_angle)]
    ax1.plot(needle_x, needle_y, 'k-', linewidth=4)
    ax1.plot(needle_x[1], needle_y[1], 'ko', markersize=8)
    
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(0, 1.1)
    ax1.set_aspect('equal')
    ax1.set_title(f'Health Score: {overall_score:.2f}\n({health_score["grade"]})', fontweight='bold')
    ax1.text(0, -0.2, f'{overall_score:.2f}', ha='center', va='center', fontsize=20, fontweight='bold')
    ax1.axis('off')
    
    # 2. Component Scores Radar Chart
    ax2 = axes[0, 1]
    categories = ['Initiation\nBalance', 'Responsiveness', 'Response\nBalance', 'Participation\nBalance']
    values = list(health_score['component_scores'].values())
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax2.plot(angles, values, 'o-', linewidth=2, color='#3498DB')
    ax2.fill(angles, values, alpha=0.25, color='#3498DB')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 1)
    ax2.set_title('Component Breakdown', fontweight='bold')
    ax2.grid(True)
    
    # 3. Message Distribution
    ax3 = axes[0, 2]
    if 'message_distribution' in analysis_results['dominance_analysis']:
        msg_dist = analysis_results['dominance_analysis']['message_distribution']
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38BA8', '#A8E6CF'][:len(msg_dist)]
        ax3.pie(msg_dist.values(), labels=msg_dist.keys(), autopct='%1.1f%%', colors=colors, startangle=90)
        ax3.set_title('Message Distribution', fontweight='bold')
    
    # 4. Response Time Comparison
    ax4 = axes[1, 0]
    if 'response_stats' in analysis_results['response_analysis']:
        response_stats = analysis_results['response_analysis']['response_stats']
        if 'mean' in response_stats:
            names = list(response_stats['mean'].keys())
            values = list(response_stats['mean'].values())
            colors = ['#FF6B6B', '#4ECDC4'][:len(names)]
            
            bars = ax4.bar(names, values, color=colors, alpha=0.8)
            ax4.set_ylabel('Average Response Time (minutes)')
            ax4.set_title('Response Speed Comparison', fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                         f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Conversation Initiation
    ax5 = axes[1, 1]
    if 'initiator_counts' in analysis_results['initiator_analysis']:
        init_counts = analysis_results['initiator_analysis']['initiator_counts']
        colors = ['#E74C3C', '#3498DB'][:len(init_counts)]
        bars = ax5.bar(init_counts.keys(), init_counts.values(), color=colors, alpha=0.8)
        ax5.set_ylabel('Conversations Started')
        ax5.set_title('Conversation Initiation', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, init_counts.values()):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                     f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Health Summary Text
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"Grade: {health_score['grade']}\n\n"
    summary_text += "Strengths:\n"
    for strength in health_score['strengths'][:3]:  # Limit to 3 for space
        summary_text += f"{strength}\n"
    
    if health_score['areas_for_improvement']:
        summary_text += f"\nAreas to improve:\n"
        for improvement in health_score['areas_for_improvement'][:2]:
            summary_text += f"{improvement}\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax6.set_title('Summary', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# DAY 9 FUNCTIONS - Rolling Health Score Tracker (NEW)
# ============================================================================

def calculate_rolling_health_scores(
    df: pd.DataFrame, 
    window_days: int = 1,
    gap_threshold_minutes: int = 60
) -> pd.DataFrame:
    """
    Calculate relationship health scores in rolling time windows.
    
    **NEW DAY 9 FUNCTION**
    
    Args:
        df: DataFrame with datetime, sender, message columns
        window_days: Size of rolling window in days
        gap_threshold_minutes: Time gap to identify conversation starters
        
    Returns:
        DataFrame with rolling health scores and component metrics
    """
    # Prepare data with conversation markers
    df_prepared = identify_conversation_starters(df, gap_threshold_minutes)
    
    # Get date range
    start_date = df_prepared['datetime'].min().date()
    end_date = df_prepared['datetime'].max().date()
    
    # Create rolling windows
    current_date = start_date
    rolling_scores = []
    
    while current_date <= end_date:
        window_end = current_date + timedelta(days=window_days)
        
        # Get data for this window
        window_data = df_prepared[
            (df_prepared['datetime'].dt.date >= current_date) &
            (df_prepared['datetime'].dt.date < window_end)
        ]
        
        if len(window_data) > 0:
            # Calculate metrics for this window
            # 1. Initiation balance
            starters = window_data[window_data['is_conversation_starter']==True]
            if len(starters) > 0:
                init_counts = starters['sender'].value_counts()
                if len(init_counts) >= 2:
                    init_balance = 1 - abs(init_counts.values[0] - init_counts.values[1]) / len(starters)
                else:
                    init_balance = 0.0
            else:
                init_balance = 0.5  # Neutral if no conversations started
            
            # 2. Response metrics
            responses = window_data[window_data['is_conversation_starter'] == False].copy()
            valid_responses = []
            for i, row in responses.iterrows():
                if row['prev_sender'] != row['sender'] and pd.notna(row['time_diff_minutes']):
                    valid_responses.append(row['time_diff_minutes'])
            
            if len(valid_responses) > 0:
                avg_response = np.mean(valid_responses)
                responsiveness = max(0, 1 - (avg_response / 120))
            else:
                responsiveness = 0.5
            
            # 3. Participation balance
            msg_counts = window_data['sender'].value_counts()
            if len(msg_counts) >= 2:
                participation_balance = 1 - abs(msg_counts.values[0] - msg_counts.values[1]) / len(window_data)
            else:
                participation_balance = 0.0
            
            # Calculate weighted score
            window_score = (
                0.25 * init_balance +
                0.40 * responsiveness +
                0.35 * participation_balance
            )
            
            rolling_scores.append({
                'date': current_date,
                'window_end': window_end,
                'health_score': window_score * 100,
                'initiation_balance': init_balance,
                'responsiveness': responsiveness,
                'participation_balance': participation_balance,
                'message_count': len(window_data),
                'conversation_count': len(starters) if len(starters) > 0 else 0
            })
        
        # Move to next window
        current_date = window_end
    
    return pd.DataFrame(rolling_scores)


def analyze_health_trend(rolling_health_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze trends in rolling health scores.
    
    **NEW DAY 9 FUNCTION**
    
    Args:
        rolling_health_df: DataFrame from calculate_rolling_health_scores()
        
    Returns:
        Dictionary with trend analysis metrics
    """
    from scipy.stats import linregress
    
    if len(rolling_health_df) < 2:
        return {'error': 'Insufficient data for trend analysis'}
    
    # Linear regression on scores
    x = np.arange(len(rolling_health_df))
    y = rolling_health_df['health_score'].values
    
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # Trend classification
    if slope > 2:
        trend_direction = "STRONG_IMPROVEMENT"
        trend_emoji = "📈"
    elif slope > 0:
        trend_direction = "SLIGHT_IMPROVEMENT"
        trend_emoji = "↗️"
    elif slope > -2:
        trend_direction = "STABLE"
        trend_emoji = "➡️"
    elif slope > -5:
        trend_direction = "SLIGHT_DECLINE"
        trend_emoji = "↘️"
    else:
        trend_direction = "NOTABLE_DECLINE"
        trend_emoji = "📉"
    
    # Calculate statistics
    avg_score = rolling_health_df['health_score'].mean()
    max_score = rolling_health_df['health_score'].max()
    min_score = rolling_health_df['health_score'].min()
    volatility = rolling_health_df['health_score'].std()
    
    first_score = rolling_health_df.iloc[0]['health_score']
    last_score = rolling_health_df.iloc[-1]['health_score']
    total_change = last_score - first_score
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'trend_direction': trend_direction,
        'trend_emoji': trend_emoji,
        'average_score': avg_score,
        'max_score': max_score,
        'min_score': min_score,
        'volatility': volatility,
        'first_score': first_score,
        'last_score': last_score,
        'total_change': total_change,
        'change_percentage': (total_change / first_score * 100) if first_score > 0 else 0
    }


def forecast_health_scores(rolling_health_df: pd.DataFrame, periods: int = 2) -> pd.DataFrame:
    """
    Forecast future health scores based on trend.
    
    **NEW DAY 9 FUNCTION**
    
    Args:
        rolling_health_df: DataFrame from calculate_rolling_health_scores()
        periods: Number of future periods to forecast
        
    Returns:
        DataFrame with forecasted dates and scores
    """
    from scipy.stats import linregress
    
    if len(rolling_health_df) < 2:
        return pd.DataFrame()
    
    # Fit trend line
    x = np.arange(len(rolling_health_df))
    y = rolling_health_df['health_score'].values
    slope, intercept, _, _, _ = linregress(x, y)
    
    # Generate forecasts
    last_date = rolling_health_df.iloc[-1]['date']
    forecast_data = []
    
    for
