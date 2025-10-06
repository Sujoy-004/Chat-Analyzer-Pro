"""
Visualization Module - Day 13
Comprehensive visualization utilities for chat analysis.

Provides heatmaps, word clouds, timeline plots, sentiment visualizations,
and interactive elements for Streamlit integration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple, Union
import warnings
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ChatVisualizer:
    """
    Comprehensive visualization class for chat analysis.
    Provides various plotting methods for different aspects of chat data.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6), style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize the ChatVisualizer.
        
        Args:
            figsize: Default figure size for plots
            style: Matplotlib style to use
        """
        self.figsize = figsize
        self.style = style
        plt.style.use(style)
        
        # Color schemes
        self.colors = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'positive': '#4caf50',
            'neutral': '#ffc107',
            'negative': '#f44336',
            'accent': '#00bcd4'
        }
    
    def plot_message_timeline(
        self,
        df: pd.DataFrame,
        resample_freq: str = 'D',
        figsize: Optional[Tuple[int, int]] = None,
        title: str = 'Message Timeline',
        show_trend: bool = True
    ) -> plt.Figure:
        """
        Plot message volume over time.
        
        Args:
            df: DataFrame with 'timestamp' column
            resample_freq: Resampling frequency ('D'=daily, 'W'=weekly, 'M'=monthly)
            figsize: Figure size (uses default if None)
            title: Plot title
            show_trend: Whether to show trend line
            
        Returns:
            matplotlib Figure object
        """
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        # Ensure timestamp is datetime
        if 'timestamp' not in df.columns:
            logger.error("DataFrame must have 'timestamp' column")
            return fig
        
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
        
        # Resample and count messages
        message_counts = df_copy.set_index('timestamp').resample(resample_freq).size()
        
        # Plot
        ax.plot(message_counts.index, message_counts.values, 
                linewidth=2, color=self.colors['primary'], marker='o', markersize=4)
        
        # Add trend line
        if show_trend and len(message_counts) > 1:
            z = np.polyfit(range(len(message_counts)), message_counts.values, 1)
            p = np.poly1d(z)
            ax.plot(message_counts.index, p(range(len(message_counts))), 
                   "--", alpha=0.5, color=self.colors['secondary'], linewidth=2, label='Trend')
            ax.legend()
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Number of Messages', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_activity_heatmap(
        self,
        df: pd.DataFrame,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = 'Activity Heatmap (Hour vs Day)'
    ) -> plt.Figure:
        """
        Create heatmap showing activity by hour and day of week.
        
        Args:
            df: DataFrame with 'timestamp' column
            figsize: Figure size
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        figsize = figsize or (14, 8)
        fig, ax = plt.subplots(figsize=figsize)
        
        if 'timestamp' not in df.columns:
            logger.error("DataFrame must have 'timestamp' column")
            return fig
        
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
        
        # Extract hour and day
        df_copy['hour'] = df_copy['timestamp'].dt.hour
        df_copy['day'] = df_copy['timestamp'].dt.day_name()
        
        # Create pivot table
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = df_copy.groupby(['day', 'hour']).size().unstack(fill_value=0)
        heatmap_data = heatmap_data.reindex(day_order)
        
        # Plot heatmap
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='g', 
                   cbar_kws={'label': 'Number of Messages'}, ax=ax, linewidths=0.5)
        
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Day of Week', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_wordcloud(
        self,
        df: pd.DataFrame,
        message_col: str = 'message',
        figsize: Optional[Tuple[int, int]] = None,
        title: str = 'Word Cloud',
        max_words: int = 100,
        background_color: str = 'white',
        colormap: str = 'viridis'
    ) -> plt.Figure:
        """
        Generate word cloud from messages.
        
        Args:
            df: DataFrame with message column
            message_col: Name of message column
            figsize: Figure size
            title: Plot title
            max_words: Maximum number of words
            background_color: Background color
            colormap: Color scheme
            
        Returns:
            matplotlib Figure object
        """
        try:
            from wordcloud import WordCloud, STOPWORDS
        except ImportError:
            logger.error("wordcloud package not installed. Install with: pip install wordcloud")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'wordcloud package not installed', 
                   ha='center', va='center', fontsize=14)
            return fig
        
        figsize = figsize or (14, 8)
        fig, ax = plt.subplots(figsize=figsize)
        
        if message_col not in df.columns:
            logger.error(f"Column '{message_col}' not found")
            return fig
        
        # Combine all messages
        text = ' '.join(df[message_col].dropna().astype(str))
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            max_words=max_words,
            background_color=background_color,
            colormap=colormap,
            stopwords=STOPWORDS,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig
    
    def plot_emoji_distribution(
        self,
        df: pd.DataFrame,
        message_col: str = 'message',
        top_n: int = 15,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = 'Top Emoji Usage'
    ) -> plt.Figure:
        """
        Plot most frequently used emojis.
        
        Args:
            df: DataFrame with message column
            message_col: Name of message column
            top_n: Number of top emojis to show
            figsize: Figure size
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        import re
        
        figsize = figsize or (12, 8)
        fig, ax = plt.subplots(figsize=figsize)
        
        if message_col not in df.columns:
            logger.error(f"Column '{message_col}' not found")
            return fig
        
        # Extract emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        
        all_emojis = []
        for msg in df[message_col].dropna():
            emojis = emoji_pattern.findall(str(msg))
            all_emojis.extend(emojis)
        
        if not all_emojis:
            ax.text(0.5, 0.5, 'No emojis found', ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Count and plot
        emoji_series = pd.Series(all_emojis).value_counts().head(top_n)
        
        bars = ax.barh(range(len(emoji_series)), emoji_series.values, 
                       color=self.colors['primary'])
        ax.set_yticks(range(len(emoji_series)))
        ax.set_yticklabels(emoji_series.index, fontsize=16)
        ax.set_xlabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, emoji_series.values)):
            ax.text(value, i, f' {value}', va='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_sentiment_distribution(
        self,
        df: pd.DataFrame,
        sentiment_col: str = 'sentiment',
        figsize: Optional[Tuple[int, int]] = None,
        title: str = 'Sentiment Distribution'
    ) -> plt.Figure:
        """
        Plot sentiment distribution as pie chart.
        
        Args:
            df: DataFrame with sentiment column
            sentiment_col: Name of sentiment column
            figsize: Figure size
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        figsize = figsize or (10, 8)
        fig, ax = plt.subplots(figsize=figsize)
        
        if sentiment_col not in df.columns:
            logger.error(f"Column '{sentiment_col}' not found")
            return fig
        
        sentiment_counts = df[sentiment_col].value_counts()
        
        colors = [self.colors['positive'], self.colors['neutral'], self.colors['negative']]
        colors = colors[:len(sentiment_counts)]
        
        wedges, texts, autotexts = ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 12}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig
    
    def plot_sentiment_timeline(
        self,
        df: pd.DataFrame,
        sentiment_score_col: str = 'sentiment_score',
        resample_freq: str = 'D',
        figsize: Optional[Tuple[int, int]] = None,
        title: str = 'Sentiment Over Time'
    ) -> plt.Figure:
        """
        Plot sentiment score timeline with moving average.
        
        Args:
            df: DataFrame with timestamp and sentiment_score columns
            sentiment_score_col: Name of sentiment score column
            resample_freq: Resampling frequency
            figsize: Figure size
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        if 'timestamp' not in df.columns or sentiment_score_col not in df.columns:
            logger.error("DataFrame must have 'timestamp' and sentiment score columns")
            return fig
        
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
        
        # Resample and calculate mean sentiment
        sentiment_over_time = df_copy.set_index('timestamp')[sentiment_score_col].resample(resample_freq).mean()
        
        # Plot
        ax.plot(sentiment_over_time.index, sentiment_over_time.values,
               linewidth=2, color=self.colors['primary'], alpha=0.6, label='Daily Average')
        
        # Add moving average
        if len(sentiment_over_time) > 7:
            ma = sentiment_over_time.rolling(window=7, center=True).mean()
            ax.plot(ma.index, ma.values, linewidth=3, 
                   color=self.colors['secondary'], label='7-Day Moving Average')
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sentiment Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_user_activity(
        self,
        df: pd.DataFrame,
        sender_col: str = 'sender',
        top_n: int = 10,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = 'Top Active Users'
    ) -> plt.Figure:
        """
        Plot top active users by message count.
        
        Args:
            df: DataFrame with sender column
            sender_col: Name of sender column
            top_n: Number of top users to show
            figsize: Figure size
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        figsize = figsize or (12, 8)
        fig, ax = plt.subplots(figsize=figsize)
        
        if sender_col not in df.columns:
            logger.error(f"Column '{sender_col}' not found")
            return fig
        
        user_counts = df[sender_col].value_counts().head(top_n)
        
        bars = ax.barh(range(len(user_counts)), user_counts.values,
                      color=self.colors['primary'])
        ax.set_yticks(range(len(user_counts)))
        ax.set_yticklabels(user_counts.index)
        ax.set_xlabel('Number of Messages', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, user_counts.values)):
            ax.text(value, i, f' {value:,}', va='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_response_time_distribution(
        self,
        df: pd.DataFrame,
        response_time_col: str = 'response_time_minutes',
        bins: int = 50,
        max_minutes: int = 1440,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = 'Response Time Distribution'
    ) -> plt.Figure:
        """
        Plot distribution of response times.
        
        Args:
            df: DataFrame with response_time column
            response_time_col: Name of response time column
            bins: Number of histogram bins
            max_minutes: Maximum minutes to show
            figsize: Figure size
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        if response_time_col not in df.columns:
            logger.error(f"Column '{response_time_col}' not found")
            return fig
        
        # Filter valid response times
        response_times = df[response_time_col].dropna()
        response_times = response_times[response_times <= max_minutes]
        
        if len(response_times) == 0:
            ax.text(0.5, 0.5, 'No response time data available', 
                   ha='center', va='center', fontsize=14)
            return fig
        
        ax.hist(response_times, bins=bins, color=self.colors['primary'], 
               edgecolor='black', alpha=0.7)
        
        ax.set_xlabel('Response Time (minutes)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add median line
        median = response_times.median()
        ax.axvline(median, color='red', linestyle='--', linewidth=2, 
                  label=f'Median: {median:.1f} min')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_relationship_health_trend(
        self,
        df: pd.DataFrame,
        health_score_col: str = 'health_score',
        resample_freq: str = 'W',
        figsize: Optional[Tuple[int, int]] = None,
        title: str = 'Relationship Health Score Trend'
    ) -> plt.Figure:
        """
        Plot relationship health score over time.
        
        Args:
            df: DataFrame with timestamp and health_score columns
            health_score_col: Name of health score column
            resample_freq: Resampling frequency
            figsize: Figure size
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        if 'timestamp' not in df.columns or health_score_col not in df.columns:
            logger.error("DataFrame must have 'timestamp' and health_score columns")
            return fig
        
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
        
        # Resample
        health_over_time = df_copy.set_index('timestamp')[health_score_col].resample(resample_freq).mean()
        
        # Plot with filled area
        ax.fill_between(health_over_time.index, health_over_time.values, 
                       alpha=0.3, color=self.colors['primary'])
        ax.plot(health_over_time.index, health_over_time.values,
               linewidth=3, color=self.colors['primary'], marker='o')
        
        # Add threshold lines
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Healthy (>0.7)')
        ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.4-0.7)')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Health Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def create_summary_dashboard(
        self,
        df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple plots.
        
        Args:
            df: Chat DataFrame
            output_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Message Timeline
        ax1 = fig.add_subplot(gs[0, :])
        if 'timestamp' in df.columns:
            df_copy = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
                df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            message_counts = df_copy.set_index('timestamp').resample('D').size()
            ax1.plot(message_counts.index, message_counts.values, 
                    linewidth=2, color=self.colors['primary'])
            ax1.set_title('Message Timeline', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Messages')
            ax1.grid(True, alpha=0.3)
        
        # 2. Top Users
        ax2 = fig.add_subplot(gs[1, 0])
        if 'sender' in df.columns:
            user_counts = df['sender'].value_counts().head(5)
            ax2.barh(range(len(user_counts)), user_counts.values, color=self.colors['primary'])
            ax2.set_yticks(range(len(user_counts)))
            ax2.set_yticklabels(user_counts.index, fontsize=8)
            ax2.set_title('Top 5 Users', fontsize=12, fontweight='bold')
            ax2.invert_yaxis()
        
        # 3. Sentiment Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            colors = [self.colors['positive'], self.colors['neutral'], self.colors['negative']]
            ax3.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                   autopct='%1.1f%%', colors=colors[:len(sentiment_counts)])
            ax3.set_title('Sentiment Distribution', fontsize=12, fontweight='bold')
        
        # 4. Hourly Activity
        ax4 = fig.add_subplot(gs[1, 2])
        if 'timestamp' in df.columns:
            df_copy = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
                df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            hourly = df_copy.groupby(df_copy['timestamp'].dt.hour).size()
            ax4.bar(hourly.index, hourly.values, color=self.colors['primary'])
            ax4.set_title('Activity by Hour', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Hour')
            ax4.set_ylabel('Messages')
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Daily Activity
        ax5 = fig.add_subplot(gs[2, :])
        if 'timestamp' in df.columns:
            df_copy = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
                df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            daily = df_copy.groupby(df_copy['timestamp'].dt.day_name()).size()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily = daily.reindex([d for d in day_order if d in daily.index])
            ax5.bar(range(len(daily)), daily.values, color=self.colors['primary'])
            ax5.set_xticks(range(len(daily)))
            ax5.set_xticklabels(daily.index, rotation=45)
            ax5.set_title('Activity by Day of Week', fontsize=12, fontweight='bold')
            ax5.set_ylabel('Messages')
            ax5.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Chat Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dashboard saved to {output_path}")
        
        return fig


# Utility functions for quick plotting

def quick_timeline(df: pd.DataFrame, **kwargs) -> plt.Figure:
    """Quick message timeline plot."""
    viz = ChatVisualizer()
    return viz.plot_message_timeline(df, **kwargs)


def quick_heatmap(df: pd.DataFrame, **kwargs) -> plt.Figure:
    """Quick activity heatmap."""
    viz = ChatVisualizer()
    return viz.plot_activity_heatmap(df, **kwargs)


def quick_wordcloud(df: pd.DataFrame, **kwargs) -> plt.Figure:
    """Quick word cloud generation."""
    viz = ChatVisualizer()
    return viz.plot_wordcloud(df, **kwargs)


def quick_sentiment(df: pd.DataFrame, **kwargs) -> plt.Figure:
    """Quick sentiment distribution plot."""
    viz = ChatVisualizer()
    return viz.plot_sentiment_distribution(df, **kwargs)


def quick_dashboard(df: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    """Generate complete dashboard quickly."""
    viz = ChatVisualizer()
    return viz.create_summary_dashboard(df, output_path)


# Example usage for Google Colab
if __name__ == "__main__":
    """
    Example usage in Google Colab:
    
    import pandas as pd
    from src.utils.visualization import ChatVisualizer, quick_dashboard
    
    # Load data
    df = pd.read_csv('data/processed/example_parsed.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create visualizer
    viz = ChatVisualizer()
    
    # Individual plots
    fig1 = viz.plot_message_timeline(df)
    plt.show()
    
    fig2 = viz.plot_activity_heatmap(df)
    plt.show()
    
    fig3 = viz.plot_wordcloud(df)
    plt.show()
    
    fig4 = viz.plot_sentiment_distribution(df)
    plt.show()
    
    # Quick plots
    quick_timeline(df)
    plt.show()
    
    # Complete dashboard
    dashboard = quick_dashboard(df, output_path='dashboard.png')
    plt.show()
    """
    print("ChatVisualizer module loaded. Use the functions above for visualization.")
