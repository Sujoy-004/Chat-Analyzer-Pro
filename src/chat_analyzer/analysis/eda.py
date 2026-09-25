import itertools
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ChatEDA:
    """
    Comprehensive Exploratory Data Analysis for Chat Data
    Based on the analysis patterns developed in 02_exploratory_analysis.ipynb
    """
    
    def __init__(self, df):
        """Initialize with chat DataFrame"""
        self.df = self.prepare_data(df.copy())
        self.summary = None
    
    def prepare_data(self, df):
        """Prepare and enhance data for analysis"""
        # Convert datetime columns
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Add time-based features
        df['day_of_week'] = df['datetime'].dt.day_name()
        df['month'] = df['datetime'].dt.month_name()
        df['weekday'] = df['datetime'].dt.weekday
        df['is_weekend'] = df['weekday'] >= 5
        df['time_period'] = df['hour'].apply(self._categorize_time)
        
        # Add message features
        df['word_count'] = df['message'].str.split().str.len()
        df['is_media'] = df['message'].str.contains('<Media omitted>', case=False, na=False)
        df['has_emoji'] = df['message'].str.contains(r'[😀-🙏🌀-🗿]', regex=True, na=False)
        
        return df
    
    def _categorize_time(self, hour):
        """Categorize hour into time periods"""
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon' 
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    
    def analyze_message_volume(self):
        """Analyze message volume and activity patterns"""
        return {
            'daily_messages': self.df.groupby('date').size(),
            'hourly_activity': self.df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0),
            'time_period_counts': self.df['time_period'].value_counts(),
            'sender_counts': self.df['sender'].value_counts()
        }
    
    def analyze_conversation_dynamics(self):
        """Analyze conversation flow and response patterns (vectorized)."""
        # Response time calculation
        df_sorted = self.df.sort_values('datetime').reset_index(drop=True)
        switch = df_sorted['sender'].ne(df_sorted['sender'].shift())
        minutes = df_sorted['datetime'].diff().dt.total_seconds() / 60
        response_times = minutes[switch & minutes.notna()].tolist()
        
        return {
            'response_times': response_times,
            'avg_response_time': np.mean(response_times) if response_times else None,
            'balance_ratio': min(self.df['sender'].value_counts()) / max(self.df['sender'].value_counts())
        }
    
    def analyze_content(self):
        """Analyze message content and vocabulary (vectorized)."""
        msgs = self.df['message']
        cleaned = msgs.where(~msgs.isna() & (msgs != '<Media omitted>'), pd.NA)
        cleaned = cleaned.str.lower().str.replace(r'[^\w\s]', ' ', regex=True)
        tok = cleaned.str.split().explode().dropna()
        tok = tok[tok.str.len() > 2]
        counts = tok.value_counts()
        # Counter key order = first occurrence in the token stream (ties in most_common break on it); do not sort.
        first_seen = tok.drop_duplicates(keep='first')
        word_freq = Counter(dict(zip(first_seen.tolist(), counts.loc[first_seen.tolist()].astype(int).tolist())))

        emoji_rows = self.df['message'].str.findall(r'[😀-🙏🌀-🗿]').dropna()
        emoji_frequency = Counter(itertools.chain.from_iterable(emoji_rows))

        return {
            'word_frequency': word_freq,
            'emoji_frequency': emoji_frequency,
            'total_words': sum(self.df['word_count']),
            'unique_words': len(word_freq)
        }
    
    def _clean_text(self, text):
        """Clean text for analysis"""
        if pd.isna(text) or text == '<Media omitted>':
            return ""
        return re.sub(r'[^\w\s]', ' ', text.lower())
    
    def generate_comprehensive_summary(self, volume_analysis=None, dynamics_analysis=None, content_analysis=None):
        """Generate comprehensive analysis summary (precomputed results may be passed in)."""
        if volume_analysis is None:
            volume_analysis = self.analyze_message_volume()
        if dynamics_analysis is None:
            dynamics_analysis = self.analyze_conversation_dynamics()
        if content_analysis is None:
            content_analysis = self.analyze_content()
        
        summary = {
            'dataset_info': {
                'total_messages': len(self.df),
                'participants': self.df['sender'].unique().tolist(),
                'duration_days': (self.df['datetime'].max() - self.df['datetime'].min()).days + 1
            },
            'activity_patterns': {
                'peak_hour': self.df.groupby('hour').size().idxmax(),
                'most_active_period': volume_analysis['time_period_counts'].idxmax()
            },
            'conversation_quality': {
                'balance_ratio': dynamics_analysis['balance_ratio'],
                'avg_response_time': dynamics_analysis['avg_response_time']
            },
            'content_insights': {
                'total_words': content_analysis['total_words'],
                'unique_vocabulary': content_analysis['unique_words'],
                'emoji_usage_rate': (self.df['has_emoji'].sum() / len(self.df)) * 100
            }
        }
        
        self.summary = summary
        return summary
    
    def create_dashboard(self, figsize=(15, 10)):
        """Create comprehensive EDA dashboard"""
        fig, _ = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Chat Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Implementation would include all visualization code from the notebook
        # This is a template structure
        
        plt.tight_layout()
        return fig
