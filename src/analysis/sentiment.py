import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Sentiment analysis imports
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️ VADER not available. Install with: pip install vaderSentiment")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️ TextBlob not available. Install with: pip install textblob")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Install with: pip install transformers torch")

class SentimentConfig:
    """Configuration settings for sentiment analysis"""
    HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    POSITIVE_THRESHOLD = 0.05
    NEGATIVE_THRESHOLD = -0.05
    MAX_TEXT_LENGTH = 512  # For HF models

# Global analyzers (initialized once)
_vader_analyzer = None
_hf_analyzer = None

def initialize_analyzers(hf_model=None):
    """Initialize sentiment analyzers"""
    global _vader_analyzer, _hf_analyzer
    
    print("🚀 Initializing Sentiment Analyzers...")
    
    # Initialize VADER
    if VADER_AVAILABLE:
        _vader_analyzer = SentimentIntensityAnalyzer()
        print("✅ VADER analyzer loaded")
    
    # Initialize HuggingFace
    if TRANSFORMERS_AVAILABLE:
        try:
            model = hf_model or SentimentConfig.HF_MODEL
            _hf_analyzer = pipeline(
                "sentiment-analysis",
                model=model,
                return_all_scores=True,
                device=-1  # Use CPU
            )
            print("✅ HuggingFace analyzer loaded")
        except Exception as e:
            print(f"⚠️ HuggingFace analyzer failed: {e}")
            _hf_analyzer = None
    
    print("🎯 Sentiment analyzers ready!")

def analyze_vader(text):
    """
    Analyze sentiment using VADER
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: VADER sentiment scores
    """
    if not VADER_AVAILABLE or _vader_analyzer is None:
        return {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}
    
    if pd.isna(text) or str(text).strip() == "":
        return {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}
    
    return _vader_analyzer.polarity_scores(str(text))

def analyze_textblob(text):
    """
    Analyze sentiment using TextBlob
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: TextBlob sentiment scores
    """
    if not TEXTBLOB_AVAILABLE:
        return {'polarity': 0, 'subjectivity': 0}
    
    if pd.isna(text) or str(text).strip() == "":
        return {'polarity': 0, 'subjectivity': 0}
    
    blob = TextBlob(str(text))
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }

def analyze_huggingface(text):
    """
    Analyze sentiment using HuggingFace transformers
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: HuggingFace sentiment scores
    """
    if not TRANSFORMERS_AVAILABLE or _hf_analyzer is None:
        return {'positive': 0, 'negative': 0, 'neutral': 1}
    
    if pd.isna(text) or str(text).strip() == "":
        return {'positive': 0, 'negative': 0, 'neutral': 1}
    
    try:
        # Truncate text for model limits
        text_truncated = str(text)[:SentimentConfig.MAX_TEXT_LENGTH]
        results = _hf_analyzer(text_truncated)
        
        # Convert to consistent format
        scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        for result in results[0]:
            label = result['label'].lower()
            if 'pos' in label:
                scores['positive'] = result['score']
            elif 'neg' in label:
                scores['negative'] = result['score']
            else:
                scores['neutral'] = result['score']
        
        return scores
    
    except Exception as e:
        print(f"HF analysis error: {e}")
        return {'positive': 0, 'negative': 0, 'neutral': 1}

def categorize_sentiment(score, positive_threshold=None, negative_threshold=None):
    """
    Categorize sentiment based on score
    
    Args:
        score (float): Sentiment score
        positive_threshold (float): Threshold for positive sentiment
        negative_threshold (float): Threshold for negative sentiment
        
    Returns:
        str: Sentiment category
    """
    pos_thresh = positive_threshold or SentimentConfig.POSITIVE_THRESHOLD
    neg_thresh = negative_threshold or SentimentConfig.NEGATIVE_THRESHOLD
    
    if score >= pos_thresh:
        return 'Positive'
    elif score <= neg_thresh:
        return 'Negative'
    else:
        return 'Neutral'

def analyze_sentiment(df, message_col='message'):
    """
    Simple sentiment analysis function for Streamlit integration.
    Returns a DataFrame with basic sentiment columns added.
    
    Args:
        df (pd.DataFrame): DataFrame with messages
        message_col (str): Column name containing messages
        
    Returns:
        pd.DataFrame: DataFrame with sentiment columns added
    """
    # Initialize VADER if not already done
    global _vader_analyzer
    if VADER_AVAILABLE and _vader_analyzer is None:
        _vader_analyzer = SentimentIntensityAnalyzer()
    
    df = df.copy()
    
    # VADER Analysis (most reliable without external downloads)
    if VADER_AVAILABLE:
        vader_results = df[message_col].apply(analyze_vader)
        df['sentiment_score'] = [r['compound'] for r in vader_results]
        df['sentiment_positive'] = [r['pos'] for r in vader_results]
        df['sentiment_neutral'] = [r['neu'] for r in vader_results]
        df['sentiment_negative'] = [r['neg'] for r in vader_results]
        df['sentiment_label'] = df['sentiment_score'].apply(categorize_sentiment)
    else:
        # Fallback if VADER not available
        df['sentiment_score'] = 0.0
        df['sentiment_label'] = 'Neutral'
    
    return df

def add_sentiment_analysis(df, message_col='message', initialize_first=True):
    """
    Add comprehensive sentiment analysis to DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame with messages
        message_col (str): Column name containing messages
        initialize_first (bool): Whether to initialize analyzers first
        
    Returns:
        pd.DataFrame: DataFrame with sentiment columns added
    """
    if initialize_first:
        initialize_analyzers()
    
    print(f"🔍 Analyzing sentiment for {len(df)} messages...")
    df = df.copy()
    
    # VADER Analysis
    if VADER_AVAILABLE:
        print("Running VADER analysis...")
        vader_results = df[message_col].apply(analyze_vader)
        df['vader_compound'] = [r['compound'] for r in vader_results]
        df['vader_pos'] = [r['pos'] for r in vader_results]
        df['vader_neu'] = [r['neu'] for r in vader_results]
        df['vader_neg'] = [r['neg'] for r in vader_results]
        df['vader_sentiment'] = df['vader_compound'].apply(categorize_sentiment)
    
    # TextBlob Analysis
    if TEXTBLOB_AVAILABLE:
        print("Running TextBlob analysis...")
        textblob_results = df[message_col].apply(analyze_textblob)
        df['textblob_polarity'] = [r['polarity'] for r in textblob_results]
        df['textblob_subjectivity'] = [r['subjectivity'] for r in textblob_results]
        df['textblob_sentiment'] = df['textblob_polarity'].apply(categorize_sentiment)
    
    # HuggingFace Analysis
    if TRANSFORMERS_AVAILABLE and _hf_analyzer is not None:
        print("Running HuggingFace analysis...")
        hf_results = df[message_col].apply(analyze_huggingface)
        df['hf_positive'] = [r['positive'] for r in hf_results]
        df['hf_negative'] = [r['negative'] for r in hf_results]
        df['hf_neutral'] = [r['neutral'] for r in hf_results]
        
        # Determine dominant sentiment
        df['hf_sentiment'] = df[['hf_positive', 'hf_negative', 'hf_neutral']].apply(
            lambda x: ['Positive', 'Negative', 'Neutral'][x.argmax()], axis=1
        )
    
    # Create consensus sentiment (majority vote)
    sentiment_cols = []
    if 'vader_sentiment' in df.columns:
        sentiment_cols.append('vader_sentiment')
    if 'textblob_sentiment' in df.columns:
        sentiment_cols.append('textblob_sentiment')
    if 'hf_sentiment' in df.columns:
        sentiment_cols.append('hf_sentiment')
    
    if sentiment_cols:
        df['consensus_sentiment'] = df[sentiment_cols].mode(axis=1)[0]
    
    print("✅ Sentiment analysis complete!")
    return df

def get_sentiment_summary(df):
    """
    Generate comprehensive sentiment analysis summary
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment analysis
        
    Returns:
        dict: Sentiment summary statistics
    """
    summary = {
        'total_messages': len(df),
        'sentiment_distribution': {},
        'average_scores': {},
        'by_sender': {},
        'temporal_analysis': {}
    }
    
    # Overall sentiment distribution
    if 'consensus_sentiment' in df.columns:
        summary['sentiment_distribution'] = df['consensus_sentiment'].value_counts().to_dict()
    elif 'sentiment_label' in df.columns:
        summary['sentiment_distribution'] = df['sentiment_label'].value_counts().to_dict()
    
    # Average scores
    score_columns = ['vader_compound', 'textblob_polarity', 'sentiment_score']
    for col in score_columns:
        if col in df.columns:
            summary['average_scores'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
    
    # By sender analysis
    sentiment_col = 'consensus_sentiment' if 'consensus_sentiment' in df.columns else 'sentiment_label'
    if 'sender' in df.columns and sentiment_col in df.columns:
        for sender in df['sender'].unique():
            sender_data = df[df['sender'] == sender]
            summary['by_sender'][sender] = {
                'message_count': len(sender_data),
                'sentiment_distribution': sender_data[sentiment_col].value_counts().to_dict(),
                'avg_scores': {}
            }
            
            for col in score_columns:
                if col in sender_data.columns:
                    summary['by_sender'][sender]['avg_scores'][col] = sender_data[col].mean()
    
    # Temporal analysis (if datetime available)
    if 'datetime' in df.columns:
        df_temp = df.copy()
        df_temp['date'] = pd.to_datetime(df_temp['datetime']).dt.date
        df_temp['hour'] = pd.to_datetime(df_temp['datetime']).dt.hour
        
        # Daily averages
        score_col = 'vader_compound' if 'vader_compound' in df.columns else 'sentiment_score'
        if score_col in df.columns:
            daily_sentiment = df_temp.groupby('date')[score_col].mean()
            summary['temporal_analysis']['daily_avg_sentiment'] = daily_sentiment.to_dict()
        
        # Hourly patterns
        if score_col in df.columns:
            hourly_sentiment = df_temp.groupby('hour')[score_col].mean()
            summary['temporal_analysis']['hourly_avg_sentiment'] = hourly_sentiment.to_dict()
    
    return summary

def plot_sentiment_analysis(df, figsize=(15, 10)):
    """
    Create comprehensive sentiment analysis visualizations
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment analysis
        figsize (tuple): Figure size for plots
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Chat Sentiment Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Determine which sentiment column to use
    sentiment_col = 'consensus_sentiment' if 'consensus_sentiment' in df.columns else 'sentiment_label'
    score_col = 'vader_compound' if 'vader_compound' in df.columns else 'sentiment_score'
    
    # 1. Sentiment Distribution
    if sentiment_col in df.columns:
        ax1 = axes[0, 0]
        sentiment_counts = df[sentiment_col].value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        wedges, texts, autotexts = ax1.pie(sentiment_counts.values,
                                          labels=sentiment_counts.index,
                                          autopct='%1.1f%%',
                                          colors=colors[:len(sentiment_counts)],
                                          startangle=90)
        ax1.set_title('Overall Sentiment Distribution', fontweight='bold')
    
    # 2. Sentiment Timeline
    if 'datetime' in df.columns and score_col in df.columns:
        ax2 = axes[0, 1]
        df_temp = df.copy()
        df_temp['datetime'] = pd.to_datetime(df_temp['datetime'])
        df_temp['date'] = df_temp['datetime'].dt.date
        
        daily_sentiment = df_temp.groupby('date')[score_col].mean().reset_index()
        
        ax2.plot(daily_sentiment['date'], daily_sentiment[score_col],
                marker='o', linewidth=2, label='Sentiment', color='#3498db')
        
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax2.set_title('Sentiment Timeline', fontweight='bold')
        ax2.set_ylabel('Sentiment Score')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. Sentiment by Sender
    if 'sender' in df.columns and sentiment_col in df.columns:
        ax3 = axes[1, 0]
        sender_sentiment = df.groupby('sender')[sentiment_col].value_counts().unstack(fill_value=0)
        sender_sentiment.plot(kind='bar', ax=ax3, color=colors[:len(sender_sentiment.columns)])
        ax3.set_title('Sentiment Distribution by Sender', fontweight='bold')
        ax3.set_ylabel('Number of Messages')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Sentiment')
    
    # 4. Hourly Sentiment Pattern
    if 'datetime' in df.columns and score_col in df.columns:
        ax4 = axes[1, 1]
        df_temp = df.copy()
        df_temp['hour'] = pd.to_datetime(df_temp['datetime']).dt.hour
        hourly_sentiment = df_temp.groupby('hour')[score_col].mean()
        
        colors_hourly = ['red' if x < -0.05 else 'orange' if x < 0.05 else 'green' 
                        for x in hourly_sentiment.values]
        ax4.bar(hourly_sentiment.index, hourly_sentiment.values, color=colors_hourly)
        ax4.set_title('Average Sentiment by Hour', fontweight='bold')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Average Score')
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def find_extreme_messages(df, n=3):
    """
    Find most positive and negative messages
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment analysis
        n (int): Number of messages to return for each extreme
        
    Returns:
        dict: Most positive and negative messages
    """
    results = {'most_positive': [], 'most_negative': []}
    
    score_col = 'vader_compound' if 'vader_compound' in df.columns else 'sentiment_score'
    sentiment_col = 'consensus_sentiment' if 'consensus_sentiment' in df.columns else 'sentiment_label'
    
    if score_col in df.columns:
        # Most positive
        most_pos = df.nlargest(n, score_col)
        for _, row in most_pos.iterrows():
            results['most_positive'].append({
                'message': row.get('message', ''),
                'sender': row.get('sender', ''),
                'score': row.get(score_col, 0),
                'sentiment': row.get(sentiment_col, 'Unknown')
            })
        
        # Most negative
        most_neg = df.nsmallest(n, score_col)
        for _, row in most_neg.iterrows():
            results['most_negative'].append({
                'message': row.get('message', ''),
                'sender': row.get('sender', ''),
                'score': row.get(score_col, 0),
                'sentiment': row.get(sentiment_col, 'Unknown')
            })
    
    return results

# Utility functions for easy usage
def quick_sentiment_analysis(df, message_col='message', plot=True):
    """
    Quick sentiment analysis with default settings
    
    Args:
        df (pd.DataFrame): DataFrame with messages
        message_col (str): Message column name
        plot (bool): Whether to create plots
        
    Returns:
        tuple: (analyzed_df, summary_dict)
    """
    # Analyze sentiment
    df_analyzed = add_sentiment_analysis(df, message_col)
    
    # Get summary
    summary = get_sentiment_summary(df_analyzed)
    
    # Plot if requested
    if plot:
        plot_sentiment_analysis(df_analyzed)
    
    return df_analyzed, summary

if __name__ == "__main__":
    print("📊 Sentiment Analysis Module - Chat Analyzer Pro")
    print("This module provides comprehensive sentiment analysis for chat data")
    print("Usage: import sentiment; sentiment.analyze_sentiment(df)")
