"""
emotion.py - Emotion Classification Module
Chat Analyzer Pro - Day 8 Implementation

This module provides advanced emotion classification using HuggingFace transformers.
Classifies messages into 6 emotions: Joy, Sadness, Anger, Fear, Surprise, Love

Dependencies:
    - transformers>=4.30.0
    - torch>=2.0.0
    - pandas>=1.5.0
    - numpy>=1.24.0

Usage:
    from src.analysis.emotion import EmotionAnalyzer
    
    analyzer = EmotionAnalyzer()
    df_with_emotions = analyzer.analyze_emotions(df)
    summary = analyzer.get_emotion_summary(df_with_emotions)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Global analyzer instance for reuse
_emotion_analyzer = None
_emotion_model_loaded = False


class EmotionAnalyzer:
    """
    Advanced emotion classification using HuggingFace transformers.
    Optimized for batch processing and cloud deployment.
    """
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        """
        Initialize the emotion analyzer.
        
        Args:
            model_name: HuggingFace model identifier for emotion classification
        """
        self.model_name = model_name
        self.pipeline = None
        self.emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love']
        self._initialize_model()
    
    def _initialize_model(self):
        """Load the emotion classification model."""
        global _emotion_analyzer, _emotion_model_loaded
        
        if _emotion_model_loaded and _emotion_analyzer is not None:
            self.pipeline = _emotion_analyzer
            print("✅ Using cached emotion model")
            return
        
        try:
            from transformers import pipeline
            print(f"🚀 Loading emotion classification model: {self.model_name}")
            print("   This may take a moment on first run...")
            
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,  # Return all emotion scores
                device=-1  # CPU (use 0 for GPU if available)
            )
            
            _emotion_analyzer = self.pipeline
            _emotion_model_loaded = True
            print("✅ Emotion model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading emotion model: {e}")
            print("   Falling back to rule-based emotion detection...")
            self.pipeline = None
    
    def analyze_single_message(self, text: str) -> Dict[str, float]:
        """
        Analyze emotion in a single message.
        
        Args:
            text: Message text to analyze
            
        Returns:
            Dictionary with emotion scores
        """
        # Handle empty or invalid messages
        if not text or not isinstance(text, str) or text.strip() == "":
            return self._get_neutral_emotions()
        
        # Skip media messages
        if "<Media omitted>" in text or "<media omitted>" in text.lower():
            return self._get_neutral_emotions()
        
        # Skip very short messages (likely just emojis or punctuation)
        if len(text.strip()) < 3:
            return self._get_neutral_emotions()
        
        try:
            if self.pipeline is not None:
                # Use transformer model
                result = self.pipeline(text[:512])[0]  # Limit to 512 chars for efficiency
                
                # Convert to our standard format
                emotion_scores = {item['label']: item['score'] for item in result}
                
                # Ensure all 6 emotions are present
                for emotion in self.emotions:
                    if emotion not in emotion_scores:
                        emotion_scores[emotion] = 0.0
                
                return emotion_scores
            else:
                # Fallback to rule-based detection
                return self._rule_based_emotion(text)
                
        except Exception as e:
            print(f"⚠️ Error analyzing message: {e}")
            return self._get_neutral_emotions()
    
    def _get_neutral_emotions(self) -> Dict[str, float]:
        """Return neutral emotion scores."""
        return {emotion: 1/len(self.emotions) for emotion in self.emotions}
    
    def _rule_based_emotion(self, text: str) -> Dict[str, float]:
        """
        Simple rule-based emotion detection as fallback.
        Uses keyword matching.
        """
        text_lower = text.lower()
        scores = {emotion: 0.0 for emotion in self.emotions}
        
        # Joy keywords
        joy_words = ['happy', 'great', 'wonderful', 'amazing', 'love', 'excellent', 
                     'perfect', 'awesome', '😊', '😄', '🎉', '❤️', '😍']
        scores['joy'] = sum(1 for word in joy_words if word in text_lower)
        
        # Sadness keywords
        sad_words = ['sad', 'sorry', 'unfortunately', 'miss', 'lost', 'cry', 
                     '😢', '😭', '☹️']
        scores['sadness'] = sum(1 for word in sad_words if word in text_lower)
        
        # Anger keywords
        anger_words = ['angry', 'annoyed', 'frustrated', 'hate', 'terrible', 
                       'worst', '😠', '😡', '🤬']
        scores['anger'] = sum(1 for word in anger_words if word in text_lower)
        
        # Fear keywords
        fear_words = ['scared', 'afraid', 'worry', 'anxious', 'nervous', 'fear',
                      '😨', '😰', '😱']
        scores['fear'] = sum(1 for word in fear_words if word in text_lower)
        
        # Surprise keywords
        surprise_words = ['wow', 'amazing', 'unexpected', 'surprise', 'shocked',
                         '😮', '😲', '🤯', '!']
        scores['surprise'] = sum(1 for word in surprise_words if word in text_lower)
        
        # Love keywords
        love_words = ['love', 'adore', 'cherish', 'care', 'heart', '❤️', '😘', '💕']
        scores['love'] = sum(1 for word in love_words if word in text_lower)
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            scores = self._get_neutral_emotions()
        
        return scores
    
    def analyze_emotions(self, 
                        df: pd.DataFrame, 
                        text_column: str = 'message',
                        batch_size: int = 32) -> pd.DataFrame:
        """
        Analyze emotions for all messages in a DataFrame.
        
        Args:
            df: DataFrame with messages
            text_column: Column name containing message text
            batch_size: Number of messages to process at once (for efficiency)
            
        Returns:
            DataFrame with added emotion columns
        """
        print(f"\n🎭 Analyzing emotions for {len(df)} messages...")
        
        df_copy = df.copy()
        
        # Initialize emotion columns
        for emotion in self.emotions:
            df_copy[f'emotion_{emotion}'] = 0.0
        
        # Process messages
        for idx, row in df_copy.iterrows():
            message = row[text_column]
            emotion_scores = self.analyze_single_message(message)
            
            # Add scores to dataframe
            for emotion, score in emotion_scores.items():
                df_copy.at[idx, f'emotion_{emotion}'] = score
        
        # Add dominant emotion column
        emotion_cols = [f'emotion_{e}' for e in self.emotions]
        df_copy['dominant_emotion'] = df_copy[emotion_cols].idxmax(axis=1).str.replace('emotion_', '')
        df_copy['emotion_confidence'] = df_copy[emotion_cols].max(axis=1)
        
        print("✅ Emotion analysis complete!")
        return df_copy
    
    def get_emotion_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive emotion summary statistics.
        
        Args:
            df: DataFrame with emotion analysis results
            
        Returns:
            Dictionary containing emotion statistics
        """
        emotion_cols = [f'emotion_{e}' for e in self.emotions]
        
        summary = {
            'total_messages': len(df),
            'emotion_distribution': df['dominant_emotion'].value_counts().to_dict(),
            'average_emotion_scores': {
                emotion: df[f'emotion_{emotion}'].mean() 
                for emotion in self.emotions
            },
            'emotion_intensity': {
                emotion: {
                    'mean': df[f'emotion_{emotion}'].mean(),
                    'std': df[f'emotion_{emotion}'].std(),
                    'max': df[f'emotion_{emotion}'].max(),
                    'min': df[f'emotion_{emotion}'].min()
                }
                for emotion in self.emotions
            }
        }
        
        # Add per-sender statistics if sender column exists
        if 'sender' in df.columns:
            summary['by_sender'] = {}
            for sender in df['sender'].unique():
                sender_df = df[df['sender'] == sender]
                summary['by_sender'][sender] = {
                    'message_count': len(sender_df),
                    'dominant_emotions': sender_df['dominant_emotion'].value_counts().to_dict(),
                    'avg_emotion_scores': {
                        emotion: sender_df[f'emotion_{emotion}'].mean()
                        for emotion in self.emotions
                    }
                }
        
        # Add temporal analysis if datetime column exists
        if 'datetime' in df.columns:
            df_temp = df.copy()
            df_temp['datetime'] = pd.to_datetime(df_temp['datetime'])
            df_temp['date'] = df_temp['datetime'].dt.date
            
            summary['temporal_analysis'] = {
                'by_date': df_temp.groupby('date')['dominant_emotion'].apply(
                    lambda x: x.value_counts().to_dict()
                ).to_dict()
            }
        
        return summary
    
    def find_most_emotional_messages(self, 
                                     df: pd.DataFrame, 
                                     emotion: str = None,
                                     n: int = 5) -> pd.DataFrame:
        """
        Find messages with highest scores for specific emotion(s).
        
        Args:
            df: DataFrame with emotion analysis
            emotion: Specific emotion to filter (None for all)
            n: Number of top messages to return
            
        Returns:
            DataFrame with most emotional messages
        """
        if emotion:
            if emotion not in self.emotions:
                raise ValueError(f"Invalid emotion. Choose from: {self.emotions}")
            
            return df.nlargest(n, f'emotion_{emotion}')[
                ['datetime', 'sender', 'message', f'emotion_{emotion}', 'dominant_emotion']
            ]
        else:
            # Return top messages by confidence score
            return df.nlargest(n, 'emotion_confidence')[
                ['datetime', 'sender', 'message', 'dominant_emotion', 'emotion_confidence']
            ]
    
    def get_emotion_timeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create emotion timeline showing emotion evolution over time.
        
        Args:
            df: DataFrame with emotion analysis and datetime
            
        Returns:
            DataFrame with temporal emotion aggregations
        """
        if 'datetime' not in df.columns:
            raise ValueError("DataFrame must contain 'datetime' column")
        
        df_temp = df.copy()
        df_temp['datetime'] = pd.to_datetime(df_temp['datetime'])
        
        # Group by date and calculate average emotion scores
        df_temp['date'] = df_temp['datetime'].dt.date
        
        emotion_cols = [f'emotion_{e}' for e in self.emotions]
        timeline = df_temp.groupby('date')[emotion_cols].mean().reset_index()
        
        return timeline


# Convenience functions for quick analysis
def quick_emotion_analysis(df: pd.DataFrame, 
                           text_column: str = 'message',
                           plot: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform complete emotion analysis with visualization.
    
    Args:
        df: DataFrame with messages
        text_column: Column containing message text
        plot: Whether to generate visualizations
        
    Returns:
        Tuple of (analyzed DataFrame, summary statistics)
    """
    analyzer = EmotionAnalyzer()
    df_analyzed = analyzer.analyze_emotions(df, text_column)
    summary = analyzer.get_emotion_summary(df_analyzed)
    
    if plot:
        try:
            plot_emotion_analysis(df_analyzed, summary)
        except Exception as e:
            print(f"⚠️ Could not generate plots: {e}")
    
    return df_analyzed, summary


def plot_emotion_analysis(df: pd.DataFrame, summary: Dict):
    """
    Create comprehensive emotion visualizations.
    
    Args:
        df: DataFrame with emotion analysis
        summary: Summary statistics dictionary
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🎭 Emotion Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Emotion Distribution Pie Chart
    ax1 = axes[0, 0]
    emotion_dist = summary['emotion_distribution']
    colors = plt.cm.Set3(range(len(emotion_dist)))
    ax1.pie(emotion_dist.values(), labels=emotion_dist.keys(), autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax1.set_title('Overall Emotion Distribution', fontweight='bold')
    
    # 2. Average Emotion Scores Bar Chart
    ax2 = axes[0, 1]
    emotions = list(summary['average_emotion_scores'].keys())
    scores = list(summary['average_emotion_scores'].values())
    bars = ax2.bar(emotions, scores, color=plt.cm.Set3(range(len(emotions))))
    ax2.set_title('Average Emotion Scores', fontweight='bold')
    ax2.set_ylabel('Average Score')
    ax2.set_xlabel('Emotion')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Emotion Timeline
    ax3 = axes[1, 0]
    if 'datetime' in df.columns:
        analyzer = EmotionAnalyzer()
        timeline = analyzer.get_emotion_timeline(df)
        
        for emotion in analyzer.emotions:
            ax3.plot(timeline['date'], timeline[f'emotion_{emotion}'], 
                    marker='o', label=emotion.capitalize(), linewidth=2)
        
        ax3.set_title('Emotion Timeline', fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Average Emotion Score')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax3.text(0.5, 0.5, 'Timeline requires datetime column', 
                ha='center', va='center')
        ax3.set_title('Emotion Timeline', fontweight='bold')
    
    # 4. Per-Sender Emotion Comparison
    ax4 = axes[1, 1]
    if 'sender' in df.columns and 'by_sender' in summary:
        senders = list(summary['by_sender'].keys())
        emotions = analyzer.emotions
        
        x = np.arange(len(emotions))
        width = 0.35
        
        if len(senders) >= 2:
            sender1_scores = [summary['by_sender'][senders[0]]['avg_emotion_scores'][e] 
                            for e in emotions]
            sender2_scores = [summary['by_sender'][senders[1]]['avg_emotion_scores'][e] 
                            for e in emotions]
            
            ax4.bar(x - width/2, sender1_scores, width, label=senders[0], alpha=0.8)
            ax4.bar(x + width/2, sender2_scores, width, label=senders[1], alpha=0.8)
            
            ax4.set_title('Emotion Comparison by Sender', fontweight='bold')
            ax4.set_ylabel('Average Score')
            ax4.set_xlabel('Emotion')
            ax4.set_xticks(x)
            ax4.set_xticklabels([e.capitalize() for e in emotions])
            ax4.legend()
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax4.text(0.5, 0.5, 'Need at least 2 senders', ha='center', va='center')
    else:
        ax4.text(0.5, 0.5, 'Sender comparison unavailable', 
                ha='center', va='center')
        ax4.set_title('Emotion by Sender', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def combine_sentiment_emotion(df_sentiment: pd.DataFrame, 
                              df_emotion: pd.DataFrame) -> pd.DataFrame:
    """
    Combine sentiment and emotion analysis results.
    
    Args:
        df_sentiment: DataFrame with sentiment analysis
        df_emotion: DataFrame with emotion analysis
        
    Returns:
        Combined DataFrame with both sentiment and emotion features
    """
    # Merge on index
    df_combined = df_sentiment.copy()
    
    # Add emotion columns
    emotion_cols = [col for col in df_emotion.columns 
                   if col.startswith('emotion_') or col == 'dominant_emotion']
    
    for col in emotion_cols:
        if col in df_emotion.columns:
            df_combined[col] = df_emotion[col]
    
    return df_combined


# Module info
print("🎭 Emotion Classification Module - Chat Analyzer Pro")
print("This module provides advanced emotion analysis using transformers")
print("Usage: from emotion import EmotionAnalyzer; analyzer = EmotionAnalyzer()")
