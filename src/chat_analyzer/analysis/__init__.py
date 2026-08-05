"""
Chat Analysis Package

This package contains analysis modules for chat data:
- EDA: eda.py
- Sentiment Analysis: sentiment.py
- Emotion Classification: emotion.py
- Relationship Health: relationship_health.py

Usage:
    from chat_analyzer.analysis.relationship_health import analyze_relationship_health
    from chat_analyzer.analysis.sentiment import quick_sentiment_analysis
    from chat_analyzer.analysis.emotion import EmotionAnalyzer
"""

from .relationship_health import (
    analyze_relationship_health,
    calculate_relationship_health_score,
    plot_relationship_health_dashboard_enhanced,
)

__all__ = [
    'analyze_relationship_health',
    'calculate_relationship_health_score',
    'plot_relationship_health_dashboard_enhanced'
]

__version__ = '1.0.0'
__author__ = 'Chat Analyzer Pro Team'
