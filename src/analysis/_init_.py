"""
Chat Analysis Package

This package contains analysis modules for chat data:
- EDA: eda.py
- Sentiment Analysis: sentiment.py  
- Emotion Classification: emotion.py
- Relationship Health: relationship_health.py

Usage:
    from src.analysis.relationship_health import analyze_relationship_health
    from src.analysis.sentiment import analyze_sentiment
    from src.analysis.emotion import classify_emotions
"""

from .relationship_health import (
    analyze_relationship_health,
    calculate_relationship_health_score,
    plot_relationship_health_dashboard
)

# Import other modules when they're available
try:
    from .sentiment import analyze_sentiment
    from .eda import perform_eda
    from .emotion import classify_emotions
except ImportError:
    # Modules not yet implemented
    pass

__all__ = [
    'analyze_relationship_health',
    'calculate_relationship_health_score', 
    'plot_relationship_health_dashboard'
]

__version__ = '1.0.0'
__author__ = 'Chat Analyzer Pro Team'
