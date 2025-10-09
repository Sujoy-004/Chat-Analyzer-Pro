"""
Chat Analyzer Pro - Main Package
Complete chat analysis suite with NLP, visualization, and gamification.
"""

__version__ = "1.0.0"
__author__ = "Sujoy"
__project__ = "Chat Analyzer Pro"

# Package metadata
__all__ = [
    'parser',
    'analysis',
    'reporting',
    'utils'
]

# Import subpackages
from . import parser
from . import analysis
from . import reporting
from . import utils
