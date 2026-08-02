"""
Chat Analyzer Pro - Main Package
Complete chat analysis suite with NLP, visualization, and gamification.
"""

import logging

__version__ = "0.1.0"
__author__ = "Sujoy"
__project__ = "Chat Analyzer Pro"

# Package metadata
__all__ = [
    "analysis",
    "ingest",
    "parser",
    "reporting",
    "utils"
]

# Canonical package-level NullHandler (Python logging best practice): without a
# handler, every logger.exception() in a degraded path (chart encoding failure,
# browser auto-open failure) falls through to logging's lastResort handler and
# dumps a traceback to stderr. The CLI's UX is console narration (pipeline /
# render / main), so analysis-side logs are deliberately silent - same decision
# as utils/visualization.py's NullHandler (Phase 2, Task 5).
logging.getLogger("chat_analyzer").addHandler(logging.NullHandler())
