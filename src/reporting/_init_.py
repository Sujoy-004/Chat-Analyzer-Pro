"""
Reporting Package

This package contains report generation modules:
- PDF Report Generator: pdf_report.py
- Weekly Digest: weekly_digest.py (future)

Usage:
    from src.reporting.pdf_report import generate_chat_analysis_pdf
"""

from .pdf_report import (
    generate_chat_analysis_pdf,
    ChatAnalysisPDFGenerator
)

__all__ = [
    'generate_chat_analysis_pdf',
    'ChatAnalysisPDFGenerator'
]

__version__ = '1.0.0' 
__author__ = 'Chat Analyzer Pro Team'
