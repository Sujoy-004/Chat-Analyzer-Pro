"""
Chat Parser Package

This package contains parsers for different chat platforms:
- WhatsApp parser: whatsapp_parser.py
- Telegram parser: telegram_parser.py

Usage:
    from src.parser.whatsapp_parser import parse_whatsapp_chat
    from src.parser.telegram_parser import parse_telegram_chat
"""

from .whatsapp_parser import parse_whatsapp_chat
from .telegram_parser import parse_telegram_json

__all__ = [
    'parse_whatsapp_chat',
    'parse_telegram_json'
]

__version__ = '1.0.0'
__author__ = 'Chat Analyzer Pro Team'
