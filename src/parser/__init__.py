"""
Parser package for Chat Analyzer Pro
Handles WhatsApp and Telegram chat parsing
"""

from .whatsapp_parser import WhatsAppParser, parse_whatsapp_chat
from .telegram_parser import parse_telegram_chat

__all__ = [
    'WhatsAppParser',
    'parse_whatsapp_chat',
    'parse_telegram_chat',
]
