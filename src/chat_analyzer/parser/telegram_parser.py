import json
import pandas as pd
from datetime import datetime
import requests

def parse_telegram_chat(source):
    """
    Parse Telegram chat from JSON file/URL into structured DataFrame
    
    Args:
        source (str): File path or URL to Telegram JSON export
    
    Returns:
        pd.DataFrame: Parsed chat data
    """
    
    # Load data
    if source.startswith('http'):
        response = requests.get(source)
        data = response.json()
    else:
        with open(source, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    messages = data.get('messages', [])
    parsed_messages = []
    
    for msg in messages:
        if msg.get('type') != 'message':
            continue
            
        date_str = msg.get('date', '')
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            continue
        
        text = msg.get('text', '')
        if isinstance(text, list):
            text_parts = []
            for part in text:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and 'text' in part:
                    text_parts.append(part['text'])
            text = ''.join(text_parts)
        
        if not text and ('photo' in msg or 'video' in msg or 'document' in msg):
            text = '<Media omitted>'
        
        parsed_msg = {
            'datetime': dt,
            'sender': msg.get('from', 'Unknown'),
            'message': text,
            'date': dt.date(),
            'time': dt.time(),
            'hour': dt.hour,
            'message_length': len(text),
            'message_id': msg.get('id'),
            'type': msg.get('type', 'message')
        }
        
        parsed_messages.append(parsed_msg)
    
    return pd.DataFrame(parsed_messages)

