"""
Text Preprocessing Utilities
Functions for cleaning and preparing chat messages for analysis.
"""

import re
import pandas as pd
from typing import List, Optional


def preprocess_text(text: str, lowercase: bool = True, remove_urls: bool = True,
                   remove_mentions: bool = False) -> str:
    """
    Preprocess text message for analysis.
    
    Args:
        text: Input text string
        lowercase: Convert to lowercase
        remove_urls: Remove URLs
        remove_mentions: Remove @mentions
        
    Returns:
        Preprocessed text string
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    if remove_urls:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove @mentions
    if remove_mentions:
        text = re.sub(r'@\w+', '', text)
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def clean_messages(df: pd.DataFrame, message_col: str = 'message',
                   min_length: int = 1) -> pd.DataFrame:
    """
    Clean message DataFrame by removing empty/invalid messages.
    
    Args:
        df: Input DataFrame
        message_col: Name of message column
        min_length: Minimum message length to keep
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Remove null messages
    df = df[df[message_col].notna()]
    
    # Remove empty messages after stripping
    df[message_col] = df[message_col].astype(str).str.strip()
    df = df[df[message_col].str.len() >= min_length]
    
    # Filter system messages
    system_patterns = [
        r'Messages and calls are end-to-end encrypted',
        r'<Media omitted>',
        r'This message was deleted',
        r'You deleted this message',
        r'joined using this group\'s invite link',
        r'left',
        r'changed the subject',
        r'changed this group\'s icon',
        r'added you',
        r'removed'
    ]
    
    for pattern in system_patterns:
        df = df[~df[message_col].str.contains(pattern, case=False, na=False)]
    
    return df.reset_index(drop=True)


def extract_emojis(text: str) -> List[str]:
    """
    Extract all emojis from text.
    
    Args:
        text: Input text string
        
    Returns:
        List of emojis found in text
    """
    if not isinstance(text, str):
        return []
    
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    return emoji_pattern.findall(text)


def remove_emojis(text: str) -> str:
    """
    Remove all emojis from text.
    
    Args:
        text: Input text string
        
    Returns:
        Text with emojis removed
    """
    if not isinstance(text, str):
        return ""
    
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r'', text)


def extract_urls(text: str) -> List[str]:
    """
    Extract all URLs from text.
    
    Args:
        text: Input text string
        
    Returns:
        List of URLs found in text
    """
    if not isinstance(text, str):
        return []
    
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.findall(text)


def tokenize_simple(text: str) -> List[str]:
    """
    Simple tokenization by splitting on whitespace.
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens
    """
    if not isinstance(text, str):
        return []
    
    return text.split()


def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Input text string
        
    Returns:
        Number of words
    """
    if not isinstance(text, str):
        return 0
    
    return len(tokenize_simple(text))


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Input text string
        
    Returns:
        Text with normalized whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    return text.strip()


def contains_question(text: str) -> bool:
    """
    Check if text contains a question.
    
    Args:
        text: Input text string
        
    Returns:
        True if text contains question mark or question words
    """
    if not isinstance(text, str):
        return False
    
    # Check for question mark
    if '?' in text:
        return True
    
    # Check for question words
    question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 'whose']
    text_lower = text.lower()
    
    return any(text_lower.startswith(qw) for qw in question_words)


def is_short_response(text: str, threshold: int = 10) -> bool:
    """
    Check if text is a short response.
    
    Args:
        text: Input text string
        threshold: Character length threshold
        
    Returns:
        True if text is shorter than threshold
    """
    if not isinstance(text, str):
        return True
    
    return len(text.strip()) < threshold


# Example usage
if __name__ == "__main__":
    # Test preprocessing
    sample_text = "Check out this link: https://example.com @user 😊"
    print("Original:", sample_text)
    print("Preprocessed:", preprocess_text(sample_text))
    print("Emojis:", extract_emojis(sample_text))
    print("URLs:", extract_urls(sample_text))
