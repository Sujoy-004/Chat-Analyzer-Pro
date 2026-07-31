import re
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import os

class WhatsAppParser:
    """
    Parser for WhatsApp chat export files (.txt format)
    """
    
    def __init__(self):
        # Regex pattern to match WhatsApp message format
        # Handles various date/time formats
        self.message_pattern = re.compile(
            r'(\d{1,2}/\d{1,2}/\d{2,4}),?\s(\d{1,2}:\d{2}(?::\d{2})?)?\s?([AaPp][Mm])?\s?-\s([^:]+):\s(.*)'
        )
        
        # Alternative pattern for different date formats
        self.alt_pattern = re.compile(
            r'\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s(\d{1,2}:\d{2}(?::\d{2})?)?\s?([AaPp][Mm])?\]\s([^:]+):\s(.*)'
        )
    
    def parse_line(self, line: str) -> Optional[Dict]:
        """
        Parse a single line from WhatsApp chat export
        
        Args:
            line (str): Single line from chat export
            
        Returns:
            Optional[Dict]: Parsed message data or None if not a message
        """
        line = line.strip()
        
        # Try main pattern first
        match = self.message_pattern.match(line)
        if not match:
            # Try alternative pattern
            match = self.alt_pattern.match(line)
        
        if not match:
            return None
        
        date_str, time_str, ampm, sender, message = match.groups()
        
        # Handle different time formats
        if time_str and ampm:
            datetime_str = f"{date_str} {time_str} {ampm}"
            try:
                # Try different datetime formats
                for fmt in ['%m/%d/%y %I:%M %p', '%d/%m/%y %I:%M %p', 
                           '%m/%d/%Y %I:%M %p', '%d/%m/%Y %I:%M %p',
                           '%m/%d/%y %I:%M:%S %p', '%d/%m/%y %I:%M:%S %p']:
                    try:
                        timestamp = datetime.strptime(datetime_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    timestamp = datetime.now()  # Fallback
            except:
                timestamp = datetime.now()
        else:
            # Handle 24-hour format
            datetime_str = f"{date_str} {time_str or '00:00'}"
            try:
                for fmt in ['%m/%d/%y %H:%M', '%d/%m/%y %H:%M', 
                           '%m/%d/%Y %H:%M', '%d/%m/%Y %H:%M',
                           '%m/%d/%y %H:%M:%S', '%d/%m/%y %H:%M:%S']:
                    try:
                        timestamp = datetime.strptime(datetime_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    timestamp = datetime.now()
            except:
                timestamp = datetime.now()
        
        return {
            'timestamp': timestamp,
            'sender': sender.strip(),
            'message': message.strip(),
            'date': timestamp.date(),
            'time': timestamp.time(),
            'hour': timestamp.hour,
            'day_of_week': timestamp.strftime('%A'),
            'message_length': len(message.strip()),
            'word_count': len(message.strip().split()),
        }
    
    def parse_file(self, file_path: str) -> pd.DataFrame:
        """
        Parse entire WhatsApp chat export file
        
        Args:
            file_path (str): Path to WhatsApp export file
            
        Returns:
            pd.DataFrame: Parsed chat data
        """
        messages = []
        current_message = None
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parsed_line = self.parse_line(line)
                
                if parsed_line:
                    # New message
                    if current_message:
                        messages.append(current_message)
                    current_message = parsed_line
                else:
                    # Continuation of previous message (multiline)
                    if current_message and line.strip():
                        current_message['message'] += '\n' + line.strip()
                        current_message['message_length'] = len(current_message['message'])
                        current_message['word_count'] = len(current_message['message'].split())
        
        # Don't forget the last message
        if current_message:
            messages.append(current_message)
        
        df = pd.DataFrame(messages)
        
        if not df.empty:
            # Add additional features
            df = self._add_features(df)
        
        return df
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional features to the parsed DataFrame
        
        Args:
            df (pd.DataFrame): Base parsed DataFrame
            
        Returns:
            pd.DataFrame: Enhanced DataFrame with additional features
        """
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add message sequence number
        df['message_id'] = range(1, len(df) + 1)
        
        # Add is_media flag (common WhatsApp media indicators)
        media_patterns = [
            '<Media omitted>',
            'image omitted',
            'video omitted',
            'audio omitted',
            'document omitted',
            'GIF omitted',
            'sticker omitted'
        ]
        df['is_media'] = df['message'].str.contains('|'.join(media_patterns), case=False, na=False)
        
        # Add emoji count (basic emoji detection)
        emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+')
        df['emoji_count'] = df['message'].apply(lambda x: len(emoji_pattern.findall(str(x))))
        df['has_emoji'] = df['emoji_count'] > 0
        
        # Time-based features
        df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
        df['time_period'] = df['hour'].apply(self._categorize_time_period)
        
        return df
    
    def _categorize_time_period(self, hour: int) -> str:
        """Categorize hour into time periods"""
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save processed DataFrame to CSV
        
        Args:
            df (pd.DataFrame): Processed chat DataFrame
            output_path (str): Output file path
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Processed data saved to: {output_path}")
    
    def get_basic_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get basic statistics from parsed data
        
        Args:
            df (pd.DataFrame): Parsed chat DataFrame
            
        Returns:
            Dict: Basic statistics
        """
        if df.empty:
            return {"error": "No data to analyze"}
        
        stats = {
            "total_messages": len(df),
            "date_range": {
                "start": df['timestamp'].min().strftime('%Y-%m-%d'),
                "end": df['timestamp'].max().strftime('%Y-%m-%d')
            },
            "participants": df['sender'].nunique(),
            "participant_list": df['sender'].unique().tolist(),
            "messages_per_participant": df['sender'].value_counts().to_dict(),
            "media_messages": df['is_media'].sum(),
            "total_words": df['word_count'].sum(),
            "avg_message_length": df['message_length'].mean(),
            "total_emojis": df['emoji_count'].sum()
        }
        
        return stats


# Utility function for easy usage
def parse_whatsapp_chat(file_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Quick function to parse WhatsApp chat and optionally save to CSV
    
    Args:
        file_path (str): Path to WhatsApp export file
        output_path (str, optional): Path to save processed CSV
        
    Returns:
        pd.DataFrame: Parsed chat data
    """
    parser = WhatsAppParser()
    df = parser.parse_file(file_path)
    
    if output_path:
        parser.save_processed_data(df, output_path)
    
    # Print basic stats
    stats = parser.get_basic_stats(df)
    print("\n=== WhatsApp Chat Analysis Summary ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return df


if __name__ == "__main__":
    # Example usage
    sample_file = "data/sample_chats/whatsapp_sample.txt"
    output_file = "data/processed/example_parsed.csv"
    
    if os.path.exists(sample_file):
        df = parse_whatsapp_chat(sample_file, output_file)
        print(f"\nParsed {len(df)} messages successfully!")
        print(f"Columns: {list(df.columns)}")
    else:
        print(f"Sample file not found: {sample_file}")
        print("Please place your WhatsApp export file in the data/sample_chats/ directory")
