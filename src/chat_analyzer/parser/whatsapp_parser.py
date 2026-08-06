import os
import re
from datetime import datetime

import pandas as pd

# Known WhatsApp timestamp formats (C-17 — no M/D-vs-D/M disambiguation
# heuristics beyond a fixed locale bias; %d/%m is tried first because the
# dominant DD/MM/YYYY export (incl. iOS "[14/06/2024, 2:30:45 PM]") parses
# correctly there and a US-first order would silently misread e.g. 02/11/25
# as Feb 11 instead of Nov 2 whenever the day <= 12). Covers 2/4-digit year,
# with and without seconds, 12h AM/PM and 24h variants.
DATE_FORMATS = (
    "%d/%m/%y %I:%M %p", "%m/%d/%y %I:%M %p",
    "%d/%m/%Y %I:%M %p", "%m/%d/%Y %I:%M %p",
    "%d/%m/%y %I:%M:%S %p", "%m/%d/%y %I:%M:%S %p",
    "%d/%m/%Y %I:%M:%S %p", "%m/%d/%Y %I:%M:%S %p",
    "%d/%m/%y %H:%M", "%m/%d/%y %H:%M",
    "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M",
    "%d/%m/%y %H:%M:%S", "%m/%d/%y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
)


class WhatsAppParser:
    """
    Parser for WhatsApp chat export files (.txt format)
    """

    def __init__(self):
        # Regex pattern to match WhatsApp message format
        # Handles various date/time formats. `:\s?(.*)` makes the whitespace
        # after the sender colon optional so empty-body messages (deleted or
        # media-removed lines like "7:08 PM - sujoy:" after strip()) still
        # count as real messages instead of falling through to system lines.
        self.message_pattern = re.compile(
            r'(\d{1,2}/\d{1,2}/\d{2,4}),?\s(\d{1,2}:\d{2}(?::\d{2})?)?\s?([AaPp][Mm])?\s?-\s([^:]+):\s?(.*)'
        )

        # Alternative pattern for different date formats
        self.alt_pattern = re.compile(
            r'\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s(\d{1,2}:\d{2}(?::\d{2})?)?\s?([AaPp][Mm])?\]\s([^:]+):\s?(.*)'
        )

        # System-line classification (D-18): header-without-sender lines,
        # bare no-header system phrases, and the encryption notice.
        self.system_header_pattern = re.compile(
            r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s(\d{1,2}:\d{2}(?::\d{2})?)\s?([AaPp][Mm])?\s?-\s(.+)$'
        )
        self.system_phrase_pattern = re.compile(
            r'^(.+?)\s+(added|removed|left|joined|created (the )?group|changed (the )?(group|subject|name)|named)\b',
            re.IGNORECASE,
        )
        self.encryption_notice = re.compile(
            r'^Messages and calls are end-to-end encrypted\.?$', re.IGNORECASE
        )

        # Honest counters (D-15/D-16/D-18) — reset per parse_file_with_report call.
        self.skipped_lines = 0
        self.system_messages = 0
        self.total_lines = 0

    def _parse_datetime_strict(self, datetime_str: str) -> datetime | None:
        """Parse a WhatsApp timestamp strictly — None on total failure.

        Never falls back to fabricating a current timestamp: an unparseable
        date is corrupt data and must be counted as skipped (D-15).
        """
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(datetime_str, fmt)  # noqa: DTZ007 - WhatsApp exports carry no timezone; the naive datetime is deliberate and normalized to naive UTC downstream
            except ValueError:
                continue
        return None

    def parse_line_strict(self, line: str) -> dict | None:
        """
        Strictly classify a single line: message row, system line, or None.

        System lines and the encryption notice increment self.system_messages
        and return None (the caller must not treat them as continuations).
        A header match whose date fails to parse increments self.skipped_lines
        and returns None (never a continuation, never a fabricated timestamp).
        """
        line = line.strip()
        if not line:
            return None

        # Try main pattern first, then alternative (iOS bracket)
        match = self.message_pattern.match(line)
        if not match:
            match = self.alt_pattern.match(line)

        if match:
            date_str, time_str, ampm, sender, message = match.groups()
            if time_str and ampm:
                datetime_str = f"{date_str} {time_str} {ampm}"
            else:
                datetime_str = f"{date_str} {time_str or '00:00'}"

            timestamp = self._parse_datetime_strict(datetime_str)
            if timestamp is None:
                self.skipped_lines += 1
                return None

            return {
                'datetime': timestamp,
                'sender': sender.strip(),
                'message': message.strip(),
                'message_length': len(message.strip()),
                'type': 'message',
                'date': timestamp.date(),
                'time': timestamp.time(),
                'hour': timestamp.hour,
                'day_of_week': timestamp.strftime('%A'),
                'word_count': len(message.strip().split()),
            }

        if self.encryption_notice.match(line):
            self.system_messages += 1
            return None

        if self.system_header_pattern.match(line):
            self.system_messages += 1
            return None

        if len(line) <= 120 and self.system_phrase_pattern.match(line):
            self.system_messages += 1
            return None

        return None

    def parse_line(self, line: str) -> dict | None:
        """
        Parse a single line from WhatsApp chat export (QUAL-01 delegate).

        Kept for backward compatibility: returns the legacy dict shape with a
        'timestamp' key (not 'datetime').
        """
        row = self.parse_line_strict(line)
        if row is None:
            return None
        row = dict(row)
        row['timestamp'] = row.pop('datetime')
        return row

    def parse_file_with_report(self, file_path: str) -> tuple[list[dict], dict]:
        """
        Parse an entire WhatsApp export with honest counters.

        Returns (rows, counts) where counts has total_lines, parsed_messages,
        skipped_lines and system_messages. System rows never enter rows.
        """
        self.skipped_lines = 0
        self.system_messages = 0
        self.total_lines = 0

        rows: list[dict] = []
        current_message = None

        with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as file:
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                self.total_lines += 1

                before_system = self.system_messages
                before_skipped = self.skipped_lines
                parsed = self.parse_line_strict(line)

                if parsed is not None:
                    # New message
                    if current_message:
                        rows.append(current_message)
                    current_message = parsed
                elif self.system_messages > before_system:
                    # System line — already counted, never a continuation (D-18)
                    continue
                elif self.skipped_lines > before_skipped:
                    # Header line with unparseable date — already counted (D-15)
                    continue
                elif current_message:
                    # Continuation of previous message (multiline)
                    current_message['message'] += '\n' + stripped
                    current_message['message_length'] = len(current_message['message'])
                    current_message['word_count'] = len(current_message['message'].split())
                else:
                    # Orphan line with no current message — honest count (D-16)
                    self.skipped_lines += 1

        if current_message:
            rows.append(current_message)

        return rows, {
            'total_lines': self.total_lines,
            'parsed_messages': len(rows),
            'skipped_lines': self.skipped_lines,
            'system_messages': self.system_messages,
        }

    def parse_file(self, file_path: str) -> pd.DataFrame:
        """
        Parse entire WhatsApp chat export file (QUAL-01 entry point).

        Hardened internals: failed-date lines are dropped (counted), never
        fabricated with a current timestamp. Returns a DataFrame with the legacy
        'timestamp' column so the existing _add_features pipeline works.
        """
        rows, _ = self.parse_file_with_report(file_path)
        df = pd.DataFrame(rows)

        if df.empty:
            return df

        df = df.rename(columns={'datetime': 'timestamp'})
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

    def get_basic_stats(self, df: pd.DataFrame) -> dict:
        """
        Get basic statistics from parsed data

        Args:
            df (pd.DataFrame): Parsed chat DataFrame

        Returns:
            dict: Basic statistics
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
def parse_whatsapp_chat(file_path: str, output_path: str | None = None) -> pd.DataFrame:
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
