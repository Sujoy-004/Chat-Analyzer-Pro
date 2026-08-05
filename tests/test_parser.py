"""
Unit Tests for Chat Parser Modules (rewired to real chat_analyzer.* modules — D-16).

Tests exercise the REAL WhatsAppParser and Telegram parse functions on inline
fixture exports (D-16). The legacy hand-rolled regex/split parsing copies were
removed in this rewire; assertions target the shipped parser contracts from
Phase 2 (strict date parsing, system-message classification, honest counters,
tz-aware -> naive UTC normalization).
"""

import json
import os
import tempfile
import unittest
from datetime import datetime

import pandas as pd

from chat_analyzer.parser.telegram_parser import parse_telegram_chat_with_report
from chat_analyzer.parser.whatsapp_parser import WhatsAppParser


class TestWhatsAppParser(unittest.TestCase):
    """Test cases for the real WhatsAppParser (inline fixture export)."""

    def setUp(self):
        """Write an inline WhatsApp export fixture to a temp file."""
        self.sample_whatsapp_text = """12/25/23, 9:30 AM - Alice: Hey! How are you?
12/25/23, 9:35 AM - Bob: I'm good, thanks! 😊
12/25/23, 10:15 AM - Messages and calls are end-to-end encrypted.
12/25/23, 10:20 AM - Alice: That's great to hear
12/25/23, 10:25 AM - Bob: What are you up to today?
12/25/23, 10:30 AM - Bob: This is line 1
and line 2 continues
12/26/23, 8:00 AM - Alice: Going to the park!"""

        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', delete=False, suffix='.txt'
        ) as self.temp_file:
            self.temp_file.write(self.sample_whatsapp_text)

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def _parse(self):
        """Parse the fixture with the REAL parser, returning (rows, counts)."""
        return WhatsAppParser().parse_file_with_report(self.temp_file.name)

    def test_basic_parsing(self):
        """Test basic WhatsApp message parsing via the real parser."""
        rows, _ = self._parse()

        self.assertIsInstance(rows, list)
        self.assertEqual(len(rows), 6, "Should parse 6 messages (system line excluded)")
        self.assertIn('datetime', rows[0])
        self.assertIn('sender', rows[0])
        self.assertIn('message', rows[0])

    def test_parse_report_counters(self):
        """The honest report counters match the inline fixture (D-16)."""
        _, counts = self._parse()

        self.assertEqual(counts, {
            'total_lines': 8,
            'parsed_messages': 6,
            'skipped_lines': 0,
            'system_messages': 1,
        })

    def test_system_messages_excluded(self):
        """The encryption notice is counted as system and never a row (D-18)."""
        rows, counts = self._parse()

        self.assertEqual(counts['system_messages'], 1)
        for row in rows:
            self.assertNotIn('end-to-end encrypted', row['message'])

    def test_sender_extraction(self):
        """Test correct sender extraction via the real parser."""
        rows, _ = self._parse()
        df = pd.DataFrame(rows)

        senders = df['sender'].unique().tolist()
        self.assertIn('Alice', senders)
        self.assertIn('Bob', senders)
        self.assertEqual(len(senders), 2)

    def test_datetime_parsing(self):
        """Test datetime parsing accuracy via the real parser."""
        rows, _ = self._parse()
        df = pd.DataFrame(rows)

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['datetime']))

        # Check first message datetime
        first_datetime = df.iloc[0]['datetime']
        self.assertEqual(first_datetime.month, 12)
        self.assertEqual(first_datetime.day, 25)

    def test_message_content_and_emoji(self):
        """Test message content extraction and emoji preservation."""
        rows, _ = self._parse()
        df = pd.DataFrame(rows)

        first_message = df.iloc[0]['message']
        self.assertIn('Hey', first_message)

        # Check emoji preservation
        emoji_message = df[df['message'].str.contains('😊', na=False)]
        self.assertGreater(len(emoji_message), 0, "Should preserve emojis")

    def test_multiline_messages(self):
        """Test handling of multiline messages via the real parser."""
        rows, _ = self._parse()

        multiline = next(r for r in rows if 'line 1' in r['message'])
        self.assertIn('line 1', multiline['message'])
        self.assertIn('line 2 continues', multiline['message'])
        self.assertIn('\n', multiline['message'], "continuation joins with a newline (D-17)")

    def test_message_count_accuracy(self):
        """Test message count per sender via the real parser."""
        rows, _ = self._parse()
        df = pd.DataFrame(rows)

        alice_count = len(df[df['sender'] == 'Alice'])
        bob_count = len(df[df['sender'] == 'Bob'])

        self.assertEqual(alice_count, 3, "Alice should have 3 messages")
        self.assertEqual(bob_count, 3, "Bob should have 3 messages")

    def test_regional_datetime_format(self):
        """Phase 2 strict-parse behavior: EU 24h + iOS bracket formats parse
        to the expected naive UTC datetimes (no fabricated timestamps)."""
        parser = WhatsAppParser()

        eu_row = parser.parse_line_strict('25/12/2023, 21:07 - Bob: EU 24h')
        self.assertIsNotNone(eu_row)
        self.assertEqual(eu_row['datetime'], datetime(2023, 12, 25, 21, 7))  # noqa: DTZ001 - naive constant compared against parser output, naive by design

        ios_row = parser.parse_line_strict('[14/06/2024, 2:30:45 PM] Maria: iOS bracket')
        self.assertIsNotNone(ios_row)
        self.assertEqual(ios_row['datetime'], datetime(2024, 6, 14, 14, 30, 45))  # noqa: DTZ001 - naive constant compared against parser output, naive by design


class TestTelegramParser(unittest.TestCase):
    """Test cases for the real Telegram parser (inline JSON export)."""

    def setUp(self):
        """Write an inline Telegram JSON export fixture to a temp file."""
        self.sample_telegram_json = {
            "messages": [
                {
                    "id": 1,
                    "type": "message",
                    "date": "2023-12-25T09:30:00",
                    "from": "Alice",
                    "text": "Hey! How are you?"
                },
                {
                    "id": 2,
                    "type": "message",
                    "date": "2023-12-25T09:35:00",
                    "from": "Bob",
                    "text": [{"text": "I'm good, "}, "thanks!"]
                },
                {
                    "id": 3,
                    "type": "service",
                    "date": "2023-12-25T09:36:00",
                    "action": "phone_call"
                }
            ]
        }

        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', delete=False, suffix='.json'
        ) as self.temp_file:
            json.dump(self.sample_telegram_json, self.temp_file)

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def _parse(self):
        """Parse the fixture with the REAL parser, returning (rows, counts)."""
        return parse_telegram_chat_with_report(self.temp_file.name)

    def test_json_parsing(self):
        """Test basic Telegram JSON parsing via the real parser."""
        rows, _ = self._parse()

        self.assertIsInstance(rows, list)
        self.assertEqual(len(rows), 2, "Should parse 2 messages (service filtered)")

    def test_parse_report_counters(self):
        """The honest report counters match the inline fixture (D-19)."""
        _, counts = self._parse()

        self.assertEqual(counts, {
            'total_lines': 3,
            'parsed_messages': 2,
            'skipped_lines': 0,
            'system_messages': 1,
        })

    def test_service_messages_filtered(self):
        """Test that service messages are filtered out via the real parser."""
        rows, counts = self._parse()

        self.assertEqual(counts['system_messages'], 1)
        self.assertTrue(all(row.get('type') != 'service' for row in rows))
        self.assertTrue(all(row.get('message_id') != 3 for row in rows))

    def test_datetime_conversion(self):
        """Test datetime conversion is naive UTC via the real parser."""
        rows, _ = self._parse()

        for row in rows:
            self.assertIsNotNone(row['datetime'])
            self.assertIsNone(row['datetime'].tzinfo, "datetimes normalized to naive UTC (D-20)")
        self.assertEqual(rows[0]['datetime'], datetime(2023, 12, 25, 9, 30))  # noqa: DTZ001 - naive constant compared against parser output, naive by design

    def test_sender_extraction(self):
        """Test sender name extraction via the real parser."""
        rows, _ = self._parse()

        senders = [row['sender'] for row in rows]
        self.assertIn('Alice', senders)
        self.assertIn('Bob', senders)

    def test_text_extraction(self):
        """Test message text extraction (incl. entity-array join) via the real parser."""
        rows, _ = self._parse()

        messages = [row['message'] for row in rows]
        self.assertIn('Hey! How are you?', messages)
        self.assertIn("I'm good, thanks!", messages, "entity-array text must join (D-19)")


class TestParserEdgeCases(unittest.TestCase):
    """Test edge cases and error handling with the REAL parsers."""

    def _parse_whatsapp(self, text):
        """Write an inline WhatsApp fixture and parse it with the real parser."""
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', delete=False, suffix='.txt'
        ) as temp:
            temp.write(text)
        try:
            return WhatsAppParser().parse_file_with_report(temp.name)
        finally:
            if os.path.exists(temp.name):
                os.unlink(temp.name)

    def test_special_characters_in_names(self):
        """Test parsing with special characters in sender names."""
        rows, _ = self._parse_whatsapp(
            "12/25/23, 9:30 AM - Alice-123: Message 1\n"
            "12/25/23, 9:35 AM - Bob_Test: Message 2"
        )

        senders = [row['sender'] for row in rows]
        self.assertIn('Alice-123', senders)
        self.assertIn('Bob_Test', senders)

    def test_unicode_handling(self):
        """Test handling of Unicode characters via the real parser."""
        rows, _ = self._parse_whatsapp(
            "12/25/23, 9:30 AM - Alice: Hello! 你好 مرحبا\n"
            "12/25/23, 9:35 AM - Bob: Привет! Bonjour 🌍"
        )

        messages = [row['message'] for row in rows]
        self.assertIn('你好', messages[0])
        self.assertIn('🌍', messages[1])

    def test_very_long_messages(self):
        """Test handling of very long messages without crashing."""
        long_message = "A" * 10000
        rows, counts = self._parse_whatsapp(
            f"12/25/23, 9:30 AM - Alice: {long_message}"
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['message'], long_message)
        self.assertEqual(counts['parsed_messages'], 1)

    def test_empty_whatsapp_file(self):
        """An empty WhatsApp export returns zero rows — no crash (D-16)."""
        rows, counts = self._parse_whatsapp("")

        self.assertEqual(len(rows), 0, "Empty file should return zero rows")
        self.assertEqual(counts['parsed_messages'], 0)

    def test_empty_telegram_export(self):
        """An empty Telegram export raises the friendly ValueError (MEDIUM #3)."""
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', delete=False, suffix='.json'
        ) as temp:
            json.dump({"messages": []}, temp)
        try:
            with self.assertRaises(ValueError) as ctx:
                parse_telegram_chat_with_report(temp.name)
            self.assertIn("Not a Telegram chat export", str(ctx.exception))
        finally:
            if os.path.exists(temp.name):
                os.unlink(temp.name)


def run_parser_tests():
    """Run all parser tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestWhatsAppParser))
    suite.addTests(loader.loadTestsFromTestCase(TestTelegramParser))
    suite.addTests(loader.loadTestsFromTestCase(TestParserEdgeCases))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    unittest.main()
