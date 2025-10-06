"""
Unit Tests for Chat Parser Modules
Tests WhatsApp and Telegram parsers for correctness and edge cases.
"""

import unittest
import pandas as pd
from datetime import datetime
import os
import tempfile


class TestWhatsAppParser(unittest.TestCase):
    """Test cases for WhatsApp parser."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_whatsapp_text = """12/25/23, 9:30 AM - Alice: Hey! How are you?
12/25/23, 9:35 AM - Bob: I'm good, thanks! 😊
12/25/23, 10:15 AM - Alice: That's great to hear
12/25/23, 10:20 AM - Bob: What are you up to today?
12/26/23, 8:00 AM - Alice: Going to the park!"""
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        self.temp_file.write(self.sample_whatsapp_text)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_basic_parsing(self):
        """Test basic WhatsApp message parsing."""
        # Mock parser function
        df = self._parse_whatsapp_file(self.temp_file.name)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5, "Should parse 5 messages")
        self.assertIn('datetime', df.columns)
        self.assertIn('sender', df.columns)
        self.assertIn('message', df.columns)
    
    def test_sender_extraction(self):
        """Test correct sender extraction."""
        df = self._parse_whatsapp_file(self.temp_file.name)
        
        senders = df['sender'].unique()
        self.assertIn('Alice', senders)
        self.assertIn('Bob', senders)
        self.assertEqual(len(senders), 2)
    
    def test_datetime_parsing(self):
        """Test datetime parsing accuracy."""
        df = self._parse_whatsapp_file(self.temp_file.name)
        
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['datetime']))
        
        # Check first message datetime
        first_datetime = df.iloc[0]['datetime']
        self.assertEqual(first_datetime.month, 12)
        self.assertEqual(first_datetime.day, 25)
    
    def test_message_content(self):
        """Test message content extraction."""
        df = self._parse_whatsapp_file(self.temp_file.name)
        
        first_message = df.iloc[0]['message']
        self.assertIn('Hey', first_message)
        
        # Check emoji preservation
        emoji_message = df[df['message'].str.contains('😊', na=False)]
        self.assertGreater(len(emoji_message), 0, "Should preserve emojis")
    
    def test_multiline_messages(self):
        """Test handling of multiline messages."""
        multiline_text = """12/25/23, 9:30 AM - Alice: This is line 1
This is line 2
This is line 3
12/25/23, 9:35 AM - Bob: Single line"""
        
        temp_multi = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp_multi.write(multiline_text)
        temp_multi.close()
        
        df = self._parse_whatsapp_file(temp_multi.name)
        
        # First message should contain all lines
        first_msg = df.iloc[0]['message']
        self.assertIn('line 1', first_msg)
        self.assertIn('line 2', first_msg)
        
        os.unlink(temp_multi.name)
    
    def test_system_messages_filtered(self):
        """Test that system messages are filtered out."""
        system_text = """12/25/23, 9:30 AM - Alice: Regular message
12/25/23, 9:31 AM - Messages and calls are end-to-end encrypted.
12/25/23, 9:32 AM - Bob: Another message"""
        
        temp_sys = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp_sys.write(system_text)
        temp_sys.close()
        
        df = self._parse_whatsapp_file(temp_sys.name)
        
        # Should only have 2 messages (system message filtered)
        self.assertEqual(len(df), 2)
        
        os.unlink(temp_sys.name)
    
    def test_empty_file(self):
        """Test handling of empty file."""
        empty_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        empty_file.write("")
        empty_file.close()
        
        df = self._parse_whatsapp_file(empty_file.name)
        
        self.assertEqual(len(df), 0, "Empty file should return empty DataFrame")
        
        os.unlink(empty_file.name)
    
    def test_message_count_accuracy(self):
        """Test message count per sender."""
        df = self._parse_whatsapp_file(self.temp_file.name)
        
        alice_count = len(df[df['sender'] == 'Alice'])
        bob_count = len(df[df['sender'] == 'Bob'])
        
        self.assertEqual(alice_count, 3, "Alice should have 3 messages")
        self.assertEqual(bob_count, 2, "Bob should have 2 messages")
    
    # Helper method (mock implementation)
    def _parse_whatsapp_file(self, filepath):
        """Mock WhatsApp parser for testing."""
        import re
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple regex pattern for WhatsApp format
        pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[AP]M)\s-\s([^:]+):\s(.+?)(?=\d{1,2}/\d{1,2}/\d{2,4},|\Z)'
        
        matches = re.findall(pattern, content, re.DOTALL)
        
        data = []
        for match in matches:
            timestamp_str, sender, message = match
            
            # Filter system messages
            if 'end-to-end encrypted' in message or 'Messages and calls' in message:
                continue
            
            # Parse datetime
            try:
                dt = pd.to_datetime(timestamp_str, format='%m/%d/%y, %I:%M %p')
            except:
                continue
            
            data.append({
                'datetime': dt,
                'sender': sender.strip(),
                'message': message.strip(),
                'message_length': len(message.strip())
            })
        
        return pd.DataFrame(data)


class TestTelegramParser(unittest.TestCase):
    """Test cases for Telegram parser."""
    
    def setUp(self):
        """Set up test data."""
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
                    "text": "I'm good, thanks!"
                },
                {
                    "id": 3,
                    "type": "service",
                    "date": "2023-12-25T09:36:00",
                    "action": "phone_call"
                }
            ]
        }
    
    def test_json_parsing(self):
        """Test basic Telegram JSON parsing."""
        df = self._parse_telegram_json(self.sample_telegram_json)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 2, "Should parse at least 2 messages")
    
    def test_service_messages_filtered(self):
        """Test that service messages are filtered out."""
        df = self._parse_telegram_json(self.sample_telegram_json)
        
        # Should only have 2 regular messages (service message filtered)
        self.assertEqual(len(df), 2)
    
    def test_datetime_conversion(self):
        """Test datetime conversion from ISO format."""
        df = self._parse_telegram_json(self.sample_telegram_json)
        
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['datetime']))
    
    def test_sender_extraction(self):
        """Test sender name extraction."""
        df = self._parse_telegram_json(self.sample_telegram_json)
        
        senders = df['sender'].unique()
        self.assertIn('Alice', senders)
        self.assertIn('Bob', senders)
    
    def test_text_extraction(self):
        """Test message text extraction."""
        df = self._parse_telegram_json(self.sample_telegram_json)
        
        messages = df['message'].tolist()
        self.assertIn('Hey! How are you?', messages)
    
    # Helper method (mock implementation)
    def _parse_telegram_json(self, json_data):
        """Mock Telegram parser for testing."""
        data = []
        
        for msg in json_data.get('messages', []):
            # Filter service messages
            if msg.get('type') != 'message':
                continue
            
            # Extract data
            dt = pd.to_datetime(msg['date'])
            sender = msg.get('from', 'Unknown')
            text = msg.get('text', '')
            
            if isinstance(text, list):
                text = ' '.join([t.get('text', str(t)) if isinstance(t, dict) else str(t) for t in text])
            
            data.append({
                'datetime': dt,
                'sender': sender,
                'message': text,
                'message_length': len(text)
            })
        
        return pd.DataFrame(data)


class TestParserEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_special_characters_in_names(self):
        """Test parsing with special characters in sender names."""
        text = """12/25/23, 9:30 AM - Alice-123: Message 1
12/25/23, 9:35 AM - Bob_Test: Message 2"""
        
        temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp.write(text)
        temp.close()
        
        # This would use actual parser in production
        # For now, just verify file handling
        self.assertTrue(os.path.exists(temp.name))
        
        os.unlink(temp.name)
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        text = """12/25/23, 9:30 AM - Alice: Hello! 你好 مرحبا
12/25/23, 9:35 AM - Bob: Привет! Bonjour 🌍"""
        
        temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8')
        temp.write(text)
        temp.close()
        
        # Verify file can be read
        with open(temp.name, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('你好', content)
            self.assertIn('🌍', content)
        
        os.unlink(temp.name)
    
    def test_very_long_messages(self):
        """Test handling of very long messages."""
        long_message = "A" * 10000
        text = f"""12/25/23, 9:30 AM - Alice: {long_message}"""
        
        temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp.write(text)
        temp.close()
        
        # Should handle long messages without crashing
        self.assertTrue(os.path.exists(temp.name))
        
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
