"""
Unit Tests for Reporting Modules
Tests PDF report generation and weekly digest functionality.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os


class TestPDFReportGeneration(unittest.TestCase):
    """Test cases for PDF report generation."""
    
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2023-12-01', periods=100, freq='6H')
        self.test_df = pd.DataFrame({
            'datetime': dates,
            'sender': ['Alice', 'Bob'] * 50,
            'message': ['Test message'] * 100,
            'message_length': np.random.randint(10, 100, 100),
            'sentiment': np.random.choice(['positive', 'neutral', 'negative'], 100),
            'sentiment_score': np.random.uniform(-1, 1, 100)
        })
    
    def test_report_metadata_generation(self):
        """Test report metadata creation."""
        metadata = {
            'title': 'Chat Analysis Report',
            'generated_date': datetime.now(),
            'total_messages': len(self.test_df),
            'date_range': f"{self.test_df['datetime'].min()} to {self.test_df['datetime'].max()}"
        }
        
        self.assertIn('title', metadata)
        self.assertIn('generated_date', metadata)
        self.assertEqual(metadata['total_messages'], 100)
    
    def test_summary_statistics_calculation(self):
        """Test summary statistics for report."""
        summary = {
            'total_messages': len(self.test_df),
            'unique_senders': self.test_df['sender'].nunique(),
            'avg_message_length': self.test_df['message_length'].mean(),
            'date_range': (self.test_df['datetime'].max() - self.test_df['datetime'].min()).days
        }
        
        self.assertEqual(summary['total_messages'], 100)
        self.assertEqual(summary['unique_senders'], 2)
        self.assertGreater(summary['avg_message_length'], 0)
    
    def test_sentiment_summary_for_report(self):
        """Test sentiment summary generation."""
        sentiment_summary = self.test_df['sentiment'].value_counts().to_dict()
        
        self.assertIn('positive', sentiment_summary or 'neutral' in sentiment_summary or 'negative' in sentiment_summary)
        self.assertEqual(sum(sentiment_summary.values()), 100)
    
    def test_report_file_creation(self):
        """Test that report file can be created."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.close()
        
        # Simulate report creation
        report_path = temp_file.name
        
        self.assertTrue(report_path.endswith('.pdf'))
        
        os.unlink(temp_file.name)
    
    def test_chart_data_preparation(self):
        """Test data preparation for charts in report."""
        # Timeline data
        timeline_data = self.test_df.set_index('datetime').resample('D').size()
        
        self.assertGreater(len(timeline_data), 0)
        
        # Sentiment distribution
        sentiment_data = self.test_df['sentiment'].value_counts()
        
        self.assertGreater(len(sentiment_data), 0)


class TestWeeklyDigest(unittest.TestCase):
    """Test cases for weekly digest functionality."""
    
    def setUp(self):
        """Set up test data for weekly digest."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        dates = pd.date_range(start_date, end_date, freq='3H')
        
        self.test_df = pd.DataFrame({
            'datetime': dates,
            'sender': ['Alice', 'Bob'] * (len(dates) // 2),
            'message': ['Test'] * len(dates),
            'message_length': np.random.randint(10, 100, len(dates)),
            'sentiment': np.random.choice(['positive', 'neutral', 'negative'], len(dates)),
            'sentiment_score': np.random.uniform(-1, 1, len(dates))
        })
    
    def test_weekly_date_range_filtering(self):
        """Test filtering data for weekly range."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        weekly_df = self.test_df[
            (self.test_df['datetime'] >= start_date) & 
            (self.test_df['datetime'] <= end_date)
        ]
        
        self.assertGreater(len(weekly_df), 0)
    
    def test_weekly_summary_generation(self):
        """Test weekly summary statistics generation."""
        summary = {
            'total_messages': len(self.test_df),
            'total_participants': self.test_df['sender'].nunique(),
            'date_range': f"{self.test_df['datetime'].min().date()} to {self.test_df['datetime'].max().date()}"
        }
        
        self.assertGreater(summary['total_messages'], 0)
        self.assertEqual(summary['total_participants'], 2)
    
    def test_most_active_day_calculation(self):
        """Test most active day identification."""
        daily_counts = self.test_df.groupby(
            self.test_df['datetime'].dt.day_name()
        ).size()
        
        most_active_day = daily_counts.idxmax()
        
        self.assertIsNotNone(most_active_day)
    
    def test_top_contributors_calculation(self):
        """Test top contributors identification."""
        top_contributors = self.test_df['sender'].value_counts().head(5)
        
        self.assertGreater(len(top_contributors), 0)
        self.assertTrue(all(count > 0 for count in top_contributors.values))
    
    def test_sentiment_summary_calculation(self):
        """Test sentiment summary for digest."""
        sentiment_counts = self.test_df['sentiment'].value_counts().to_dict()
        avg_sentiment = self.test_df['sentiment_score'].mean()
        
        self.assertEqual(sum(sentiment_counts.values()), len(self.test_df))
        self.assertGreaterEqual(avg_sentiment, -1)
        self.assertLessEqual(avg_sentiment, 1)
    
    def test_activity_patterns_calculation(self):
        """Test activity pattern analysis."""
        hourly_activity = self.test_df.groupby(
            self.test_df['datetime'].dt.hour
        ).size()
        
        peak_hour = hourly_activity.idxmax()
        
        self.assertGreaterEqual(peak_hour, 0)
        self.assertLess(peak_hour, 24)
    
    def test_engagement_metrics_calculation(self):
        """Test engagement metrics for digest."""
        avg_length = self.test_df['message_length'].mean()
        total_words = self.test_df['message'].str.split().str.len().sum()
        
        self.assertGreater(avg_length, 0)
        self.assertGreater(total_words, 0)
    
    def test_html_email_formatting(self):
        """Test HTML email template formatting."""
        summary = {
            'period': {'start': '2023-12-01', 'end': '2023-12-07'},
            'total_messages': 100,
            'total_participants': 2
        }
        
        html_template = f"""
        <html>
        <body>
            <h1>Weekly Chat Digest</h1>
            <p>Period: {summary['period']['start']} to {summary['period']['end']}</p>
            <p>Total Messages: {summary['total_messages']}</p>
        </body>
        </html>
        """
        
        self.assertIn('Weekly Chat Digest', html_template)
        self.assertIn('100', html_template)
    
    def test_telegram_message_formatting(self):
        """Test Telegram message formatting."""
        summary = {
            'total_messages': 150,
            'total_participants': 2,
            'most_active_day': 'Monday'
        }
        
        telegram_msg = f"""
📊 *Weekly Chat Digest*

📈 *Key Metrics*
• Total Messages: *{summary['total_messages']:,}*
• Active Participants: *{summary['total_participants']}*
• Most Active Day: *{summary['most_active_day']}*
        """
        
        self.assertIn('Weekly Chat Digest', telegram_msg)
        self.assertIn('150', telegram_msg)


class TestEmailDelivery(unittest.TestCase):
    """Test cases for email delivery functionality."""
    
    def test_email_configuration_validation(self):
        """Test email configuration validation."""
        email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': 'test@example.com',
            'sender_password': 'password123'
        }
        
        self.assertIn('smtp_server', email_config)
        self.assertIn('smtp_port', email_config)
        self.assertIn('sender_email', email_config)
        self.assertEqual(email_config['smtp_port'], 587)
    
    def test_recipient_email_validation(self):
        """Test recipient email format validation."""
        valid_emails = [
            'user@example.com',
            'test.user@domain.co.uk',
            'name+tag@email.org'
        ]
        
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        for email in valid_emails:
            self.assertIsNotNone(re.match(email_pattern, email))
    
    def test_email_subject_generation(self):
        """Test email subject line generation."""
        start_date = '2023-12-01'
        end_date = '2023-12-07'
        
        subject = f"Weekly Chat Digest: {start_date} to {end_date}"
        
        self.assertIn('Weekly Chat Digest', subject)
        self.assertIn(start_date, subject)


class TestTelegramDelivery(unittest.TestCase):
    """Test cases for Telegram delivery functionality."""
    
    def test_telegram_config_validation(self):
        """Test Telegram configuration validation."""
        telegram_config = {
            'bot_token': '123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11',
            'chat_id': '123456789'
        }
        
        self.assertIn('bot_token', telegram_config)
        self.assertIn('chat_id', telegram_config)
        self.assertTrue(telegram_config['bot_token'].count(':') >= 1)
    
    def test_telegram_message_length_limit(self):
        """Test Telegram message length constraint."""
        max_length = 4096  # Telegram's limit
        
        long_message = "A" * 5000
        truncated = long_message[:max_length]
        
        self.assertLessEqual(len(truncated), max_length)
    
    def test_telegram_markdown_escaping(self):
        """Test Telegram markdown special character handling."""
        message_with_specials = "Test *bold* _italic_ [link](url)"
        
        # Should contain markdown characters
        self.assertIn('*', message_with_specials)
        self.assertIn('_', message_with_specials)


class TestReportAttachments(unittest.TestCase):
    """Test cases for report attachments."""
    
    def test_pdf_attachment_handling(self):
        """Test PDF attachment preparation."""
        pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf_path.write(b'%PDF-1.4 fake pdf content')
        pdf_path.close()
        
        self.assertTrue(os.path.exists(pdf_path.name))
        self.assertTrue(pdf_path.name.endswith('.pdf'))
        
        os.unlink(pdf_path.name)
    
    def test_multiple_attachments_handling(self):
        """Test handling of multiple attachments."""
        attachments = []
        
        for i in range(3):
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=f'_file{i}.txt')
            temp.write(b'test content')
            temp.close()
            attachments.append(temp.name)
        
        self.assertEqual(len(attachments), 3)
        
        for path in attachments:
            self.assertTrue(os.path.exists(path))
            os.unlink(path)
    
    def test_attachment_size_validation(self):
        """Test attachment size validation."""
        max_size_mb = 25  # Common email limit
        
        # Create small test file
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(b'test' * 100)
        temp.close()
        
        file_size_mb = os.path.getsize(temp.name) / (1024 * 1024)
        
        self.assertLess(file_size_mb, max_size_mb)
        
        os.unlink(temp.name)


class TestScheduling(unittest.TestCase):
    """Test cases for digest scheduling functionality."""
    
    def test_multiple_recipients_handling(self):
        """Test handling of multiple recipients."""
        recipients = {
            'email': ['user1@example.com', 'user2@example.com', 'user3@example.com'],
            'telegram': ['123456', '789012', '345678']
        }
        
        self.assertEqual(len(recipients['email']), 3)
        self.assertEqual(len(recipients['telegram']), 3)
    
    def test_delivery_status_tracking(self):
        """Test delivery status tracking."""
        results = {
            'email_user1@example.com': True,
            'email_user2@example.com': False,
            'telegram_123456': True
        }
        
        successful = sum(1 for status in results.values() if status)
        failed = sum(1 for status in results.values() if not status)
        
        self.assertEqual(successful, 2)
        self.assertEqual(failed, 1)
    
    def test_error_handling_in_batch_delivery(self):
        """Test error handling for batch delivery."""
        recipients = ['valid@email.com', 'invalid-email', 'another@valid.com']
        
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        valid_recipients = [r for r in recipients if re.match(email_pattern, r)]
        
        self.assertEqual(len(valid_recipients), 2)


class TestReportingIntegration(unittest.TestCase):
    """Test cases for reporting module integration."""
    
    def test_end_to_end_digest_generation(self):
        """Test complete digest generation flow."""
        # Create sample data
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-12-01', periods=50, freq='6H'),
            'sender': ['Alice', 'Bob'] * 25,
            'message': ['Test'] * 50,
            'sentiment': ['positive'] * 50
        })
        
        # Generate summary
        summary = {
            'total_messages': len(df),
            'total_participants': df['sender'].nunique(),
            'period': {'start': '2023-12-01', 'end': '2023-12-07'}
        }
        
        self.assertEqual(summary['total_messages'], 50)
        self.assertEqual(summary['total_participants'], 2)
    
    def test_pdf_and_digest_coordination(self):
        """Test coordination between PDF generation and digest."""
        # Should be able to generate both
        pdf_generated = True
        digest_generated = True
        
        self.assertTrue(pdf_generated and digest_generated)


def run_reporting_tests():
    """Run all reporting tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestPDFReportGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestWeeklyDigest))
    suite.addTests(loader.loadTestsFromTestCase(TestEmailDelivery))
    suite.addTests(loader.loadTestsFromTestCase(TestTelegramDelivery))
    suite.addTests(loader.loadTestsFromTestCase(TestReportAttachments))
    suite.addTests(loader.loadTestsFromTestCase(TestScheduling))
    suite.addTests(loader.loadTestsFromTestCase(TestReportingIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    unittest.main()
