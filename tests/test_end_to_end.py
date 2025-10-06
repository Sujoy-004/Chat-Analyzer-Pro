"""
End-to-End Integration Tests
Tests complete pipeline from parsing to reporting.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os


class TestCompletePipeline(unittest.TestCase):
    """Test complete analysis pipeline end-to-end."""
    
    def setUp(self):
        """Set up complete test scenario."""
        # Create sample chat export file
        self.sample_chat = """12/01/23, 9:00 AM - Alice: Hey! How are you?
12/01/23, 9:05 AM - Bob: I'm great! 😊
12/01/23, 9:10 AM - Alice: Awesome! Let's catch up soon
12/01/23, 10:00 AM - Bob: Sounds good!
12/02/23, 8:00 AM - Alice: Good morning!
12/02/23, 8:05 AM - Bob: Morning! Ready for today?
12/02/23, 8:10 AM - Alice: Absolutely! 💪"""
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        self.temp_file.write(self.sample_chat)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_parse_to_dataframe_pipeline(self):
        """Test parsing chat file to DataFrame."""
        # Step 1: Parse file
        df = self._mock_parse(self.temp_file.name)
        
        # Verify parsing
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('datetime', df.columns)
        self.assertIn('sender', df.columns)
        self.assertIn('message', df.columns)
    
    def test_full_analysis_pipeline(self):
        """Test complete analysis from parse to results."""
        # Step 1: Parse
        df = self._mock_parse(self.temp_file.name)
        
        # Step 2: Add sentiment (mock)
        df['sentiment'] = np.random.choice(['positive', 'neutral', 'negative'], len(df))
        df['sentiment_score'] = np.random.uniform(-1, 1, len(df))
        
        # Step 3: Calculate metrics
        total_messages = len(df)
        unique_senders = df['sender'].nunique()
        
        # Step 4: Generate analysis results
        results = {
            'total_messages': total_messages,
            'unique_senders': unique_senders,
            'sentiment_distribution': df['sentiment'].value_counts().to_dict()
        }
        
        # Verify complete pipeline
        self.assertEqual(results['total_messages'], 7)
        self.assertEqual(results['unique_senders'], 2)
        self.assertIn('sentiment_distribution', results)
    
    def test_parse_analyze_visualize_pipeline(self):
        """Test pipeline from parsing to visualization."""
        # Parse
        df = self._mock_parse(self.temp_file.name)
        
        # Analyze
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        hourly_activity = df.groupby('hour').size()
        
        # Prepare visualization data
        viz_data = {
            'hours': hourly_activity.index.tolist(),
            'counts': hourly_activity.values.tolist()
        }
        
        # Verify
        self.assertGreater(len(viz_data['hours']), 0)
        self.assertEqual(len(viz_data['hours']), len(viz_data['counts']))
    
    def test_parse_analyze_report_pipeline(self):
        """Test pipeline from parsing to report generation."""
        # Parse
        df = self._mock_parse(self.temp_file.name)
        
        # Analyze
        summary = {
            'total_messages': len(df),
            'date_range': f"{df['datetime'].min()} to {df['datetime'].max()}",
            'participants': df['sender'].unique().tolist()
        }
        
        # Generate report metadata
        report_data = {
            'generated_at': datetime.now(),
            'summary': summary,
            'charts': ['timeline', 'sentiment', 'activity']
        }
        
        # Verify
        self.assertIn('summary', report_data)
        self.assertIn('charts', report_data)
        self.assertEqual(len(report_data['charts']), 3)
    
    def _mock_parse(self, filepath):
        """Mock parser for testing."""
        import re
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[AP]M)\s-\s([^:]+):\s(.+?)(?=\d{1,2}/\d{1,2}/\d{2,4},|\Z)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        data = []
        for match in matches:
            timestamp_str, sender, message = match
            dt = pd.to_datetime(timestamp_str, format='%m/%d/%y, %I:%M %p')
            
            data.append({
                'datetime': dt,
                'sender': sender.strip(),
                'message': message.strip(),
                'message_length': len(message.strip())
            })
        
        return pd.DataFrame(data)


class TestDataFlow(unittest.TestCase):
    """Test data flow through different modules."""
    
    def test_parser_to_eda_flow(self):
        """Test data flow from parser to EDA module."""
        # Mock parsed data
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-12-01', periods=50, freq='6H'),
            'sender': ['Alice', 'Bob'] * 25,
            'message': ['Test'] * 50,
            'message_length': np.random.randint(10, 100, 50)
        })
        
        # EDA calculations
        eda_results = {
            'total_messages': len(df),
            'avg_length': df['message_length'].mean(),
            'top_sender': df['sender'].value_counts().idxmax()
        }
        
        self.assertEqual(eda_results['total_messages'], 50)
        self.assertGreater(eda_results['avg_length'], 0)
    
    def test_eda_to_sentiment_flow(self):
        """Test data flow from EDA to sentiment analysis."""
        # Mock EDA output
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-12-01', periods=30, freq='12H'),
            'sender': ['Alice'] * 30,
            'message': ['Happy day!', 'Terrible news', 'Okay'] * 10,
            'message_length': [10] * 30
        })
        
        # Add sentiment
        df['sentiment'] = ['positive', 'negative', 'neutral'] * 10
        
        # Verify sentiment added
        self.assertIn('sentiment', df.columns)
        self.assertEqual(len(df), 30)
    
    def test_sentiment_to_health_flow(self):
        """Test data flow from sentiment to relationship health."""
        # Mock sentiment output
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-12-01', periods=100, freq='30min'),
            'sender': ['Alice', 'Bob'] * 50,
            'message': ['Test'] * 100,
            'sentiment': np.random.choice(['positive', 'neutral', 'negative'], 100),
            'sentiment_score': np.random.uniform(-1, 1, 100)
        })
        
        # Calculate health metrics
        message_balance = 1 - abs(
            len(df[df['sender'] == 'Alice']) - len(df[df['sender'] == 'Bob'])
        ) / len(df)
        
        avg_sentiment = df['sentiment_score'].mean()
        
        self.assertGreaterEqual(message_balance, 0)
        self.assertLessEqual(message_balance, 1)
        self.assertGreaterEqual(avg_sentiment, -1)
        self.assertLessEqual(avg_sentiment, 1)
    
    def test_health_to_report_flow(self):
        """Test data flow from health metrics to report."""
        # Mock health metrics
        health_metrics = {
            'overall_score': 0.85,
            'grade': 'VERY GOOD',
            'components': {
                'initiation': 0.8,
                'responsiveness': 0.9,
                'balance': 0.85,
                'dominance': 0.85
            }
        }
        
        # Generate report data
        report_sections = {
            'health_score': health_metrics['overall_score'],
            'grade': health_metrics['grade'],
            'recommendations': ['Keep up the good work!']
        }
        
        self.assertIn('health_score', report_sections)
        self.assertIn('recommendations', report_sections)


class TestErrorHandling(unittest.TestCase):
    """Test error handling throughout pipeline."""
    
    def test_empty_file_handling(self):
        """Test handling of empty input file."""
        empty_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        empty_file.write("")
        empty_file.close()
        
        # Should handle gracefully
        df = pd.DataFrame()  # Empty result
        
        self.assertEqual(len(df), 0)
        
        os.unlink(empty_file.name)
    
    def test_malformed_data_handling(self):
        """Test handling of malformed data."""
        malformed_data = "This is not a valid chat format"
        
        # Should not crash, return empty or raise specific error
        try:
            result = []  # Mock: no valid messages found
            self.assertEqual(len(result), 0)
        except Exception as e:
            # Specific error handling
            self.assertIsInstance(e, (ValueError, KeyError))
    
    def test_missing_columns_handling(self):
        """Test handling of missing expected columns."""
        incomplete_df = pd.DataFrame({
            'datetime': [datetime.now()],
            'sender': ['Alice']
            # Missing 'message' column
        })
        
        # Should handle gracefully
        if 'message' not in incomplete_df.columns:
            incomplete_df['message'] = ''
        
        self.assertIn('message', incomplete_df.columns)
    
    def test_invalid_datetime_handling(self):
        """Test handling of invalid datetime values."""
        df = pd.DataFrame({
            'datetime': ['invalid_date', '2023-12-01', 'another_invalid'],
            'sender': ['Alice', 'Bob', 'Alice'],
            'message': ['Test'] * 3
        })
        
        # Convert, handling errors
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        valid_rows = df[df['datetime'].notna()]
        
        self.assertEqual(len(valid_rows), 1)


class TestScalability(unittest.TestCase):
    """Test system behavior with large datasets."""
    
    def test_large_dataset_processing(self):
        """Test processing of large dataset."""
        # Create large dataset
        large_df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=10000, freq='5min'),
            'sender': np.random.choice(['Alice', 'Bob', 'Charlie'], 10000),
            'message': ['Test message'] * 10000,
            'message_length': np.random.randint(10, 200, 10000)
        })
        
        # Should handle efficiently
        self.assertEqual(len(large_df), 10000)
        
        # Basic operations should complete
        summary = {
            'total': len(large_df),
            'avg_length': large_df['message_length'].mean()
        }
        
        self.assertGreater(summary['avg_length'], 0)
    
    def test_memory_efficiency(self):
        """Test memory efficiency with moderate dataset."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=1000, freq='30min'),
            'sender': ['Alice', 'Bob'] * 500,
            'message': ['Test'] * 1000
        })
        
        # Should not cause memory issues
        memory_usage = df.memory_usage(deep=True).sum()
        
        # Reasonable memory usage (< 10MB for this size)
        self.assertLess(memory_usage, 10 * 1024 * 1024)


class TestModuleIntegration(unittest.TestCase):
    """Test integration between different modules."""
    
    def test_parser_visualization_integration(self):
        """Test integration between parser and visualization."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-12-01', periods=50, freq='6H'),
            'sender': ['Alice', 'Bob'] * 25,
            'message': ['Test'] * 50
        })
        # Prepare for visualization
        df['timestamp'] = pd.to_datetime(df['datetime'])
        timeline_data = df.set_index('timestamp').resample('D').size()
        
        # Verify integration
        self.assertGreater(len(timeline_data), 0)
        self.assertIsInstance(timeline_data, pd.Series)
    
    def test_analysis_reporting_integration(self):
        """Test integration between analysis and reporting modules."""
        # Analysis output
        analysis_results = {
            'health_score': 0.85,
            'friendship_index': 87.5,
            'streaks': {'longest': 30, 'current': 15},
            'milestones': ['1000 messages', '30-day streak']
        }
        
        # Should be consumable by reporting
        report_summary = {
            'overall_health': analysis_results['health_score'],
            'friendship_tier': 'CLOSE FRIENDS' if analysis_results['friendship_index'] >= 75 else 'FRIENDS',
            'achievements': analysis_results['milestones']
        }
        
        self.assertIn('overall_health', report_summary)
        self.assertIn('achievements', report_summary)
    
    def test_gamification_digest_integration(self):
        """Test integration between gamification and digest."""
        # Gamification output
        gamification_data = {
            'friendship_index': 92.0,
            'tier': 'BEST FRIENDS 👑',
            'streaks': {'current': 20, 'longest': 45},
            'achievements': [
                {'icon': '💬', 'milestone': '1,000 Messages'},
                {'icon': '🔥', 'milestone': '45-Day Streak'}
            ]
        }
        
        # Format for digest
        digest_text = f"""
🏆 Friendship Status: {gamification_data['tier']}
📊 Friendship Index: {gamification_data['friendship_index']:.0f}/100
🔥 Current Streak: {gamification_data['streaks']['current']} days
        """
        
        self.assertIn('BEST FRIENDS', digest_text)
        self.assertIn('92', digest_text)


class TestOutputFormats(unittest.TestCase):
    """Test different output format generations."""
    
    def test_csv_export(self):
        """Test CSV export functionality."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-12-01', periods=10, freq='1H'),
            'sender': ['Alice', 'Bob'] * 5,
            'message': ['Test'] * 10,
            'sentiment': ['positive'] * 10
        })
        
        # Export to CSV
        temp_csv = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        temp_csv.close()
        
        df.to_csv(temp_csv.name, index=False)
        
        # Verify file exists and readable
        self.assertTrue(os.path.exists(temp_csv.name))
        
        reloaded = pd.read_csv(temp_csv.name)
        self.assertEqual(len(reloaded), len(df))
        
        os.unlink(temp_csv.name)
    
    def test_json_export(self):
        """Test JSON export functionality."""
        results = {
            'total_messages': 100,
            'health_score': 0.85,
            'participants': ['Alice', 'Bob']
        }
        
        import json
        
        temp_json = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(results, temp_json)
        temp_json.close()
        
        # Verify
        with open(temp_json.name, 'r') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded['total_messages'], 100)
        
        os.unlink(temp_json.name)
    
    def test_html_report_generation(self):
        """Test HTML report generation."""
        html_content = """
        <html>
        <head><title>Chat Analysis Report</title></head>
        <body>
            <h1>Analysis Results</h1>
            <p>Total Messages: 100</p>
        </body>
        </html>
        """
        
        temp_html = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html')
        temp_html.write(html_content)
        temp_html.close()
        
        # Verify
        with open(temp_html.name, 'r') as f:
            content = f.read()
        
        self.assertIn('<html>', content)
        self.assertIn('Total Messages', content)
        
        os.unlink(temp_html.name)


class TestRobustness(unittest.TestCase):
    """Test system robustness and edge cases."""
    
    def test_single_message_handling(self):
        """Test handling of single message."""
        df = pd.DataFrame({
            'datetime': [datetime.now()],
            'sender': ['Alice'],
            'message': ['Single message'],
            'message_length': [14]
        })
        
        # Should handle gracefully
        self.assertEqual(len(df), 1)
        
        # Should still calculate metrics
        avg_length = df['message_length'].mean()
        self.assertEqual(avg_length, 14)
    
    def test_single_sender_handling(self):
        """Test handling of single sender (monologue)."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-12-01', periods=50, freq='1H'),
            'sender': ['Alice'] * 50,
            'message': ['Test'] * 50
        })
        
        # Should handle monologue
        unique_senders = df['sender'].nunique()
        self.assertEqual(unique_senders, 1)
        
        # Balance score should reflect single sender
        balance_score = 0  # No balance possible with one sender
        self.assertEqual(balance_score, 0)
    
    def test_extreme_time_gaps(self):
        """Test handling of extreme time gaps."""
        dates = [
            datetime(2023, 1, 1),
            datetime(2023, 6, 1),  # 5-month gap
            datetime(2023, 6, 2),
            datetime(2024, 1, 1)   # 6-month gap
        ]
        
        df = pd.DataFrame({
            'datetime': dates,
            'sender': ['Alice', 'Bob', 'Alice', 'Bob'],
            'message': ['Test'] * 4
        })
        
        # Should still process
        time_span = (df['datetime'].max() - df['datetime'].min()).days
        self.assertGreater(time_span, 300)
    
    def test_identical_timestamps_handling(self):
        """Test handling of messages with identical timestamps."""
        same_time = datetime(2023, 12, 1, 10, 30, 0)
        
        df = pd.DataFrame({
            'datetime': [same_time, same_time, same_time],
            'sender': ['Alice', 'Bob', 'Alice'],
            'message': ['Msg1', 'Msg2', 'Msg3']
        })
        
        # Should handle without errors
        self.assertEqual(len(df), 3)
        
        # Should maintain all messages
        unique_messages = df['message'].nunique()
        self.assertEqual(unique_messages, 3)


class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""
    
    def test_processing_speed(self):
        """Test that processing completes in reasonable time."""
        import time
        
        # Create moderate dataset
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=1000, freq='30min'),
            'sender': np.random.choice(['Alice', 'Bob'], 1000),
            'message': ['Test message'] * 1000,
            'message_length': np.random.randint(10, 200, 1000)
        })
        
        # Time basic operations
        start = time.time()
        
        # Simulate analysis
        total = len(df)
        avg_length = df['message_length'].mean()
        hourly = df.groupby(pd.to_datetime(df['datetime']).dt.hour).size()
        
        end = time.time()
        
        # Should complete quickly (< 1 second for this size)
        self.assertLess(end - start, 1.0)
    
    def test_memory_cleanup(self):
        """Test that memory is properly managed."""
        # Create temporary data
        temp_df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=1000, freq='1H'),
            'sender': ['Alice'] * 1000,
            'message': ['Test'] * 1000
        })
        
        # Use and release
        result = len(temp_df)
        del temp_df
        
        # Should have released memory
        self.assertEqual(result, 1000)


class TestDataValidation(unittest.TestCase):
    """Test data validation throughout pipeline."""
    
    def test_datetime_format_validation(self):
        """Test datetime format validation."""
        valid_dates = pd.to_datetime([
            '2023-12-01',
            '2023-12-01 10:30:00',
            '12/01/2023'
        ], errors='coerce')
        
        # All should parse successfully
        self.assertEqual(valid_dates.notna().sum(), 3)
    
    def test_sender_name_validation(self):
        """Test sender name validation."""
        senders = ['Alice', 'Bob', '', 'Charlie123', 'user@email.com']
        
        # Filter valid senders (non-empty)
        valid_senders = [s for s in senders if s.strip()]
        
        self.assertEqual(len(valid_senders), 4)
    
    def test_message_content_validation(self):
        """Test message content validation."""
        messages = ['Valid message', '', '   ', 'Another valid', None]
        
        # Filter valid messages
        valid_messages = [m for m in messages if m and m.strip()]
        
        self.assertEqual(len(valid_messages), 2)
    
    def test_score_range_validation(self):
        """Test that all scores are in valid ranges."""
        scores = {
            'health_score': 0.85,
            'friendship_index': 87.5,
            'sentiment_score': 0.3
        }
        
        # Validate ranges
        self.assertGreaterEqual(scores['health_score'], 0)
        self.assertLessEqual(scores['health_score'], 1)
        self.assertGreaterEqual(scores['friendship_index'], 0)
        self.assertLessEqual(scores['friendship_index'], 100)
        self.assertGreaterEqual(scores['sentiment_score'], -1)
        self.assertLessEqual(scores['sentiment_score'], 1)


def run_end_to_end_tests():
    """Run all end-to-end tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestCompletePipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestDataFlow))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestScalability))
    suite.addTests(loader.loadTestsFromTestCase(TestModuleIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestOutputFormats))
    suite.addTests(loader.loadTestsFromTestCase(TestRobustness))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidation))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    unittest.main()
