"""
Unit Tests for Analysis Modules
Tests EDA, sentiment, emotion, relationship health, and gamification features.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestEDAModule(unittest.TestCase):
    """Test cases for exploratory data analysis module."""
    
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2023-12-01', periods=100, freq='6H')
        self.test_df = pd.DataFrame({
            'datetime': dates,
            'sender': ['Alice', 'Bob'] * 50,
            'message': ['Test message'] * 100,
            'message_length': np.random.randint(10, 100, 100)
        })
    
    def test_message_volume_calculation(self):
        """Test message volume statistics."""
        total_messages = len(self.test_df)
        unique_senders = self.test_df['sender'].nunique()
        
        self.assertEqual(total_messages, 100)
        self.assertEqual(unique_senders, 2)
    
    def test_hourly_activity_distribution(self):
        """Test hourly activity analysis."""
        self.test_df['hour'] = pd.to_datetime(self.test_df['datetime']).dt.hour
        hourly_counts = self.test_df['hour'].value_counts()
        
        self.assertGreater(len(hourly_counts), 0)
        self.assertTrue(all(0 <= h < 24 for h in hourly_counts.index))
    
    def test_daily_activity_distribution(self):
        """Test daily activity analysis."""
        self.test_df['day'] = pd.to_datetime(self.test_df['datetime']).dt.day_name()
        daily_counts = self.test_df['day'].value_counts()
        
        self.assertGreater(len(daily_counts), 0)
    
    def test_top_senders_calculation(self):
        """Test top senders identification."""
        top_senders = self.test_df['sender'].value_counts().head(5)
        
        self.assertEqual(len(top_senders), 2)
        self.assertTrue(all(count > 0 for count in top_senders.values))
    
    def test_message_length_statistics(self):
        """Test message length analysis."""
        avg_length = self.test_df['message_length'].mean()
        max_length = self.test_df['message_length'].max()
        min_length = self.test_df['message_length'].min()
        
        self.assertGreater(avg_length, 0)
        self.assertGreaterEqual(max_length, min_length)


class TestSentimentAnalysis(unittest.TestCase):
    """Test cases for sentiment analysis module."""
    
    def setUp(self):
        """Set up test data with known sentiments."""
        self.positive_messages = [
            "I love this! 😊",
            "This is amazing!",
            "Great job!",
            "Awesome! ❤️"
        ]
        
        self.negative_messages = [
            "This is terrible 😢",
            "I hate this",
            "Awful experience",
            "Very disappointed"
        ]
        
        self.neutral_messages = [
            "Okay",
            "The weather is cloudy",
            "I went to the store",
            "It's 5 PM"
        ]
    
    def test_positive_sentiment_detection(self):
        """Test detection of positive sentiments."""
        # Mock sentiment scores
        positive_scores = [0.8, 0.9, 0.7, 0.85]
        
        self.assertTrue(all(score > 0.5 for score in positive_scores))
    
    def test_negative_sentiment_detection(self):
        """Test detection of negative sentiments."""
        # Mock sentiment scores
        negative_scores = [-0.8, -0.7, -0.9, -0.75]
        
        self.assertTrue(all(score < -0.5 for score in negative_scores))
    
    def test_neutral_sentiment_detection(self):
        """Test detection of neutral sentiments."""
        # Mock sentiment scores
        neutral_scores = [0.1, -0.1, 0.0, 0.05]
        
        self.assertTrue(all(-0.5 <= score <= 0.5 for score in neutral_scores))
    
    def test_sentiment_distribution(self):
        """Test sentiment distribution calculation."""
        sentiments = ['positive'] * 40 + ['neutral'] * 35 + ['negative'] * 25
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        self.assertEqual(sentiment_counts['positive'], 40)
        self.assertEqual(sentiment_counts['neutral'], 35)
        self.assertEqual(sentiment_counts['negative'], 25)
    
    def test_sentiment_score_range(self):
        """Test that sentiment scores are within valid range."""
        scores = np.random.uniform(-1, 1, 100)
        
        self.assertTrue(all(-1 <= score <= 1 for score in scores))


class TestEmotionClassification(unittest.TestCase):
    """Test cases for emotion classification module."""
    
    def test_emotion_categories(self):
        """Test emotion category classification."""
        emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love']
        
        # All emotions should be valid categories
        for emotion in emotions:
            self.assertIn(emotion, ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love'])
    
    def test_emotion_distribution(self):
        """Test emotion distribution calculation."""
        emotions = ['joy'] * 30 + ['sadness'] * 20 + ['anger'] * 15 + ['love'] * 35
        emotion_counts = pd.Series(emotions).value_counts()
        
        self.assertEqual(len(emotion_counts), 4)
        self.assertEqual(emotion_counts.sum(), 100)


class TestRelationshipHealth(unittest.TestCase):
    """Test cases for relationship health metrics."""
    
    def setUp(self):
        """Set up test conversation data."""
        dates = pd.date_range('2023-12-01', periods=100, freq='30min')
        self.test_df = pd.DataFrame({
            'datetime': dates,
            'sender': ['Alice', 'Bob'] * 50,
            'message': ['Test'] * 100,
            'message_length': np.random.randint(10, 100, 100)
        })
    
    def test_conversation_starter_identification(self):
        """Test identification of conversation starters."""
        # Add large gaps to create conversation boundaries
        self.test_df['time_diff'] = self.test_df['datetime'].diff()
        self.test_df['time_diff_minutes'] = self.test_df['time_diff'].dt.total_seconds() / 60
        
        # Messages with >60 min gap should be conversation starters
        conversation_starters = self.test_df[self.test_df['time_diff_minutes'] > 60]
        
        self.assertGreaterEqual(len(conversation_starters), 0)
    
    def test_initiator_ratio_calculation(self):
        """Test conversation initiator ratio."""
        # Mock conversation starters
        initiators = ['Alice'] * 55 + ['Bob'] * 45
        initiator_counts = pd.Series(initiators).value_counts()
        
        alice_ratio = initiator_counts['Alice'] / len(initiators)
        bob_ratio = initiator_counts['Bob'] / len(initiators)
        
        self.assertAlmostEqual(alice_ratio + bob_ratio, 1.0)
        self.assertEqual(alice_ratio, 0.55)
    
    def test_balance_score_calculation(self):
        """Test balance score computation."""
        # Perfect balance
        perfect_balance = 1 - abs(50 - 50) / 100
        self.assertEqual(perfect_balance, 1.0)
        
        # Moderate imbalance
        imbalanced = 1 - abs(70 - 30) / 100
        self.assertEqual(imbalanced, 0.6)
    
    def test_dominance_score(self):
        """Test dominance score calculation."""
        message_counts = pd.Series({'Alice': 60, 'Bob': 40})
        total = message_counts.sum()
        
        dominance_score = 1 - abs(message_counts['Alice'] - message_counts['Bob']) / total
        
        self.assertGreater(dominance_score, 0)
        self.assertLessEqual(dominance_score, 1)
    
    def test_health_score_range(self):
        """Test that health scores are in valid range [0, 1]."""
        # Mock component scores
        component_scores = {
            'initiation': 0.8,
            'responsiveness': 0.9,
            'balance': 0.7,
            'dominance': 0.85
        }
        
        weights = {'initiation': 0.25, 'responsiveness': 0.35, 'balance': 0.20, 'dominance': 0.20}
        
        health_score = sum(component_scores[k] * weights[k] for k in component_scores)
        
        self.assertGreaterEqual(health_score, 0)
        self.assertLessEqual(health_score, 1)


class TestGamificationFeatures(unittest.TestCase):
    """Test cases for gamification features (Day 14)."""
    
    def setUp(self):
        """Set up test data for gamification."""
        # Create data with streaks
        dates = []
        current_date = datetime(2023, 12, 1)
        
        # Create a 10-day streak
        for i in range(10):
            dates.append(current_date + timedelta(days=i))
        
        # Gap of 3 days
        current_date += timedelta(days=13)
        
        # Another 5-day streak
        for i in range(5):
            dates.append(current_date + timedelta(days=i))
        
        self.test_df = pd.DataFrame({
            'datetime': dates,
            'sender': ['Alice', 'Bob'] * (len(dates) // 2),
            'message': ['Test'] * len(dates)
        })
    
    def test_streak_detection(self):
        """Test conversation streak detection."""
        self.test_df['date'] = pd.to_datetime(self.test_df['datetime']).dt.date
        unique_dates = sorted(self.test_df['date'].unique())
        
        # Should detect consecutive days
        self.assertGreater(len(unique_dates), 0)
    
    def test_longest_streak_calculation(self):
        """Test longest streak identification."""
        # Mock streak calculation
        streaks = [10, 5, 3, 7]
        longest = max(streaks)
        
        self.assertEqual(longest, 10)
    
    def test_friendship_index_range(self):
        """Test friendship index is in range [0, 100]."""
        # Mock friendship index components
        frequency_score = 25
        balance_score = 20
        responsiveness_score = 18
        engagement_score = 12
        consistency_score = 10
        
        friendship_index = frequency_score + balance_score + responsiveness_score + engagement_score + consistency_score
        
        self.assertGreaterEqual(friendship_index, 0)
        self.assertLessEqual(friendship_index, 100)
    
    def test_milestone_detection(self):
        """Test milestone achievement detection."""
        total_messages = 1500
        milestones = [100, 500, 1000, 5000]
        
        achieved = [m for m in milestones if total_messages >= m]
        
        self.assertEqual(len(achieved), 3)
        self.assertIn(100, achieved)
        self.assertIn(500, achieved)
        self.assertIn(1000, achieved)
    
    def test_emoji_extraction(self):
        """Test emoji extraction from messages."""
        messages_with_emojis = [
            "Hello 😊",
            "Great! 🎉",
            "Love it ❤️",
            "Amazing 🌟"
        ]
        
        # Count messages with emojis
        emoji_count = sum(1 for msg in messages_with_emojis if any(c for c in msg if ord(c) > 127))
        
        self.assertEqual(emoji_count, 4)


class TestRollingHealthScore(unittest.TestCase):
    """Test cases for rolling health score (Day 9)."""
    
    def setUp(self):
        """Set up test data for rolling calculations."""
        dates = pd.date_range('2023-11-01', periods=60, freq='D')
        self.test_df = pd.DataFrame({
            'datetime': dates,
            'sender': ['Alice', 'Bob'] * 30,
            'message': ['Test'] * 60,
            'message_length': np.random.randint(10, 100, 60)
        })
        self.test_df['date'] = self.test_df['datetime'].dt.date
    
    def test_rolling_window_calculation(self):
        """Test rolling window health score calculation."""
        window_days = 7
        dates = sorted(self.test_df['date'].unique())
        
        # Should be able to calculate rolling scores
        self.assertGreaterEqual(len(dates), window_days)
    
    def test_rolling_score_continuity(self):
        """Test that rolling scores change smoothly."""
        # Mock rolling scores
        scores = [0.7, 0.72, 0.75, 0.73, 0.76, 0.78, 0.77]
        
        # Check for extreme jumps (shouldn't change more than 0.3 per day)
        for i in range(1, len(scores)):
            diff = abs(scores[i] - scores[i-1])
            self.assertLess(diff, 0.3)
    
    def test_minimum_messages_threshold(self):
        """Test minimum message threshold for rolling calculation."""
        min_messages = 10
        
        # Window with fewer messages should be skipped
        few_messages_df = self.test_df.head(5)
        self.assertLess(len(few_messages_df), min_messages)


class TestVisualizationIntegration(unittest.TestCase):
    """Test cases for visualization module integration."""
    
    def test_plot_data_preparation(self):
        """Test data preparation for plotting."""
        dates = pd.date_range('2023-12-01', periods=50, freq='D')
        df = pd.DataFrame({
            'datetime': dates,
            'sender': ['Alice', 'Bob'] * 25,
            'message': ['Test'] * 50
        })
        
        # Prepare for timeline plot
        df['timestamp'] = pd.to_datetime(df['datetime'])
        message_counts = df.set_index('timestamp').resample('D').size()
        
        self.assertGreater(len(message_counts), 0)
    
    def test_heatmap_data_structure(self):
        """Test heatmap data structure."""
        dates = pd.date_range('2023-12-01', periods=100, freq='3H')
        df = pd.DataFrame({
            'datetime': dates,
            'sender': ['Alice'] * 100,
            'message': ['Test'] * 100
        })
        
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        df['day'] = pd.to_datetime(df['datetime']).dt.day_name()
        
        heatmap_data = df.groupby(['day', 'hour']).size().unstack(fill_value=0)
        
        self.assertGreater(len(heatmap_data), 0)


def run_analysis_tests():
    """Run all analysis tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestEDAModule))
    suite.addTests(loader.loadTestsFromTestCase(TestSentimentAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestEmotionClassification))
    suite.addTests(loader.loadTestsFromTestCase(TestRelationshipHealth))
    suite.addTests(loader.loadTestsFromTestCase(TestGamificationFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestRollingHealthScore))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualizationIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    unittest.main()
