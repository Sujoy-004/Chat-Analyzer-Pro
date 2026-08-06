"""
Unit Tests for Analysis Modules (rewired to real chat_analyzer.* modules — D-16).

Tests exercise the REAL EDA, sentiment, emotion, relationship health and
gamification modules on small fixture DataFrames. Heavy model callables
(transformers pipeline) are mocked with unittest.mock (D-17) — the real
analyzer/pipeline logic is what the assertions exercise. The legacy
duplicated-logic copies were removed in this rewire.
"""

import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

# Headless-first (Pitfall 7): the chat_analyzer.analysis.* modules import
# matplotlib.pyplot at module import, and this machine's default TkAgg backend
# is broken. Pin Agg BEFORE any pyplot import (same guarantee run_pipeline
# makes) so figure creation is headless-safe.
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

# sentiment.py prints "⚠️ TextBlob not available" at module import time (the
# missing-optional-dependency notice at sentiment.py:22). On a cp1252 console
# that emoji raises UnicodeEncodeError before the tests even run. Suppress
# import-time stdout (D-17 pattern) so the file works via unittest.main().
with redirect_stdout(StringIO()):
    from chat_analyzer.analysis import sentiment as _sentiment
    from chat_analyzer.analysis.eda import ChatEDA
    from chat_analyzer.analysis.emotion import EmotionAnalyzer
    from chat_analyzer.analysis.relationship_health import (
        analyze_relationship_health,
        calculate_rolling_health_score,
    )
    from chat_analyzer.analysis.sentiment import (
        add_sentiment_analysis,
        get_sentiment_summary,
    )
    from chat_analyzer.ingest.ingestion import messages_to_dataframe
    from chat_analyzer.utils.visualization import ChatVisualizer


class TestEDAModule(unittest.TestCase):
    """Test cases for exploratory data analysis module (real ChatEDA)."""

    def setUp(self):
        """Set up test data (fixture via messages_to_dataframe, 6h spacing)."""
        dates = pd.date_range('2023-12-01', periods=100, freq='6h')
        messages = [
            {
                'datetime': dt,
                'sender': 'Alice' if i % 2 == 0 else 'Bob',
                'message': 'Test message',
            }
            for i, dt in enumerate(dates)
        ]
        self.test_df = messages_to_dataframe(messages)

    def test_message_volume_calculation(self):
        """Test message volume statistics via the real ChatEDA."""
        eda = ChatEDA(self.test_df)
        sender_counts = eda.analyze_message_volume()['sender_counts']

        self.assertEqual(len(self.test_df), 100)
        self.assertEqual(sender_counts.to_dict(), {'Alice': 50, 'Bob': 50})

    def test_hourly_activity_distribution(self):
        """Test hourly activity analysis via the real ChatEDA."""
        eda = ChatEDA(self.test_df)
        hourly_activity = eda.analyze_message_volume()['hourly_activity']

        self.assertGreater(len(hourly_activity), 0)
        self.assertTrue(all(0 <= h < 24 for h in hourly_activity.columns))

    def test_daily_activity_distribution(self):
        """Test daily activity analysis via the real ChatEDA."""
        eda = ChatEDA(self.test_df)
        daily_messages = eda.analyze_message_volume()['daily_messages']

        self.assertGreater(len(daily_messages), 0)

    def test_top_senders_calculation(self):
        """Test top senders identification via the real ChatEDA."""
        eda = ChatEDA(self.test_df)
        top_senders = eda.analyze_message_volume()['sender_counts'].head(5)

        self.assertEqual(len(top_senders), 2)
        self.assertTrue(all(count > 0 for count in top_senders.values))

    def test_message_length_statistics(self):
        """Test content statistics via the real ChatEDA comprehensive summary."""
        eda = ChatEDA(self.test_df)
        summary = eda.generate_comprehensive_summary()

        self.assertIn('dataset_info', summary)
        self.assertIn('content_insights', summary)
        self.assertEqual(summary['dataset_info']['total_messages'], 100)
        self.assertGreaterEqual(summary['content_insights']['total_words'], 0)


class TestSentimentAnalysis(unittest.TestCase):
    """Test cases for sentiment analysis module (real add_sentiment_analysis)."""

    def setUp(self):
        """Set up a small fixture DataFrame with known sentiments and analyze it."""
        self.positive_messages = [
            "I love this!",
            "This is amazing!",
            "Great job!",
            "Awesome!",
        ]

        self.negative_messages = [
            "This is terrible",
            "I hate this",
            "Awful experience",
            "Very disappointed",
        ]

        self.neutral_messages = [
            "Okay",
            "The weather is cloudy",
            "I went to the store",
            "It's 5 PM",
        ]

        messages = []
        for i, text in enumerate(self.positive_messages):
            messages.append({'datetime': f'2023-12-01T09:{i:02d}:00', 'sender': 'Alice', 'message': text})
        for i, text in enumerate(self.negative_messages):
            messages.append({'datetime': f'2023-12-01T10:{i:02d}:00', 'sender': 'Bob', 'message': text})
        for i, text in enumerate(self.neutral_messages):
            messages.append({'datetime': f'2023-12-01T11:{i:02d}:00', 'sender': 'Carol', 'message': text})

        df = messages_to_dataframe(messages)

        # Pin the VADER path (no HF model construction/download in the suite — D-17/T-04-15).
        _sentiment.TRANSFORMERS_AVAILABLE = False
        with redirect_stdout(StringIO()):
            self.df_sent = add_sentiment_analysis(df)

    def test_positive_sentiment_detection(self):
        """Test detection of positive sentiments (real VADER columns)."""
        self.assertIn('vader_compound', self.df_sent.columns)
        self.assertIn('vader_sentiment', self.df_sent.columns)

        positive_df = self.df_sent[self.df_sent['sender'] == 'Alice']
        self.assertEqual(len(positive_df), 4)
        self.assertTrue(all(compound > 0 for compound in positive_df['vader_compound']))

    def test_negative_sentiment_detection(self):
        """Test detection of negative sentiments (real VADER columns)."""
        negative_df = self.df_sent[self.df_sent['sender'] == 'Bob']
        self.assertEqual(len(negative_df), 4)
        self.assertTrue(all(compound < 0 for compound in negative_df['vader_compound']))

    def test_neutral_sentiment_detection(self):
        """Test detection of neutral sentiments (real VADER columns)."""
        neutral_df = self.df_sent[self.df_sent['sender'] == 'Carol']
        self.assertEqual(len(neutral_df), 4)
        self.assertTrue(all(abs(compound) <= 0.5 for compound in neutral_df['vader_compound']))

    def test_sentiment_distribution(self):
        """Test sentiment distribution calculation via the real summary."""
        summary = get_sentiment_summary(self.df_sent)
        distribution = summary['sentiment_distribution']

        self.assertIsInstance(distribution, dict)
        self.assertEqual(sum(distribution.values()), len(self.df_sent))

    def test_sentiment_score_range(self):
        """Test that real VADER scores are within valid range."""
        self.assertTrue(all(-1 <= score <= 1 for score in self.df_sent['vader_compound']))


class TestEmotionClassification(unittest.TestCase):
    """Test cases for emotion classification module (real EmotionAnalyzer, mocked pipeline)."""

    # Faithful transformers-pipeline output shape: a flat list of label/score dicts
    # per class (RESEARCH 220-230). Two variants produce non-uniform dominant
    # emotions so the real parse fix surfaces. Tuples keep class attributes
    # immutable (RUF012).
    JOY_SCORES = (
        {'label': 'joy', 'score': 0.87},
        {'label': 'sadness', 'score': 0.03},
        {'label': 'anger', 'score': 0.03},
        {'label': 'fear', 'score': 0.02},
        {'label': 'surprise', 'score': 0.02},
        {'label': 'love', 'score': 0.03},
    )
    LOVE_SCORES = (
        {'label': 'love', 'score': 0.8},
        {'label': 'joy', 'score': 0.1},
        {'label': 'sadness', 'score': 0.02},
        {'label': 'anger', 'score': 0.02},
        {'label': 'fear', 'score': 0.02},
        {'label': 'surprise', 'score': 0.04},
    )

    def setUp(self):
        """Set up a 2-row fixture DataFrame."""
        self.df = messages_to_dataframe([
            {'datetime': '2023-12-01T09:00:00', 'sender': 'Alice', 'message': 'I love this!'},
            {'datetime': '2023-12-01T09:05:00', 'sender': 'Bob', 'message': 'Just checking in'},
        ])

    def _make_analyzer(self):
        """Build a real EmotionAnalyzer with the transformers pipeline mocked (D-17).

        Patches the module-level model cache (`_emotion_analyzer` /
        `_emotion_model_loaded`) instead of `transformers.pipeline`. This keeps
        the tests runnable in a clean LEAN install where `transformers` is NOT
        importable (it is an optional `[nlp]`-extra dependency): `_initialize_model`
        short-circuits to the cached fake pipeline and never executes its lazy
        `from transformers import pipeline`.
        """

        def _classifier(text):
            return self.LOVE_SCORES if 'love' in str(text).lower() else self.JOY_SCORES

        from chat_analyzer.analysis import emotion as _emotion_module

        with (
            redirect_stdout(StringIO()),
            mock.patch.object(_emotion_module, "_emotion_analyzer", _classifier),
            mock.patch.object(_emotion_module, "_emotion_model_loaded", True),
        ):
            return EmotionAnalyzer()

    def test_emotion_categories(self):
        """Test emotion category columns and the locked default model name."""
        analyzer = self._make_analyzer()
        with redirect_stdout(StringIO()):
            df_emo = analyzer.analyze_emotions(self.df)

        emotion_cols = [f'emotion_{emotion}' for emotion in analyzer.emotions]
        for col in emotion_cols:
            self.assertIn(col, df_emo.columns)

        self.assertEqual(analyzer.model_name, 'bhadresh-savani/distilbert-base-uncased-emotion')

    def test_emotion_distribution(self):
        """Test emotion distribution via the real summary — non-uniform scores."""
        analyzer = self._make_analyzer()
        with redirect_stdout(StringIO()):
            df_emo = analyzer.analyze_emotions(self.df)

        # The real parse fix surfaces: dominant emotions must NOT be uniform.
        self.assertGreater(df_emo['dominant_emotion'].nunique(), 1)

        distribution = analyzer.get_emotion_summary(df_emo)['emotion_distribution']
        self.assertIsInstance(distribution, dict)
        self.assertGreater(len(distribution), 1)

    def test_emotion_pipeline_nested_list_transformers5(self):
        """transformers 5.x nests top_k=None output one level deeper:
        [[{label, score}, ...]] instead of 4.x's flat [{label, score}, ...].
        The real EmotionAnalyzer must normalize both shapes so scores are not
        silently degraded to uniform 1/6 neutral (C-… 5.x compat regression)."""

        def _nested_classifier(text):
            if 'love' in str(text).lower():
                return [list(self.LOVE_SCORES)]
            return [list(self.JOY_SCORES)]

        from chat_analyzer.analysis import emotion as _emotion_module

        with (
            redirect_stdout(StringIO()),
            mock.patch.object(_emotion_module, "_emotion_analyzer", _nested_classifier),
            mock.patch.object(_emotion_module, "_emotion_model_loaded", True),
        ):
            analyzer = EmotionAnalyzer()

        with redirect_stdout(StringIO()):
            df_emo = analyzer.analyze_emotions(self.df)

        # Real per-message scores surfaced (not the uniform 1/6 fallback).
        self.assertGreater(df_emo['dominant_emotion'].nunique(), 1)
        dominant_col = f"emotion_{df_emo['dominant_emotion'].mode().iloc[0]}"
        self.assertGreater(df_emo[dominant_col].max(), 0.5)


class TestRelationshipHealth(unittest.TestCase):
    """Test cases for relationship health metrics (real analyze_relationship_health)."""

    def setUp(self):
        """Set up test conversation data (100 messages, 30min spacing)."""
        dates = pd.date_range('2023-12-01', periods=100, freq='30min')
        messages = [
            {
                'datetime': dt,
                'sender': 'Alice' if i % 2 == 0 else 'Bob',
                'message': 'Test',
            }
            for i, dt in enumerate(dates)
        ]
        self.test_df = messages_to_dataframe(messages)

    def _health(self):
        with redirect_stdout(StringIO()):
            return analyze_relationship_health(self.test_df)

    def test_conversation_starter_identification(self):
        """Test conversation starter identification via the real pipeline."""
        health = self._health()

        self.assertIn('conversation_stats', health)
        self.assertEqual(health['conversation_stats']['total_messages'], 100)
        self.assertIn('initiator_analysis', health)

    def test_initiator_ratio_calculation(self):
        """Test conversation initiator ratio via the real pipeline."""
        health = self._health()

        balance_score = health['initiator_analysis']['balance_score']
        self.assertIsInstance(balance_score, float)
        self.assertGreaterEqual(balance_score, 0.0)
        self.assertLessEqual(balance_score, 1.0)

    def test_balance_score_calculation(self):
        """Test balance score computation via the real pipeline."""
        health = self._health()

        balance_score = health['initiator_analysis']['balance_score']
        self.assertGreaterEqual(balance_score, 0.0)
        self.assertLessEqual(balance_score, 1.0)

    def test_dominance_score(self):
        """Test dominance analysis exists via the real pipeline."""
        health = self._health()

        self.assertIn('dominance_analysis', health)
        self.assertIsInstance(health['dominance_analysis'], dict)

    def test_health_score_range(self):
        """Test that real health scores are in valid range [0, 1]."""
        health = self._health()

        overall = health['health_score']['overall_health_score']
        self.assertIsInstance(overall, float)
        self.assertGreaterEqual(overall, 0.0)
        self.assertLessEqual(overall, 1.0)
        self.assertIsInstance(health['health_score']['grade'], str)


class TestGamificationFeatures(unittest.TestCase):
    """Test cases for gamification features via the real health pipeline."""

    def setUp(self):
        """Set up a fixture with a 10-day streak, a gap, then a 5-day streak."""
        day = pd.Timestamp('2023-12-01')
        dates = [day + pd.Timedelta(days=i) for i in range(10)]
        gap_day = day + pd.Timedelta(days=13)
        dates.extend(gap_day + pd.Timedelta(days=i) for i in range(5))

        messages = [
            {
                'datetime': d.strftime('%Y-%m-%dT%H:%M:%S'),
                'sender': 'Alice' if i % 2 == 0 else 'Bob',
                'message': 'Test',
            }
            for i, d in enumerate(dates)
        ]
        self.test_df = messages_to_dataframe(messages)

    def _health(self):
        with redirect_stdout(StringIO()):
            return analyze_relationship_health(self.test_df)

    def test_streak_detection(self):
        """Test conversation streak detection via the real pipeline."""
        health = self._health()

        streaks = health.get('streaks', {})
        self.assertIn('longest_streak', streaks)
        self.assertGreaterEqual(streaks.get('longest_streak', 0), 1)
        self.assertEqual(streaks.get('total_active_days'), 15)

    def test_longest_streak_calculation(self):
        """Test longest streak identification via the real pipeline."""
        health = self._health()

        # The fixture has a real 10-day streak — the real detector must find it.
        self.assertEqual(health['streaks']['longest_streak'], 10)

    def test_friendship_index_range(self):
        """Test friendship index is in range [0, 100] via the real pipeline."""
        health = self._health()

        friendship_index = health.get('friendship_index', {}).get('friendship_index', 0)
        self.assertGreaterEqual(friendship_index, 0)
        self.assertLessEqual(friendship_index, 100)
        self.assertIn('tier', health['friendship_index'])

    def test_milestone_detection(self):
        """Test milestone achievement detection via the real pipeline."""
        health = self._health()

        milestones = health.get('milestones', {})
        self.assertIn('total_achievements', milestones)
        self.assertIsInstance(milestones.get('total_achievements'), int)
        self.assertIsInstance(milestones.get('achievements'), list)
        self.assertIn('progress', milestones)

    def test_emoji_extraction(self):
        """Test emoji personality analysis exists via the real pipeline."""
        health = self._health()

        self.assertIn('emoji_personality', health)
        self.assertIsInstance(health['emoji_personality'], dict)


class TestRollingHealthScore(unittest.TestCase):
    """Test cases for rolling health score via the real module."""

    def setUp(self):
        """Set up a daily fixture dense enough to satisfy the real min_messages threshold."""
        dates = pd.date_range('2023-11-01', periods=30, freq='D')
        messages = []
        for dt in dates:
            messages.append({'datetime': dt, 'sender': 'Alice', 'message': 'Test'})
            messages.append({'datetime': dt, 'sender': 'Bob', 'message': 'Test'})
        self.test_df = messages_to_dataframe(messages)

    def test_rolling_window_calculation(self):
        """Test rolling window health score calculation via the real module."""
        result = calculate_rolling_health_score(self.test_df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0, 'per-window scores must exist')
        for col in ('date', 'health_score', 'grade', 'message_count'):
            self.assertIn(col, result.columns)

    def test_rolling_score_continuity(self):
        """Test that real rolling scores are valid and respect the message threshold."""
        result = calculate_rolling_health_score(self.test_df)

        self.assertTrue(all(0 <= score <= 1 for score in result['health_score']))
        self.assertTrue(all(count >= 10 for count in result['message_count']))

    def test_minimum_messages_threshold(self):
        """Test minimum message threshold for rolling calculation via the real module."""
        few_messages_df = self.test_df.head(5)
        self.assertLess(len(few_messages_df), 10)

        result = calculate_rolling_health_score(few_messages_df)
        self.assertEqual(len(result), 0, 'windows below the real threshold must be skipped')


class TestVisualizationIntegration(unittest.TestCase):
    """Test cases for visualization module integration (real ChatVisualizer)."""

    def setUp(self):
        """Set up a fixture DataFrame (with the 'timestamp' column the visualizer requires)."""
        dates = pd.date_range('2023-12-01', periods=50, freq='D')
        messages = [
            {
                'datetime': dt,
                'sender': 'Alice' if i % 2 == 0 else 'Bob',
                'message': 'Test',
            }
            for i, dt in enumerate(dates)
        ]
        self.df = messages_to_dataframe(messages)

    def test_plot_data_preparation(self):
        """Test timeline plotting returns a real matplotlib Figure."""
        fig = ChatVisualizer().plot_message_timeline(self.df)

        self.assertIsNotNone(fig)
        self.assertEqual(type(fig).__name__, 'Figure')

    def test_heatmap_data_structure(self):
        """Test heatmap plotting returns a real matplotlib Figure."""
        fig = ChatVisualizer().plot_activity_heatmap(self.df)

        self.assertIsNotNone(fig)
        self.assertEqual(type(fig).__name__, 'Figure')


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
