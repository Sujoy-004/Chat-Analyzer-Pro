"""Fast tests for the honesty workstreams WS-3 (network wording) and
WS-4 (non-Latin disclaimer). No heavy imports — plain unit checks.
"""

from __future__ import annotations

import pandas as pd

from chat_analyzer.cli.adapters import _strongest_connection_summary, build_insights
from chat_analyzer.utils.preprocessing import (
    is_non_latin_heavy,
    non_latin_message_share,
)

DISCLAIMER = "substantially non-English"


class TestStrongestConnectionSummary:
    def test_two_person_tie_is_evenly_matched(self):
        """A 2-person chat ties both directions — wording must say 'evenly'."""
        tie = [
            {"from": "Sujoy", "to": "Anu", "interactions": 8275},
            {"from": "Anu", "to": "Sujoy", "interactions": 8275},
        ]
        assert _strongest_connection_summary(tie) == (
            "Evenly matched: 8275 turn-exchanges in each direction."
        )

    def test_single_edge_names_the_pair(self):
        one = [{"from": "A", "to": "B", "interactions": 42}]
        assert _strongest_connection_summary(one) == (
            "The strongest connection is A to B (42 interactions)."
        )

    def test_empty_and_malformed_return_none(self):
        assert _strongest_connection_summary(None) is None
        assert _strongest_connection_summary([]) is None
        assert _strongest_connection_summary([{"from": "A"}]) is None


class TestNonLatinDetector:
    def test_bengali_script_flagged(self):
        assert is_non_latin_heavy("কেমন আছো ভাই")

    def test_chinese_script_flagged(self):
        assert is_non_latin_heavy("你好，世界")

    def test_english_not_flagged(self):
        assert not is_non_latin_heavy("hello world")

    def test_romanized_banglish_not_flagged(self):
        """Romanized Banglish is Latin script — the detector is script-based."""
        assert not is_non_latin_heavy("kemon acho bhai")

    def test_empty_and_non_str_never_flagged(self):
        assert not is_non_latin_heavy("")
        assert not is_non_latin_heavy(None)
        assert not is_non_latin_heavy("   ")

    def test_share_count(self):
        df = pd.DataFrame(
            {"message": ["কেমন আছো", "আমি ভালো আছি", "hello there", "fine thanks"]}
        )
        assert non_latin_message_share(df) == 0.5

    def test_share_missing_column_is_zero(self):
        df = pd.DataFrame({"nope": ["a", "b"]})
        assert non_latin_message_share(df) == 0.0
        assert non_latin_message_share(None) == 0.0


class TestNonLatinDisclaimerInInsights:
    @staticmethod
    def _leads(non_latin_share):
        return build_insights(
            {
                "busiest_day": "Mon",
                "total_messages": 10,
                "duration_days": 1,
                "peak_hour": 9,
                "avg_response_time": 5,
            },
            {"A": {"share_pct": 50}},
            {"top_words": ["hi"]},
            {"distribution": {"Positive": 8, "Neutral": 2}},
            {"overall_score": 0.8, "grade": "Good"},
            {"density": 0.5},
            {"distribution": {"joy": 5, "ok": 5}, "dominant": "joy"},
            non_latin_share=non_latin_share,
        )

    def test_disclaimer_appended_when_share_is_high(self):
        leads = self._leads(0.5)
        assert leads[4].startswith("The overall tone leans")
        assert DISCLAIMER in leads[4]
        emotion_lead = [l for l in leads if l.startswith("The dominant emotion")][0]
        assert DISCLAIMER in emotion_lead

    def test_no_disclaimer_when_share_is_low(self):
        leads = self._leads(0.1)
        assert DISCLAIMER not in leads[4]
        emotion_lead = [l for l in leads if l.startswith("The dominant emotion")][0]
        assert DISCLAIMER not in emotion_lead

    def test_indices_stay_stable_with_disclaimer(self):
        """Disclaimers append to existing sentences — tab indices never shift."""
        leads = self._leads(0.5)
        assert leads[5].startswith("Statistically this conversation scores")
        assert leads[6].startswith("The conversation network has density")
