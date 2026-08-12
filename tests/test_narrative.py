"""Tier A narrative module unit tests (DEFERRED §2.3).

Exercises the REAL ``chat_analyzer.analysis.narrative`` module on tiny pill
DataFrames built in-node (no file writes). Coverage matches the Category B
test fixtures: an arc+driver chat (A), a symmetric quick-reply chat (B), a
single-message chat (C), and Banglish question-marker recognition — plus the
non-crashing guards for empty/1-sender/NaN-sender inputs.
"""

import pandas as pd

from chat_analyzer.analysis.narrative import (
    _message_length_asymmetry,
    _question_ratio_per_sender,
    _reciprocity,
    _sensitive_topic_driver,
    analyze_narrative,
)

HEDGES = ("Possibly", "May", "Might", "This could")


def _make_df(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """Build a minimal canonical analysis DataFrame from (sender, message) rows."""
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=len(rows), freq="5min"),
            "sender": [r[0] for r in rows],
            "message": [r[1] for r in rows],
        }
    )
    df["message_length"] = df["message"].str.len()
    return df


def _hedged(text: str) -> bool:
    """Return True when observation text leads with a hedge word."""
    return any(text.startswith(hedge) for hedge in HEDGES)


class TestFixtureAArcAndDriver:
    """Fixture A: casual opening, later planning/meeting + one-sided questions.

    The same chat must surface both an ``arc`` observation (topic density
    rises in the later half) and a ``driver`` observation (one side pushes
    questions / topics), each at high or medium confidence.
    """

    def _df_a(self) -> pd.DataFrame:
        rows: list[tuple[str, str]] = []
        for i in range(16):
            rows.append(("Rima", "haha nice"))
            rows.append(("Imran", "no one"))
        for i in range(20):
            rows.append(("Rima", "ki meeting ta ekhon approved?"))
            rows.append(("Imran", "yep its set"))
        rows.append(("Rima", "kobe porer meeting hobe?"))
        rows.append(("Imran", "kol dilain"))
        return _make_df(rows)

    def _df_b(self) -> pd.DataFrame:
        rows = [
            ("Rima", "ok no"),
            ("Imran", "ok"),
            ("Rima", "hip"),
            ("Imran", "hip"),
            ("Rima", "team?"),
            ("Imran", "no"),
            ("Rima", "kill me"),
        ]
        return _make_df(rows)

    def _df_c(self) -> pd.DataFrame:
        return _make_df([("Rima", "hi")])

    def test_fixture_a_arc_and_driver_present(self):
        """Fixture A yields an arc and a driver observation at high/medium confidence."""
        result = analyze_narrative(self._df_a())
        kinds = [o["kind"] for o in result["observations"]]
        confs = [o["confidence"] for o in result["observations"]]
        assert "arc" in kinds
        assert "driver" in kinds
        assert all(c in {"high", "medium"} for c in confs)

    def test_observations_kind_stable(self):
        """All observation kinds come from the documented stable set."""
        result = analyze_narrative(self._df_a())
        for obs in result["observations"]:
            assert obs["kind"] in (
                "snapshot",
                "arc",
                "driver",
                "reciprocity",
                "engagement",
            )
            assert obs["confidence"] in {"high", "medium", "low"}

    def test_fixture_b_no_strong_driver_or_arc(self):
        """Symmetric quick-reply chat (Fixture B) must NOT claim a driver/arc.

        The always-on snapshot baseline (WS-6) is factual and excluded from
        the speculative-claim checks.
        """
        result = analyze_narrative(self._df_b())
        for obs in result["observations"]:
            if obs["kind"] == "snapshot":
                continue
            assert obs["confidence"] not in {"high", "medium"}
            assert obs["kind"] not in {"driver", "arc"}

    def test_fixture_c_single_message_never_crashes(self):
        """A single-message chat returns the base dict, never raises."""
        result = analyze_narrative(self._df_c())
        assert result["tier"] == "A"
        assert result["speculative"] is True
        assert result["observations"] == []
        assert result["narrative_summary"] == ""

    def test_empty_and_onesender_channels_never_crashes(self):
        """Empty and single-sender chats degrade to the base dict."""
        assert analyze_narrative(pd.DataFrame())["observations"] == []
        assert analyze_narrative(_make_df([("Rima", "one")]))["observations"] == []
        assert analyze_narrative(_make_df([("Rima", "a"), ("Rima", "b")]))["observations"] == []

    def test_nan_sender_never_crashes(self):
        """NaN senders quietly degrade rather than crash or mislabel.

        WS-6: the factual snapshot baseline is always present; no speculative
        observation may be emitted for a degenerate sender set.
        """
        df = _make_df([("Alice", "hello"), ("Bob", "hey there")])
        df.loc[0, "sender"] = None
        result = analyze_narrative(df)
        assert all(obs["kind"] == "snapshot" for obs in result["observations"])
        assert len(result["observations"]) == 1
        assert result["narrative_summary"].startswith("Snapshot:")

    def test_outcome_metrics_and_hedged_tone(self):
        """narrative_summary leads with the factual snapshot (WS-6); every
        speculative observation stays hedged."""
        result = analyze_narrative(self._df_a())
        assert result["narrative_summary"]
        assert result["observations"][0]["kind"] == "snapshot"
        assert result["observations"][0]["confidence"] == "high"
        for obs in result["observations"]:
            if obs["kind"] != "snapshot":
                assert _hedged(obs["text"])
        assert set(result) == {
            "observations",
            "narrative_summary",
            "tier",
            "speculative",
            "status",
        }
        assert result["status"] == {"nlp_available": False, "tier_b_generated": False}


class TestQuestionRatio:
    """Banglish question markers must drive the question ratio."""

    def test_sender_substring_not_matched(self):
        """Whole-token matching: 'ki' inside 'Sokoto' must not count."""
        df = _make_df(
            [
                ("A", "ki asche?"),
                ("B", "okkhotor niche hobe"),
                ("A", "keno suchais na?"),
                ("B", "hmm"),
                ("A", "kemon acho?"),
                ("B", "valo"),
                ("A", "kothay boscho?"),
            ]
        )
        ratios = _question_ratio_per_sender(df)["ratios"]
        assert ratios["A"] > 0.8
        assert ratios["B"] == 0.0


class TestExtractors:
    """Direct signal extractor contracts (pure pandas helpers)."""

    def test_question_ratio_recognizes_question_banglish(self):
        """Banglish markers ki/keno/kemon/koto/kothay/kobe raise the ratio."""
        df = _make_df(
            [
                ("A", "ki korcho?"),
                ("B", "kichu nah"),
                ("A", "keno skirtuje ashe?"),
                ("B", "na"),
                ("A", "kemon jeno hoy"),
                ("A", "koto deri lagbe?"),
                ("A", "kothay jaccho?"),
                ("A", "kobe ashbe?"),
            ]
        )
        ratios = _question_ratio_per_sender(df)["ratios"]
        assert ratios["A"] == 1.0
        assert ratios["B"] == 0.0

    def test_question_flag_on_english_wh(self):
        """English wh-words and '?' count as question markers too."""
        df = _make_df(
            [
                ("Ann", "what is it?"),
                ("Bob", "not much"),
                ("Ann", "how are you?"),
                ("Bob", "fine"),
            ]
        )
        ratios = _question_ratio_per_sender(df)["ratios"]
        assert ratios["Ann"] == 1.0
        assert ratios["Bob"] == 0.0

    def test_message_length_asymmetry_pairwise(self):
        """Two-sender chat reports an >= 1 ratio tag the longer writer."""
        df = _make_df(
            [("Short", "hi"), ("Long", "x" * 100)]
        )
        res = _message_length_asymmetry(df)
        assert res["ratio"] >= 1.0
        assert res["longer_sender"] == "Long"

    def test_reciprocity_exact_alternation(self):
        """Perfect alternation scores 1.0; same-sender stretches lower it."""
        alt = _make_df([("A", "x"), ("B", "y"), ("A", "x")])
        same = _make_df([("A", "x"), ("A", "y"), ("A", "z")])
        assert _reciprocity(alt)["reciprocity"] == 1.0
        assert _reciprocity(same)["reciprocity"] == 0.0

    def test_sensitive_topic_driver_identifies_leader(self):
        """The driver lexicon ties the sensitive-topic lead to one sender."""
        df = _make_df(
            [
                ("Lead", "plan the meeting deadline"),
                ("Quiet", "ok cool"),
                ("Lead", "call me about the contract"),
                ("Quiet", "nice"),
                ("Lead", "koto dilain project shuru?"),
            ]
        )
        res = _sensitive_topic_driver(df)
        assert res["leader"] == "Lead"