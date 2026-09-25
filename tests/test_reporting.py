"""Real tests for the reporting package (pdf_report + weekly_digest).

The legacy file asserted against inline dictionaries and f-strings and never
imported a single ``chat_analyzer.reporting.*`` module. These tests call the
real shipped generators — ``generate_chat_analysis_pdf`` and
``WeeklyDigestBot.generate_weekly_summary`` — on a tiny fixture and assert real
output: a %PDF-backed file on disk and a digest dict with correct scalars.

Delivery functions that need SMTP or Telegram credentials
(``send_email_digest`` / ``send_telegram_digest`` / ``send_quick_digest``)
are intentionally NOT tested here (no fake-serve): only the pure
formatting/validation helpers that exist in the source are exercised.
"""

import os
from datetime import datetime

# Headless-first (TESTING.md convention): pdf_report imports matplotlib at
# module import; pin Agg BEFORE any chat_analyzer import so chart creation is
# headless-safe on this machine's broken default TkAgg backend.
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from chat_analyzer.reporting.pdf_report import generate_chat_analysis_pdf
from chat_analyzer.reporting.weekly_digest import WeeklyDigestBot, create_digest_bot

# Naive fixture values match production's deliberate naive handling (noqa: DTZ001).
DIGEST_START = datetime(2024, 1, 1)  # noqa: DTZ001 - fixture value
DIGEST_END = datetime(2024, 1, 3, 23, 59, 59)  # noqa: DTZ001 - fixture value


def _tiny_chat_df() -> pd.DataFrame:
    """4-message fixture covering 2 senders, 3 days, 3 sentiment labels.

    Deterministic values so every digest metric can be asserted exactly:
    avg message length 28/4 = 7.0, total words 1+2+3+2 = 8, mean sentiment
    score (1.0+0.0+1.0+0.0)/4 = 0.5, peak hour 10:00, peak day Tuesday.
    """
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 09:00:00",
                    "2024-01-02 10:00:00",
                    "2024-01-02 10:00:00",
                    "2024-01-03 12:00:00",
                ]
            ),
            "sender": ["Alice", "Bob", "Alice", "Bob"],
            "message": ["hi", "hi there", "how are you", "bye all"],
            "sentiment": ["positive", "neutral", "positive", "negative"],
            "sentiment_score": [1.0, 0.0, 1.0, 0.0],
        }
    )


def _sample_analysis_results() -> dict:
    """Minimal analysis-results dict matching pdf_report's documented shape
    (the ``example_pdf_generation`` sample at pdf_report.py:563)."""
    return {
        "conversation_stats": {
            "total_messages": 4,
            "unique_senders": 2,
            "date_range": "2024-01-01 to 2024-01-07",
            "total_conversations": 1,
            "avg_response_time": 1.0,
        },
        "health_score": {
            "overall_health_score": 0.85,
            "grade": "VERY GOOD",
            "description": "Healthy communication patterns",
            "component_scores": {
                "initiation_balance": 0.9,
                "responsiveness": 0.8,
                "response_balance": 0.85,
                "dominance_balance": 0.88,
            },
            "strengths": ["Balanced conversation initiation"],
            "areas_for_improvement": ["Could improve response speed"],
        },
        "initiator_analysis": {
            "initiator_counts": {"Alice": 2, "Bob": 2},
            "balance_score": 0.9,
            "interpretation": "Good balance",
        },
        "response_analysis": {
            "response_stats": {"mean": {"Alice": 1.0, "Bob": 1.5}},
            "total_responses_analyzed": 4,
            "overall_avg_response_minutes": 1.25,
            "responsiveness_score": 0.8,
            "response_balance_score": 0.85,
        },
        "dominance_analysis": {
            "message_distribution": {"Alice": 2, "Bob": 2},
            "composite_dominance_score": 0.88,
            "interpretation": "Good balance",
            "message_count_balance": 0.9,
            "message_length_balance": 0.85,
            "conversation_control_balance": 0.89,
        },
    }


def test_generate_pdf_writes_real_pdf(tmp_path):
    """generate_chat_analysis_pdf writes a non-empty, %PDF-header file."""
    output = tmp_path / "report.pdf"
    returned = generate_chat_analysis_pdf(_sample_analysis_results(), str(output))
    assert returned == str(output)
    assert output.exists()
    data = output.read_bytes()
    assert data
    assert data.startswith(b"%PDF-")


def test_weekly_summary_scalars_and_period():
    """Real digest builder returns the documented keys and exact scalars."""
    summary = WeeklyDigestBot().generate_weekly_summary(_tiny_chat_df(), DIGEST_START, DIGEST_END)
    assert summary["period"] == {"start": "2024-01-01", "end": "2024-01-03"}
    assert summary["total_messages"] == 4
    assert summary["total_participants"] == 2
    assert summary["message_distribution"] == {
        "2024-01-01": 1,
        "2024-01-02": 2,
        "2024-01-03": 1,
    }


def test_weekly_summary_top_contributors():
    """Real _get_top_contributors ranks by message count, capped at top_n."""
    summary = WeeklyDigestBot().generate_weekly_summary(_tiny_chat_df(), DIGEST_START, DIGEST_END)
    assert summary["top_contributors"] == [
        {"name": "Alice", "message_count": 2},
        {"name": "Bob", "message_count": 2},
    ]


def test_weekly_summary_sentiment_and_engagement():
    """Real sentiment summary, activity patterns and engagement metrics."""
    summary = WeeklyDigestBot().generate_weekly_summary(_tiny_chat_df(), DIGEST_START, DIGEST_END)
    assert summary["sentiment_summary"] == {
        "positive": 2,
        "neutral": 1,
        "negative": 1,
        "average_score": 0.5,
    }
    assert summary["most_active_day"] == "Tuesday"
    assert summary["activity_patterns"] == {
        "peak_hour": "10:00",
        "peak_day": "Tuesday",
        "hourly_distribution": {9: 1, 10: 2, 12: 1},
    }
    assert summary["engagement_metrics"] == {
        "avg_message_length": 7.0,
        "total_words": 8,
        "avg_response_time": "N/A",
    }


def test_weekly_summary_outside_range_is_empty():
    """Rows outside the requested range yield the empty-summary contract."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01 09:00:00"]),
            "sender": ["Alice"],
            "message": ["old"],
        }
    )
    summary = WeeklyDigestBot().generate_weekly_summary(df, DIGEST_START, DIGEST_END)
    assert summary["total_messages"] == 0
    assert summary["total_participants"] == 0
    assert summary["top_contributors"] == []
    assert summary["most_active_day"] == "N/A"
    assert summary["engagement_metrics"] == {}


def test_format_digest_email_embeds_summary():
    """format_digest_email renders the real summary into an HTML digest."""
    bot = WeeklyDigestBot()
    html = bot.format_digest_email(
        bot.generate_weekly_summary(_tiny_chat_df(), DIGEST_START, DIGEST_END)
    )
    assert "Weekly Chat Digest" in html
    assert "2024-01-01 to 2024-01-03" in html
    assert "Total Messages" in html
    assert "Alice" in html


def test_format_telegram_message_embeds_summary():
    """_format_telegram_message renders the real summary as markdown."""
    bot = WeeklyDigestBot()
    message = bot._format_telegram_message(
        bot.generate_weekly_summary(_tiny_chat_df(), DIGEST_START, DIGEST_END)
    )
    assert "Weekly Chat Digest" in message
    assert "2024-01-01 to 2024-01-03" in message
    assert "Total Messages: *4*" in message
    assert "Alice: 2 messages" in message


def test_calculate_percentage():
    """_calculate_percentage handles normal and zero-total division."""
    bot = WeeklyDigestBot()
    assert bot._calculate_percentage(50, 200) == 25.0
    assert bot._calculate_percentage(1, 0) == 0


def test_create_digest_bot_config_wiring():
    """create_digest_bot wires credentials into the bot's config dicts."""
    bot = create_digest_bot(
        email_sender="a@b.c",
        email_password="pw",
        smtp_server="smtp.example.com",
        smtp_port=587,
    )
    assert bot.email_config == {
        "smtp_server": "smtp.example.com",
        "smtp_port": 587,
        "sender_email": "a@b.c",
        "sender_password": "pw",
    }
    assert bot.telegram_config == {}

    bare = create_digest_bot()
    assert bare.email_config == {}
    assert bare.telegram_config == {}

    tg = create_digest_bot(telegram_bot_token="tok", telegram_chat_id="123")
    assert tg.telegram_config == {"bot_token": "tok", "chat_id": "123"}


def test_schedule_weekly_digest_without_recipients():
    """schedule_weekly_digest with no recipients sends nothing and returns {}."""
    bot = WeeklyDigestBot()
    results = bot.schedule_weekly_digest(_tiny_chat_df(), {})
    assert results == {}