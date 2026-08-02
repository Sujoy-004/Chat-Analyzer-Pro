"""Phase 2 canonical DataFrame builder tests (D-20, Anti-Pattern 5).

Exercises chat_analyzer.ingest.ingestion.messages_to_dataframe — the single
dict->df builder consumed by the pipeline: schema + defaults, tz-aware input
normalized to naive UTC, the ingestion-path telegram full-ISO date bug, the
whatsapp date+time path, and unparseable-row dropping without crashing.
"""

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from chat_analyzer.ingest.ingestion import messages_to_dataframe


def test_schema_and_defaults():
    """The canonical schema carries all 9 columns with the right types."""
    df = messages_to_dataframe(
        [
            {
                "datetime": datetime(2025, 9, 15, 9, 45),  # noqa: DTZ001 - naive input is the point (D-20 contract)
                "sender": "A",
                "message": "hi",
            }
        ]
    )
    for col in (
        "datetime", "timestamp", "date", "hour", "sender",
        "message", "message_length", "source", "uid",
    ):
        assert col in df.columns, col
    assert df["timestamp"].iloc[0] == df["datetime"].iloc[0]
    assert isinstance(df["date"].iloc[0], date)
    assert df["hour"].iloc[0] == 9
    assert df["source"].iloc[0] == "unknown"
    assert isinstance(df["uid"].iloc[0], str)


def test_tz_aware_input_normalized_to_naive_utc():
    """A UTC+2 datetime becomes tz-naive UTC, 2 hours earlier (D-20)."""
    aware = datetime(2025, 9, 15, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    df = messages_to_dataframe(
        [{"datetime": aware, "sender": "A", "message": "hi"}]
    )
    assert df["datetime"].dt.tz is None
    assert df["hour"].iloc[0] == 10


def test_ingestion_path_telegram_full_iso():
    """Full ISO date in the date field (ingestion-path telegram bug) parses
    to a naive datetime."""
    df = messages_to_dataframe(
        [{"date": "2025-09-15T09:45:00", "time": "", "sender": "A", "message": "ok"}]
    )
    assert df["datetime"].iloc[0] == pd.Timestamp(2025, 9, 15, 9, 45)
    assert df["datetime"].dt.tz is None


def test_ingestion_path_whatsapp_date_time():
    """date + time fields combine into a datetime."""
    df = messages_to_dataframe(
        [{"date": "2025-09-15", "time": "09:45", "sender": "A", "message": "ok"}]
    )
    assert df["datetime"].iloc[0] == pd.Timestamp(2025, 9, 15, 9, 45)


def test_unparseable_rows_dropped():
    """Rows with no datetime/date/time are dropped from the df — no crash,
    caller owns skip accounting."""
    df = messages_to_dataframe(
        [
            {"sender": "A", "message": "no datetime at all"},
            {
                "datetime": datetime(2025, 9, 15, 9, 45),  # noqa: DTZ001 - naive input is the point (D-20 contract)
                "sender": "B",
                "message": "ok",
            },
        ]
    )
    assert len(df) == 1
    assert df["sender"].iloc[0] == "B"
