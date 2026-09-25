"""COR-02 regression: messages_to_dataframe honestly accounts dropped rows.

The builder silently dropped rows with no parseable datetime while the parser
counters (total_lines / parsed_messages / skipped_lines / system_messages)
stayed quiet about them. COR-02 adds an optional dropped_rows count to the
builder (return_counts=True) and surfaces it in the CLI with a WARN line —
never by mutating the parser counters, so the strict parsers' skipped_lines
is never double-counted against the builder's dropped_rows.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from chat_analyzer.cli.contracts import ParseReport
from chat_analyzer.ingest.ingestion import messages_to_dataframe
from chat_analyzer.parser.whatsapp_parser import WhatsAppParser


def _valid_row() -> dict:
    return {
        "datetime": datetime(2025, 9, 15, 9, 45),  # noqa: DTZ001 - naive input is the point (D-20 contract)
        "sender": "Alice",
        "message": "hi",
    }


def test_missing_datetime_row_reported_and_dropped():
    """A row with no datetime is dropped; dropped_rows == 1; the valid row
    survives untouched."""
    df, dropped = messages_to_dataframe(
        [_valid_row(), {"sender": "Bob", "message": "no datetime"}],
        return_counts=True,
    )
    assert dropped == 1
    assert len(df) == 1
    assert df["sender"].iloc[0] == "Alice"


def test_all_valid_rows_no_drops():
    """A normal all-valid list: dropped_rows == 0, every message survives."""
    df, dropped = messages_to_dataframe(
        [_valid_row(), _valid_row()], return_counts=True
    )
    assert dropped == 0
    assert len(df) == 2


def test_default_return_still_plain_dataframe():
    """Existing callers (df = messages_to_dataframe(...)) get a bare df, so
    the 35+ in-repo call sites keep working unchanged (regression-safe)."""
    df = messages_to_dataframe([_valid_row()])
    assert isinstance(df, pd.DataFrame)


def test_parser_skipped_line_counted_once_not_doubled(tmp_path: Path):
    """Pipeline-level: the strict parser's unparseable-date header is counted
    exactly once as skipped_lines (never also as dropped); the builder only
    ever sees the valid row, so dropped stays 0. The CLI's
    ParseReport(source=source, **counts) still constructs unchanged."""
    chat = tmp_path / "chat.txt"
    chat.write_text(
        "15/09/25, 14:05 - Alice: hello there\n"
        "32/13/25, 09:00 - Bob: bad date\n",
        encoding="utf-8",
    )

    rows, counts = WhatsAppParser().parse_file_with_report(str(chat))
    df, dropped = messages_to_dataframe(rows, return_counts=True)

    assert counts["parsed_messages"] == 1
    assert counts["skipped_lines"] == 1  # the bad-date header, exactly once
    assert dropped == 0                  # never also counted as dropped
    assert len(df) == 1
    assert df["sender"].iloc[0] == "Alice"

    assert "dropped_rows" not in counts  # no new key leaks into the report dict
    report = ParseReport(source="whatsapp", **counts)
    assert report.parsed_messages == 1
    assert report.skipped_lines == 1