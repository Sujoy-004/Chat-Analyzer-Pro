"""Phase 2 WhatsApp parser hardening tests (D-15/D-16/D-17/D-18, HIGH #2, QUAL-01).

Exercises the real chat_analyzer.parser.whatsapp_parser module: strict date
parsing with NO datetime.now() fabrication, system-message classification
(encryption notice, "X added Y", header-without-sender), common datetime
formats (US 12h / EU 24h / iOS bracket / 4-digit year), multiline
continuation, exact fixture counts, and the QUAL-01 parse_file DataFrame
contract (HIGH #2 — rows must feed _add_features without KeyError).
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from chat_analyzer.parser.whatsapp_parser import WhatsAppParser

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "whatsapp_system_skip.txt"


def test_unparseable_date_skipped_not_fabricated(tmp_path):
    """A line matching the header regex with an unparseable date (month 13)
    produces NO row and increments skipped_lines — never datetime.now()."""
    bad = tmp_path / "bad_date.txt"
    bad.write_text("25/13/26, 9:30 AM - Alice: hello\n", encoding="utf-8")

    parser = WhatsAppParser()
    assert parser.parse_line_strict("25/13/26, 9:30 AM - Alice: hello") is None
    assert parser.skipped_lines == 1

    rows, counts = parser.parse_file_with_report(str(bad))
    assert rows == []
    assert counts["skipped_lines"] == 1
    assert counts["parsed_messages"] == 0

    today = datetime.now().date()  # noqa: DTZ005 - test-only "now", not a pipeline timestamp
    for row in rows:
        assert row["datetime"].date() != today


def test_system_messages_counted_not_appended():
    """Encryption notice + bare 'Alice added Bob' count as system and are
    NEVER appended to the previous message body."""
    parser = WhatsAppParser()
    rows, counts = parser.parse_file_with_report(str(FIXTURE))
    assert counts["system_messages"] == 2
    for row in rows:
        assert "end-to-end encrypted" not in row["message"]
        assert "added Bob" not in row["message"]


def test_header_without_sender_is_system(tmp_path):
    """A line with a timestamp header but no 'sender: ' part is counted as
    system, never a continuation of the previous message."""
    f = tmp_path / "header_only.txt"
    f.write_text("12/26/23, 10:00 AM - Group renamed\n", encoding="utf-8")

    parser = WhatsAppParser()
    rows, counts = parser.parse_file_with_report(str(f))
    assert rows == []
    assert counts["system_messages"] == 1
    assert counts["skipped_lines"] == 0


def test_common_datetime_formats():
    """US 12h, EU 24h, iOS bracket + 4-digit year each parse to the correct
    datetime (D-17: %m/%d tried first; no M/D-vs-D/M heuristics)."""
    cases = [
        ("12/25/23, 9:30 AM - Alice: US 12h", datetime(2023, 12, 25, 9, 30)),  # noqa: DTZ001 - expected values are deliberately naive
        ("25/12/2023, 21:07 - Bob: EU 24h", datetime(2023, 12, 25, 21, 7)),  # noqa: DTZ001
        ("[14/06/2024, 2:30:45 PM] Maria: iOS bracket", datetime(2024, 6, 14, 14, 30, 45)),  # noqa: DTZ001
        ("01/15/2024, 10:00 - Carol: 4-digit year", datetime(2024, 1, 15, 10, 0)),  # noqa: DTZ001
    ]
    parser = WhatsAppParser()
    for line, expected in cases:
        row = parser.parse_line_strict(line)
        assert row is not None, line
        assert row["datetime"] == expected, line


def test_multiline_continuation_joins_with_newline():
    """A continuation line joins the previous message with '\\n'."""
    parser = WhatsAppParser()
    rows, _ = parser.parse_file_with_report(str(FIXTURE))
    second = next(r for r in rows if "Second message" in r["message"])
    assert second["message"] == "Second message\nthis is a continuation line"


def test_exact_fixture_counts():
    """parse_file_with_report on the fixture returns the locked-in counts."""
    parser = WhatsAppParser()
    rows, counts = parser.parse_file_with_report(str(FIXTURE))
    assert counts == {
        "total_lines": 7,
        "parsed_messages": 3,
        "skipped_lines": 1,
        "system_messages": 2,
    }
    assert len(rows) == 3


def test_parse_file_returns_df_with_time_period():
    """QUAL-01: parse_file still returns a DataFrame whose columns include
    'time_period' — proving strict rows carry hour into _add_features
    (HIGH #2: line 169 df['hour'].apply(...) must not KeyError)."""
    parser = WhatsAppParser()
    df = parser.parse_file(str(FIXTURE))
    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert "time_period" in df.columns
    assert len(df) == 3
