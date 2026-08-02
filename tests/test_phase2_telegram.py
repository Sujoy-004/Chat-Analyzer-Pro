"""Phase 2 Telegram parser hardening tests (D-19/D-20, MEDIUM #3, QUAL-01).

Exercises the real chat_analyzer.parser.telegram_parser module: both JSON
shapes (bare Chat + chats.list[]), recursive entity-array text join,
service-message filtering, honest malformed-drop counting (no bare
`except: continue`), tz-aware -> naive UTC normalization, empty/missing-key
exports raising "Not a Telegram chat export", and the QUAL-01
parse_telegram_chat DataFrame contract.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from chat_analyzer.parser.telegram_parser import (
    parse_telegram_chat,
    parse_telegram_chat_with_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "data" / "sample_chats" / "telegram_sample.json"
FULL_EXPORT = REPO_ROOT / "tests" / "fixtures" / "telegram_full_export.json"
BARE_ENTITY = REPO_ROOT / "tests" / "fixtures" / "telegram_bare_entity.json"


def test_bare_chat_sample_counts():
    """The existing 5-message sample parses fully with no system/skips."""
    rows, counts = parse_telegram_chat_with_report(str(SAMPLE))
    assert counts["parsed_messages"] == 5
    assert counts["system_messages"] == 0
    assert counts["skipped_lines"] == 0
    assert len(rows) == 5


def test_chats_list_shape_with_recursive_text_join():
    """chats.list[] shape: parsed=3, system=1, skipped=2; msg 1 text joins
    entity-dict + str parts into 'hello world'."""
    rows, counts = parse_telegram_chat_with_report(str(FULL_EXPORT))
    assert counts == {
        "total_lines": 6,
        "parsed_messages": 3,
        "skipped_lines": 2,
        "system_messages": 1,
    }
    msg1 = next(r for r in rows if r["message_id"] == 1)
    assert msg1["message"] == "hello world"


def test_entity_array_bare_chat():
    """Bare Chat + entity-array text: msg 1 joins to '@team check this'."""
    rows, counts = parse_telegram_chat_with_report(str(BARE_ENTITY))
    assert counts["parsed_messages"] == 2
    assert counts["system_messages"] == 0
    assert counts["skipped_lines"] == 0
    msg1 = next(r for r in rows if r["message_id"] == 1)
    assert msg1["message"] == "@team check this"


def test_service_message_excluded():
    """Service message (id 2) never appears as a row; system count increments."""
    rows, counts = parse_telegram_chat_with_report(str(FULL_EXPORT))
    assert counts["system_messages"] == 1
    assert all(r.get("message_id") != 2 for r in rows)


def test_tz_normalized_to_naive_utc():
    """Every returned row datetime is tz-naive (D-20): the Z-suffix row is
    10:00:00 naive UTC and the +05:30 row is 04:16:00 naive UTC."""
    rows, _ = parse_telegram_chat_with_report(str(FULL_EXPORT))
    z_row = next(r for r in rows if r["message_id"] == 3)
    assert z_row["datetime"].tzinfo is None
    assert z_row["datetime"].hour == 10
    assert z_row["datetime"].minute == 0

    bare_rows, _ = parse_telegram_chat_with_report(str(BARE_ENTITY))
    offset_row = next(r for r in bare_rows if r["message_id"] == 2)
    assert offset_row["datetime"].tzinfo is None
    assert (offset_row["datetime"].hour, offset_row["datetime"].minute) == (4, 16)

    for row in rows + bare_rows:
        assert row["datetime"].tzinfo is None


def test_malformed_dropped_honestly():
    """Bad-date and non-'message' type messages are counted, never silently
    dropped via a bare `except: continue`."""
    rows, counts = parse_telegram_chat_with_report(str(FULL_EXPORT))
    assert counts["skipped_lines"] == 2
    # id 5 (bad date) and id 6 (forwarded) are absent
    assert all(r.get("message_id") not in (5, 6) for r in rows)


@pytest.mark.parametrize(
    "payload",
    [
        {"chats": []},
        {"messages": []},
        {"name": "no messages or chats key"},
    ],
)
def test_not_a_chat_export_raises(tmp_path, payload):
    """Empty or missing-key exports raise ValueError (MEDIUM #3) instead of
    silently parsing to zero messages."""
    f = tmp_path / "empty.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        parse_telegram_chat_with_report(str(f))
    assert "Not a Telegram chat export" in str(exc.value)


def test_parse_telegram_chat_qual01():
    """QUAL-01: parse_telegram_chat still returns a pandas DataFrame with the
    core columns; system/service rows excluded; datetime naive UTC."""
    df = parse_telegram_chat(str(SAMPLE))
    assert isinstance(df, pd.DataFrame)
    for col in ("datetime", "sender", "message", "date", "time", "hour",
                "message_length", "message_id", "type"):
        assert col in df.columns
    assert len(df) == 5
    assert pd.api.types.is_datetime64_any_dtype(df["datetime"])
    assert df["datetime"].dt.tz is None
