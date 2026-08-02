"""Telegram chat export parser (D-19/D-20).

Supports both JSON shapes — a bare Chat export ({"messages": [...]}) and a
full export ({"chats": [{"messages": [...]}]}) — joins entity-array message
texts recursively, filters service messages, normalizes tz-aware dates to
naive UTC, and counts every dropped message honestly (no bare
`except: continue`).
"""

import json
from datetime import UTC, datetime

import pandas as pd
import requests


def _load_messages(data: dict) -> list:
    """Extract the message list from either Telegram export shape.

    Raises ValueError for exports with no usable messages (MEDIUM #3) so an
    empty export never silently parses to zero messages.
    """
    messages = data.get("messages")
    if isinstance(messages, list):
        result = list(messages)
    else:
        chats = data.get("chats")
        if isinstance(chats, list):
            result = []
            for chat in chats:
                if isinstance(chat, dict) and isinstance(chat.get("messages"), list):
                    result.extend(chat["messages"])
        else:
            raise ValueError("Not a Telegram chat export (no 'messages' or 'chats' key)")  # noqa: TRY004 - a missing expected key is a data-shape error, not a Python type error; ValueError is the contract (D-19, MEDIUM #3)
    if not result:
        raise ValueError("Not a Telegram chat export (no messages found)")
    return result


def _join_text(parts) -> str:
    """Join Telegram text parts into a single string.

    Accepts a plain str (as-is) or a list of str parts and entity dicts whose
    "text" values are appended recursively. Dict parts without a text key are
    skipped.
    """
    if isinstance(parts, str):
        return parts
    if isinstance(parts, list):
        joined = ""
        for part in parts:
            if isinstance(part, str):
                joined += part
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    joined += text
        return joined
    return ""


def _to_naive_utc(date_str: str) -> datetime:
    """Parse an ISO date string and normalize to naive UTC (D-20).

    Python >= 3.11 accepts a trailing 'Z' natively in fromisoformat, so no
    zero-offset string surgery is needed (ruff FURB162). Aware datetimes are
    shifted to UTC and stripped of tzinfo; naive values pass through unchanged.
    """
    dt = datetime.fromisoformat(date_str)
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def parse_telegram_chat_with_report(source: str) -> tuple[list[dict], dict]:
    """Parse a Telegram JSON export (file path or URL) with honest counters.

    Returns (rows, counts) where counts has total_lines, parsed_messages,
    skipped_lines and system_messages. Service rows never enter rows.
    """
    # Load data (keep the existing URL-vs-file handling)
    if source.startswith("http"):
        response = requests.get(source)
        data = response.json()
    else:
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)

    messages = _load_messages(data)
    rows: list[dict] = []
    system_messages = 0
    skipped_lines = 0

    for msg in messages:
        # Service messages (D-18/D-19): counted, excluded from rows
        if msg.get("type") == "service":
            system_messages += 1
            continue

        # Any other non-message type: honest skip, never silent (D-19)
        if msg.get("type") != "message":
            skipped_lines += 1
            continue

        try:
            dt = _to_naive_utc(msg["date"])
        except KeyError:
            # Message with no 'date' key — honest skip, never a crash
            skipped_lines += 1
            continue
        except (ValueError, TypeError):
            skipped_lines += 1
            continue

        text = _join_text(msg.get("text"))
        if not text and any(key in msg for key in ("photo", "video", "document", "audio")):
            text = "<Media omitted>"

        sender = msg.get("from") or msg.get("actor") or "Unknown"

        row: dict = {
            "datetime": dt,
            "sender": sender,
            "message": text,
            "message_length": len(text),
            "date": dt.date(),
            "time": dt.time(),
            "hour": dt.hour,
            "type": "message",
        }
        if msg.get("id") is not None:
            row["message_id"] = msg["id"]
        rows.append(row)

    return rows, {
        "total_lines": len(messages),
        "parsed_messages": len(rows),
        "skipped_lines": skipped_lines,
        "system_messages": system_messages,
    }


def parse_telegram_chat(source):
    """
    Parse Telegram chat from JSON file/URL into structured DataFrame

    Args:
        source (str): File path or URL to Telegram JSON export

    Returns:
        pd.DataFrame: Parsed chat data

    Hardened internals: system/service rows are excluded, malformed messages
    are counted and dropped (never a bare `except: continue`), and every row
    datetime is naive UTC. Empty or unparseable exports raise ValueError from
    _load_messages (previously they returned an empty DataFrame).
    """
    rows, _ = parse_telegram_chat_with_report(source)
    return pd.DataFrame(rows)
