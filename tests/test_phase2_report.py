"""Phase 2 HTML report card tests (D-08/D-09/D-10/D-11/D-12/D-14).

Exercises chat_analyzer.cli.report_html with a crafted AnalysisResults that
carries untrusted content (a `<script>alert(1)</script>` payload and a
sender named "Alice <3 Bob"): single-file self-containment, utf-8 + emoji
integrity, autoescape, filename sanitization, report location in the cwd
(D-09), auto-open degrade, skip-note surfacing, and the tab structure.
"""

import os
from pathlib import Path

from chat_analyzer.cli.report_html import (
    open_report,
    sanitize_filename,
    write_report,
)

INVALID = set('<>:"/\\|?*')


def _results() -> dict:
    return {
        "source": "whatsapp",
        "parse": {
            "total_lines": 2,
            "parsed_messages": 1,
            "skipped_lines": 1,
            "system_messages": 0,
        },
        "stats": {
            "total_messages": 1,
            "participants": 1,
            "participant_list": ["Alice <3 Bob"],
            "date_range": {"start": "2025-09-15", "end": "2025-09-15"},
            "duration_days": 1,
            "busiest_day": "Monday",
            "peak_hour": 9,
            "avg_response_time": None,
            "media_messages": 0,
        },
        "participants": {
            "Alice <3 Bob": {"messages": 1, "avg_message_length": 55.0, "share_pct": 100.0}
        },
        "content": {
            "top_words": ["hello", "world"],
            "top_emojis": ["🎉", "<script>alert(1)</script>"],
            "total_words": 2,
            "unique_words": 2,
        },
        "sentiment": {
            "distribution": {"Positive": 1},
            "avg_compound": 0.5,
            "by_sender": {"Alice <3 Bob": {"message_count": 1}},
            "daily_avg": {"2025-09-15": 0.5},
        },
        "health": {
            "overall_score": 0.82,
            "grade": "Good",
            "components": {},
            "initiator_balance": 0.6,
            "avg_response_minutes": 12,
            "response_balance": 0.7,
            "composite_dominance": 0.5,
            "total_conversations": 3,
        },
        "network": {
            "node_count": 2,
            "edge_count": 1,
            "density": 0.5,
            "reciprocity": 0.0,
            "strongest_connections": None,
            "key_participants": {},
            "subgroup_count": 1,
        },
        "charts": {
            "timeline": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=",
            "activity": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=",
            "participants": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=",
            "sentiment": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=",
            "health": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=",
            "network": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=",
        },
        "insights": [
            "Most messages land on Monday.",
            "Alice <3 Bob is the most active participant, sending 100.0% of all messages.",
            "Replies take no measurable time - mostly one-off messages.",
            "The most-used word is 'hello'.",
            "The overall tone leans Positive (100% of messages).",
        ],
        "report_path": "",
    }


def _write(tmp_path: Path) -> Path:
    src = tmp_path / "chat_analysis_test.txt"
    src.write_text("x\n", encoding="utf-8")
    # D-09: the report resolves against the cwd, so chdir into tmp_path to
    # keep LOW #8 ("the repo tree is never written to") intact.
    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        return write_report(_results(), src)
    finally:
        os.chdir(cwd)


def test_single_file_no_external_refs(tmp_path):
    out = _write(tmp_path).read_text(encoding="utf-8")
    assert "http://" not in out
    assert "https://" not in out
    assert "<script src" not in out
    assert "data:image/png;base64," in out


def test_charset_and_utf8_emoji(tmp_path):
    p = _write(tmp_path)
    out = p.read_text(encoding="utf-8")
    assert out.startswith("<!DOCTYPE html>")
    assert '<meta charset="utf-8">' in out
    assert "🎉" in out  # emoji intact through the utf-8 round trip


def test_escaping(tmp_path):
    out = _write(tmp_path).read_text(encoding="utf-8")
    # the message payload must be escaped, never raw
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out
    # '<3' in the sender name is escaped
    assert "Alice &lt;3 Bob" in out
    assert "Alice <3 Bob" not in out


def test_sanitize_filename():
    s1 = sanitize_filename("..\\..\\chat<name>:1.txt")
    assert not any(c in s1 for c in INVALID), s1
    assert not s1.startswith("."), s1
    assert not any(ord(c) < 32 for c in s1), s1
    assert "chat" in s1, s1

    s2 = sanitize_filename("con")
    assert s2 == "con" and not any(c in s2 for c in INVALID)

    s3 = sanitize_filename(".hidden")
    assert not s3.startswith(".") and s3 == "hidden"

    s4 = sanitize_filename("...")
    assert s4 == "chat_analysis"  # empty-after-strip fallback


def test_report_location_next_to_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # D-09: report lands in the cwd, not next to src
    p = _write(tmp_path)
    assert p == tmp_path / "chat_analysis_test_report.html"
    assert p.exists()


def test_open_report_degrade(monkeypatch, tmp_path):
    import webbrowser

    def boom(*args, **kwargs):
        raise OSError("no browser available")

    monkeypatch.setattr(webbrowser, "open", boom)
    assert open_report(tmp_path / "chat_report.html") is False


def test_open_report_success(monkeypatch, tmp_path):
    import webbrowser

    calls = []

    def fake_open(url):
        calls.append(url)
        return True

    monkeypatch.setattr(webbrowser, "open", fake_open)
    p = tmp_path / "chat_report.html"
    assert open_report(p) is True
    assert calls
    assert calls[0].startswith("file://")
    assert str(p.resolve()) in calls[0]


def test_skip_note_surfacing(tmp_path, monkeypatch):
    src = tmp_path / "a.txt"
    src.write_text("x\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # D-09: keep the report out of the repo tree

    res = _results()
    out1 = write_report(res, src).read_text(encoding="utf-8")
    assert "Skipped 1 lines" in out1

    res["parse"]["skipped_lines"] = 0
    out2 = write_report(res, src).read_text(encoding="utf-8")
    assert "Skipped" not in out2


def test_tabs_and_insights(tmp_path):
    out = _write(tmp_path).read_text(encoding="utf-8")
    for tab_id in ("overview", "participants", "flow", "words", "sentiment"):
        assert f'id="tab-{tab_id}"' in out, tab_id
    assert "Most messages land on Monday." in out
