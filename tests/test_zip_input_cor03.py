"""COR-03 regression: chosen_names lists ONLY successfully parsed transcripts.

Before the fix, parse_zip_with_report built chosen_names from ALL chosen
members (``chosen_names = sorted(member for member, _ in chosen)``) even when a
member failed to parse and the ``except ... continue`` skipped it. chosen_names
rides in the result-cache key (04-08), so a failed member was misreported as
analyzed and could hash differently on a later run. After the fix, only
transcripts that parse successfully end up in chosen_names.
"""

import zipfile
from io import StringIO
from pathlib import Path

from rich.console import Console

from chat_analyzer.cli.zip_input import parse_zip_with_report

VALID_CHAT = (
    "12/25/23, 9:30 AM - Alice: Hello!\n"
    "12/25/23, 9:31 AM - Bob: Hi there!\n"
)


def _corrupt_member_data(zip_path: Path, member: str) -> None:
    """Zero one zip member's deflate stream so ZipFile.read() raises on it."""
    with zipfile.ZipFile(zip_path) as zf:
        info = zf.getinfo(member)
        data_start = info.header_offset + 30 + len(info.filename.encode("utf-8")) + len(info.extra)
    with open(zip_path, "r+b") as f:
        f.seek(data_start)
        f.write(b"\x00" * info.compress_size)


def _make_zip(tmp_path: Path, bad_member: str | None = None) -> Path:
    members = ["good.txt", "b.txt"] if bad_member is None else ["good.txt", bad_member]
    zpath = tmp_path / "chat.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for member in members:
            zf.writestr(member, VALID_CHAT)
    if bad_member:
        _corrupt_member_data(zpath, bad_member)
    return zpath


def test_cor03_failed_member_excluded_from_chosen_names(tmp_path):
    """A transcript that fails to parse is absent from chosen_names."""
    z = _make_zip(tmp_path, bad_member="unparseable.txt")
    out = StringIO()

    rows, counts, source, chosen_names = parse_zip_with_report(z, Console(file=out))

    assert chosen_names == ["good.txt"]
    assert "unparseable.txt" not in chosen_names
    assert len(rows) == 2
    assert {r["sender"] for r in rows} == {"Alice", "Bob"}
    assert counts["parsed_messages"] > 0
    assert source == "whatsapp"
    assert "[WARN] Skipped transcript: unparseable.txt" in out.getvalue()


def test_cor03_all_valid_keeps_every_chosen_member(tmp_path):
    """Control: an all-valid zip still lists every chosen member (sorted)."""
    z = _make_zip(tmp_path)

    _rows, counts, source, chosen_names = parse_zip_with_report(z, Console(file=StringIO()))

    assert chosen_names == ["b.txt", "good.txt"]
    assert counts["parsed_messages"] == 4
    assert source == "whatsapp"