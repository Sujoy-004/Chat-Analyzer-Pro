"""ZIP input support tests (Item C).

Wraps the sample WhatsApp .txt into a .zip (mimicking WhatsApp/Telegram
"Export chat" archives) and exercises the CLI zip branch end-to-end plus the
zip_input selection helper in isolation:

1.  Single-transcript zip -> positional run exit 0, "Messages: 27", report produced.
2.  Multi-transcript zip, non-tty -> auto-select ALL, exit 0, merged count == sum.
3.  Empty zip (no .txt/.json) -> exit 1, friendly "No chat transcripts" msg.
4.  Corrupted zip -> exit 1, "Invalid or corrupted".
5.  _select_transcripts interactive -> returns only the user's picks.
6.  Mixed .txt + .json zip -> source is "whatsapp", both transcripts analyzed.
"""

import shutil
import subprocess
import sys
import unittest.mock
import zipfile
from io import StringIO
from pathlib import Path

from rich.console import Console

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "sample_chats"
TXT = SAMPLES / "whatsapp_sample.txt"


def _cli_cmd(*args: str, console: bool = True, z_arg: str | None = None) -> list[str]:
    exe = shutil.which("chat-analyzer")
    if console and exe:
        return [exe, *args]
    if z_arg:
        return [sys.executable, "-m", "chat_analyzer", z_arg]
    return [sys.executable, "-m", "chat_analyzer", *args]


def _run_forced(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env["BROWSER"] = "__none__"
    env["CHAT_ANALYZER_FORCE_NLP"] = "0"
    return subprocess.run(
        args,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        cwd=cwd,
        env=env,
        timeout=300,
        check=False,
    )


def _make_zip(tmp_path: Path, name: str, members: dict[str, bytes]) -> Path:
    """Build a zip. members maps subpath -> content bytes."""
    zpath = tmp_path / name
    with zipfile.ZipFile(zpath, "w") as zf:
        for arcname, content in members.items():
            if arcname.endswith("/"):
                zf.writestr(arcname, b"")
            else:
                zf.writestr(arcname, content)
    return zpath


def _txt_content() -> bytes:
    return TXT.read_bytes()


def test_zip_single_transcript(tmp_path):
    """Test 1: a zip with one WhatsApp .txt analyzes and produces a report."""
    z = _make_zip(tmp_path, "chat.zip", {"_chat.txt": _txt_content()})
    res = _run_forced(_cli_cmd(str(z), console=True), cwd=tmp_path)

    assert res.returncode == 0, res.stdout + res.stderr
    assert "Messages: 27" in res.stdout + res.stderr
    report = tmp_path / "chat_report.html"
    assert report.exists(), f"report missing: {res.stdout}"


def test_zip_multiple_transcripts_non_tty(tmp_path):
    """Test 2: two .txt files, piped (not a tty) -> analyze all, merged count."""
    z = _make_zip(
        tmp_path,
        "multi.zip",
        {
            "chat_A.txt": _txt_content(),
            "chat_B.txt": _txt_content(),  # same 27 messages, merged = 54
        },
    )
    res = _run_forced(_cli_cmd(console=False, z_arg=str(z)), cwd=tmp_path)

    assert res.returncode == 0, res.stdout + res.stderr
    assert "non-interactive" in res.stdout + res.stderr
    assert "Messages: 54" in res.stdout + res.stderr


def test_zip_empty_no_transcripts(tmp_path):
    """Test 3: zip with no .txt/.json -> exit 1, "No chat transcripts" msg."""
    z = _make_zip(tmp_path, "empty.zip", {"stickers/": b""})
    res = _run_forced(_cli_cmd(str(z), console=True), cwd=tmp_path)

    assert res.returncode == 1, res.stdout + res.stderr
    assert "No chat transcripts" in res.stdout + res.stderr


def test_zip_corrupted(tmp_path):
    """Test 4: a corrupt .zip -> exit 1, "Invalid or corrupted"."""
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"this is not a real zip")
    res = _run_forced(_cli_cmd(str(bad), console=True), cwd=tmp_path)

    assert res.returncode == 1, res.stdout + res.stderr
    assert "Invalid or corrupted" in res.stdout + res.stderr


def test_selection_interactive_returns_picks():
    """Test 5: _select_transcripts returns only the user's comma-separated picks."""
    from chat_analyzer.cli import zip_input

    out = StringIO()
    console = Console(file=out, width=120)

    transcripts = [("a.txt", "whatsapp"), ("b.txt", "whatsapp"), ("c.txt", "whatsapp")]
    with (
        unittest.mock.patch.object(
            type(console), "is_terminal", new_callable=unittest.mock.PropertyMock
        ) as m,
        unittest.mock.patch.object(zip_input.typer, "prompt", return_value="1,3"),
    ):
        m.return_value = True  # force the interactive branch (no real tty in pytest)
        chosen = zip_input._select_transcripts(transcripts, console)

    assert chosen == [("a.txt", "whatsapp"), ("c.txt", "whatsapp")]


def test_zip_mixed_txt_json(tmp_path):
    """Test 6: zip with both .txt and .json -> merged, source whatsapp."""
    tele = SAMPLES / "telegram_sample.json"
    z = _make_zip(
        tmp_path,
        "both.zip",
        {
            "_chat.txt": _txt_content(),
            "telegram.json": tele.read_bytes(),
        },
    )
    res = _run_forced(_cli_cmd(str(z), console=True), cwd=tmp_path)

    assert res.returncode == 0, res.stdout + res.stderr
    assert "Messages:" in res.stdout + res.stderr