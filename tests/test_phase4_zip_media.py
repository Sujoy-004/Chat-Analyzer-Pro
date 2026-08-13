"""ZIP media-file counting tests (D3 / P17).

WhatsApp/Telegram "Export chat" zips hold real media FILES (jpg/mp4/opus/...)
alongside the transcripts; those were previously ignored, so the report's
"Media messages" stat undercounted a zip export. Verifies:

1.  count_zip_media_members counts the actual media files and excludes
    directory entries and .txt/.json transcript members.
2.  End-to-end: a zip with a .txt transcript + media files reports
    media_messages = max("<Media omitted>" markers, media file count) in the
    produced chat_report.html (D3 review: markers and files are the same 1:1
    messages in a real export, so max(), not sum(), avoids double-counting).
"""

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

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
    env["CHAT_ANALYZER_NO_OPEN"] = "1"
    env["CHAT_ANALYZER_FORCE_NLP"] = "0"
    return subprocess.run(
        args,
        input="",  # non-tty stdin: the PH2 tier menu never appears (silent tier 1)
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


def _marker_count() -> int:
    return len(re.findall(r"Media omitted", TXT.read_text(encoding="utf-8"), re.IGNORECASE))


def test_count_zip_media_members(tmp_path):
    """Media files are counted; dirs and .txt/.json transcripts are not."""
    from chat_analyzer.cli.zip_input import count_zip_media_members

    z = _make_zip(
        tmp_path,
        "media.zip",
        {
            "_chat.txt": _txt_content(),
            "chat.json": b"{}",
            "Media/": b"",
            "Media/photo_1.jpg": b"\xff\xd8\xff\xe0",
            "Media/video_2.mp4": b"fake video",
            "Media/voice_3.opus": b"fake audio",
            "Photos/IMG_4.png": b"fake png",
            "Docs/report.pdf": b"%PDF",
        },
    )
    assert count_zip_media_members(z) == 5


@pytest.mark.slow
def test_zip_media_count_in_report(tmp_path):
    """E2E: media_messages = max(transcript markers, media file count).

    The zip carries 4 media files but the transcript has only 2 "<Media
    omitted>" markers, so max() must surface the file count (proving zip
    members are counted) without double-counting in the 1:1 case.
    """
    z = _make_zip(
        tmp_path,
        "chat.zip",
        {
            "_chat.txt": _txt_content(),
            "Media/": b"",
            "Media/photo_1.jpg": b"\xff\xd8\xff\xe0",
            "Media/video_2.mp4": b"fake video",
            "Media/voice_3.opus": b"fake audio",
            "Photos/IMG_4.png": b"fake png",
        },
    )
    res = _run_forced(_cli_cmd(str(z), console=True), cwd=tmp_path)

    assert res.returncode == 0, res.stdout + res.stderr
    report = tmp_path / "chat_report.html"
    assert report.exists(), f"report missing: {res.stdout}"
    html = report.read_text(encoding="utf-8")
    m = re.search(r"<th>Media messages</th><td>(\d+)</td>", html)
    assert m, html
    assert int(m.group(1)) == max(_marker_count(), 4)


def _txt_content() -> bytes:
    return TXT.read_bytes()
