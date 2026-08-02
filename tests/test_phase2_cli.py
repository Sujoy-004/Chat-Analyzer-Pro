"""Phase 2 CLI end-to-end tests for chat-analyzer.

Subprocess tests against the real installed `chat-analyzer` console script and
the `python -m chat_analyzer` fallback (test_phase1_smoke.py style). Every test
that generates a report first COPIES the sample export into `tmp_path` and runs
the CLI against the copy — the repo tree is never written to (LOW #8).

Maps to 02-PLAN.md Task 9 tests 1-10 (ROADMAP criteria 1-5):
1.  One command end-to-end: positional whatsapp run -> "Parsed 27 messages",
    "Messages: 27" (smoke-contract token, CRITICAL #1) and "Report:".
2.  Stage narration (CLI-03): the three stage lines appear (rich Status renders
    nothing on non-tty; the CLI degrades to plain `[OK] <stage>` lines) and the
    "Messages: 27" token appears BEFORE the "Total messages:" summary panel.
3.  Report lands next to the input copy (D-08/LOW #8); nothing written in repo.
4.  Report card is well-formed: 5 tab ids, >= 4 base64 chart URIs, charset.
5.  Interactive path (D-01): piped path -> exit 0, "Messages: 27".
6.  --version (D-03): "chat-analyzer <semver>", exit 0.
7.  Unsupported extension re-prompt (D-06/MEDIUM #4) + positional error paths:
    no crash, no traceback, friendly messages, exit 1 for malformed file.
8.  Telegram round-trip (D-19/D-20): "Parsed 5 messages", "Messages: 5".
9.  Skip surfacing (D-15/D-16): skipped line counted in terminal + report skip
    note; no fabricated (today) timestamp in the report stats.
10. No console pollution (Pitfall 5): no emoji / no "Initializing Sentiment".
"""

import os
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = REPO_ROOT / "data" / "sample_chats"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _cli_cmd(*args: str, console: bool = True) -> list[str]:
    """Build the CLI argv: the installed console script, else `python -m`."""
    exe = shutil.which("chat-analyzer")
    if console and exe:
        return [exe, *args]
    return [sys.executable, "-m", "chat_analyzer", *args]


def _run(
    args: list[str], stdin_text: str | None = None, cwd: Path | None = None
) -> subprocess.CompletedProcess:
    """Run the CLI, suppressing the auto-open browser (D-09 degrade path)."""
    env = dict(os.environ)
    env["BROWSER"] = "__none__"  # webbrowser.get() raises -> open_report degrades
    return subprocess.run(
        args,
        input=stdin_text,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        cwd=cwd,
        env=env,
        timeout=300,
        check=False,  # assertions inspect returncode/stderr below
    )


def _copy_sample(tmp_path: Path, name: str) -> Path:
    dst = tmp_path / name
    shutil.copyfile(SAMPLES / name, dst)
    return dst


def test_positional_whatsapp_roundtrip(tmp_path):
    """Test 1 (ROADMAP crit 1): one command runs the full pipeline e2e."""
    dst = _copy_sample(tmp_path, "whatsapp_sample.txt")
    res = _run(_cli_cmd(str(dst), console=True))

    assert res.returncode == 0, res.stdout + res.stderr
    out = res.stdout + res.stderr
    assert "Parsed 27 messages" in out
    assert "Messages: 27" in out  # smoke-contract token, CRITICAL #1
    assert "Report:" in out


def test_no_console_pollution(tmp_path):
    """Test 10 (Pitfall 5): analysis-stage prints are captured (T-02-05);
    the VADER pin keeps the transformers 'Initializing Sentiment' warning
    off stderr."""
    dst = _copy_sample(tmp_path, "whatsapp_sample.txt")
    res = _run(_cli_cmd(str(dst), console=True))

    assert res.returncode == 0, res.stdout + res.stderr
    out = res.stdout + res.stderr
    assert "🚀" not in out
    assert "Initializing Sentiment" not in out


def test_stage_narration_and_order(tmp_path):
    """Test 2 (ROADMAP crit 4): stage lines present; token before panel."""
    dst = _copy_sample(tmp_path, "whatsapp_sample.txt")
    res = _run(_cli_cmd(str(dst), console=True))

    assert res.returncode == 0, res.stdout + res.stderr
    out = res.stdout + res.stderr
    for stage in ("Parsing chat", "Computing insights", "Writing report"):
        assert stage in out, f"missing stage narration: {stage}"
    assert "Messages: 27" in out
    assert "Total messages: 27" in out
    assert out.index("Messages: 27") < out.index("Total messages: 27")


def test_report_written_next_to_input(tmp_path):
    """Test 3 (D-08/LOW #8): report next to the tmp copy; repo untouched."""
    dst = _copy_sample(tmp_path, "whatsapp_sample.txt")
    res = _run(_cli_cmd(str(dst), console=True))

    assert res.returncode == 0, res.stdout + res.stderr
    report = tmp_path / "whatsapp_sample_report.html"
    assert report.exists(), f"report missing at {report}"
    assert not list(REPO_ROOT.rglob("*_report.html")), "CLI wrote into the repo"


def test_report_card_wellformed(tmp_path):
    """Test 4 (ROADMAP crit 2+3): 5 tabs, >= 4 charts, utf-8 declaration."""
    dst = _copy_sample(tmp_path, "whatsapp_sample.txt")
    res = _run(_cli_cmd(str(dst), console=True))

    assert res.returncode == 0, res.stdout + res.stderr
    html = (tmp_path / "whatsapp_sample_report.html").read_text(encoding="utf-8")
    for tab in ("overview", "participants", "flow", "words", "sentiment"):
        assert f'id="tab-{tab}"' in html, f"missing tab: {tab}"
    assert html.count("data:image/png;base64,") >= 4, "fewer than 4 charts"
    assert '<meta charset="utf-8">' in html


def test_interactive_path(tmp_path):
    """Test 5 (D-01): no-arg interactive prompt analyzes a piped path."""
    dst = _copy_sample(tmp_path, "whatsapp_sample.txt")
    res = _run(_cli_cmd(console=False), stdin_text=f"{dst}\n")

    assert res.returncode == 0, res.stdout + res.stderr
    assert "Messages: 27" in res.stdout + res.stderr


def test_version(tmp_path):
    """Test 6 (D-03): --version prints the installed semver and exits 0."""
    res = _run(_cli_cmd("--version", console=True))

    assert res.returncode == 0, res.stdout + res.stderr
    assert re.search(r"chat-analyzer \d+\.\d+\.\d+", res.stdout), res.stdout


def test_unsupported_and_error_paths(tmp_path):
    """Test 7 (D-06/MEDIUM #4): re-prompt + positional error paths, no traceback."""
    bad = tmp_path / "chat.pdf"
    bad.write_text("", encoding="utf-8")

    # Interactive: bad extension re-prompts, missing file re-prompts, EOF ends.
    res = _run(
        _cli_cmd(console=False),
        stdin_text="chat.pdf\nnonexistent.txt\n",
        cwd=tmp_path,
    )
    out = res.stdout + res.stderr
    assert "expected a WhatsApp .txt or Telegram .json" in out
    assert "Traceback" not in out
    assert res.returncode in (0, 1)

    # Positional: existing file with an unsupported extension -> exit 1.
    res2 = _run(_cli_cmd(str(bad), console=True))
    assert res2.returncode == 1, res2.stdout + res2.stderr
    assert "expected a WhatsApp .txt or Telegram .json" in res2.stdout + res2.stderr
    assert "Traceback" not in res2.stdout + res2.stderr

    # Positional: every line fails to parse -> friendly ValueError, exit 1.
    unparseable = tmp_path / "all_unparseable.txt"
    shutil.copyfile(FIXTURES / "whatsapp_all_skipped.txt", unparseable)
    res3 = _run(_cli_cmd(str(unparseable), console=True))
    assert res3.returncode == 1, res3.stdout + res3.stderr
    assert "No messages could be parsed" in res3.stdout + res3.stderr
    assert "Traceback" not in res3.stdout + res3.stderr


def test_telegram_roundtrip(tmp_path):
    """Test 8 (D-19/D-20): telegram export parses end-to-end."""
    dst = _copy_sample(tmp_path, "telegram_sample.json")
    res = _run(_cli_cmd(str(dst), console=False))

    assert res.returncode == 0, res.stdout + res.stderr
    out = res.stdout + res.stderr
    assert "Parsed 5 messages" in out
    assert "Messages: 5" in out


def test_skip_surfacing(tmp_path):
    """Test 9 (D-15/D-16): skipped lines counted + report skip note; no today."""
    src = tmp_path / "mixed.txt"
    src.write_text(
        "1/15/2024, 10:00:00 AM - Alice: a perfectly valid message\n"
        "13/40/2024, 10:00:00 AM - Bob: this date cannot exist\n",
        encoding="utf-8",
    )
    res = _run(_cli_cmd(str(src), console=False))

    assert res.returncode == 0, res.stdout + res.stderr
    assert "Skipped 1 lines that couldn't be parsed" in res.stdout + res.stderr

    report = tmp_path / "mixed_report.html"
    assert report.exists()
    html = report.read_text(encoding="utf-8")
    assert "Skipped 1 lines that couldn't be parsed" in html
    # No fabricated timestamps: the only parsed date is 2024-01-15, so the
    # report stats must not contain today's date (T-02-04).
    assert (
        date.today().strftime("%Y-%m-%d") not in html  # noqa: DTZ011 - deliberate: asserting no today-timestamp
    )
