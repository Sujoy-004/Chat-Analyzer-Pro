"""Phase 4 CLI UX tests: hint line, tty download menu, friendly-error taxonomy.

Maps to 04-03-PLAN.md tasks 1-3 (D-04/D-05/D-06, D-13/D-14/D-15, CLI-04):

1.  Positional hint (D-06): NLP forced off -> exit 0, exactly one hint line,
    no menu text, "Messages: 27" smoke token still present.
2.  Piped no-arg hint (D-06): piped path (not a tty) -> hint line, no menu.
3.  Missing file (D-13): positional nonexistent.txt -> exit 1, "File not
    found" + inline export instructions, no traceback.
4.  Wrong format (D-13): positional chat.pdf -> exit 1, "Unsupported file
    type" + export instructions, no traceback.
5.  Empty/unparseable (D-13): positional all-skipped fixture -> exit 1,
    "No messages could be parsed" + export instructions, no traceback.
6.  Interactive re-prompt (D-15): piped bad suffix then valid path -> no
    exit 1, "Messages: 27", exit 0.
7.  Menu on tty (D-04): in-process unit test of `_nlp_menu` -- all three
    options render and the patched choice is returned.

CHAT_ANALYZER_FORCE_NLP=0 forces the basic path deterministically (RESEARCH
Pitfall 5: the dev machine has transformers but no cached emotion model, so
the raw probe could vary by machine; the env hook removes the flake).
"""

import os
import shutil
import subprocess
import sys
import unittest.mock
from io import StringIO
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


def _run_forced(
    args: list[str], stdin_text: str | None = None, cwd: Path | None = None
) -> subprocess.CompletedProcess:
    """Run the CLI with NLP forced OFF and the auto-open browser suppressed."""
    env = dict(os.environ)
    env["BROWSER"] = "__none__"  # webbrowser.get() raises -> open_report degrades
    env["CHAT_ANALYZER_FORCE_NLP"] = "0"  # deterministic basic path (Pitfall 5)
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
        check=False,  # assertions inspect returncode/stdout below
    )


def _copy_sample(tmp_path: Path, name: str) -> Path:
    dst = tmp_path / name
    shutil.copyfile(SAMPLES / name, dst)
    return dst


def test_positional_hint_line(tmp_path):
    """Test 1 (D-06): positional run with NLP missing hints once, never prompts."""
    dst = _copy_sample(tmp_path, "whatsapp_sample.txt")
    res = _run_forced(_cli_cmd(str(dst), console=True), cwd=tmp_path)

    assert res.returncode == 0, res.stdout + res.stderr
    out = res.stdout + res.stderr
    assert "pip install chat-analyzer-pro[nlp]" in out  # the single hint line
    assert "torch (~3GB)" not in out  # no menu in positional output
    assert "Messages: 27" in out  # smoke-contract token preserved
    assert out.count("pip install chat-analyzer-pro[nlp]") == 1  # exactly once


def test_piped_noarg_hint(tmp_path):
    """Test 2 (D-06): piped no-arg run hints; menu suppressed (not a tty)."""
    dst = _copy_sample(tmp_path, "whatsapp_sample.txt")
    res = _run_forced(_cli_cmd(console=False), stdin_text=f"{dst}\n", cwd=tmp_path)

    assert res.returncode == 0, res.stdout + res.stderr
    out = res.stdout + res.stderr
    assert "pip install chat-analyzer-pro[nlp]" in out  # the single hint line
    assert "torch (~3GB)" not in out  # piped stdin is not a tty -> no menu


def test_positional_missing_file(tmp_path):
    """Test 3 (D-13): nonexistent file -> exit 1, friendly msg + export steps."""
    missing = tmp_path / "nonexistent.txt"
    res = _run_forced(_cli_cmd(str(missing), console=True), cwd=tmp_path)

    assert res.returncode == 1, res.stdout + res.stderr
    out = res.stdout + res.stderr
    assert "File not found" in out
    assert "Export chat" in out  # inline WhatsApp/Telegram export instructions
    assert "Traceback" not in out


def test_positional_wrong_format(tmp_path):
    """Test 4 (D-13): unsupported suffix -> exit 1, friendly msg + export steps."""
    bad = tmp_path / "chat.pdf"
    bad.write_text("", encoding="utf-8")
    res = _run_forced(_cli_cmd(str(bad), console=True), cwd=tmp_path)

    assert res.returncode == 1, res.stdout + res.stderr
    out = res.stdout + res.stderr
    assert "Unsupported file type" in out
    assert "Export chat" in out  # inline export instructions present
    assert "Traceback" not in out


def test_positional_empty_chat(tmp_path):
    """Test 5 (D-13): unparseable export -> exit 1, friendly msg + export steps."""
    unparseable = tmp_path / "all_skipped.txt"
    shutil.copyfile(FIXTURES / "whatsapp_all_skipped.txt", unparseable)
    res = _run_forced(_cli_cmd(str(unparseable), console=True), cwd=tmp_path)

    assert res.returncode == 1, res.stdout + res.stderr
    out = res.stdout + res.stderr
    assert "No messages could be parsed" in out
    assert "Export chat" in out  # inline export instructions present
    assert "Traceback" not in out


def test_interactive_reprompts_on_bad_file(tmp_path):
    """Test 6 (D-15): bad suffix re-prompts; a valid path then analyzes."""
    bad = tmp_path / "chat.pdf"
    bad.write_text("", encoding="utf-8")
    dst = _copy_sample(tmp_path, "whatsapp_sample.txt")
    res = _run_forced(
        _cli_cmd(console=False), stdin_text=f"{bad}\n{dst}\n", cwd=tmp_path
    )

    assert res.returncode == 0, res.stdout + res.stderr
    assert "Messages: 27" in res.stdout + res.stderr


def test_menu_renders_three_options_on_tty():
    """Test 7 (D-04): the 3-option menu renders on a tty when NLP is missing.

    Subprocess tests cannot fake a tty without a pty, so this is an in-process
    unit test of the module-level `_nlp_menu` function: the availability probe
    is forced off, stdin claims to be a tty, the menu's prompt answers "3",
    and the rendered console output shows all three options.
    """
    from rich.console import Console

    import chat_analyzer.cli.main as cli_main
    import chat_analyzer.cli.nlp_gate as nlp_gate

    out = StringIO()
    console = Console(file=out, width=120)

    with (
        unittest.mock.patch.object(cli_main.sys.stdin, "isatty", return_value=True),
        unittest.mock.patch.object(nlp_gate, "nlp_available", return_value=False),
        unittest.mock.patch.object(cli_main.typer, "prompt", return_value="3"),
    ):
        choice = cli_main._nlp_menu(console)

    rendered = out.getvalue()
    assert "torch (~3GB)" in rendered
    assert "CPU-only torch" in rendered
    assert "No download" in rendered
    assert choice == "3"
