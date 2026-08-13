"""Phase 4 CLI UX tests: hint line, tty tier menu, friendly-error taxonomy.

Maps to 04-03-PLAN.md tasks 1-3 and the PH2 always-on tier menu (D-04/D-05/D-06,
D-13/D-14/D-15, CLI-04):

1.  Positional hint (D-06): non-tty tier 1 (silent) -> exit 0, exactly one
    hint line, no menu text, "Messages: 27" smoke token still present.
2.  Piped no-arg hint (D-06): piped path (not a tty) -> hint line, no menu.
3.  Missing file (D-13): positional nonexistent.txt -> exit 1, "File not
    found" + inline export instructions, no traceback.
4.  Wrong format (D-13): positional chat.pdf -> exit 1, "Unsupported file
    type" + export instructions, no traceback.
5.  Empty/unparseable (D-13): positional all-skipped fixture -> exit 1,
    "No messages could be parsed" + export instructions, no traceback.
6.  Interactive re-prompt (D-15): piped bad suffix then valid path -> no
    exit 1, "Messages: 27", exit 0.
7.  Tier menu on tty (PH2): in-process unit test of `_tier_menu` -- all three
    options render in the new wording, default is 1, and the patched choice
    is returned.

All subprocess runs force a non-tty stdin (input="") so the PH2 tier
resolution takes the silent tier-1 path deterministically (RESEARCH Pitfall 5:
a real tty would block on the menu prompt). CHAT_ANALYZER_FORCE_NLP=0 is kept
as the legacy override and maps to tier 1.
"""

import os
import shutil
import subprocess
import sys
import unittest.mock
from io import StringIO
from pathlib import Path

import pytest

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
    """Run the CLI with NLP forced OFF and the auto-open browser suppressed.

    stdin is always a pipe (input="") so the subprocess is non-interactive:
    the PH2 tier menu never appears and the silent tier-1 hint path is taken.
    """
    env = dict(os.environ)
    env["BROWSER"] = "__none__"  # webbrowser.get() raises -> open_report degrades
    env["CHAT_ANALYZER_NO_OPEN"] = "1"  # never pop a browser (B3 opt-out)
    env["CHAT_ANALYZER_FORCE_NLP"] = "0"  # deterministic basic path (Pitfall 5)
    return subprocess.run(
        args,
        input=stdin_text if stdin_text is not None else "",
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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
    """Test 7 (PH2): the 3-option tier menu renders with the new wording.

    Subprocess tests cannot fake a tty without a pty, so this is an in-process
    unit test of the module-level `_tier_menu`: the menu's prompt answers "2",
    and the rendered console output shows the three PH2 tier options with the
    new wording (default choice is 1, checked via the prompt default).
    """
    from rich.console import Console

    import chat_analyzer.cli.main as cli_main

    out = StringIO()
    console = Console(file=out, width=120)

    with unittest.mock.patch.object(cli_main.typer, "prompt", return_value="2"):
        choice = cli_main._tier_menu(console)

    rendered = out.getvalue()
    assert "NLP tier:" in rendered
    assert "1) Without NLP" in rendered
    assert "2) Minimal (~0.6 GB)" in rendered
    assert "3) Full-fledged (~3 GB)" in rendered
    assert choice == "2"


def test_tier_menu_default_is_one(monkeypatch):
    """PH2: the tier menu defaults to 1 (Without NLP) — typer.prompt default."""
    from rich.console import Console

    import chat_analyzer.cli.main as cli_main

    out = StringIO()
    console = Console(file=out, width=120)
    captured: dict = {}

    def _fake_prompt(text, default=None, **kwargs):
        captured["default"] = default
        return default

    with monkeypatch.context() as m:
        m.setattr(cli_main.typer, "prompt", _fake_prompt)
        choice = cli_main._tier_menu(console)

    assert captured["default"] == "1"
    assert choice == "1"


def test_tier_from_env_override(monkeypatch):
    """PH2: non-tty tier resolution honors CHAT_ANALYZER_TIER and the legacy
    CHAT_ANALYZER_FORCE_NLP mapping, defaulting to silent tier 1."""
    import chat_analyzer.cli.main as cli_main

    monkeypatch.delenv("CHAT_ANALYZER_TIER", raising=False)
    monkeypatch.delenv("CHAT_ANALYZER_FORCE_NLP", raising=False)
    assert cli_main._tier_from_env() == "1"  # default silent tier 1

    monkeypatch.setenv("CHAT_ANALYZER_TIER", "2")
    assert cli_main._tier_from_env() == "2"
    monkeypatch.setenv("CHAT_ANALYZER_TIER", "3")
    assert cli_main._tier_from_env() == "3"
    monkeypatch.setenv("CHAT_ANALYZER_TIER", "1")
    assert cli_main._tier_from_env() == "1"  # accepted for explicitness

    monkeypatch.delenv("CHAT_ANALYZER_TIER", raising=False)
    monkeypatch.setenv("CHAT_ANALYZER_FORCE_NLP", "0")
    assert cli_main._tier_from_env() == "1"  # legacy 0 -> tier 1
    monkeypatch.setenv("CHAT_ANALYZER_FORCE_NLP", "1")
    assert cli_main._tier_from_env() == "3"  # legacy 1 -> tier 3


def test_ensure_tier_ready_proceeds(monkeypatch):
    """PH2: a READY nlp_status() proceeds without installing."""
    from rich.console import Console

    import chat_analyzer.cli.main as cli_main
    from chat_analyzer.cli import nlp_gate

    out = StringIO()
    console = Console(file=out, width=120)
    monkeypatch.setattr(nlp_gate, "nlp_status", lambda: ("READY", {}))
    with (
        unittest.mock.patch.object(nlp_gate, "install_nlp") as install,
        unittest.mock.patch.object(nlp_gate, "download_models") as dl,
    ):
        ok = cli_main._ensure_nlp_for_tier("2", console)

    assert ok is True
    install.assert_not_called()
    dl.assert_not_called()
    assert "NLP ready" in out.getvalue()


def test_ensure_tier_missing_installs_and_downloads(monkeypatch):
    """PH2: a MISSING status installs the tier flavor then downloads models."""
    from rich.console import Console

    import chat_analyzer.cli.main as cli_main
    from chat_analyzer.cli import nlp_gate

    out = StringIO()
    console = Console(file=out, width=120)
    monkeypatch.setattr(nlp_gate, "nlp_status", lambda: ("MISSING", {}))
    calls: list[str] = []

    def _fake_install(cpu_only: bool):
        calls.append(f"install cpu_only={cpu_only}")

    def _fake_download():
        calls.append("download")

    monkeypatch.setattr(nlp_gate, "install_nlp", _fake_install)
    monkeypatch.setattr(nlp_gate, "download_models", _fake_download)

    ok = cli_main._ensure_nlp_for_tier("2", console)
    assert ok is True
    assert calls == ["install cpu_only=True", "download"]

    ok = cli_main._ensure_nlp_for_tier("3", console)
    assert calls == [
        "install cpu_only=True",
        "download",
        "install cpu_only=False",
        "download",
    ]


def test_ensure_tier_outdated_prompts_tty(monkeypatch):
    """PH2: OUTDATED on a tty prompts update-or-keep; 'n' keeps current."""
    from rich.console import Console

    import chat_analyzer.cli.main as cli_main
    from chat_analyzer.cli import nlp_gate

    out = StringIO()
    console = Console(file=out, width=120)
    monkeypatch.setattr(nlp_gate, "nlp_status", lambda: ("OUTDATED", {}))
    monkeypatch.setattr(cli_main.sys.stdin, "isatty", lambda: True)

    with unittest.mock.patch.object(cli_main.typer, "prompt", return_value="n"):
        ok = cli_main._ensure_nlp_for_tier("2", console)
    assert ok is True
    assert "continuing with current install" not in out.getvalue()
    assert "NLP ready" not in out.getvalue()


def test_ensure_tier_outdated_non_tty_keeps_current(monkeypatch):
    """PH2: OUTDATED on non-tty cannot prompt — proceeds with current install."""
    from rich.console import Console

    import chat_analyzer.cli.main as cli_main
    from chat_analyzer.cli import nlp_gate

    out = StringIO()
    console = Console(file=out, width=120)
    monkeypatch.setattr(nlp_gate, "nlp_status", lambda: ("OUTDATED", {}))
    monkeypatch.setattr(cli_main.sys.stdin, "isatty", lambda: False)

    with (
        unittest.mock.patch.object(nlp_gate, "update_nlp") as upd,
        unittest.mock.patch.object(nlp_gate, "download_models") as dl,
    ):
        ok = cli_main._ensure_nlp_for_tier("2", console)

    assert ok is True
    upd.assert_not_called()
    dl.assert_not_called()
    assert "continuing with current install" in out.getvalue()
