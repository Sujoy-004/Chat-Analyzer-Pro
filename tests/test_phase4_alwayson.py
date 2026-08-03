"""Phase 4 always-on insights tests (D-07 / D-07b) for plan 04-01.

Proves the vertical slice: relationship health (ANAL-07) and network graph
(ANAL-09) ship with the LEAN install — no [nlp] extra, no torch. Two tests:

- Test A (in-process): run_pipeline returns health + network blocks and both
  chart keys with base64 PNG data URIs.
- Test B (subprocess, cwd=tmp_path): the CLI writes the report to the CURRENT
  WORKING DIRECTORY (D-09) and it contains the Relationship Health and Network
  tabs plus the health-grade lead-in sentence (D-11).
"""

import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from chat_analyzer.cli.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = REPO_ROOT / "data" / "sample_chats"


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False)


def _cli_cmd(*args: str) -> list[str]:
    """Build the CLI argv: the installed console script, else `python -m`."""
    exe = shutil.which("chat-analyzer")
    if exe:
        return [exe, *args]
    return [sys.executable, "-m", "chat_analyzer", *args]


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run the CLI, suppressing the auto-open browser (D-09 degrade path)."""
    env = dict(os.environ)
    env["BROWSER"] = "__none__"  # webbrowser.get() raises -> open_report degrades
    return subprocess.run(
        args,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        cwd=cwd,
        env=env,
        timeout=300,
        check=False,  # assertions inspect returncode/output below
    )


def _copy_sample(tmp_path: Path, name: str) -> Path:
    dst = tmp_path / name
    shutil.copyfile(SAMPLES / name, dst)
    return dst


def test_health_network_in_report():
    """Test A: run_pipeline returns health + network blocks and charts."""
    results = run_pipeline(SAMPLES / "whatsapp_sample.txt", _console())

    # Health: score is a float in [0, 1]
    overall_score = results["health"]["overall_score"]
    assert isinstance(overall_score, float)
    assert 0.0 <= overall_score <= 1.0

    # Network: density is a float
    assert isinstance(results["network"]["density"], float)

    # Charts: both new keys exist and carry base64 PNG data URIs
    for key in ("health", "network"):
        uri = results["charts"][key]
        assert uri.startswith("data:image/png;base64,")


def test_report_tabs_and_cwd_location(tmp_path):
    """Test B: report lands in cwd with the two new tabs (D-09/D-11)."""
    dst = _copy_sample(tmp_path, "whatsapp_sample.txt")
    res = _run(_cli_cmd(str(dst)), cwd=tmp_path)

    assert res.returncode == 0, res.stdout + res.stderr
    report = tmp_path / "whatsapp_sample_report.html"
    assert report.exists(), f"report missing at {report} (cwd location, D-09)"
    html = report.read_text(encoding="utf-8")
    assert 'id="tab-health"' in html
    assert 'id="tab-network"' in html
    assert "overall health" in html  # health-grade lead-in sentence
