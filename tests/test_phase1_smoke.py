"""Phase 1 smoke tests for the chat-analyzer CLI and analysis core.

Covers, in order:
1. D-01 console script answers --help instantly
2. D-02 python -m fallback answers --help
3. Interactive prompt happy path (piped valid export -> message count)
4. Re-prompt on invalid path (no crash, second attempt succeeds)
5. Friendly exit code 1 + no traceback for unprocessable input
6. QUAL-01: all chat_analyzer.* modules import after install
7. QUAL-04: no web-app tokens (exec(code / unsafe_allow_html / streamlit / plotly)
8. ROADMAP criterion 4: the analysis core PRODUCES results, not just imports
9. PKG-02/03: lean base install — heavy deps structurally confined to [nlp]
10. D-10/D-11: reporting modules importable but NOT wired into the CLI
"""

import importlib.util
import re
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_WHATSAPP = "data/sample_chats/whatsapp_sample.txt"
KNOWN_WHATSAPP_COUNT = 27

ALL_MODULES = [
    "chat_analyzer",
    "chat_analyzer.parser",
    "chat_analyzer.parser.whatsapp_parser",
    "chat_analyzer.parser.telegram_parser",
    "chat_analyzer.ingest",
    "chat_analyzer.ingest.ingestion",
    "chat_analyzer.analysis",
    "chat_analyzer.analysis.eda",
    "chat_analyzer.analysis.sentiment",
    "chat_analyzer.analysis.emotion",
    "chat_analyzer.analysis.network_graph",
    "chat_analyzer.analysis.relationship_health",
    "chat_analyzer.analysis.summarizer",
    "chat_analyzer.utils",
    "chat_analyzer.utils.preprocessing",
    "chat_analyzer.utils.visualization",
    "chat_analyzer.reporting",
    "chat_analyzer.reporting.pdf_report",
    "chat_analyzer.reporting.weekly_digest",
    "chat_analyzer.cli",
]

FORBIDDEN_TOKENS = [
    "exec(code",
    "unsafe_allow_html",
    "import streamlit",
    "import plotly",
]


def run_cli(args, stdin_text):
    """Run the chat-analyzer console script (D-01) with piped stdin."""
    return subprocess.run(
        ["chat-analyzer", *args],
        input=stdin_text,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )


def run_python_m(args, stdin_text=""):
    """Run python -m chat_analyzer (D-02) with piped stdin."""
    return subprocess.run(
        [sys.executable, "-m", "chat_analyzer", *args],
        input=stdin_text,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )


def message_count(output):
    """Extract the processed message count from CLI output, or None."""
    match = re.search(r"Messages:\s*(\d+)", output)
    return int(match.group(1)) if match else None


def test_console_script_help():
    """chat-analyzer --help exits 0 and identifies the tool."""
    result = run_cli(["--help"], "")
    assert result.returncode == 0
    assert "chat-analyzer" in result.stdout


def test_python_m_help():
    """python -m chat_analyzer --help exits 0 (D-02 fallback)."""
    result = run_python_m(["--help"])
    assert result.returncode == 0


def test_prompt_happy_path():
    """Piped valid export is processed and reports a message count."""
    result = run_cli([], f"{SAMPLE_WHATSAPP}\n")
    assert result.returncode == 0
    count = message_count(result.stdout)
    assert count is not None and count > 0


def test_invalid_path_reprompts():
    """First invalid path re-prompts; the valid second path gets processed."""
    result = run_cli([], f"nonexistent_export.txt\n{SAMPLE_WHATSAPP}\n")
    assert result.returncode == 0
    assert message_count(result.stdout) == KNOWN_WHATSAPP_COUNT


def test_unprocessable_input_exits_one():
    """A directory path exits 1 with a friendly message, never a traceback."""
    result = run_cli([], "src\n")
    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "File not found" in output or "Could not process" in output
    assert "Traceback" not in output


def test_import_matrix():
    """Every chat_analyzer module imports cleanly in a subprocess (QUAL-01)."""
    imports = ", ".join(ALL_MODULES)
    # -X utf8: legacy sentiment.py prints an emoji fallback warning at module
    # load when textblob is absent; a bare python -c has no encoding bootstrap
    # (research Pitfall 5) and would crash on cp1252 consoles. UTF-8 mode is
    # the same mitigation family as the CLI's T-01-07 stdout reconfigure.
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", f"import {imports}; print('IMPORT-MATRIX-OK')"],
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "IMPORT-MATRIX-OK" in result.stdout


def test_no_web_app_tokens_in_package():
    """Installed package tree contains no web-app-only patterns (QUAL-04)."""
    spec = importlib.util.find_spec("chat_analyzer")
    assert spec is not None
    package_root = Path(spec.origin).parent
    offenders = []
    for py_file in package_root.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8", errors="replace")
        for token in FORBIDDEN_TOKENS:
            if token in source:
                offenders.append(f"{py_file}: {token}")
    assert not offenders, f"Forbidden tokens found: {offenders}"


def test_analysis_core_produces_results():
    """The analysis core runs on real data, not just imports."""
    from chat_analyzer.analysis.eda import ChatEDA
    from chat_analyzer.analysis.relationship_health import analyze_relationship_health
    from chat_analyzer.analysis.sentiment import add_sentiment_analysis

    df = pd.DataFrame(
        {
            "datetime": [
                pd.Timestamp("2024-01-01 09:00:00"),
                pd.Timestamp("2024-01-01 10:00:00"),
                pd.Timestamp("2024-01-01 11:00:00"),
                pd.Timestamp("2024-01-01 12:00:00"),
            ],
            "date": ["2024-01-01"] * 4,
            "hour": [9, 10, 11, 12],
            "sender": ["Alice", "Bob", "Alice", "Bob"],
            "message": ["hello!", "hi there", "how are you", "good thanks"],
        }
    )

    df_sentiment = add_sentiment_analysis(df)
    assert "vader_sentiment" in df_sentiment.columns
    assert "vader_compound" in df_sentiment.columns

    eda = ChatEDA(df)
    volume = eda.analyze_message_volume()
    assert volume
    assert "sender_counts" in volume
    assert volume["sender_counts"].sum() > 0

    health = analyze_relationship_health(df)
    assert health
    assert "health_score" in health
    assert "conversation_stats" in health
    assert health["conversation_stats"]["total_messages"] == 4


def _dependency_names(entries):
    """Strip version specifiers from pyproject dependency entries."""
    names = []
    for entry in entries:
        match = re.match(r"^[A-Za-z0-9._-]+", entry)
        if match:
            names.append(match.group(0))
    return names


def test_lean_base_structural():
    """Heavy deps confined to [nlp]; base install stays lean (PKG-02/03)."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)

    project = pyproject["project"]
    assert project["requires-python"] == ">=3.11"

    base_names = _dependency_names(project["dependencies"])
    for heavy in ("torch", "transformers", "streamlit", "plotly"):
        assert heavy not in base_names, f"{heavy} must not be a base dependency"

    nlp_names = _dependency_names(project["optional-dependencies"]["nlp"])
    assert nlp_names == ["torch", "transformers"]

    # Environment half: confirm torch is absent from the current base env.
    # Non-fatal if a pre-existing torch happens to be installed (old app era).
    if importlib.util.find_spec("torch") is None:
        assert importlib.util.find_spec("transformers") is None
    else:
        warnings.warn(
            "torch is importable in this base environment (pre-existing install); "
            "only the pyproject structural confinement is asserted.",
            stacklevel=2,
        )


def test_reporting_importable_but_not_wired():
    """Reporting modules ship importable; the CLI does not reference them (D-10/11)."""
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-c",
            (
                "import chat_analyzer.reporting.pdf_report, "
                "chat_analyzer.reporting.weekly_digest; print('REPORTING-OK')"
            ),
        ],
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "REPORTING-OK" in result.stdout

    cli_source = (REPO_ROOT / "src" / "chat_analyzer" / "cli" / "main.py").read_text(
        encoding="utf-8"
    )
    for token in ("pdf_report", "weekly_digest", "reportlab"):
        assert token not in cli_source, f"CLI must not reference {token}"
