"""Windows long-path guard tests for the NLP installer (A1).

The venv path must be checked BEFORE a multi-GB pip install starts: torch
2.x extraction crashes with WinError 206 past the Windows 260-char limit.
These tests exercise the real chat_analyzer.cli.nlp_gate module and never
spawn pip — the _pip_install recorder in the last test proves the guard
fires first. No torch or [nlp] extra is needed.
"""

import os
import sys

import pytest

from chat_analyzer.cli import nlp_gate

SHORT_PREFIX = "C:\\nlp"
DEEP_PREFIX = (
    "C:\\Users\\SomeLongUserName\\Documents\\project with long name"
    "\\Chat-Analyzer-Pro"
)


def _deep_windows_env(monkeypatch):
    """A deep Windows venv prefix with the LongPathsEnabled flag stubbed off
    so the guard does not read the live machine registry (deterministic)."""
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(sys, "prefix", DEEP_PREFIX)
    monkeypatch.setattr(nlp_gate, "registry_long_paths_enabled", lambda: False)
    monkeypatch.delenv("CHAT_ANALYZER_ALLOW_LONG_PATH", raising=False)


def test_short_path_returns_none(monkeypatch):
    """A short Windows venv path (well under 260 chars) warns nothing."""
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(sys, "prefix", SHORT_PREFIX)
    monkeypatch.setattr(nlp_gate, "registry_long_paths_enabled", lambda: False)
    monkeypatch.delenv("CHAT_ANALYZER_ALLOW_LONG_PATH", raising=False)
    assert nlp_gate.windows_long_path_message() is None


def test_non_windows_returns_none(monkeypatch):
    """Non-Windows platforms never warn regardless of path depth."""
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(sys, "prefix", DEEP_PREFIX)
    assert nlp_gate.windows_long_path_message() is None


def test_deep_windows_path_returns_message(monkeypatch):
    """A deep Windows venv path returns relocation guidance, non-empty."""
    _deep_windows_env(monkeypatch)
    message = nlp_gate.windows_long_path_message()
    assert message
    assert "LongPathsEnabled" in message
    assert "make_nlp_env" in message


def test_long_paths_enabled_registry_silences_guard(monkeypatch):
    """The registry LongPathsEnabled flag turns the guard off (README (a))."""
    _deep_windows_env(monkeypatch)
    monkeypatch.setattr(nlp_gate, "registry_long_paths_enabled", lambda: True)
    assert nlp_gate.windows_long_path_message() is None


def test_allow_long_path_override_silences_guard(monkeypatch):
    """CHAT_ANALYZER_ALLOW_LONG_PATH=1 bypasses the heuristic over-block."""
    _deep_windows_env(monkeypatch)
    monkeypatch.setenv("CHAT_ANALYZER_ALLOW_LONG_PATH", "1")
    assert nlp_gate.windows_long_path_message() is None


def test_install_nlp_raises_on_long_path(monkeypatch):
    """install_nlp refuses before pip runs when the venv path is too deep."""
    _deep_windows_env(monkeypatch)
    calls = []

    def recorder(args):
        calls.append(args)

    monkeypatch.setattr(nlp_gate, "_pip_install", recorder)
    with pytest.raises(RuntimeError):
        nlp_gate.install_nlp()
    assert calls == []