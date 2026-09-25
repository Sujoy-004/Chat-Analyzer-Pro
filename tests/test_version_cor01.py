"""COR-01: `--version` must not traceback when package metadata is missing.

The eager `--version` callback reads `importlib.metadata.version`, which
raises `PackageNotFoundError` when the `chat-analyzer-pro` distribution is
not installed (dev checkout, future rename). The callback must degrade to a
clear fallback line instead of surfacing a traceback.

Exercised through the real typer app (CliRunner) so the callback is driven as
a user would, with `importlib.metadata.version` monkeypatched in-process.
"""

from importlib.metadata import PackageNotFoundError

from typer.testing import CliRunner

from chat_analyzer.cli.main import app

runner = CliRunner()


def _patch_version(monkeypatch, behavior):
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", behavior)


def _raise_package_not_found(name):
    raise PackageNotFoundError(name)


def test_version_fallback_when_metadata_missing(monkeypatch):
    _patch_version(monkeypatch, _raise_package_not_found)
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "chat-analyzer (dev" in result.output
    assert "package metadata not found" in result.output
    assert "PackageNotFoundError" not in result.output
    assert "Traceback" not in result.output


def test_version_prints_semver_when_metadata_present(monkeypatch):
    _patch_version(monkeypatch, lambda name: "1.2.3")
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == "chat-analyzer 1.2.3"