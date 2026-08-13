"""PH2 nlp_gate tier-resolution tests (version satisfaction + status mapping).

Exercises the pure logic behind the always-on tier menu:
- nlp_installed_versions / nlp_versions_satisfied against the pinned ranges
  mirrored from pyproject.toml.
- nlp_status() mapping: MISSING (package not installed) vs OUTDATED (version
  outside pins, or models not cached) vs READY (both satisfied).
- nlp_models_cached() honours the two locked model IDs.

All functions are pure (no subprocess, no network) so these are fast tests.
"""

import unittest.mock

from chat_analyzer.cli import nlp_gate


def test_versions_satisfied_accepts_in_pin(monkeypatch):
    """A version inside the pinned range satisfies its pin."""
    installed = {"torch": "2.1.0", "transformers": "4.40.0", "sentencepiece": "0.2.0"}
    monkeypatch.setattr(nlp_gate, "nlp_installed_versions", lambda: installed)
    assert nlp_gate.nlp_versions_satisfied() == {
        "torch": True,
        "transformers": True,
        "sentencepiece": True,
    }


def test_versions_satisfied_rejects_out_of_pin(monkeypatch):
    """Versions outside the pins fail satisfaction (transformers>=4.30,<5.15)."""
    installed = {"torch": "1.13.0", "transformers": "5.16.0", "sentencepiece": "0.1.98"}
    monkeypatch.setattr(nlp_gate, "nlp_installed_versions", lambda: installed)
    assert nlp_gate.nlp_versions_satisfied() == {
        "torch": False,
        "transformers": False,
        "sentencepiece": False,
    }


def test_status_missing_when_package_absent(monkeypatch):
    """A missing package is MISSING regardless of the other pins."""
    installed = {"torch": None, "transformers": "4.40.0", "sentencepiece": "0.2.0"}
    monkeypatch.setattr(nlp_gate, "nlp_installed_versions", lambda: installed)
    monkeypatch.setattr(nlp_gate, "nlp_models_cached", lambda: False)
    status, detail = nlp_gate.nlp_status()
    assert status == "MISSING"
    assert detail["installed_versions"]["torch"] is None


def test_status_outdated_when_models_not_cached(monkeypatch):
    """All packages in pin but weights missing -> OUTDATED (fresh install)."""
    installed = {"torch": "2.1.0", "transformers": "4.40.0", "sentencepiece": "0.2.0"}
    monkeypatch.setattr(nlp_gate, "nlp_installed_versions", lambda: installed)
    monkeypatch.setattr(nlp_gate, "nlp_models_cached", lambda: False)
    status, detail = nlp_gate.nlp_status()
    assert status == "OUTDATED"
    assert detail["models_cached"] is False


def test_status_outdated_when_version_outside_pin(monkeypatch):
    """Package installed but outside pins -> OUTDATED even with models cached."""
    installed = {"torch": "2.1.0", "transformers": "5.16.0", "sentencepiece": "0.2.0"}
    monkeypatch.setattr(nlp_gate, "nlp_installed_versions", lambda: installed)
    monkeypatch.setattr(nlp_gate, "nlp_models_cached", lambda: True)
    status, detail = nlp_gate.nlp_status()
    assert status == "OUTDATED"
    assert detail["versions_satisfied"]["transformers"] is False


def test_status_ready_when_all_satisfied(monkeypatch):
    """Packages in pin AND both models cached -> READY."""
    installed = {"torch": "2.1.0", "transformers": "4.40.0", "sentencepiece": "0.2.0"}
    monkeypatch.setattr(nlp_gate, "nlp_installed_versions", lambda: installed)
    monkeypatch.setattr(nlp_gate, "nlp_models_cached", lambda: True)
    status, _ = nlp_gate.nlp_status()
    assert status == "READY"


def test_models_cached_requires_both(monkeypatch):
    """nlp_models_cached is True only when BOTH locked models are cached."""
    with unittest.mock.patch.object(
        nlp_gate, "model_cached", side_effect=lambda mid: mid == nlp_gate.MODEL_ID
    ):
        assert nlp_gate.nlp_models_cached() is False

    with unittest.mock.patch.object(
        nlp_gate,
        "model_cached",
        side_effect=lambda mid: mid in (nlp_gate.MODEL_ID, nlp_gate.TIER_B_MODEL_ID),
    ):
        assert nlp_gate.nlp_models_cached() is True
