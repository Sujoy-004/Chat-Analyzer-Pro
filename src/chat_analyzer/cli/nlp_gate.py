"""Silent NLP availability probe + locked model constants (D-02/D-05/D-07c).

The pipeline always *prepares* for NLP (D-01); whether the heavy models
actually run depends on this pure importability probe: transformers+torch
importable. Never raises — import failures and missing installs all resolve
to False so the caller silently runs basic analysis (D-02/D-06), with no
prompt and no hint (that is main.py's job in 05-03). Model weights are NOT
required up front: they download on first use (announced via model_cached in
pipeline.py), the "first install gap" fix for fresh [nlp] installs.

The CHAT_ANALYZER_FORCE_NLP env override makes either branch deterministic
in tests (RESEARCH Pitfall 5: the dev machine has transformers and no cached
emotion model, so the probe alone would not reliably exercise both branches).

install_nlp is the guarded runtime installer for the D-04 download menu: a
subprocess pip re-install of the already-declared [nlp] extras (torch +
transformers), CPU-only or full torch, that raises RuntimeError on failure so
the caller degrades to basic analysis + hint (never a frozen terminal).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

try:
    from huggingface_hub.constants import HF_HUB_CACHE
except ImportError:  # huggingface_hub ships with transformers ([nlp] extra only)
    HF_HUB_CACHE = None

logger = logging.getLogger(__name__)

# Locked model constants (CONTEXT D-07c — do NOT use RESEARCH.md's typo
# spellings). Announced with their sizes BEFORE any download/construction
# (D-05, Pitfall 4).
MODEL_ID = "bhadresh-savani/distilbert-base-uncased-emotion"
EMOTION_MODEL_SIZE_MB = 255
SUMMARY_MODEL_ID = "t5-small"
SUMMARY_MODEL_SIZE_MB = 231
TIER_B_MODEL_ID = "google/flan-t5-small"
TIER_B_MODEL_SIZE_MB = 340   # approx flan-t5-small total disk (~308 MB weights)

_FORCE_NLP = "CHAT_ANALYZER_FORCE_NLP"
_ALLOW_LONG_PATH = "CHAT_ANALYZER_ALLOW_LONG_PATH"

# Windows MAX_PATH guard (A1): torch 2.x wheel extraction crashes with
# WinError 206 when the venv's site-packages path plus torch's own reserved
# extraction depth crosses the 260-char limit.
TORCH_PATH_RESERVE = 180
WINDOWS_MAX_PATH = 260


def model_cached(model_id: str) -> bool:
    """True when the model weights are already in the local HF cache.

    Uses huggingface_hub's canonical cache root when available, else the raw
    ~/.cache/huggingface/hub fallback (RESEARCH A3).
    """
    if HF_HUB_CACHE:
        cache = Path(HF_HUB_CACHE)
    else:
        cache = Path.home() / ".cache" / "huggingface" / "hub"
    return (cache / ("models--" + model_id.replace("/", "--"))).exists()


def nlp_available(model_id: str = MODEL_ID) -> bool:
    """Silent importability probe (D-02): never raises, never prompts.

    Env override CHAT_ANALYZER_FORCE_NLP wins ("0" -> False, "1" -> True) so
    tests can force either branch deterministically (RESEARCH Pitfall 5).
    Otherwise the probe returns True when transformers+torch import — it no
    longer requires the weights to be cached locally. The model download
    happens on first use (announced via model_cached by pipeline.py). This is
    the "first install gap" fix: install_nlp only pip-installs the extras (no
    weights), so the old model_cached requirement meant a fresh [nlp] user's
    very first run silently skipped NLP.
    """
    force = os.environ.get(_FORCE_NLP)
    if force is not None:
        if force == "1":
            return True
        if force == "0":
            return False

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False

    return True


_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
_INSTALL_TIMEOUT = 900


def _pip_install(args: list[str]) -> None:
    """Run a guarded pip install, raising RuntimeError on failure/timeout.

    Output is captured, never echoed raw (WR-02: a multi-GB download must not
    hang the terminal forever — timeout expires to the same friendly error as
    a pip failure).
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=_INSTALL_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - slow path
        raise RuntimeError(
            "Model install timed out — run basic analysis, or install: "
            "pip install chat-analyzer-pro[nlp]"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            "Model install failed — run basic analysis, or install: "
            "pip install chat-analyzer-pro[nlp]"
        )


def registry_long_paths_enabled() -> bool:
    """True when the Windows LongPathsEnabled system flag is set (nt only).

    Reads the LongPathsEnabled DWORD the README tells the user to enable;
    when it is on, officially long paths are permitted and the MAX_PATH
    guard must not warn or block.
    """
    if os.name != "nt":
        return False
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
            return value == 1
    except OSError:
        return False


def windows_long_path_message() -> str | None:
    """Instructive warning when the venv path is too deep for torch on Windows.

    Returns None on non-Windows platforms, when the user has enabled Windows'
    LongPathsEnabled flag, when the CHAT_ANALYZER_ALLOW_LONG_PATH override is
    set to "1", or when the site-packages path leaves enough headroom under
    the 260-char MAX_PATH limit. Otherwise returns a short ASCII remediation
    string (WinError 206) — callers print or raise it so the user relocates
    before a multi-GB download. The reserve is heuristic (a global Python3xx
    user-site install is not measured), which is why the override exists.
    """
    if os.name != "nt":
        return None
    if os.environ.get(_ALLOW_LONG_PATH) == "1":
        return None
    if registry_long_paths_enabled():
        return None
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    if len(str(site_packages)) + TORCH_PATH_RESERVE <= WINDOWS_MAX_PATH:
        return None
    return (
        "This Python environment lives at a deep path, so the torch "
        "extraction will hit the Windows 260-character limit (WinError 206). "
        "Either enable the LongPathsEnabled registry flag, or use the "
        "reliable fix: create a short-path venv in your temp folder with "
        "scripts/make_nlp_env.ps1 (for example %TEMP%\\chat-analyzer-nlp), "
        "well under 260 characters."
    )


def install_nlp(cpu_only: bool = False) -> None:
    """Runtime install of the already-declared [nlp] extras (D-05).

    Guarded subprocess pip — never shell=True, never os.system (T-04-10).
    Installs torch + transformers at runtime when the user picks the download
    option from the interactive menu. CPU-only torch pulls PyTorch's CPU wheel
    index (~0.6 GB install); the default is the full torch build (~3 GB). No
    new package names enter the dependency graph — these are the already
    audited [nlp] extras (T-04-SC).

    Raises RuntimeError on any failure (offline, no pip, timeout) so the caller
    degrades to basic analysis plus the hint line — never a frozen terminal
    (Pitfall 4).
    """
    reason = windows_long_path_message()
    if reason is not None:
        raise RuntimeError(
            reason + " Basic analysis still works without the NLP models."
        )
    if cpu_only:
        # WR-01: --index-url REPLACES PyPI, so transformers would never
        # resolve from the PyTorch CPU wheel index. Install torch from the
        # CPU index first, then transformers from PyPI separately.
        _pip_install(["torch", "--index-url", _CPU_INDEX])
        _pip_install(["transformers>=4.30,<5.15"])
    else:
        _pip_install(["torch", "transformers>=4.30,<5.15"])
