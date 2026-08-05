"""Silent NLP availability probe + locked model constants (D-02/D-05/D-07c).

The pipeline always *prepares* for NLP (D-01); whether the heavy models
actually run depends on this pure probe: transformers+torch importable AND
the emotion model cached locally. Never raises — import failures and missing
caches all resolve to False so the caller silently runs basic analysis
(D-02/D-06), with no prompt and no hint (that is main.py's job in 04-03).

The CHAT_ANALYZER_FORCE_NLP env override makes either branch deterministic
in tests (RESEARCH Pitfall 5: the dev machine has transformers but no cached
emotion model, so the probe alone would always report "unavailable").

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

_FORCE_NLP = "CHAT_ANALYZER_FORCE_NLP"


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
    """Silent availability probe (D-02): never raises, never prompts.

    Env override CHAT_ANALYZER_FORCE_NLP wins ("0" -> False, "1" -> True) so
    tests can force either branch deterministically (RESEARCH Pitfall 5).
    Otherwise the probe requires transformers+torch to be importable AND the
    model to be cached locally.
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

    return model_cached(model_id)


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
    if cpu_only:
        # WR-01: --index-url REPLACES PyPI, so transformers would never
        # resolve from the PyTorch CPU wheel index. Install torch from the
        # CPU index first, then transformers from PyPI separately.
        _pip_install(["torch", "--index-url", _CPU_INDEX])
        _pip_install(["transformers>=4.30,<6"])
    else:
        _pip_install(["torch", "transformers>=4.30,<6"])
