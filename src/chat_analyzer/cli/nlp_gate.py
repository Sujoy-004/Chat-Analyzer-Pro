"""Silent NLP availability probe + locked model constants (D-02/D-05/D-07c).

The pipeline always *prepares* for NLP (D-01); whether the heavy models
actually run depends on this pure probe: transformers+torch importable AND
the emotion model cached locally. Never raises — import failures and missing
caches all resolve to False so the caller silently runs basic analysis
(D-02/D-06), with no prompt and no hint (that is main.py's job in 04-03).

The CHAT_ANALYZER_FORCE_NLP env override makes either branch deterministic
in tests (RESEARCH Pitfall 5: the dev machine has transformers but no cached
emotion model, so the probe alone would always report "unavailable").
"""

from __future__ import annotations

import logging
import os
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
