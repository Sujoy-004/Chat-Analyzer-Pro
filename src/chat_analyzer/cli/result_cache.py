"""Opt-in result cache keyed by sha256 of the input file bytes (04-08).

Repeat runs of the same chat file complete in seconds instead of minutes or
hours: the full AnalysisResults contract (stats, charts, narrative, ...) is
persisted to a user-local directory keyed by the input file's hash plus a
config signature, and run_pipeline serves the cached payload on a hit.

Privacy/security contract:
- The cache lives OUTSIDE the repo tree (never the working tree): the default
  dir is %LOCALAPPDATA%\\chat-analyzer\\cache on Windows and
  ~/.cache/chat-analyzer elsewhere (resolved by nlp_gate.result_cache_dir);
  a non-keyword CHAT_ANALYZER_RESULT_CACHE value overrides the location.
- The payload is the AnalysisResults contract ONLY (derived data — sender
  names, top words, charts); raw chat text never enters it.
- json.load only — never pickle, never eval (T-04-20). Corrupt or
  schema-mismatched entries degrade to a miss, never a crash; corrupt files
  are self-healed (deleted); stale entries are pruned by a 30-day TTL.
- Key filenames are hex digests only, so no user input ever enters a path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from chat_analyzer.cli import nlp_gate

try:  # numpy is optional — the sanitizer guards on it (Q2.3)
    import numpy as _np
except ImportError:  # pragma: no cover - non-numpy env
    _np = None

logger = logging.getLogger(__name__)

RESULT_CACHE_TTL_DAYS = 30

_SHA_CHUNK = 1024 * 1024  # 1 MiB blocks — a 20 MB file hashes in ~0.1-0.3 s


def sha256_file(path: Path) -> str:
    """sha256 hexdigest of a file's raw bytes, read in 1 MiB chunks.

    Never touches mtime/size — a `touch` on the input must still hit the
    cache (research Q1.1). Zip inputs hash the WHOLE archive bytes, media
    members included (research Q1.2).
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_SHA_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _app_version() -> str:
    """Installed chat-analyzer-pro version, or "dev" for editable installs.

    The same call main.py uses; the PackageNotFoundError guard keeps editable
    (uninstalled) installs safe. Folds into the cache key so app upgrades
    invalidate (research A2).
    """
    try:
        return version("chat-analyzer-pro")
    except PackageNotFoundError:
        return "dev"


def cache_key(
    path,
    *,
    nlp_on,
    sample_cap,
    emotion_workers,
    chosen_transcripts,
) -> str:
    """Deterministic key: sha256 of the file bytes + a config signature.

    signature (JSON sort_keys=True) = {schema, app_version, nlp_on, sample_cap
    (None = exact), emotion_workers, chosen_transcripts (sorted)}. Returns the
    hex digest only — the filename is `<key>.json`, so no user input ever
    enters a path (traversal impossible).
    """
    file_sha256 = sha256_file(path)
    signature = {
        "schema": nlp_gate.RESULT_CACHE_SCHEMA,
        "app_version": _app_version(),
        "nlp_on": bool(nlp_on),
        "sample_cap": sample_cap,
        "emotion_workers": int(emotion_workers),
        "chosen_transcripts": sorted(chosen_transcripts or []),
    }
    payload = json.dumps(
        {"file_sha256": file_sha256, "signature": signature}, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sanitize(obj):
    """Recursively coerce a payload to JSON-safe plain data (Q2.3).

    - dict/list/tuple walk; everything else passes through unchanged
    - dict KEYS: numpy scalars -> .item(); anything else not JSON-serializable
      (verified leak: sentiment.daily_avg is keyed by datetime.date) -> str()
    - numpy scalars -> .item(): integer -> int, floating -> float (NaN/Inf ->
      None), bool_ -> bool; ndarray -> tolist() then recurse
    - non-finite plain floats (NaN/+-Inf) -> None (chart_json _float_or_none
      precedent)

    The numpy checks are guarded so a non-numpy environment never crashes the
    walk (the module imports without numpy).
    """
    if isinstance(obj, dict):
        out: dict = {}
        for key, value in obj.items():
            if _np is not None:
                if isinstance(key, _np.bool_):
                    key = bool(key.item())
                elif isinstance(key, _np.integer):
                    key = int(key.item())
                elif isinstance(key, _np.floating):
                    key = float(key.item())
            if isinstance(key, (str, int, float, bool)) or key is None:
                out[key] = _sanitize(value)
            else:
                # Non-serializable keys (verified leak: sentiment.daily_avg is
                # keyed by datetime.date) become their str() form — the same
                # normalization json itself applies to int keys.
                out[str(key)] = _sanitize(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [_sanitize(value) for value in obj]
    if _np is not None:
        if isinstance(obj, _np.ndarray):
            return _sanitize(obj.tolist())
        if isinstance(obj, _np.bool_):
            return bool(obj.item())
        if isinstance(obj, _np.integer):
            return int(obj.item())
        if isinstance(obj, _np.floating):
            value = float(obj.item())
            return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj


def load(cache_dir: Path, key: str) -> dict | None:
    """Load the cached AnalysisResults for key, or None on any failure (miss).

    json.load only — never pickle/eval (T-04-20). A corrupt file is unlinked
    (self-healing); a schema mismatch or wrong-shape payload returns None and
    is NOT deleted (a newer-format file stays for a future version). The
    loaded results never carry a stale report_path (Q2.4). Never raises.
    """
    path = cache_dir / f"{key}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        try:  # self-heal: delete the corrupt file so the next run re-analyzes
            path.unlink(missing_ok=True)
        except OSError:  # best-effort — never raise out of load
            pass
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema") != nlp_gate.RESULT_CACHE_SCHEMA:
        return None
    results = data.get("results")
    if not isinstance(results, dict):
        return None
    results["report_path"] = ""  # main.py fills it fresh after write_report (D-09)
    return results


def store(cache_dir: Path, key: str, results: dict) -> None:
    """Best-effort atomic store of AnalysisResults under key; never raises.

    Envelope = {schema, app_version, created_at (UTC ISO-8601), results
    (sanitized)}. Written via a same-dir temp file + os.replace (atomic on
    the same volume, Windows-safe; last-write-wins for concurrent same-file
    runs — Q5.6). Prunes stale entries on every store (TTL). Any failure is
    logged and swallowed — the pipeline never crashes because of the cache.
    """
    tmp_path: Path | None = None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _prune(cache_dir)
        envelope = {
            "schema": nlp_gate.RESULT_CACHE_SCHEMA,
            "app_version": _app_version(),
            "created_at": datetime.now(UTC).isoformat(),
            "results": _sanitize(results),
        }
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=cache_dir, delete=False
        ) as fh:
            tmp_path = Path(fh.name)
            json.dump(envelope, fh)
        os.replace(tmp_path, cache_dir / f"{key}.json")
    except Exception:
        logger.exception("result cache store failed; continuing without cache")
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:  # best-effort tmp cleanup
                pass


def _prune(cache_dir: Path, max_age_days: int = RESULT_CACHE_TTL_DAYS) -> None:
    """Delete .json cache entries whose mtime is older than max_age_days.

    Best-effort per entry (try/except OSError); never raises. Disk-growth
    control (T-04-24) — a 30-day rolling window keeps the dir in the tens of
    MB for a handful of weekly chats.
    """
    cutoff = time.time() - max_age_days * 86400
    try:
        with os.scandir(cache_dir) as it:
            for entry in it:
                if not entry.name.endswith(".json"):
                    continue
                try:
                    if entry.stat().st_mtime < cutoff:
                        os.unlink(entry.path)
                except OSError:
                    continue
    except OSError:
        return