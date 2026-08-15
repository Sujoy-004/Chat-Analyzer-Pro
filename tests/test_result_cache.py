"""Result cache keyed by file hash — fast, mocked (D-17), no real models.

Drives the REAL run_pipeline (QUAL-02) with mocked model callables exactly
like tests/test_emotion_sampling.py: patch the module-level _emotion_analyzer
singleton + _emotion_model_loaded flag and the T5 from_pretrained classes, so
the cache hit/miss/key/corruption/TTL contracts are proven end-to-end without
torch inference.

Covered behavior (plan Task 5, research Q7 a-o):

- a. default OFF: no env var -> no cache dir created, no Loaded label
- b. env parser whitelist (absent/empty/off-words disable; on-words -> default
       dir; explicit path; garbage -> that dir; never raises; default dir never
       under the repo tree)
- c. miss runs all stages and stores the entry
- d. hit skips the NLP stages (mock call counter unchanged) and returns an
       output deep-equal to the first run
- e. hit prints the honest '[INFO] Loaded analysis from cache' label
- f. editing the file bytes invalidates (new hash -> miss; old entry kept)
- g. config change (sampling knob) invalidates (distinct keys)
- h. tier change (nlp_on) invalidates
- i. the tty y/N sample choice collapses into the key (two entries)
- j. corrupt cache (garbage text + pickle payload) degrades to a miss, never
       a crash, and self-heals (file deleted) — json-only proof
- k. schema-mismatched payload -> miss, file NOT deleted
- l. sanitizer round-trips numpy scalars/arrays and NaN/Inf -> None
- m. 30-day TTL: a back-dated entry is pruned on the next store
- n. zip transcript selection rides in the key; mtime-only changes do not
- o. report_path is reset to '' on load (never a stale cwd path)
"""

import contextlib
import io
import json
import os
import pickle
import time
import unittest.mock
import zipfile
from pathlib import Path

# Headless-first (Pitfall 7): importing chat_analyzer pulls matplotlib; pin
# Agg BEFORE any pyplot import so chart building is headless-safe.
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest
from rich.console import Console

from chat_analyzer.analysis import emotion as _emotion_module
from chat_analyzer.cli import nlp_gate, result_cache
from chat_analyzer.cli.pipeline import run_pipeline

try:
    import transformers
except ImportError:  # [nlp] extra not installed — pipeline mocks skip (D-17)
    transformers = None

EMOTIONS = ("joy", "sadness", "anger", "fear", "surprise", "love")

JOY_SCORES = [
    {"label": "joy", "score": 0.87},
    {"label": "sadness", "score": 0.03},
    {"label": "anger", "score": 0.03},
    {"label": "fear", "score": 0.02},
    {"label": "surprise", "score": 0.02},
    {"label": "love", "score": 0.03},
]
LOVE_SCORES = [
    {"label": "love", "score": 0.8},
    {"label": "joy", "score": 0.1},
    {"label": "sadness", "score": 0.02},
    {"label": "anger", "score": 0.02},
    {"label": "fear", "score": 0.02},
    {"label": "surprise", "score": 0.04},
]
SAD_SCORES = [
    {"label": "sadness", "score": 0.8},
    {"label": "joy", "score": 0.1},
    {"label": "anger", "score": 0.02},
    {"label": "fear", "score": 0.03},
    {"label": "surprise", "score": 0.02},
    {"label": "love", "score": 0.03},
]


def _per_text(text):
    lowered = str(text).lower()
    if "love" in lowered:
        return LOVE_SCORES
    if "sad" in lowered or "sorry" in lowered:
        return SAD_SCORES
    return JOY_SCORES


def _classifier(texts, **kwargs):
    """Batch-faithful: a list input gets a list of per-message results; a bare
    string gets a single result (so analyze_single_message works unchanged)."""
    if isinstance(texts, str):
        return _per_text(texts)
    return [_per_text(t) for t in texts]


class _CountingClassifier:
    """Classifier wrapper that counts calls (D-17) — proves hits never re-run."""

    def __init__(self):
        self.calls = 0

    def __call__(self, texts, **kwargs):
        self.calls += 1
        return _classifier(texts, **kwargs)


class _FakeT5Model:
    """Stand-in for T5ForConditionalGeneration mocking the direct
    generate() path the summarizer uses (no pipeline() abstraction)."""

    def generate(self, **kwargs):
        return [[0]]


class _FakeT5Tokenizer:
    """Stand-in for T5Tokenizer — callable and decodable, mirroring how
    ConversationSummarizer builds inputs and decodes outputs."""

    def __call__(self, text, **kwargs):
        return {"input_ids": [[0]]}

    def decode(self, output, skip_special_tokens=True):
        return "A test summary."


@contextlib.contextmanager
def _mocked_models(counter=None):
    """Patch the heavy model callables (D-17): the emotion singletons plus the
    T5 summarizer's direct from_pretrained path, so run_pipeline runs the real
    modules fast and offline. Mirrors test_emotion_sampling.py::_mocked_models;
    `counter` (optional) is swapped in as the classifier to count NLP calls."""
    if transformers is None:
        pytest.skip("transformers not installed — model-load mocks need the [nlp] extra (D-17)")
    classifier = counter if counter is not None else _classifier
    with (
        unittest.mock.patch.object(_emotion_module, "_emotion_analyzer", classifier),
        unittest.mock.patch.object(_emotion_module, "_emotion_model_loaded", True),
        unittest.mock.patch("transformers.pipeline", return_value=classifier),
        unittest.mock.patch.object(
            transformers.T5Tokenizer, "from_pretrained", return_value=_FakeT5Tokenizer()
        ),
        unittest.mock.patch.object(
            transformers.T5ForConditionalGeneration,
            "from_pretrained",
            return_value=_FakeT5Model(),
        ),
    ):
        yield


def _big_whatsapp_file(tmp_path, n=40):
    """Generate a 40-message WhatsApp-style export (all messages scorable)."""
    lines = []
    for i in range(n):
        sender = "Alice" if i % 3 == 0 else "Bob"
        msg = f"I love this one {i}" if i % 4 == 0 else f"regular message {i} here"
        hour = 9 + (i // 60) % 3
        lines.append(f"12/25/23, {hour}:{i % 60:02d} AM - {sender}: {msg}")
    path = tmp_path / "big.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _non_tty_console():
    """Off-tty console plus its capture buffer (rich renders [INFO]/[OK] tags
    literally off-tty, so assertions read the raw narration text)."""
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False), buf


def _expected_key(path, nlp_on=True, sample_cap=None, chosen=()):
    """The key run_pipeline computes for this file/config (mirror contract)."""
    return result_cache.cache_key(
        path,
        nlp_on=nlp_on,
        sample_cap=sample_cap,
        emotion_workers=nlp_gate.emotion_worker_count(),
        chosen_transcripts=chosen,
    )


# --- a. default OFF ---------------------------------------------------------


def test_cache_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("CHAT_ANALYZER_RESULT_CACHE", raising=False)
    assert nlp_gate.result_cache_dir() is None, "no env var -> cache disabled"

    # the would-be default dir must not exist anywhere (privacy lock)
    monkeypatch.setenv("CHAT_ANALYZER_RESULT_CACHE", "1")
    default_dir = nlp_gate.result_cache_dir()
    monkeypatch.delenv("CHAT_ANALYZER_RESULT_CACHE")
    assert default_dir is not None
    assert not default_dir.exists(), (
        "no cache dir may be created while the feature is disabled"
    )

    path = _big_whatsapp_file(tmp_path)
    console, buf = _non_tty_console()
    with _mocked_models():
        run_pipeline(path, console, nlp_enabled=True)
    assert "Loaded analysis from cache" not in buf.getvalue()
    assert not default_dir.exists(), "a default-OFF run must create nothing"


# --- b. env parser ----------------------------------------------------------


def test_env_parser(monkeypatch, tmp_path):
    env = "CHAT_ANALYZER_RESULT_CACHE"
    monkeypatch.delenv(env, raising=False)
    assert nlp_gate.result_cache_dir() is None  # absent -> disabled

    monkeypatch.setenv(env, "")
    assert nlp_gate.result_cache_dir() is None  # empty -> disabled

    for off in ("0", "off", "OFF", "false", "FALSE", "no", "NO"):
        monkeypatch.setenv(env, off)
        assert nlp_gate.result_cache_dir() is None, f"{off!r} must disable"

    for on in ("1", "on", "true", "yes"):
        monkeypatch.setenv(env, on)
        d = nlp_gate.result_cache_dir()
        assert d is not None and "chat-analyzer" in str(d), f"{on!r} -> default dir"

    explicit = tmp_path / "explicit-cache"
    monkeypatch.setenv(env, str(explicit))
    assert nlp_gate.result_cache_dir() == explicit, "a path value IS the dir"

    monkeypatch.setenv(env, "garbage")
    assert str(nlp_gate.result_cache_dir()) == "garbage"  # garbage -> that dir

    # privacy lock (research Pitfall 4): the DEFAULT dir is never under the repo
    monkeypatch.setenv(env, "1")
    repo_root = Path(__file__).resolve().parent.parent
    assert not nlp_gate.result_cache_dir().is_relative_to(repo_root), (
        "the default cache dir must always live outside the repo tree"
    )


# --- c. miss runs and stores ------------------------------------------------


def test_miss_runs_and_stores(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_ANALYZER_RESULT_CACHE", str(tmp_path))
    path = _big_whatsapp_file(tmp_path)
    counter = _CountingClassifier()
    console, _ = _non_tty_console()
    with _mocked_models(counter):
        results = run_pipeline(path, console, nlp_enabled=True)

    key = _expected_key(path, nlp_on=True, sample_cap=None)
    assert (tmp_path / f"{key}.json").exists(), "a miss must store the entry"
    assert counter.calls > 0, "a miss must run the NLP (emotion) stages"
    assert results["report_path"] == "", "adapt() hands report_path to main.py"


# --- d. hit skips the NLP stages --------------------------------------------


def test_hit_skips_nlp_stages(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_ANALYZER_RESULT_CACHE", str(tmp_path))
    path = _big_whatsapp_file(tmp_path)
    counter = _CountingClassifier()
    console, _ = _non_tty_console()
    with _mocked_models(counter):
        first = run_pipeline(path, console, nlp_enabled=True)
    calls_after_first = counter.calls
    assert calls_after_first > 0

    with _mocked_models(counter):
        second = run_pipeline(path, console, nlp_enabled=True)
    assert counter.calls == calls_after_first, (
        "a hit must NOT re-run the NLP stages (call counter unchanged)"
    )
    # Deep-equal modulo the JSON normalization every cache must apply: the
    # sanitizer coerces numpy scalars to plain numbers and non-serializable
    # dict keys (sentiment.daily_avg is keyed by datetime.date) to str, so the
    # cached payload equals the first run's payload through _sanitize().
    assert second == result_cache._sanitize(first), (
        "the cached payload must equal the first run (JSON-normalized)"
    )


# --- e. hit prints the loaded label -----------------------------------------


def test_hit_prints_loaded_label(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_ANALYZER_RESULT_CACHE", str(tmp_path))
    path = _big_whatsapp_file(tmp_path)
    console, buf = _non_tty_console()
    with _mocked_models():
        run_pipeline(path, console, nlp_enabled=True)
    with _mocked_models():
        run_pipeline(path, console, nlp_enabled=True)
    assert "Loaded analysis from cache" in buf.getvalue(), (
        "a hit must narrate the honest loaded-from-cache label"
    )


# --- f. file edit invalidates ------------------------------------------------


def test_file_edit_invalidates(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_ANALYZER_RESULT_CACHE", str(tmp_path))
    path = _big_whatsapp_file(tmp_path)
    counter = _CountingClassifier()
    console, _ = _non_tty_console()
    with _mocked_models(counter):
        run_pipeline(path, console, nlp_enabled=True)
    key_before = _expected_key(path)
    assert (tmp_path / f"{key_before}.json").exists()
    calls_before = counter.calls

    with path.open("a", encoding="utf-8") as fh:  # append one byte -> new hash
        fh.write("\n")
    key_after = _expected_key(path)
    assert key_after != key_before, "any byte change must produce a new key"

    with _mocked_models(counter):
        run_pipeline(path, console, nlp_enabled=True)
    assert counter.calls > calls_before, "an edited file must be re-analyzed"
    assert (tmp_path / f"{key_after}.json").exists(), "new entry stored"
    assert (tmp_path / f"{key_before}.json").exists(), "old entry kept"


# --- g. config change (sampling knob) invalidates ----------------------------


def test_config_change_invalidates(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_ANALYZER_RESULT_CACHE", str(tmp_path))
    path = _big_whatsapp_file(tmp_path)
    console, _ = _non_tty_console()

    # Over-cap sampling env -> non-tty AUTO-SAMPLE; the EFFECTIVE cap (5) folds
    # into the key. (The plan's literal '0 vs unset' pair collapses to the same
    # effective cap below the 50000 default, so an over-cap value is used to
    # make the config delta real — the code path under test is identical.)
    monkeypatch.setenv("CHAT_ANALYZER_EMOTION_SAMPLE", "5")
    with _mocked_models():
        run_pipeline(path, console, nlp_enabled=True)
    key_sampled = _expected_key(path, sample_cap=5)
    assert (tmp_path / f"{key_sampled}.json").exists()

    # unset -> below the default cap -> exact (effective cap None)
    monkeypatch.delenv("CHAT_ANALYZER_EMOTION_SAMPLE", raising=False)
    with _mocked_models():
        run_pipeline(path, console, nlp_enabled=True)
    key_exact = _expected_key(path, sample_cap=None)
    assert key_exact != key_sampled, "a config change must invalidate the key"
    assert (tmp_path / f"{key_exact}.json").exists()


# --- h. tier change invalidates ----------------------------------------------


def test_tier_change_invalidates(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_ANALYZER_RESULT_CACHE", str(tmp_path))
    path = _big_whatsapp_file(tmp_path)
    console, _ = _non_tty_console()
    with _mocked_models():
        run_pipeline(path, console, nlp_enabled=True)
    key_nlp = _expected_key(path, nlp_on=True)
    assert (tmp_path / f"{key_nlp}.json").exists()

    with _mocked_models():
        run_pipeline(path, console, nlp_enabled=False)
    key_basic = _expected_key(path, nlp_on=False)
    assert key_basic != key_nlp, "nlp_on (tier) must be part of the key"
    assert (tmp_path / f"{key_basic}.json").exists()


# --- i. tty y/N sample choice collapses into the key -------------------------


def test_sample_choice_in_key(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_ANALYZER_RESULT_CACHE", str(tmp_path))
    monkeypatch.setenv("CHAT_ANALYZER_EMOTION_SAMPLE", "10")
    path = _big_whatsapp_file(tmp_path)

    # 'y' -> sampled
    console = Console(file=io.StringIO(), force_terminal=True)
    monkeypatch.setattr(console, "input", lambda prompt="": "y")
    with _mocked_models():
        results_y = run_pipeline(path, console, nlp_enabled=True)
    assert results_y["emotion"]["sample"]["sampled"] is True

    # 'N' -> exact
    console = Console(file=io.StringIO(), force_terminal=True)
    monkeypatch.setattr(console, "input", lambda prompt="": "N")
    with _mocked_models():
        results_n = run_pipeline(path, console, nlp_enabled=True)
    assert results_n["emotion"]["sample"] is None

    key_y = _expected_key(path, sample_cap=10)
    key_n = _expected_key(path, sample_cap=None)
    assert key_y != key_n, "y and N on the same file must produce two entries"
    assert (tmp_path / f"{key_y}.json").exists()
    assert (tmp_path / f"{key_n}.json").exists()


# --- j. corrupt cache is a miss, never a crash -------------------------------


def test_corrupt_cache_is_miss_not_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_ANALYZER_RESULT_CACHE", str(tmp_path))
    path = _big_whatsapp_file(tmp_path)
    key = _expected_key(path)
    target = tmp_path / f"{key}.json"
    console, _ = _non_tty_console()

    # unit-level self-heal: load() itself deletes a corrupt file
    target.write_text("{ this is not json !!!", encoding="utf-8")
    assert result_cache.load(tmp_path, key) is None, "corrupt -> miss"
    assert not target.exists(), "load() self-heals: corrupt file deleted"

    # garbage text at the expected key path -> run succeeds, full pipeline
    # runs, and the stale corrupt bytes never surface (the miss store writes
    # a fresh valid entry over them)
    target.write_text("{ this is not json !!!", encoding="utf-8")
    counter = _CountingClassifier()
    with _mocked_models(counter):
        run_pipeline(path, console, nlp_enabled=True)  # must succeed (miss)
    assert counter.calls > 0, "a corrupt entry must degrade to a full re-run"
    raw = json.loads(target.read_text(encoding="utf-8"))
    assert raw["schema"] == nlp_gate.RESULT_CACHE_SCHEMA, (
        "the corrupt bytes were replaced by a valid fresh entry"
    )

    # pickle payload — inert data under a json-only loader (T-04-20 proof)
    target.write_bytes(pickle.dumps({"exec": "evil"}))
    with _mocked_models(counter):
        run_pipeline(path, console, nlp_enabled=True)
    raw = json.loads(target.read_text(encoding="utf-8"))
    assert raw["schema"] == nlp_gate.RESULT_CACHE_SCHEMA, (
        "a pickle payload is never unpickled; replaced by a valid entry"
    )


# --- k. schema mismatch is a miss, file kept --------------------------------


def test_schema_mismatch_is_miss(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_ANALYZER_RESULT_CACHE", str(tmp_path))
    path = _big_whatsapp_file(tmp_path)
    key = _expected_key(path)
    target = tmp_path / f"{key}.json"
    target.write_text(
        json.dumps({"schema": 999, "results": {"source": "x"}}), encoding="utf-8"
    )
    console, _ = _non_tty_console()
    with _mocked_models():
        run_pipeline(path, console, nlp_enabled=True)  # miss, no crash
    assert target.exists(), (
        "a schema-mismatch file is NOT deleted (a newer-format entry stays)"
    )


# --- l. sanitizer round-trip -------------------------------------------------


def test_sanitizer_roundtrip():
    from datetime import date

    import numpy as np

    payload = {
        "stats": {
            "peak_hour": np.int64(17),
            "avg_response_time": np.float64(1.5),
            "nan_value": np.float64(float("nan")),
            "inf_value": np.float64(float("inf")),
            "neg_inf": np.float64(float("-inf")),
        },
        "flag": np.bool_(True),
        "arr": np.array([1, 2, 3]),
        "nested": {"matrix": np.array([[1.0, 2.0], [3.0, 4.0]])},
        "plain": 42,
        "text": "hello",
        "none": None,
        # verified leak: sentiment.daily_avg is keyed by datetime.date
        "daily_avg": {date(2023, 12, 25): np.float64(0.159)},
    }
    sanitized = result_cache._sanitize(payload)
    loaded = json.loads(json.dumps(sanitized))  # must never raise TypeError
    assert loaded["stats"]["peak_hour"] == 17
    assert isinstance(loaded["stats"]["peak_hour"], int)
    assert loaded["stats"]["avg_response_time"] == 1.5
    assert isinstance(loaded["stats"]["avg_response_time"], float)
    assert loaded["stats"]["nan_value"] is None
    assert loaded["stats"]["inf_value"] is None
    assert loaded["stats"]["neg_inf"] is None
    assert loaded["flag"] is True
    assert isinstance(loaded["flag"], bool)
    assert loaded["arr"] == [1, 2, 3]
    assert loaded["nested"]["matrix"] == [[1.0, 2.0], [3.0, 4.0]]
    assert loaded["plain"] == 42
    assert loaded["text"] == "hello"
    assert loaded["none"] is None
    assert loaded["daily_avg"] == {"2023-12-25": 0.159}, (
        "date keys must normalize to their str() form"
    )


# --- m. TTL prune on store ---------------------------------------------------


def test_ttl_prune(tmp_path):
    cache_dir = tmp_path / "cache"
    old_key = "f" * 64
    result_cache.store(cache_dir, old_key, {"stats": {"peak_hour": 1}})
    old_file = cache_dir / f"{old_key}.json"
    assert old_file.exists()

    backdate = time.time() - 31 * 86400  # 31 days ago — past the 30-day TTL
    os.utime(old_file, (backdate, backdate))

    new_key = "0" * 64
    result_cache.store(cache_dir, new_key, {"stats": {"peak_hour": 2}})
    assert not old_file.exists(), "a stale entry must be pruned on the next store"
    assert (cache_dir / f"{new_key}.json").exists()


# --- n. zip transcript selection rides in the key ----------------------------


def test_zip_transcript_selection_in_key(tmp_path):
    path = tmp_path / "chat.txt"
    path.write_text("12/25/23, 9:00 AM - Alice: hi\n", encoding="utf-8")

    k_all = result_cache.cache_key(
        path, nlp_on=True, sample_cap=None, emotion_workers=2,
        chosen_transcripts=["a.txt", "b.txt"],
    )
    k_one = result_cache.cache_key(
        path, nlp_on=True, sample_cap=None, emotion_workers=2,
        chosen_transcripts=["a.txt"],
    )
    assert k_all != k_one, "different chosen transcript lists must invalidate"

    # mtime-only change -> SAME key (a touch must still hit; research Q1.1)
    old = time.time() - 3600
    os.utime(path, (old, old))
    k_touched = result_cache.cache_key(
        path, nlp_on=True, sample_cap=None, emotion_workers=2,
        chosen_transcripts=["a.txt", "b.txt"],
    )
    assert k_touched == k_all, "mtime is not part of the key"

    # zip-level media property: the whole-zip hash captures media changes
    def _zip_with_media(media_name):
        z = tmp_path / f"{media_name}.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("chat.txt", "12/25/23, 9:00 AM - Alice: hi\n")
            zf.writestr(media_name, b"\x89PNG fake bytes")
        return z

    z1 = _zip_with_media("photo1.jpg")
    z2 = _zip_with_media("photo2.jpg")
    kz1 = result_cache.cache_key(
        z1, nlp_on=True, sample_cap=None, emotion_workers=2,
        chosen_transcripts=["chat.txt"],
    )
    kz2 = result_cache.cache_key(
        z2, nlp_on=True, sample_cap=None, emotion_workers=2,
        chosen_transcripts=["chat.txt"],
    )
    assert kz1 != kz2, "different media members must invalidate (whole-zip hash)"


# --- o. report_path reset on load --------------------------------------------


def test_report_path_reset_on_load(tmp_path):
    cache_dir = tmp_path / "cache"
    key = "a" * 64
    stale = "C:\\old\\cwd\\report.html"
    result_cache.store(cache_dir, key, {"stats": {"peak_hour": 1}, "report_path": stale})

    loaded = result_cache.load(cache_dir, key)
    assert loaded is not None
    assert loaded["report_path"] == "", "a stale cwd path must never surface"
    # the stored file itself keeps the raw value — the load API blanks it
    raw = json.loads((cache_dir / f"{key}.json").read_text(encoding="utf-8"))
    assert raw["results"]["report_path"] == stale