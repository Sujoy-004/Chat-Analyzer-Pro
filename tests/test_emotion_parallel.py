"""Parallel emotion worker coverage: env contract, gating, chunking, degrade.

Covers the parallel unique-text scoring branch (research delta #5 — ZERO
coverage at research time) of the exact-path emotion scorer:

- test_emotion_worker_count_env_contract: CHAT_ANALYZER_EMOTION_WORKERS parsing
  (absent -> 3, "0"/"1" -> 1, "4" -> 4, "99" -> min(99, cpu, 8), garbage and
  "-5" -> 3, never raises).
- test_mocked_pipeline_stays_sequential: the real-pipeline gate — a mocked
  classifier is NOT a transformers.Pipeline, so _scoring_workers returns 0 and
  _score_unique_texts never invokes the parallel driver.
- test_threshold_below_stays_sequential / test_threshold_above_invokes_parallel:
  _EMOTION_PARALLEL_THRESHOLD (20_000) boundary, both directions.
- test_parallel_chunks_contiguous_fixed_order: the pool slots unique texts
  into contiguous fixed-order chunks (deterministic output).
- test_parallel_pool_failure_degrades_to_sequential: a pool exception degrades
  to None (logger.exception) and the caller falls back to sequential
  _score_batch — output identical to the sequential reference.
- test_tokenizers_parallelism_disabled_in_worker: TOKENIZERS_PARALLELISM=false
  is set before the transformers import inside _score_text_chunk (A1).
- test_spawn_parallel_matches_sequential_smoke (@pytest.mark.slow): the REAL
  cached DistilBERT pipeline with the threshold forced to 1 and 2 pool
  workers — parallel output matches the sequential reference within float
  tolerance. Skipped when the [nlp] extra / model cache is missing (D-17).

The fast tests mock the model callables exactly like test_perf_parity_emotion.py
(module-level ``_emotion_analyzer`` + ``_emotion_model_loaded`` patched with a
batch-faithful classifier) and never spawn real processes — the ProcessPool
driver is exercised via a fake pool; real-model inference happens ONLY in the
slow-marked smoke test (D-17).
"""

import inspect
import logging
import os
import unittest.mock
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from io import StringIO

# Headless-first (Pitfall 7): importing chat_analyzer.analysis.emotion pulls
# matplotlib; pin Agg BEFORE any pyplot import so figures are headless-safe.
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd
import pytest

from chat_analyzer.analysis import emotion as _emotion_module
from chat_analyzer.analysis.emotion import EmotionAnalyzer
from chat_analyzer.cli import nlp_gate
from chat_analyzer.ingest.ingestion import messages_to_dataframe

EMOTIONS = ("joy", "sadness", "anger", "fear", "surprise", "love")
EMOTION_COLS = [f"emotion_{e}" for e in EMOTIONS]

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


def _per_text(text):
    return LOVE_SCORES if "love" in str(text).lower() else JOY_SCORES


def _classifier(texts, **kwargs):
    """Batch-faithful: a list input gets a list of per-message results; a bare
    string gets a single result (so analyze_single_message works unchanged)."""
    if isinstance(texts, str):
        return _per_text(texts)
    return [_per_text(t) for t in texts]


def _make_analyzer(classifier=_classifier):
    """Real EmotionAnalyzer with the transformers pipeline mocked (D-17),
    mirroring tests/test_perf_parity_emotion.py: patch the module-level model
    cache so _initialize_model short-circuits to the fake pipeline."""
    with (
        redirect_stdout(StringIO()),
        unittest.mock.patch.object(_emotion_module, "_emotion_analyzer", classifier),
        unittest.mock.patch.object(_emotion_module, "_emotion_model_loaded", True),
    ):
        return EmotionAnalyzer()


def _pending(n, prefix="unique message text number"):
    """(idx, text) pending rows with n UNIQUE scorable texts."""
    return [(i, f"{prefix} {i}") for i in range(n)]


class _FakePool:
    """ProcessPoolExecutor stand-in: records the chunks handed to map() and
    either invokes the worker inline (chunk test) or raises (degrade test).
    No real processes are spawned (fast, D-17)."""

    def __init__(self, max_workers, fail=False):
        self.max_workers = max_workers
        self.fail = fail
        self.seen_chunks = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, chunks, *iterables):
        self.seen_chunks = list(chunks)
        if self.fail:
            raise RuntimeError("pool exploded")
        # zip() stops at len(chunks), so the itertools.repeat args are finite
        # here even though they are infinite iterables in the real call.
        return [fn(*args) for args in zip(chunks, *iterables)]


def _inline_fake_worker(texts, model_name, batch_size, num_threads):
    """Lightweight _score_text_chunk stand-in: scores without loading a model."""
    return [(t, _per_text(t)) for t in texts]


# --- env contract -----------------------------------------------------------


def test_emotion_worker_count_env_contract(monkeypatch):
    env = "CHAT_ANALYZER_EMOTION_WORKERS"
    monkeypatch.delenv(env, raising=False)
    assert nlp_gate.emotion_worker_count() == 3  # absent -> default

    for off in ("0", "1"):
        monkeypatch.setenv(env, off)
        assert nlp_gate.emotion_worker_count() == 1  # no pool

    monkeypatch.setenv(env, "4")
    assert nlp_gate.emotion_worker_count() == 4

    monkeypatch.setenv(env, "99")
    assert nlp_gate.emotion_worker_count() == min(99, os.cpu_count() or 1, 8)

    for garbage in ("off", "false", "abc", "-5"):
        monkeypatch.setenv(env, garbage)
        assert nlp_gate.emotion_worker_count() == 3  # never raises


# --- real-pipeline gate ------------------------------------------------------


def test_mocked_pipeline_stays_sequential(monkeypatch):
    """A mocked classifier is NOT a transformers.Pipeline -> _scoring_workers
    returns 0 and the parallel driver is never invoked, even above the
    threshold (mocked tests stay sequential by construction, D-17)."""
    analyzer = _make_analyzer(_classifier)
    assert analyzer._scoring_workers(workers=3) == 0

    driver = unittest.mock.Mock(wraps=analyzer._score_unique_texts_parallel)
    monkeypatch.setattr(analyzer, "_score_unique_texts_parallel", driver)

    with redirect_stdout(StringIO()):
        out = analyzer._score_unique_texts(_pending(20_001), batch_size=64, workers=3)

    driver.assert_not_called()
    assert len(out) == 20_001
    assert all(set(scores) == set(EMOTIONS) for scores in out.values())


# --- threshold gating --------------------------------------------------------


def test_threshold_below_stays_sequential(monkeypatch):
    """< 20_000 unique texts -> sequential even with a real pipeline and
    workers >= 2: the parallel driver must not be invoked."""
    analyzer = _make_analyzer(_classifier)
    monkeypatch.setattr(analyzer, "_is_real_pipeline", lambda: True)

    driver = unittest.mock.Mock(wraps=analyzer._score_unique_texts_parallel)
    monkeypatch.setattr(analyzer, "_score_unique_texts_parallel", driver)

    with redirect_stdout(StringIO()):
        out = analyzer._score_unique_texts(_pending(100), batch_size=32, workers=2)

    driver.assert_not_called()
    assert len(out) == 100


def test_threshold_above_invokes_parallel(monkeypatch):
    """> 20_000 unique texts -> the parallel driver IS invoked (the boundary
    is strict: below it stays sequential, above it goes parallel)."""
    analyzer = _make_analyzer(_classifier)
    monkeypatch.setattr(analyzer, "_is_real_pipeline", lambda: True)

    fake = {"some text": {"joy": 1.0}}
    driver = unittest.mock.Mock(return_value=fake)
    monkeypatch.setattr(analyzer, "_score_unique_texts_parallel", driver)

    with redirect_stdout(StringIO()):
        out = analyzer._score_unique_texts(_pending(20_001), batch_size=32, workers=2)

    driver.assert_called_once()
    assert out is fake


# --- chunking + degrade ------------------------------------------------------


def test_parallel_chunks_contiguous_fixed_order(monkeypatch):
    """The pool slots unique texts into contiguous fixed-order chunks so the
    flattened result restores the original unique order (deterministic output
    regardless of worker count)."""
    analyzer = _make_analyzer(_classifier)
    pool = _FakePool(max_workers=3)
    monkeypatch.setattr(
        "concurrent.futures.ProcessPoolExecutor", lambda max_workers: pool
    )
    monkeypatch.setattr(_emotion_module, "_score_text_chunk", _inline_fake_worker)

    unique_texts = [f"unique message text number {i}" for i in range(10)]
    result = analyzer._score_unique_texts_parallel(
        unique_texts, batch_size=4, workers=3
    )

    step = (10 + 3 - 1) // 3  # ceil(10/3) = 4
    assert pool.seen_chunks == [
        unique_texts[0:step],
        unique_texts[step : 2 * step],
        unique_texts[2 * step :],
    ]
    assert [t for chunk in pool.seen_chunks for t in chunk] == unique_texts
    assert list(result) == unique_texts


def test_parallel_pool_failure_degrades_to_sequential(monkeypatch, caplog):
    """A raised exception inside the pool degrades to None (logger.exception);
    the caller falls back to sequential _score_batch, so the output is
    identical to the sequential reference even when the parallel driver fails."""
    analyzer = _make_analyzer(_classifier)
    monkeypatch.setattr(analyzer, "_is_real_pipeline", lambda: True)
    monkeypatch.setattr(
        "concurrent.futures.ProcessPoolExecutor",
        lambda max_workers: _FakePool(max_workers, fail=True),
    )

    pending = _pending(20_001)
    with (
        caplog.at_level(logging.ERROR, logger="chat_analyzer.analysis.emotion"),
        redirect_stdout(StringIO()),
    ):
        out = analyzer._score_unique_texts(pending, batch_size=64, workers=2)
        reference = analyzer._score_unique_texts(pending, batch_size=64, workers=1)

    assert "parallel emotion scoring failed" in caplog.text
    assert out == reference
    assert len(out) == 20_001


# --- worker hardening (A1) ---------------------------------------------------


def test_tokenizers_parallelism_disabled_in_worker():
    """TOKENIZERS_PARALLELISM=false must be set right after `import os` and
    BEFORE the transformers import inside _score_text_chunk, so the spawn
    child's tokenizers rayon pool never oversubscribes the box."""
    source = inspect.getsource(_emotion_module._score_text_chunk)
    marker = 'os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")'
    assert marker in source
    assert (
        source.index("import os")
        < source.index(marker)
        < source.index("from transformers import pipeline")
    )


# --- slow: real-model spawn-parity smoke -------------------------------------


def _smoke_fixture_df(n=20):
    """Small deterministic frame of n unique scorable messages."""
    rows = []
    base = datetime(2024, 1, 1, 9, 0, 0)  # noqa: DTZ001 - naive fixture base
    for i in range(n):
        rows.append({
            "datetime": base + timedelta(minutes=i),
            "sender": "Alice" if i % 2 == 0 else "Bob",
            "message": f"this is a perfectly ordinary message number {i} to score",
        })
    return messages_to_dataframe(rows)


@pytest.mark.slow
def test_spawn_parallel_matches_sequential_smoke(monkeypatch):
    """Spawn-parity smoke (research delta #5): forces _EMOTION_PARALLEL_THRESHOLD
    to 1 and scores a tiny fixture through the REAL cached DistilBERT pipeline
    with 2 pool workers, asserting the parallel output matches the sequential
    reference within float tolerance (last-bit thread-dependent noise < 1e-6).

    Skipped when the [nlp] extra or the model cache is missing (D-17: real-model
    inference is allowed ONLY in this slow-marked test).
    """
    model_cache = os.path.expanduser(
        "~/.cache/huggingface/hub/models--bhadresh-savani--distilbert-base-uncased-emotion"
    )
    if not nlp_gate.nlp_available() or not os.path.isdir(model_cache):
        pytest.skip(
            "real DistilBERT model cache missing — spawn smoke needs [nlp] weights (D-17)"
        )

    monkeypatch.setenv("CHAT_ANALYZER_EMOTION_WORKERS", "2")
    monkeypatch.setattr(_emotion_module, "_EMOTION_PARALLEL_THRESHOLD", 1)

    with redirect_stdout(StringIO()):
        analyzer = EmotionAnalyzer()
    df = _smoke_fixture_df(20)
    with redirect_stdout(StringIO()):
        parallel = analyzer.analyze_emotions(df, batch_size=8, workers=2)
        sequential = analyzer.analyze_emotions(df, batch_size=8, workers=1)

    cols = EMOTION_COLS + ["dominant_emotion", "emotion_confidence"]
    pd.testing.assert_frame_equal(
        parallel[cols], sequential[cols], check_exact=False, rtol=1e-4, atol=1e-6
    )