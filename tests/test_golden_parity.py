"""Golden-CSV parity harness — clone-friendly regression gate (no NLP).

The two committed tiny fixtures (data/sample_chats/) have tiny golden CSVs
(tests/golden/) holding expected aggregate stats captured on the author
machine by running the REAL pipeline (nlp_enabled=False, tier-1 — no torch,
no transformers). This test re-runs the live pipeline on the same fixtures
and asserts the same projection matches the goldens, so a fresh clone can
verify the whole parse -> dataframe -> EDA -> VADER sentiment path is
correct in seconds without any NLP models.

The projection mirrors the AnalysisResults contract (contracts.py / the
adapt() assembly in adapters.py): parse counters, the stats block scalars
(total_messages, participant count, date_range, media_messages) and the
sentiment block's avg_compound (the average_scores.vader_compound mean the
adapter exposes). Float metrics compare with pytest.approx (rel=1e-4,
abs=1e-6) since the golden rounds to 6 decimal places; ints/strings compare
exactly. The metric-key set must match the golden exactly — a contract
rename (e.g. moving media_messages out of stats) trips the key-assert first.
"""

import io
from pathlib import Path

import pytest
from rich.console import Console

from chat_analyzer.cli.pipeline import run_pipeline

DATA = Path(__file__).resolve().parents[1] / "data" / "sample_chats"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# Contract-path names whose golden value is a float rounded to 6dp; everything
# else in the golden is an exact int or string.
FLOAT_METRICS = {"sentiment.avg_compound"}

FIXTURES = [
    ("whatsapp_sample.csv", DATA / "whatsapp_sample.txt"),
    ("telegram_sample.csv", DATA / "telegram_sample.json"),
]


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False)


def project(results: dict) -> dict:
    """Project the stable deterministic metric set from AnalysisResults."""
    return {
        "parse.parsed_messages": results["parse"]["parsed_messages"],
        "parse.skipped_lines": results["parse"]["skipped_lines"],
        "parse.system_messages": results["parse"]["system_messages"],
        # media_messages lives under stats in the real contract (adapters
        # computes max(marker count, zip file count)) — NOT in the parse block.
        "stats.media_messages": results["stats"]["media_messages"],
        "stats.total_messages": results["stats"]["total_messages"],
        "stats.participants": results["stats"]["participants"],
        "stats.date_range.start": results["stats"]["date_range"]["start"],
        "stats.date_range.end": results["stats"]["date_range"]["end"],
        "sentiment.avg_compound": results["sentiment"]["avg_compound"],
    }


def _read_golden(path: Path) -> dict[str, str]:
    """Read the long-format metric,value golden CSV into a dict."""
    golden: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        metric, _, value = line.partition(",")
        golden[metric.strip()] = value.strip()
    return golden


def _assert_parity(golden_name: str, fixture: Path) -> None:
    results = run_pipeline(fixture, _console(), nlp_enabled=False)
    projected = project(results)
    golden = _read_golden(GOLDEN_DIR / golden_name)

    # The projection key set must match the golden exactly — a drift in the
    # contract shape (added/renamed/removed metric) fails here first.
    assert set(projected) == set(golden), (
        f"projection keys {sorted(projected)} != golden keys {sorted(golden)}"
    )

    for metric, live in projected.items():
        expected = golden[metric]
        if metric in FLOAT_METRICS:
            assert live is not None, f"{metric} must never be None on a parsed chat"
            assert float(live) == pytest.approx(
                float(expected), rel=1e-4, abs=1e-6
            ), f"{metric}: live {live} vs golden {expected}"
        else:
            assert str(live) == expected, (
                f"{metric}: live {live!r} vs golden {expected!r}"
            )


def test_golden_parity_whatsapp():
    """Live pipeline matches the committed golden for the WhatsApp fixture."""
    _assert_parity("whatsapp_sample.csv", DATA / "whatsapp_sample.txt")


def test_golden_parity_telegram():
    """Live pipeline matches the committed golden for the Telegram fixture."""
    _assert_parity("telegram_sample.csv", DATA / "telegram_sample.json")