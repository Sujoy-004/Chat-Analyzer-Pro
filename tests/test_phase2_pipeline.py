"""Phase 2 pipeline orchestration tests (Pattern 1, D-05/D-16, LOW #9).

Exercises chat_analyzer.cli.pipeline.run_pipeline end-to-end on the sample
exports plus edge cases: an all-skipped file raising a friendly ValueError,
unsupported suffixes, emoji-print capture during analysis (Pitfall 5), the
Agg headless backend, and the None avg_response_time single-message edge
(LOW #9).
"""

import io
from pathlib import Path

import matplotlib
import pytest
from rich.console import Console

from chat_analyzer.cli.adapters import adapt
from chat_analyzer.cli.contracts import ParseReport
from chat_analyzer.cli.pipeline import run_pipeline

DATA = Path(__file__).resolve().parents[1] / "data" / "sample_chats"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False)


def test_whatsapp_e2e():
    """The WhatsApp sample produces a complete AnalysisResults."""
    results = run_pipeline(DATA / "whatsapp_sample.txt", _console())
    assert results["source"] == "whatsapp"
    assert results["parse"]["parsed_messages"] == 27
    assert results["parse"]["skipped_lines"] == 0
    assert results["stats"]["total_messages"] == 27
    assert results["stats"]["participants"] == 2
    assert results["stats"]["date_range"]["start"] == "2023-12-25"
    assert results["stats"]["date_range"]["end"] == "2023-12-27"
    assert len(results["participants"]) == 2
    assert results["content"]["top_words"]
    assert results["sentiment"]["distribution"]
    assert set(results["charts"]) == {"timeline", "activity", "participants", "sentiment"}
    for uri in results["charts"].values():
        assert uri.startswith("data:image/png;base64,")
    assert results["insights"] and all(isinstance(i, str) and i for i in results["insights"])


def test_telegram_e2e():
    """The Telegram sample produces a complete AnalysisResults."""
    results = run_pipeline(DATA / "telegram_sample.json", _console())
    assert results["source"] == "telegram"
    assert results["parse"]["parsed_messages"] == 5
    assert results["stats"]["total_messages"] == 5


def test_all_skipped_raises_friendly():
    """A file with zero parseable messages raises a friendly ValueError."""
    with pytest.raises(ValueError, match="No messages could be parsed"):
        run_pipeline(FIXTURES / "whatsapp_all_skipped.txt", _console())


def test_no_emoji_print_pollution(capsys):
    """Analysis-stage prints (emoji lines) never reach the terminal stdout."""
    run_pipeline(DATA / "whatsapp_sample.txt", _console())
    captured = capsys.readouterr().out
    for token in ("🚀", "✅", "🔍", "Running VADER"):
        assert token not in captured, token


def test_unsupported_format():
    """Non .txt/.json suffixes raise an Unsupported error."""
    with pytest.raises(ValueError, match="Unsupported"):
        run_pipeline(Path("chat.pdf"), _console())


def test_agg_headless():
    """run_pipeline works headless — matplotlib backend is Agg during the run."""
    run_pipeline(DATA / "whatsapp_sample.txt", _console())
    assert matplotlib.get_backend().upper() == "AGG"


def test_single_message_no_response_time():
    """LOW #9: a single-message chat has no avg_response_time — insights
    never print 'None' and the response insight reads 'no measurable'."""
    from chat_analyzer.analysis.eda import ChatEDA
    from chat_analyzer.analysis.sentiment import (
        add_sentiment_analysis,
        get_sentiment_summary,
    )
    from chat_analyzer.ingest.ingestion import messages_to_dataframe

    df = messages_to_dataframe(
        [{"datetime": "2025-09-15T09:45:00", "sender": "A", "message": "just me here"}]
    )
    eda = ChatEDA(df)
    summary = eda.generate_comprehensive_summary()
    volume = eda.analyze_message_volume()
    dynamics = eda.analyze_conversation_dynamics()
    dynamics.pop("avg_response_time", None)  # simulate the edge: key absent
    content = eda.analyze_content()
    df_sent = add_sentiment_analysis(df)
    sent = get_sentiment_summary(df_sent)

    results = adapt(
        "whatsapp",
        ParseReport(source="whatsapp", parsed_messages=1),
        df, summary, volume, dynamics, content, sent, {},
    )
    joined = "\n".join(results["insights"])
    assert "None" not in joined
    assert "no measurable" in joined
