"""Pipeline orchestration for the chat-analyzer CLI (research Pattern 1).

run_pipeline is the ONLY orchestration path: parse -> canonical DataFrame ->
EDA + VADER sentiment -> base64 PNG charts -> AnalysisResults contract.

Design rules honored here:
- matplotlib is pinned to the Agg backend BEFORE anything imports pyplot
  (Pitfall 7) — done lazily inside run_pipeline so importing this module
  never pulls matplotlib into the interpreter (Anti-Pattern 2).
- Analysis modules (eda/sentiment/visualization) are imported lazily inside
  the analysis stage, whose stdout prints are captured (Pitfall 5).
- A chart crash degrades to an empty string — it must never kill the report
  (Pitfall 6 degrade spirit).
"""

from __future__ import annotations

import base64
import contextlib
import io
import logging
from pathlib import Path

from chat_analyzer.cli.contracts import AnalysisResults, ParseReport
from chat_analyzer.ingest.ingestion import messages_to_dataframe

logger = logging.getLogger(__name__)


def fig_to_data_uri(fig) -> str:
    """Encode a matplotlib figure as a base64 PNG data URI (D-12)."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)  # never leak figures between runs
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _safe_chart(fig) -> str:
    """Encode a chart or degrade to an empty string on failure."""
    try:
        return fig_to_data_uri(fig)
    except Exception:
        logger.exception("chart encoding failed; substituting empty string")
        return ""


def run_pipeline(path: Path, console) -> AnalysisResults:
    """Parse, analyze and assemble the full AnalysisResults for one export."""
    import matplotlib

    matplotlib.use("Agg")  # headless first — before any pyplot import (Pitfall 7)

    source = ""
    rows: list[dict] = []
    counts: dict = {}

    with console.status("Parsing chat...", spinner="line"):
        if path.suffix.lower() == ".txt":
            from chat_analyzer.parser.whatsapp_parser import WhatsAppParser

            rows, counts = WhatsAppParser().parse_file_with_report(str(path))
            source = "whatsapp"
        elif path.suffix.lower() == ".json":
            from chat_analyzer.parser.telegram_parser import (
                parse_telegram_chat_with_report,
            )

            rows, counts = parse_telegram_chat_with_report(str(path))
            source = "telegram"
        else:
            raise ValueError(
                f"Unsupported file type: {path.suffix} — expected .txt (WhatsApp) "
                "or .json (Telegram)"
            )

    # D-05 / CLI-03: surface the parsed count immediately after parsing
    senders = {row.get("sender") or row.get("author") for row in rows}
    console.print(
        f"[OK] Parsed {counts['parsed_messages']} messages from {len(senders)} participants"
    )
    if counts["skipped_lines"] > 0:
        console.print(
            f"[WARN] Skipped {counts['skipped_lines']} lines that couldn't be parsed"
        )

    parse_report = ParseReport(source=source, **counts)

    df = messages_to_dataframe(rows)
    if df.empty:
        raise ValueError("No messages could be parsed from this file")

    with console.status("Computing insights...", spinner="line"):
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            from chat_analyzer.analysis.eda import ChatEDA

            eda = ChatEDA(df)
            summary = eda.generate_comprehensive_summary()
            volume = eda.analyze_message_volume()
            dynamics = eda.analyze_conversation_dynamics()
            content = eda.analyze_content()

            from chat_analyzer.analysis.sentiment import (
                add_sentiment_analysis,
                get_sentiment_summary,
            )

            df_sent = add_sentiment_analysis(df)
            sent_summary = get_sentiment_summary(df_sent)

            from chat_analyzer.utils.visualization import ChatVisualizer

            viz = ChatVisualizer()
            charts = {
                "timeline": _safe_chart(viz.plot_message_timeline(df, resample_freq="D")),
                "activity": _safe_chart(viz.plot_activity_heatmap(df)),
                "participants": _safe_chart(viz.plot_user_activity(df, top_n=10)),
                "sentiment": _safe_chart(
                    viz.plot_sentiment_timeline(
                        df_sent, sentiment_score_col="vader_compound"
                    )
                ),
            }
        if captured.getvalue():
            logger.debug("Captured analysis-stage output:\n%s", captured.getvalue())

    from chat_analyzer.cli.adapters import adapt

    return adapt(
        source, parse_report, df, summary, volume, dynamics, content, sent_summary, charts
    )
