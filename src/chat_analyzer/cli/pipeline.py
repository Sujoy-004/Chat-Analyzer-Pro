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


def stage_status(console, label: str):
    """Stage narration (D-05 / CLI-03).

    On a real terminal, rich's Status shows the ASCII 'line' spinner. When
    stdout is piped/not a tty, rich Status renders NOTHING — degrade to a
    plain '[OK] <stage>' line so stage narration still reaches the user
    (and captured output in CI/tests).
    """
    if console.is_terminal:
        return console.status(label, spinner="line")
    console.print(f"[OK] {label}...")
    return contextlib.nullcontext()


def run_pipeline(path: Path, console) -> AnalysisResults:
    """Parse, analyze and assemble the full AnalysisResults for one export."""
    import matplotlib

    matplotlib.use("Agg")  # headless first — before any pyplot import (Pitfall 7)

    source = ""
    rows: list[dict] = []
    counts: dict = {}

    with stage_status(console, "Parsing chat"):
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

    with stage_status(console, "Computing insights"):
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            from chat_analyzer.analysis import sentiment as _sentiment

            # Plan contract (A6): the CLI always uses the VADER sentiment
            # path. A pre-existing transformers install in the env would
            # otherwise make initialize_analyzers() load a HF model on every
            # run (slow per-message inference + stderr warnings) — pin the
            # flag before the analyzer decides (consensus degrades to VADER,
            # exactly as in a clean base install).
            _sentiment.TRANSFORMERS_AVAILABLE = False

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

            # Always-on analysis (D-07/D-07b): relationship health and the
            # network graph are pandas/numpy/networkx/matplotlib only — no
            # torch, no [nlp] extra. Both are computed inside the
            # redirect_stdout capture so module prints never reach the user.
            from chat_analyzer.analysis.network_graph import (
                analyze_network,
                network_figure,
            )
            from chat_analyzer.analysis.relationship_health import (
                analyze_relationship_health,
            )

            health_res = analyze_relationship_health(df)
            network_res = analyze_network(df)

            from chat_analyzer.utils.visualization import ChatVisualizer

            viz = ChatVisualizer()

            # The health trend chart needs a 'health_score' column — the
            # rolling-health gamification output provides exactly that
            # (date -> timestamp rename). Fall back to the raw df (blank
            # figure) when the rolling series is empty.
            rolling = health_res.get("rolling_health")
            if rolling is not None and not rolling.empty:
                health_trend_df = rolling.rename(columns={"date": "timestamp"})
            else:
                health_trend_df = df

            charts = {
                "timeline": _safe_chart(viz.plot_message_timeline(df, resample_freq="D")),
                "activity": _safe_chart(viz.plot_activity_heatmap(df)),
                "participants": _safe_chart(viz.plot_user_activity(df, top_n=10)),
                "sentiment": _safe_chart(
                    viz.plot_sentiment_timeline(
                        df_sent, sentiment_score_col="vader_compound"
                    )
                ),
                "health": _safe_chart(viz.plot_relationship_health_trend(health_trend_df)),
                "network": _safe_chart(network_figure(df)),
            }
        if captured.getvalue():
            logger.debug("Captured analysis-stage output:\n%s", captured.getvalue())

    from chat_analyzer.cli.adapters import adapt

    return adapt(
        source,
        parse_report,
        df,
        summary,
        volume,
        dynamics,
        content,
        sent_summary,
        charts,
        health=health_res,
        network=network_res,
    )
