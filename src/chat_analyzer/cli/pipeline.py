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
- Narration is a live determinate progress bar on a real terminal (D-12)
  that degrades to plain '[OK] <label>' stage lines off-tty (Pitfall 8).
"""

from __future__ import annotations

import base64
import contextlib
import io
import logging
from pathlib import Path

from chat_analyzer.cli.chart_json import build_chart_specs, build_emotion_spec
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


def stage(console, progress, task_id: int | None, label: str):
    """Narrate one run stage: live bar on tty, plain '[OK]' line off-tty.

    D-12 + Pitfall 8: on a real terminal the determinate progress task is
    advanced to the stage label (its body runs while the bar shows it); when
    stdout is piped/not a tty (CI, tests) this falls back to stage_status so
    narration still reaches captured output. Stage labels MUST stay exactly
    "Parsing chat" / "Computing insights" / "Analyzing emotions" /
    "Generating narrative" — test_stage_narration_and_order asserts the
    first two (plus main.py's "Writing report") verbatim.
    """
    if progress is None:
        return stage_status(console, label)
    progress.update(task_id, description=label, advance=1)
    return contextlib.nullcontext()


def _tier_b_digest(df, narrative: dict):
    """Compose the compact Tier-B signal digest (B5) for flan-t5-small.

    The generative narrative summarizes a small ASCII digest driven by the
    Tier A observations plus one factual line — never raw chat messages.
    This keeps the model's input tiny, ASCII-only and private (the same
    "signal digest" constraint DEFERRED 2.5 sets).
    """
    import pandas as pd

    lines: list[str] = []
    for obs in narrative.get("observations") or []:
        text = str(obs.get("text", "")).strip()
        if text:
            lines.append(text)
    if not lines and not df.empty:
        lines.append(
            f"This conversation has {df['sender'].nunique()} participants "
            f"and {len(df)} messages in total."
        )
    if not lines:
        lines.append("This conversation has very few messages to comment on.")
    rows = [{"sender": "analysis", "message": line} for line in lines]
    return pd.DataFrame(rows)


_CODE_LIKE_TOKENS = ("[", "]", "int(", "input()", "def ", "import ", "for ", "=>", "->")


def _acceptable_narrative_text(text: str) -> bool:
    """WS-6 sanity gate for Tier B output: sane length, no code-like
    tokens, must read as a sentence. Garbage is never surfaced."""
    if not text or len(text) < 5 or len(text) > 250:
        return False
    if any(token in text for token in _CODE_LIKE_TOKENS):
        return False
    return text.rstrip()[-1:] in (".", "!", "?")


def run_pipeline(path: Path, console, nlp_enabled: bool | None = None) -> AnalysisResults:
    """Parse, analyze and assemble the full AnalysisResults for one export."""
    import matplotlib

    matplotlib.use("Agg")  # headless first — before any pyplot import (Pitfall 7)

    # Silent availability gate BEFORE any heavy import (D-02/D-05): the probe
    # never raises and never prompts (no hint — that is main.py's job in
    # 04-03); it only decides whether the emotion/summary stages below run at
    # all (D-06 silent degrade). Computed once up front so the progress task
    # total matches the stages the pipeline actually runs.
    from chat_analyzer.cli import nlp_gate

    # PH2 tier menu: the CLI resolves the tier in main.py and passes a forced
    # nlp_enabled here (True/False). None keeps the old auto-detect probe so
    # direct/test callers stay deterministic without touching the environment.
    nlp_on = nlp_gate.nlp_available(nlp_gate.MODEL_ID) if nlp_enabled is None else nlp_enabled

    # D-12: live determinate progress bar on a real terminal; off-tty the
    # stage() helper degrades to stage_status's plain '[OK]' lines (Pitfall 8).
    if console.is_terminal:
        from rich.progress import Progress

        progress = Progress(console=console, transient=False)
        task_id = progress.add_task("Starting", total=4 if nlp_on else 3)
        progress.start()
    else:
        progress = None
        task_id = None

    try:
        source = ""
        rows: list[dict] = []
        counts: dict = {}

        with stage(console, progress, task_id, "Parsing chat"):
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
            elif path.suffix.lower() == ".zip":
                # Item C: a WhatsApp/Telegram zip export may contain transcripts
                # (.txt -> WhatsApp, .json -> Telegram). The user picks which to
                # analyze; rows/counts come back in the same contract shape.
                from chat_analyzer.cli.zip_input import parse_zip_with_report

                rows, counts, source = parse_zip_with_report(path, console)
            else:
                raise ValueError(
                    f"Unsupported file type: {path.suffix} — expected .txt (WhatsApp), "
                    ".json (Telegram), or .zip (exported archive)"
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

        with stage(console, progress, task_id, "Computing insights"):
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
                volume = eda.analyze_message_volume()
                dynamics = eda.analyze_conversation_dynamics()
                content = eda.analyze_content()
                summary = eda.generate_comprehensive_summary(
                    volume_analysis=volume,
                    dynamics_analysis=dynamics,
                    content_analysis=content,
                )

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

                health_res = analyze_relationship_health(df, include_gamification=False)
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

                # Interactive ECharts specs for the same charts (charts_json).
                # A spec failing to build degrades to the PNG above — each
                # chart is skipped individually and never kills the report.
                charts_json = build_chart_specs(df, df_sent, health_trend_df, network_res)
            if captured.getvalue():
                logger.debug("Captured analysis-stage output:\n%s", captured.getvalue())

        # Tier A narrative (B2): the always-on, pandas-only "what is going on"
        # heuristic. It never raises and needs no heavy deps, so the narrative
        # tab has content even on a base install. observations feed Tier B's
        # digest below (B5); status is overwritten with the real gate state.
        from chat_analyzer.analysis.narrative import analyze_narrative

        narrative = analyze_narrative(df)
        narrative_tier_b = False

        # Gated NLP stages (ANAL-06 emotion / ANAL-08 summary, D-07c). When the
        # gate is OFF they are skipped silently — no error, no prompt, no hint
        # (D-02/D-06): the pipeline always prepares for NLP, availability decides.
        emotion_summary = None
        if nlp_on:
            # Option C: sampled emotion scoring for large chats. Resolve the
            # cap once; when the frame exceeds it, prompt on interactive
            # terminals (default NO = exact) and AUTO-SAMPLE off-tty
            # (piped/CI/tests). The exact path stays untouched below the cap.
            sample_cap = None
            emotion_sample_cap = nlp_gate.emotion_sample_cap()
            if emotion_sample_cap is not None and len(df) > emotion_sample_cap:
                if console.is_terminal:
                    answer = console.input(
                        f"{len(df)} messages — exact emotion scoring can take "
                        f"~2 hours; sampled (up to {emotion_sample_cap}) takes "
                        f"minutes. Sample? [y/N] "
                    )
                    if answer.strip().lower() in ("y", "yes"):
                        sample_cap = emotion_sample_cap
                else:
                    sample_cap = emotion_sample_cap  # AUTO-SAMPLE (piped/CI)
            with stage(console, progress, task_id, "Analyzing emotions"):
                # D-05/Pitfall 4: announce model name + size BEFORE any
                # construction that triggers from_pretrained — and outside the
                # redirect capture so the message reaches piped output too
                # (Pitfall 8). ASCII only, no emoji.
                console.print(
                    "[INFO] Models run locally on your device - "
                    "no chat data leaves this machine."
                )
                if not nlp_gate.model_cached(nlp_gate.MODEL_ID):
                    console.print(
                        "[INFO] Downloading model weights on first "
                        "run - this may take a few minutes."
                    )
                console.print(
                    f"Emotion model: {nlp_gate.MODEL_ID} "
                    f"(~{nlp_gate.EMOTION_MODEL_SIZE_MB} MB)"
                )
                with contextlib.redirect_stdout(io.StringIO()) as captured_nlp:
                    try:
                        from chat_analyzer.analysis.emotion import (
                            EmotionAnalyzer,
                            emotion_figure,
                        )

                        emo_analyzer = EmotionAnalyzer()
                        df_emo = emo_analyzer.analyze_emotions(
                            df, sample_cap=sample_cap
                        )
                        if df_emo.attrs.get("emotion_sample"):
                            # Option C: the summary is computed over the
                            # SCORED sample rows only, and the sample metadata
                            # rides along so adapters/render/report can label
                            # "based on a sample of N of M messages".
                            emotion_summary = emo_analyzer.get_emotion_summary(
                                df_emo[df_emo["emotion_scored"]]
                            )
                            emotion_summary["sampled"] = df_emo.attrs["emotion_sample"]
                        else:
                            emotion_summary = emo_analyzer.get_emotion_summary(df_emo)
                    except Exception:
                        logger.exception("emotion analysis failed; degrading to None")
                        emotion_summary = None
                if captured_nlp.getvalue():
                    logger.debug(
                        "Captured emotion-stage output:\n%s", captured_nlp.getvalue()
                    )
                if emotion_summary is not None:
                    charts["emotion"] = _safe_chart(emotion_figure(emotion_summary))
                    emotion_spec = build_emotion_spec(emotion_summary)
                    if emotion_spec:
                        charts_json["emotion"] = emotion_spec

            with stage(console, progress, task_id, "Generating narrative"):
                console.print(
                    "[INFO] Models run locally on your device - "
                    "no chat data leaves this machine."
                )
                # Tier B narrative (B2, D-05/Pitfall 4): announce the small
                # local model before any construction that could download it.
                if not nlp_gate.model_cached(nlp_gate.TIER_B_MODEL_ID):
                    console.print(
                        "[INFO] Downloading model weights on first "
                        "run - this may take a few minutes."
                    )
                console.print(
                    f"Narrative model: {nlp_gate.TIER_B_MODEL_ID} "
                    f"(~{nlp_gate.TIER_B_MODEL_SIZE_MB} MB)"
                )
                with contextlib.redirect_stdout(io.StringIO()) as captured_nlp:
                    try:
                        # Pitfall 7: construct ONLY now — the ctor downloads
                        # flan-t5-small; degrade instead of failing the run.
                        from chat_analyzer.analysis.summarizer import (
                            ConversationSummarizer,
                        )

                        # Tier B narrative (B2/B5): the same local T5 backend on
                        # the compact signal digest, not raw chat. Any failure
                        # degrades back to the Tier A observations already held
                        # in `narrative` — the run never crashes.
                        tier_res = ConversationSummarizer(
                            model_name=nlp_gate.TIER_B_MODEL_ID,
                            max_length=90,
                            min_length=20,
                        ).summarize_conversation(_tier_b_digest(df, narrative))
                        tier_text = str(tier_res.get("summary", "")).strip()
                        if _acceptable_narrative_text(tier_text):
                            narrative["narrative_summary"] = tier_text
                            narrative_tier_b = True
                    except Exception:
                        logger.exception(
                            "Tier B narrative failed; keeping Tier A observations"
                        )
                if captured_nlp.getvalue():
                    logger.debug(
                        "Captured narrative-stage output:\n%s", captured_nlp.getvalue()
                    )

        # B1: the always-visible status notice reflects the REAL gate state —
        # never a silent skip. Tier A ran regardless; tier_b_generated only
        # after a successful flan-t5-small narrative paragraph above.
        narrative["status"] = {
            "nlp_available": nlp_on,
            "tier_b_generated": narrative_tier_b,
        }

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
            charts_json=charts_json,
            health=health_res,
            network=network_res,
            emotion=emotion_summary,
            narrative=narrative,
        )
    finally:
        if progress is not None:
            progress.stop()
