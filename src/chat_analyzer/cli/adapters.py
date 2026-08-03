"""Adapters between analysis-module dicts and the AnalysisResults contract.

Research Pattern 2 — the ONLY place that knows each module's internal dict
shape (ChatEDA summaries, sentiment summary, chat visualizer charts). Every
access is a defensive .get(): an empty edge-case dict must never KeyError
here. build_insights formats the narrative lead-ins (D-11) — every value
comes from the stats, never "None".
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from chat_analyzer.cli.contracts import AnalysisResults, ParseReport


def adapt(
    source,
    parse: ParseReport,
    df,
    summary,
    volume,
    dynamics,
    content,
    sentiment,
    charts,
    *,
    health=None,
    network=None,
    emotion=None,
    conversation_summary=None,
) -> AnalysisResults:
    """Assemble the AnalysisResults contract from the analysis module dicts.

    health/network/emotion/conversation_summary are keyword-only with None
    defaults (reconciliation note #2) so Phase 2 direct-call tests stay green.
    The always-on health + network blocks arrive in plan 04-01; emotion and
    conversation_summary arrive in plan 04-02.
    """
    total_messages = len(df)

    # --- stats -----------------------------------------------------------
    sender_counts = df["sender"].value_counts()
    participant_list = [str(s) for s in sender_counts.index]

    hourly = volume.get("hourly_activity")
    busiest_day = None
    if hourly is not None and not hourly.empty:
        busiest_day = str(hourly.sum(axis=1).idxmax())

    dt_min = df["datetime"].min()
    dt_max = df["datetime"].max()

    avg_response_time = dynamics.get("avg_response_time")
    if avg_response_time is not None and pd.isna(avg_response_time):
        avg_response_time = None  # LOW #9: single-message chats have no response time

    stats: dict = {
        "total_messages": total_messages,
        "participants": len(participant_list),
        "participant_list": participant_list,
        "date_range": {
            "start": dt_min.strftime("%Y-%m-%d"),
            "end": dt_max.strftime("%Y-%m-%d"),
        },
        "duration_days": (dt_max - dt_min).days + 1,
        "busiest_day": busiest_day,
        "peak_hour": summary.get("activity_patterns", {}).get("peak_hour"),
        "avg_response_time": avg_response_time,
        "media_messages": int(
            df["message"].str.contains("<Media omitted>", case=False, na=False).sum()
        ),
    }

    # --- participants (sorted desc by message count) ---------------------
    participant_dict: dict = {}
    for sender, count in sender_counts.items():
        sender_df = df[df["sender"] == sender]
        participant_dict[str(sender)] = {
            "messages": int(count),
            "avg_message_length": float(sender_df["message_length"].mean()),
            "share_pct": round(float(count) / total_messages * 100, 1),
        }
    participant_dict = dict(
        sorted(participant_dict.items(), key=lambda kv: kv[1]["messages"], reverse=True)
    )

    # --- content ---------------------------------------------------------
    word_freq = content.get("word_frequency") or Counter()
    emoji_freq = content.get("emoji_frequency") or Counter()
    content_block: dict = {
        "top_words": [w for w, _ in word_freq.most_common(15)],
        "top_emojis": [e for e, _ in emoji_freq.most_common(15)],
        "total_words": int(content.get("total_words", 0)),
        "unique_words": int(content.get("unique_words", 0)),
    }

    # --- sentiment -------------------------------------------------------
    sent_dist = sentiment.get("sentiment_distribution") or {}
    vader = (sentiment.get("average_scores") or {}).get("vader_compound") or {}
    sentiment_block: dict = {
        "distribution": {str(k): int(v) for k, v in sent_dist.items()},
        "avg_compound": vader.get("mean"),
        "by_sender": sentiment.get("by_sender") or {},
        "daily_avg": (sentiment.get("temporal_analysis") or {}).get(
            "daily_avg_sentiment"
        )
        or {},
    }

    # --- health + network (always-on, D-07/D-07b) ------------------------
    health_block = _build_health_block(health) if health is not None else None
    network_block = _build_network_block(network) if network is not None else None

    return AnalysisResults(
        source=parse.source,
        parse={
            "total_lines": parse.total_lines,
            "parsed_messages": parse.parsed_messages,
            "skipped_lines": parse.skipped_lines,
            "system_messages": parse.system_messages,
        },
        stats=stats,
        participants=participant_dict,
        content=content_block,
        sentiment=sentiment_block,
        health=health_block,
        network=network_block,
        charts=dict(charts),
        insights=build_insights(
            stats, participant_dict, content_block, sentiment_block,
            health_block, network_block,
        ),
        report_path="",
    )


def _build_health_block(health: dict) -> dict:
    """Extract serializable scalars from analyze_relationship_health (Pattern 3).

    The module returns a 'prepared_data' DataFrame plus nested dicts — the
    DataFrame is NEVER leaked into AnalysisResults (Jinja-consumed).
    """
    health_score = health.get("health_score") or {}
    initiator = health.get("initiator_analysis") or {}
    response = health.get("response_analysis") or {}
    dominance = health.get("dominance_analysis") or {}
    return {
        "overall_score": health_score.get("overall_health_score"),
        "grade": health_score.get("grade"),
        "components": health_score.get("components")
        or health_score.get("component_scores")
        or {},
        "initiator_balance": initiator.get("balance_score"),
        "total_conversations": initiator.get("total_conversations"),
        "avg_response_minutes": response.get("overall_avg_response_minutes"),
        "response_balance": response.get("response_balance_score"),
        "composite_dominance": dominance.get("composite_dominance_score"),
    }


def _build_network_block(network: dict) -> dict:
    """Extract serializable scalars from analyze_network (Pattern 3).

    The module returns a networkx.DiGraph and an interaction-matrix
    DataFrame inside patterns — neither is leaked into AnalysisResults.
    """
    metrics = network.get("metrics") or {}
    patterns = network.get("patterns") or {}
    subgroups = network.get("subgroups") or {}
    return {
        "node_count": metrics.get("num_nodes"),
        "edge_count": metrics.get("num_edges"),
        "density": metrics.get("density"),
        "reciprocity": patterns.get("reciprocity_score")
        or metrics.get("reciprocity"),
        "strongest_connections": patterns.get("strongest_connections"),
        "key_participants": network.get("key_participants") or {},
        "subgroup_count": len(subgroups),
    }


def build_insights(stats, participants, content, sentiment, health=None, network=None) -> list[str]:
    """Narrative lead-ins, one per report tab (D-11).

    Natural-language sentences driven entirely by stats values — no hardcoded
    numbers, and never the string "None" (LOW #9: avg_response_time may be
    None on single-message chats). Health and network lead-ins slot in at
    tab indices 5 and 6 (after the five Phase-2 tabs); the duration and
    busiest-hour sentences follow them.
    """
    insights: list[str] = []

    busiest_day = stats.get("busiest_day")
    if busiest_day:
        insights.append(f"Most messages land on {busiest_day}.")

    if participants:
        top_sender, top_data = next(iter(participants.items()))
        insights.append(
            f"{top_sender} is the most active participant, "
            f"sending {top_data.get('share_pct', 0)}% of all messages."
        )

    avg = stats.get("avg_response_time")
    if avg:
        insights.append(f"Replies take on average {avg:.0f} minutes when they come at all.")
    else:
        insights.append("Replies take no measurable time — mostly one-off messages.")

    top_words = content.get("top_words") or []
    if top_words:
        insights.append(f"The most-used word is '{top_words[0]}'.")

    dist = sentiment.get("distribution") or {}
    if dist:
        dominant = max(dist, key=dist.get)
        pct = float(dist[dominant]) / max(sum(dist.values()), 1) * 100
        insights.append(f"The overall tone leans {dominant} ({pct:.0f}% of messages).")

    # --- health + network lead-ins (D-11) — tab indices 5 and 6 -----------
    score = (health or {}).get("overall_score")
    grade = (health or {}).get("grade")
    if score is not None and grade is not None:
        insights.append(f"This conversation scores {grade} — overall health {score:.2f}.")

    density = (network or {}).get("density")
    strongest = (network or {}).get("strongest_connections") or []
    if density is not None:
        if strongest and strongest[0].get("from") and strongest[0].get("to"):
            top = strongest[0]
            insights.append(
                f"The strongest connection is {top['from']} to {top['to']} "
                f"({top['interactions']} interactions); the network has density "
                f"{density:.2f}."
            )
        else:
            insights.append(f"The conversation network has density {density:.2f}.")

    insights.append(
        f"This conversation spans {stats.get('duration_days', 0)} days "
        f"and {stats.get('total_messages', 0)} messages."
    )

    peak = stats.get("peak_hour")
    if peak is not None:
        insights.append(f"The busiest hour is {peak}:00.")

    return insights[:11]
