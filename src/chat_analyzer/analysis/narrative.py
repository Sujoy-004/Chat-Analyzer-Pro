"""Tier A narrative analysis for Chat-Analyzer-Pro (DEFERRED §2.2).

The always-on, heuristic "what is going on" layer. It turns basic chat
signals into clearly-disclaimed, hedged observations about the conversation:
arc (casual -> planning/decision), driver (one side pushes questions or
sensitive topics forward) and reciprocity (turn-taking balance).

This is Tier A: pandas + re only, never imports anything from
``chat_analyzer.cli`` and never touches torch/transformers. Tier B (local
generative summary via flan-t5-small) is defined in DEFERRED §2.5 and fills
the ``status`` block once available — here ``status`` always reports that
only the Tier A path ran.

Every observation text is speculative by design (DEFERRED §2.7, risk guard):
it starts with a hedge ("Possibly", "May", "Might", "This could") and never
states a bare factual claim. Confidence comes from a documented
conservative table (see ``_confidence``).

Observation kinds (stable, small set — DEFERRED §2.2):
    arc, driver, reciprocity, engagement   — emitted by Tier A
    sentiment, visibility                  — reserved for Tier B

Banglish keyword lexicons (DEFERRED §2.3 / B3). Whole-token matches only:

    CARE_TRUST_BANGLISH    = ("bhalobashi", "bishash", "dukho",
                              "khyal", "kharap")     care / trust signals
    PLANNING_BANGLISH      = ("miting", "plen", "chakri", "dilain", "kol")
                              planning / decision signals
    QUESTION_BANGLISH      = ("ki", "keno", "kemon", "koto", "kothay",
                              "kobe")                 Banglish question words

ASCII-only everywhere the module exposes text. ``analyze_narrative`` never
raises: an empty frame, one sender, one message, all-media or NaN senders
all degrade to the base dict with empty observations.
"""

import itertools
import re
from typing import Any

import pandas as pd

# --- Stable observation vocabulary ---------------------------------------------

OBSERVATION_KINDS = ("arc", "driver", "reciprocity", "engagement", "sentiment", "visibility")

HEDGES = ("Possibly", "May", "Might", "This could")

# --- Banglish + English lexicons -----------------------------------------------

QUESTION_BANGLISH = ("ki", "keno", "kemon", "koto", "kothay", "kobe")

CARE_TRUST_BANGLISH = ("bhalobashi", "bishash", "dukho", "khyal", "kharap")

PLANNING_BANGLISH = ("miting", "plen", "chakri", "dilain", "kol")

_CARE_TRUST_RE = re.compile(
    r"\b(?:" + "|".join(
        (
            "trust",
            "believe",
            "care",
            "love",
            "miss",
            "missed",
            "worried",
            "hurt",
            "sorry",
            "forgive",
            "promise",
            "secret",
            "confidential",
        )
        + CARE_TRUST_BANGLISH
    ) + r")\b",
    re.IGNORECASE,
)

_PLANNING_RE = re.compile(
    r"\b(?:"
    + "|".join(
        (
            "meeting",
            "plan",
            "deadline",
            "decision",
            "call",
            "job",
            "interview",
            "schedule",
            "appointment",
            "contract",
            "project",
            "apply",
        )
        + PLANNING_BANGLISH
    )
    + r")\b",
    re.IGNORECASE,
)

_QUESTION_RE = re.compile(
    r"(?:\b(?:" + "|".join(QUESTION_BANGLISH + ("what", "why", "who", "whom", "when",
                             "where", "which", "how")) + r")\b)|\?+",
    re.IGNORECASE,
)


def _ascii_safe(value: Any) -> str:
    """Return an ASCII-only string label, falling back to a neutral phrase.

    Args:
        value: Any value (sender name, etc.) to convert.

    Returns:
        An ASCII-only string; never empty.
    """
    cleaned = "".join(ch for ch in str(value) if ord(ch) < 128).strip()
    return cleaned or "one participant"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a safe copy with sender/message/message_length normalized.

    Args:
        df: Raw canonical DataFrame (datetime/sender/message/message_length).

    Returns:
        Copy with NaN senders renamed to "unknown", messages coerced to str
        and a numeric ``message_length`` column guaranteed to exist.
    """
    norm = df.copy()
    norm["sender"] = norm["sender"].fillna("unknown").astype(str)
    norm["message"] = norm["message"].fillna("").astype(str)
    if "message_length" not in norm.columns:
        norm["message_length"] = norm["message"].str.len()
    else:
        norm["message_length"] = pd.to_numeric(norm["message_length"], errors="coerce").fillna(0)
    return norm


def _confidence(score: float, n: int) -> str:
    """Map a signal strength and message volume to a confidence level.

    Conservative threshold table (DEFERRED §2.2):
        | level  | condition                                |
        |--------|-------------------------------------------|
        | high   | ``n >= 30`` and ``abs(score) >= 0.5``      |
        | medium | ``n >= 10`` and ``abs(score) >= 0.3``      |
        | low    | anything else                              |

    Args:
        score: Signed signal magnitude (e.g. density delta, ratio delta).
        n: Number of messages backing the signal.

    Returns:
        One of "high", "medium", "low".
    """
    magnitude = abs(score)
    if n >= 30 and magnitude >= 0.5:
        return "high"
    if n >= 10 and magnitude >= 0.3:
        return "medium"
    return "low"


def _message_length_asymmetry(df: pd.DataFrame) -> dict[str, Any]:
    """Return the mean-length ratio between the two most active senders.

    Compares the mean ``message_length`` of the two senders with the most
    messages. Ratio is always >= 1.0 and tags the longer-writing sender.

    Args:
        df: Normalized chat DataFrame.

    Returns:
        Dict with sender_a, sender_b, ratio and longer_sender, or an empty
        dict when no pairwise comparison is possible (0-1 unique senders).
    """
    if df.empty or df["sender"].nunique() < 2:
        return {}
    sender_counts = df["sender"].value_counts()
    if len(sender_counts) < 2:
        return {}
    first, second = sender_counts.index[0], sender_counts.index[1]
    mean_lengths = df.groupby("sender")["message_length"].mean()
    len_a, len_b = mean_lengths.get(first, 0.0), mean_lengths.get(second, 0.0)
    if min(len_a, len_b) <= 0:
        return {"sender_a": first, "sender_b": second, "ratio": 1.0,
                "longer_sender": None}
    if len_a >= len_b:
        ratio, longer = len_a / len_b, first
    else:
        ratio, longer = len_b / len_a, second
    return {
        "sender_a": first,
        "sender_b": second,
        "ratio": round(ratio, 4),
        "longer_sender": longer,
    }


def _question_ratio_per_sender(df: pd.DataFrame) -> dict[str, Any]:
    """Return question-marker ratios per sender and a "much more" flag.

    A message "has a question" when it contains a ``?`` or a whole-token
    English wh-word or Banglish marker (ki/keno/kemon/koto/kothay/kobe).

    Args:
        df: Normalized chat DataFrame.

    Returns:
        Dict: ``ratios`` (sender -> float), ``much_more`` bool, the leading
        question-asker, the ratio gap and the total number of questioned rows.
        ``much_more`` is True only when the leading ratio clears 0.4 and
        beats the second-best ratio by >= 0.25 with both sides >= 2 messages.
    """
    totals = df.groupby("sender").size()
    totals = totals[totals >= 1]
    if df.empty or len(totals) < 2:
        return {"ratios": {}, "much_more": False, "leading_sender": None,
                "gap": 0.0, "total_questions": 0}
    flagged = df["message"].str.contains(_QUESTION_RE, na=False)
    question_counts = flagged.groupby(df["sender"]).sum().astype(int)
    ratio = (question_counts / totals).round(4)
    total_questions = int(question_counts.sum())
    ratios = ratio.to_dict()
    ranked = ratio.sort_values(ascending=False)
    if len(ranked) < 2:
        return {"ratios": ratios, "much_more": False, "leading_sender": None,
                "gap": 0.0, "total_questions": total_questions}
    top_name, top_ratio = ranked.index[0], float(ranked.iloc[0])
    second_ratio = float(ranked.iloc[1])
    gap = round(top_ratio - second_ratio, 4)
    much_more = (
        top_ratio >= 0.4
        and gap >= 0.25
        and int(totals.get(top_name, 0)) >= 2
        and int(totals.get(ranked.index[1], 0)) >= 2
    )
    return {
        "ratios": ratios,
        "much_more": much_more,
        "leading_sender": top_name if much_more else None,
        "gap": gap,
        "total_questions": total_questions,
    }


def _reciprocity(df: pd.DataFrame) -> dict[str, Any]:
    """Compute turn-taking reciprocity as an adjacent-alternation share.

    Reciprocity is the fraction of adjacent sender pairs where the sender
    changes (i.e. the other side replied in the flow). Scores near 1.0 for a
    balanced back-and-forth; near 0 for stacked monologues.

    Args:
        df: Normalized chat DataFrame.

    Returns:
        Dict ``reciprocity`` (0.0-1.0), ``alternating_pairs``, ``total_pairs``.
    """
    if df.empty:
        return {"reciprocity": 1.0, "alternating_pairs": 0, "total_pairs": 0}
    senders = df["sender"].tolist()
    total_pairs = len(senders) - 1
    if total_pairs <= 0:
        return {"reciprocity": 1.0, "alternating_pairs": 0, "total_pairs": 0}
    alternating = sum(a != b for a, b in itertools.pairwise(senders))
    return {
        "reciprocity": round(alternating / total_pairs, 4),
        "alternating_pairs": alternating,
        "total_pairs": total_pairs,
    }


def _sensitive_topic_driver(df: pd.DataFrame) -> dict[str, Any]:
    """Identify who introduces sensitive/planning topics and how strongly.

    Each row counts as a ``topic`` message if it is a spec/imply-ng topic hit
    from the care/trust (SENSITIVE) or planning/decision lexicons. A driver
    score = topic messages / total messages per sender; the difference between
    the top and second sender is the "strength".

    Args:
        df: Normalized chat DataFrame.

    Returns:
        Dict with per-sender broken-down hits, the leader (or None) and the
        strength of their lead. ``total_topic_messages`` backs confidence.
    """
    if df.empty or df["sender"].nunique() < 2:
        return {
            "per_sender": {},
            "leader": None,
            "strength": 0.0,
            "total_topic_messages": 0,
        }
    sensitive_hits = df["message"].str.contains(_CARE_TRUST_RE, na=False)
    planning_hits = df["message"].str.contains(_PLANNING_RE, na=False)
    topic_any = sensitive_hits | planning_hits
    totals = df.groupby("sender").size()
    topic_counts = topic_any.groupby(df["sender"]).sum().astype(int)
    sensit_counts = sensitive_hits.groupby(df["sender"]).sum().astype(int)
    planning_counts = planning_hits.groupby(df["sender"]).sum().astype(int)

    per_sender: dict[str, dict[str, Any]] = {}
    densities: dict[str, float] = {}
    for name in df["sender"].unique():
        msgs = int(totals.get(name, 0))
        count = int(topic_counts.get(name, 0))
        per_sender[name] = {
            "sensitive": int(sensit_counts.get(name, 0)),
            "planning": int(planning_counts.get(name, 0)),
            "topic_messages": count,
            "density": round(count / msgs, 4) if msgs else 0.0,
        }
        densities[name] = per_sender[name]["density"]

    total_topic_messages = int(topic_counts.sum())
    leader, strength = None, 0.0
    if len(densities) >= 2 and total_topic_messages > 0:
        ordered = sorted(densities.items(), key=lambda kv: kv[1], reverse=True)
        leader_name, leader_density = ordered[0]
        second_density = ordered[1][1]
        strength = round(leader_density - second_density, 4)
        if strength >= 0.3 and int(topic_counts.get(leader_name, 0)) >= 1:
            leader = leader_name
    return {
        "per_sender": per_sender,
        "leader": leader,
        "strength": strength,
        "total_topic_messages": total_topic_messages,
    }


def _arc_signal(df: pd.DataFrame) -> dict[str, Any] | None:
    """Return an arc signal when topic density rises in the later half.

    Splits the chat at its midpoint, counts the share of topic-carrying
    messages (sensitive/planning lexicons) per half and keeps a signal only
    when the second half is clearly denser. Never a later density alone.

    Args:
        df: Normalized chat DataFrame.

    Returns:
        Dict with first/later density, delta and total hits, or None when
        there is no reliable arc (few messages / no density shift).
    """
    n = len(df)
    if n < 8:
        return None
    hits = df["message"].str.contains(_CARE_TRUST_RE, na=False) | df["message"].str.contains(
        _PLANNING_RE, na=False
    )
    half = n // 2
    first_density = float(hits.iloc[:half].mean())
    later_density = float(hits.iloc[half:].mean())
    delta = round(later_density - first_density, 4)
    total_hits = int(hits.sum())
    if later_density <= 0.2 or delta < 0.2 or total_hits < 4:
        return None
    return {
        "first_density": first_density,
        "later_density": later_density,
        "delta": delta,
        "total_hits": total_hits,
    }


def _build_observations(df: pd.DataFrame, signals: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Compose hedged observations from the extractor signal set.

    Deterministic and conservative (DEFERRED §2.7). Every ``text`` leads with
    a classic hedge and every observation carries a stable ``kind`` plus a
    confidence level from ``_confidence`` thresholds. Well-balanced input
    produces no strong claim.

    Args:
        df: Normalized chat DataFrame (>= 2 unique senders).
        signals: Optional precomputed signal dict that overrides extractors
            (values keyed by extractor name) for Tier B reuse.

    Returns:
        List of observation dicts: ``{"text", "kind", "confidence"}``.
    """
    observations: list[dict[str, Any]] = []

    def override(key: str, default: Any) -> Any:
        if signals and key in signals:
            return signals[key]
        return default

    arc = _arc_signal(df)
    if arc is not None:
        observations.append({
            "text": "Possibly this conversation shifts from casual chat toward "
                    "planning/decision, since topic-loaded messages are denser "
                    "in the later half.",
            "kind": "arc",
            "confidence": _confidence(arc["delta"], arc["total_hits"]),
        })

    question = override("question_ratio", _question_ratio_per_sender(df))
    if isinstance(question, dict) and question.get("much_more") and question.get("leading_sender"):
        observations.append({
            "text": f"Possibly {_ascii_safe(question['leading_sender'])} drives this "
                    "conversation forward by asking questions much more often.",
            "kind": "driver",
            "confidence": _confidence(question.get("gap", 0.0),
                                      question.get("total_questions", 0)),
        })

    driver_signal = override("sensitive_topic_driver", _sensitive_topic_driver(df))
    if isinstance(driver_signal, dict) and driver_signal.get("leader") and driver_signal.get("strength", 0.0) >= 0.3:
        observations.append({
            "text": f"Possibly {_ascii_safe(driver_signal['leader'])} more actively "
                    "introduces sensitive or planning topics.",
            "kind": "driver",
            "confidence": _confidence(driver_signal["strength"],
                                     driver_signal.get("total_topic_messages", 0)),
        })

    reciprocity = override("reciprocity", _reciprocity(df))
    if isinstance(reciprocity, dict) and reciprocity.get("alternating_pairs", 0) >= 4 and reciprocity.get("reciprocity", 1.0) < 0.45:
        strength = round(0.6 - reciprocity["reciprocity"], 4)
        observations.append({
            "text": "This could be a somewhat one-sided chat, with one side "
                    "replying to meet other less often.",
            "kind": "reciprocity",
            "confidence": _confidence(strength, reciprocity["total_pairs"]),
        })

    asymmetry = override("message_length_asymmetry", _message_length_asymmetry(df))
    if isinstance(asymmetry, dict) and asymmetry.get("ratio", 1.0) >= 2.5 and asymmetry.get("longer_sender"):
        observations.append({
            "text": f"Possibly {_ascii_safe(asymmetry['longer_sender'])} writes much "
                    "longer messages, which may signal stronger engagement.",
            "kind": "engagement",
            "confidence": _confidence(asymmetry["ratio"] - 1.0, len(df)),
        })

    return observations


def analyze_narrative(df: pd.DataFrame, *, signals: dict[str, Any] | None = None) -> dict[str, Any]:
    """Produce the Tier A narrative analysis for a chat DataFrame.

    Heuristic, always-on narrative. Degrades to the base dict (empty
    observations, empty summary) for empty frames, <2 unique senders, a
    single message or NaN senders — it never raises.

    Args:
        df: Canonical chat DataFrame (datetime, sender, message,
            message_length; see `chat_analyzer.ingest.ingestion`).
        signals: Optional precomputed signal dict. If provided, extractor
            results are overridden by values keyed by extractor name
            (reserved for Tier B's signal digest).

    Returns:
        Dict with ``observations``, ``narrative_summary``, ``confidence``
        fields: {"observations", "narrative_summary", "tier", "speculative",
        "status"} per DEFERRED §2.2.
    """
    base = {
        "observations": [],
        "narrative_summary": "",
        "tier": "A",
        "speculative": True,
        "status": {"nlp_available": False, "tier_b_generated": False},
    }
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return base
    if "sender" not in df.columns or "message" not in df.columns:
        return base
    normalized = _normalize(df)
    if normalized["sender"].nunique() < 2 or len(normalized) < 2:
        return base
    try:
        observations = _build_observations(normalized, signals)
    except Exception:  # noqa: BLE001 - analysis never crashes the run (DEFERRED §2.7)
        observations = []
    narrative_summary = " ".join(obs["text"] for obs in observations).strip()
    return {
        "observations": observations,
        "narrative_summary": narrative_summary,
        "tier": "A",
        "speculative": True,
        "status": {"nlp_available": False, "tier_b_generated": False},
    }