"""Interface contracts for the chat-analyzer CLI pipeline.

Single source of truth consumed by pipeline.py, adapters.py, render.py and
report_html.py. The analysis core (parser/analysis/ingest) never imports this
module — parsers return plain counts dicts that pipeline.py wraps into a
ParseReport, so the core stays cli-agnostic.
"""

from dataclasses import dataclass
from typing import Any, TypedDict


@dataclass
class ParseReport:
    """Aggregated parse counters for one export file."""

    source: str
    total_lines: int = 0
    parsed_messages: int = 0
    skipped_lines: int = 0
    system_messages: int = 0


class AnalysisResults(TypedDict):
    """The complete analysis payload produced by run_pipeline.

    charts maps chart name -> base64 PNG data URI. report_path is filled by
    main.py after report_html.write_report succeeds. health and network hold
    serializable scalars extracted by adapters.py from the always-on analysis
    modules (D-07/D-07b) — never the raw prepared_data DataFrame or the
    networkx DiGraph (Pattern 3). emotion and summary hold the gated NLP
    blocks (D-07c, ANAL-06/ANAL-08) — None when the silent availability probe
    (nlp_gate.nlp_available) reports the models unavailable (D-02/D-06).
    """

    source: str
    parse: dict[str, int]
    stats: dict[str, Any]
    participants: dict[str, Any]
    content: dict[str, Any]
    sentiment: dict[str, Any]
    health: dict[str, Any]
    network: dict[str, Any]
    emotion: dict[str, Any] | None
    summary: dict[str, Any] | None
    charts: dict[str, str]
    insights: list[str]
    report_path: str
