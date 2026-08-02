"""Terminal narration for the chat-analyzer CLI (D-05/D-07/D-16/D-18).

Thin, ASCII-first rendering layer — the ONLY terminal insights are the
compact ASCII summary panel (no sentiment, no tables, no charts in the
terminal). render reads AnalysisResults only and owns no business logic.
Single-source narration: pipeline owns the stage lines, main.py owns the
'Messages:' smoke token, render owns the end summary.
"""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel

from chat_analyzer.cli.contracts import AnalysisResults


def show_summary(results: AnalysisResults, console: Console) -> None:
    """Print the end-of-run summary: skip/system lines + ASCII panel + path.

    ASCII-safe symbols only ([WARN]/[INFO], +-| box) — no emoji, no
    box-drawing glyphs (Pitfall 5: the utf-8 reconfigure is a safety net,
    not a license to ship non-ASCII).
    """
    parse = results["parse"]
    stats = results["stats"]

    if parse["skipped_lines"] > 0:
        console.print(
            f"[WARN] Skipped {parse['skipped_lines']} lines that couldn't be parsed"
        )

    if parse["system_messages"] > 0:
        console.print(
            f"[INFO] Excluded {parse['system_messages']} system messages from stats"
        )

    date_range = stats["date_range"]
    console.print(
        Panel(
            f"Total messages: {stats['total_messages']}\n"
            f"Participants: {stats['participants']}\n"
            f"Date range: {date_range['start']} to {date_range['end']}",
            title="Summary",
            box=box.ASCII,
        )
    )

    console.print(f"Report: {results['report_path']}")
