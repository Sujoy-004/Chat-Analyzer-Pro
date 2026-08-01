"""Command-line interface for chat-analyzer.

Interactive flow (D-03): the user runs `chat-analyzer` (or
`python -m chat_analyzer`) with no arguments, is prompted for the path to a
WhatsApp or Telegram chat export, and the existing ingestion core processes it
and reports message counts.

The heavy analysis modules are imported lazily inside the command handler so
that `--help` stays instant (research Anti-Pattern 2).
"""

import sys
from pathlib import Path

import typer

app = typer.Typer(
    add_completion=False,
    help="Analyze WhatsApp and Telegram chat exports from the terminal.",
)


@app.command()
def main() -> None:
    """Prompt for a chat export path and process it interactively."""
    # Windows console encoding bootstrap: default CMD cp1252 must never crash
    # the tool or its error messages (research Pitfall 5).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    while True:
        path = Path(typer.prompt("Enter path to chat export").strip().strip('"').strip("'"))
        if not path.is_file():
            typer.echo(f"File not found: {path}", err=True)
            continue
        try:
            from chat_analyzer.ingest.ingestion import process_uploaded_file

            messages, media = process_uploaded_file(str(path))
            typer.echo(f"Processed {path}:")
            typer.echo(f"Messages: {len(messages)}")
            typer.echo(f"Media items: {len(media)}")
        except Exception as exc:
            typer.echo(f"Could not process {path}: {exc}", err=True)
            raise typer.Exit(code=1)
        raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
