"""Command-line interface for chat-analyzer.

One command — `analyze <chat_file>` — turns a WhatsApp .txt or Telegram .json
export into terminal insights plus a self-contained HTML report.

- Positional (D-02): `chat-analyzer <path>` analyzes and exits 0 on success,
  1 with a friendly line on failure (D-06 — never a traceback).
- Interactive (D-01): no-arg runs the re-prompt loop; each successful path
  goes through `_analyze_path`; ValueError re-prompts (D-06).
- `--version` (D-03): typer 0.27 has no built-in version flag — the eager
  callback closes the gap.

The heavy analysis modules are imported lazily inside `_analyze_path` so that
`--help`/`--version` stay instant (research Anti-Pattern 2).
"""

import sys
from pathlib import Path

import typer

app = typer.Typer(
    add_completion=False,
    help="Analyze WhatsApp and Telegram chat exports from the terminal.",
)


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version

        typer.echo(f"chat-analyzer {version('chat-analyzer-pro')}")
        raise typer.Exit()


def _analyze_path(path: Path) -> None:
    """Run the full pipeline for one export and render the report."""
    from rich.console import Console

    from chat_analyzer.cli.pipeline import run_pipeline
    from chat_analyzer.cli.render import show_summary
    from chat_analyzer.cli.report_html import open_report, write_report

    console = Console()
    results = run_pipeline(path, console)

    # D-05 / CRITICAL #1 — the smoke-contract count line, printed ONCE here
    # so the token appears in both positional and interactive stdout
    # (pipeline.py and render.py must never print a second one).
    console.print(f"Messages: {results['parse']['parsed_messages']}")

    with console.status("Writing report...", spinner="line"):
        results["report_path"] = str(write_report(results, path).resolve())

    show_summary(results, console)
    open_report(Path(results["report_path"]))


@app.command()
def main(
    chat_file: Path | None = typer.Argument(  # noqa: B008 - typer idiom: argument defaults must be typer.Argument() calls
        None, help="Path to WhatsApp .txt or Telegram .json export"
    ),
    version: bool | None = typer.Option(
        None,
        "--version",
        is_eager=True,
        callback=_version_callback,
        help="Show version and exit",
    ),
) -> None:
    """Analyze a WhatsApp or Telegram chat export from the terminal."""
    # Windows console encoding bootstrap: default CMD cp1252 must never crash
    # the tool or its error messages (research Pitfall 5).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if chat_file is not None:
        if not chat_file.is_file():
            typer.echo(f"File not found: {chat_file}", err=True)
            raise typer.Exit(code=1)
        if chat_file.suffix.lower() not in {".txt", ".json"}:
            typer.echo(
                "Unsupported file type: expected a WhatsApp .txt or Telegram .json export",
                err=True,
            )
            raise typer.Exit(code=1)
        try:
            _analyze_path(chat_file)
        except ValueError as exc:
            # MEDIUM #4 — a malformed file (zero parsed rows, bad export,
            # unsupported format) exits 1 with a friendly line, never a
            # traceback (D-06).
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from None
        raise typer.Exit(code=0)

    while True:
        path = Path(typer.prompt("Enter path to chat export").strip().strip('"').strip("'"))
        if not path.is_file():
            typer.echo(f"File not found: {path}", err=True)
            continue
        if path.suffix.lower() not in {".txt", ".json"}:
            typer.echo(
                "Unsupported file type: expected a WhatsApp .txt or Telegram .json export",
                err=True,
            )
            continue
        try:
            _analyze_path(path)
        except ValueError as exc:
            # D-06 — friendly message, loop back to re-prompt.
            typer.echo(str(exc), err=True)
            continue
        raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
