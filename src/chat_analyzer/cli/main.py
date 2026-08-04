"""Command-line interface for chat-analyzer.

One command — `analyze <chat_file>` — turns a WhatsApp .txt or Telegram .json
export into terminal insights plus a self-contained HTML report.

- Positional (D-02): `chat-analyzer <path>` analyzes and exits 0 on success,
  1 with a friendly line on failure (D-06 — never a traceback).
- Interactive (D-01): no-arg runs the re-prompt loop; each successful path
  goes through `_analyze_path`; ValueError re-prompts (D-06).
- `--version` (D-03): typer 0.27 has no built-in version flag — the eager
  callback closes the gap.
- Always-integrated NLP (D-01/D-02/D-04/D-06): a silent availability check at
  startup decides the UX. Interactive tty runs with NLP missing get the
  D-04 3-option download menu (dispatched to the guarded installer);
  positional and piped runs never prompt and print a single D-06 hint line.

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


def _nlp_menu(console) -> str:
    """Show the D-04 3-option download menu and return the choice.

    Only ever called on a real tty with NLP missing — the caller gates on
    `sys.stdin.isatty()` and the availability probe. Option 2 (CPU-only,
    ~0.6 GB) is the default (D-04, T-04-13); option 3 (no download) is
    always available. Returns one of "1"/"2"/"3".
    """
    console.print("NLP extras are not installed. Choose how to proceed:")
    console.print("  1) Download full torch (~3GB) - best quality")
    console.print("  2) Download CPU-only torch + model (~0.6GB)")
    console.print("  3) No download - run basic analysis")
    while True:
        choice = typer.prompt("Choice", default="2").strip()
        if choice in ("1", "2", "3"):
            return choice
        console.print("[WARN] Please choose 1, 2, or 3.")


def _analyze_path(path: Path) -> None:
    """Run the full pipeline for one export and render the report."""
    from rich.console import Console

    from chat_analyzer.cli.pipeline import run_pipeline, stage_status
    from chat_analyzer.cli.render import show_summary
    from chat_analyzer.cli.report_html import open_report, write_report

    console = Console()
    results = run_pipeline(path, console)

    # D-05 / CRITICAL #1 — the smoke-contract count line, printed ONCE here
    # so the token appears in both positional and interactive stdout
    # (pipeline.py and render.py must never print a second one).
    console.print(f"Messages: {results['parse']['parsed_messages']}")

    with stage_status(console, "Writing report"):
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

    # D-02: silent startup availability check — no prompt, no hint here.
    # Computed once and shared by the menu gate and the hint lines (04-03).
    from rich.console import Console

    from chat_analyzer.cli import nlp_gate

    nlp_on = nlp_gate.nlp_available(nlp_gate.MODEL_ID)
    console = Console()

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
        # D-06: single hint line after the report path, never before the
        # "Messages: N" smoke token (ASCII only, no emoji). soft_wrap keeps
        # the hint one line even on a narrow non-tty console (tests, pipes).
        if not nlp_on:
            console.print(
                "[INFO] Tip: richer insights need the NLP extra - "
                "pip install chat-analyzer-pro\\[nlp]",
                soft_wrap=True,
            )
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
            # D-04: the download menu shows ONLY on a real tty with NLP
            # missing (a piped run cannot answer a menu — D-06 hint instead).
            menu_shown = False
            if (not nlp_on) and sys.stdin.isatty():
                menu_shown = True
                choice = _nlp_menu(console)
                if choice in ("1", "2"):
                    # D-05/Pitfall 4: announce name + size BEFORE the install
                    # and model download start — never a frozen terminal.
                    console.print(
                        "[INFO] Installing NLP extras: torch + transformers "
                        "(~0.6 GB CPU / ~3 GB full), then model "
                        f"{nlp_gate.MODEL_ID} (~{nlp_gate.EMOTION_MODEL_SIZE_MB} MB)"
                    )
                    try:
                        nlp_gate.install_nlp(cpu_only=(choice == "2"))
                        nlp_on = True  # install succeeded; pipeline may load models
                    except RuntimeError as exc:
                        typer.echo(f"[WARN] {exc}", err=True)
                        console.print("[INFO] Continuing with basic analysis.")
            _analyze_path(path)
        except ValueError as exc:
            # D-06 — friendly message, loop back to re-prompt.
            typer.echo(str(exc), err=True)
            continue
        # D-06 hint for the piped/no-menu path: the user never saw the menu.
        if (not nlp_on) and (not menu_shown):
            console.print(
                "[INFO] Tip: richer insights need the NLP extra - "
                "pip install chat-analyzer-pro\\[nlp]",
                soft_wrap=True,
            )
        raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
