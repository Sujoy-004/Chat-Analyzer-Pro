"""Command-line interface for chat-analyzer.

One command — `analyze <chat_file>` — turns a WhatsApp .txt or Telegram .json
export into terminal insights plus a self-contained HTML report.

- Positional (D-02): `chat-analyzer <path>` analyzes and exits 0 on success,
  1 with a friendly line on failure (D-06 — never a traceback).
- Interactive (D-01): no-arg runs the re-prompt loop; each successful path
  goes through `_analyze_path`; ValueError re-prompts (D-06).
- `--version` (D-03): typer 0.27 has no built-in version flag — the eager
  callback closes the gap.
- Always-on NLP tier menu (PH2): EVERY interactive run (positional or in the
  no-arg loop) shows the 3-option tier menu regardless of install state —
  tier 1 forces NLP off, tiers 2/3 make the chosen tier ready before
  analyzing (`_ensure_nlp_for_tier`). Non-interactive runs never prompt:
  `CHAT_ANALYZER_TIER` env override, else silent tier 1 + the hint line.

The heavy analysis modules are imported lazily inside `_analyze_path` so that
`--help`/`--version` stay instant (research Anti-Pattern 2). nlp_gate itself
is lightweight (stdlib + guarded huggingface_hub import) and safe to import
eagerly.
"""

import os
import sys
from pathlib import Path

import typer

from chat_analyzer.cli import nlp_gate

app = typer.Typer(
    add_completion=False,
    help="Analyze WhatsApp and Telegram chat exports from the terminal.",
)

# D-14: inline export instructions composed into every failure message —
# what went wrong, why (1 line), and the exact how-to-export steps. Not a
# README pointer, not auto-open.
_EXPORT_WHATSAPP = (
    "To export a chat: open the chat in WhatsApp, tap the \u22ee menu, "
    "More -> Export chat, and save the .txt file."
)
_EXPORT_TELEGRAM = (
    "To export a chat: in Telegram Desktop open Settings -> Advanced -> "
    "Export Telegram data, select only Messages, choose JSON format, and "
    "save the folder export."
)


def _friendly_error(chat_file: Path, exc: Exception) -> str:
    """Compose a distinct, instructive message for one failure class (D-13).

    Classifies by the exception type/text: file-not-found, unsupported file
    type, empty/unparseable export, and a defensive catch-all — each with
    inline WhatsApp/Telegram export instructions (D-14). Every failure path
    in main() ends in `typer.Exit(code=1) from None` so no traceback ever
    reaches the user (T-04-11).
    """
    if isinstance(exc, FileNotFoundError) or not chat_file.is_file():
        return (
            f"File not found: {chat_file}. The file must exist before running. "
            f"{_EXPORT_WHATSAPP} {_EXPORT_TELEGRAM}"
        )
    text = str(exc)
    if "Unsupported file type" in text:
        return (
            "Unsupported file type: expected a WhatsApp .txt, Telegram .json, "
            "or a .zip export archive. "
            f"{_EXPORT_WHATSAPP} {_EXPORT_TELEGRAM}"
        )
    if "No messages could be parsed" in text:
        return (
            "No messages could be parsed from this file — the export may be "
            f"empty or a system-only export. {_EXPORT_WHATSAPP} {_EXPORT_TELEGRAM}"
        )
    return (
        f"Could not process this file. {text} {_EXPORT_WHATSAPP} {_EXPORT_TELEGRAM}"
    )


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import PackageNotFoundError, version

        try:
            ver = version("chat-analyzer-pro")
        except PackageNotFoundError:
            typer.echo("chat-analyzer (dev \u2014 package metadata not found)")
        else:
            typer.echo(f"chat-analyzer {ver}")
        raise typer.Exit()


def _tier_menu(console) -> str:
    """Show the 3-option NLP tier menu and return the choice.

    Shown on EVERY interactive run, regardless of current install state (PH2).
    Default is 1 (without NLP). Returns one of "1"/"2"/"3".
    """
    console.print("NLP tier:")
    console.print("  1) Without NLP")
    console.print("  2) Minimal (~0.6 GB)")
    console.print("  3) Full-fledged (~3 GB)")
    while True:
        choice = typer.prompt("Choice", default="1").strip()
        if choice in ("1", "2", "3"):
            return choice
        console.print("[WARN] Please choose 1, 2, or 3.")


def _tier_from_env() -> str:
    """Non-interactive tier resolution (PH2).

    CHAT_ANALYZER_TIER=1|2|3 wins when set (1 accepted for explicitness).
    Legacy CHAT_ANALYZER_FORCE_NLP maps 0->1 and 1->3 so existing automation
    and tests stay valid. Default is silent tier 1 (no NLP).
    """
    env_tier = os.environ.get("CHAT_ANALYZER_TIER")
    if env_tier in ("1", "2", "3"):
        return env_tier
    force = os.environ.get("CHAT_ANALYZER_FORCE_NLP")
    if force == "1":
        return "3"
    return "1"


def _announce_install(console, cpu_only: bool) -> None:
    """D-05/Pitfall 4: announce name + size BEFORE the install and model
    download start — never a frozen terminal. Called once before any pip run."""
    flavor = "CPU-only (~0.6 GB)" if cpu_only else "full (~3 GB)"
    console.print(
        "[INFO] Installing NLP extras: torch + transformers "
        f"({flavor}), then model {nlp_gate.MODEL_ID} "
        f"(~{nlp_gate.EMOTION_MODEL_SIZE_MB} MB) and "
        f"{nlp_gate.TIER_B_MODEL_ID} (~{nlp_gate.TIER_B_MODEL_SIZE_MB} MB)"
    )


def _ensure_nlp_for_tier(tier: str, console) -> bool:
    """Make sure the chosen tier (2/3) is ready; True means NLP can run.

    Resolves nlp_gate.nlp_status():
    - READY   -> announce, proceed.
    - MISSING -> announce + install (tier flavor) + download both models.
    - OUTDATED-> prompt "Update now / Go with current"; update on request,
                otherwise proceed with what's installed (pipeline downloads
                missing weights on first use as today).
    Any install/download failure degrades to basic analysis (returns False)
    with the friendly hint — never a frozen terminal (Pitfall 4).
    """
    status, _ = nlp_gate.nlp_status()
    cpu_only = tier == "2"
    if status == "READY":
        console.print("[INFO] NLP ready.")
        return True
    if status == "MISSING":
        _announce_install(console, cpu_only)
        try:
            nlp_gate.install_nlp(cpu_only=cpu_only)
            nlp_gate.download_models()
        except RuntimeError as exc:
            typer.echo(f"[WARN] {exc}", err=True)
            console.print("[INFO] Continuing with basic analysis.")
            return False
        console.print("[INFO] NLP ready.")
        return True
    # OUTDATED
    if sys.stdin.isatty():
        choice = typer.prompt("Update NLP packages/models now?", default="n").strip().lower()
        if choice in ("y", "yes", "update"):
            _announce_install(console, cpu_only)
            try:
                nlp_gate.update_nlp(cpu_only=cpu_only)
                nlp_gate.download_models()
            except RuntimeError as exc:
                typer.echo(f"[WARN] {exc}", err=True)
                console.print("[INFO] Continuing with basic analysis.")
                return False
            console.print("[INFO] NLP ready.")
    else:
        # Non-interactive (piped/CI): cannot prompt — proceed with current.
        console.print("[INFO] NLP out of date; continuing with current install.")
    return True


def _analyze_path(path: Path, nlp_enabled: bool) -> None:
    """Run the full pipeline for one export and render the report."""
    from rich.console import Console

    from chat_analyzer.cli.pipeline import run_pipeline, stage_status
    from chat_analyzer.cli.render import show_summary
    from chat_analyzer.cli.report_html import open_report, write_report

    console = Console()
    results = run_pipeline(path, console, nlp_enabled=nlp_enabled)

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

    from rich.console import Console

    console = Console()

    def _resolve_nlp() -> tuple[bool, bool]:
        """Return (nlp_enabled, menu_shown) for the current run (PH2).

        tty -> always show the tier menu; non-tty -> CHAT_ANALYZER_TIER env or
        silent tier 1. Tier 2/3 call _ensure_nlp_for_tier to make NLP ready.
        """
        if sys.stdin.isatty():
            # A1: warn BEFORE the menu offers a multi-GB download that torch
            # extraction could never finish (WinError 206).
            long_path = nlp_gate.windows_long_path_message()
            if long_path is not None:
                console.print(f"[WARN] {long_path}")
            tier = _tier_menu(console)
            if tier == "1":
                return False, True
            return _ensure_nlp_for_tier(tier, console), True
        tier = _tier_from_env()
        if tier == "1":
            return False, False
        return _ensure_nlp_for_tier(tier, console), False

    if chat_file is not None:
        if not chat_file.is_file():
            typer.echo(_friendly_error(chat_file, FileNotFoundError()), err=True)
            raise typer.Exit(code=1)
        if chat_file.suffix.lower() not in {".txt", ".json", ".zip"}:
            typer.echo(
                _friendly_error(chat_file, ValueError("Unsupported file type")),
                err=True,
            )
            raise typer.Exit(code=1)
        try:
            nlp_enabled, menu_shown = _resolve_nlp()
            _analyze_path(chat_file, nlp_enabled=nlp_enabled)
        except ValueError as exc:
            # MEDIUM #4 — a malformed file (zero parsed rows, bad export,
            # unsupported format) exits 1 with a friendly, instructive line,
            # never a traceback (D-13/D-14, T-04-11).
            typer.echo(_friendly_error(chat_file, exc), err=True)
            raise typer.Exit(code=1) from None
        # D-06: single hint line after the report path, only when the user
        # never saw the menu (silent non-tty tier 1). ASCII only, no emoji.
        if not nlp_enabled and not menu_shown:
            console.print(
                "[INFO] Tip: richer insights need the NLP extra - "
                "pip install chat-analyzer-pro\\[nlp]",
                soft_wrap=True,
            )
        raise typer.Exit(code=0)

    while True:
        path = Path(typer.prompt("Enter path to chat export").strip().strip('"').strip("'"))
        if not path.is_file():
            typer.echo(_friendly_error(path, FileNotFoundError()), err=True)
            continue
        if path.suffix.lower() not in {".txt", ".json", ".zip"}:
            typer.echo(
                _friendly_error(path, ValueError("Unsupported file type")),
                err=True,
            )
            continue
        try:
            nlp_enabled, menu_shown = _resolve_nlp()
            _analyze_path(path, nlp_enabled=nlp_enabled)
        except ValueError as exc:
            # D-15 — friendly message with export instructions, then loop
            # back to re-prompt (never exits on a bad file).
            typer.echo(_friendly_error(path, exc), err=True)
            continue
        # D-06 hint for the silent non-tty path: the user never saw the menu.
        if not nlp_enabled and not menu_shown:
            console.print(
                "[INFO] Tip: richer insights need the NLP extra - "
                "pip install chat-analyzer-pro\\[nlp]",
                soft_wrap=True,
            )
        raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
