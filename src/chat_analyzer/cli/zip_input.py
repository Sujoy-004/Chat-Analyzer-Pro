"""ZIP export input support for the chat-analyzer CLI (Item C).

WhatsApp and Telegram "Export chat / Export Telegram data" can deliver a .zip
container holding the chat transcripts plus media. run_pipeline previously
accepted only bare .txt (WhatsApp) / .json (Telegram) paths, so a zip export
was unusable. This module:

- enumerates the chat transcripts inside the zip (.txt -> WhatsApp,
  .json -> Telegram); media members are ignored (the tool analyzes the
  conversation text, not the images/videos),
- lets the user choose which transcripts to analyze (interactively on a real
  terminal: skip any or selectively merge; non-tty runs fall back to "all"),
- parses each chosen transcript with the existing hardened parsers and MERGES
  rows + counts into one ParseReport, so the rest of run_pipeline is
  unchanged.

Design rules honored (matching pipeline.py / parser conventions):
- Reuses WhatsAppParser.parse_file_with_report / parse_telegram_chat_with_report
  (Item C decision: keep both formats — Telegram never exports .txt).
- zip extraction is path-traversal-safe: only the named member is read via
  zf.read(), and the extracted filename is the member's BASENAME, never a
  reconstructed archive path (security posture).
- Never crashes on a bad member; a failing transcript degrades to a warn and
  analysis continues on the others (Pitfall 6 degrade spirit).
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import typer


def _list_transcripts(zip_path: Path) -> list[tuple[str, str]]:
    """Return [(member_name, kind)] for chat transcripts inside the zip.

    kind is "whatsapp" for `.txt` members and "telegram" for `.json` members.
    Scanned regardless of archive directory depth.
    """
    found: list[tuple[str, str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue  # directory entry
            lower = name.lower()
            if lower.endswith(".txt"):
                found.append((name, "whatsapp"))
            elif lower.endswith(".json"):
                found.append((name, "telegram"))
    return found


def _select_transcripts(
    transcripts: list[tuple[str, str]], console
) -> list[tuple[str, str]]:
    """Let the user pick which transcripts to analyze (skip / merge).

    Interactive UX (Item C decision): list the found transcripts and let the
    user choose, comma-separated, which to keep. Never silently guess. On a
    non-tty run (piped/CI) we cannot prompt — fall back to ALL transcripts
    with a clear notice.
    """
    if len(transcripts) == 1:
        return transcripts  # nothing to choose

    if not console.is_terminal:
        console.print(
            f"[INFO] ZIP contains {len(transcripts)} chat transcripts; "
            "analyzing all (non-interactive)."
        )
        return transcripts

    console.print("[INFO] The ZIP contains multiple chat transcripts:")
    for i, (name, kind) in enumerate(transcripts, 1):
        console.print(f"  {i}) {name} [{kind}]")
    console.print(
        "Enter the numbers to analyze (comma-separated, e.g. 1,3). "
        "Press <Enter> for all."
    )
    n = len(transcripts)
    while True:
        raw = typer.prompt("Transcripts to analyze", default="").strip()
        if not raw:
            return transcripts
        try:
            picks = {int(part) - 1 for part in raw.replace(" ", "").split(",") if part}
        except ValueError:
            console.print("[WARN] Please enter numbers separated by commas.")
            continue
        if not picks or any(p < 0 or p >= n for p in picks):
            console.print(f"[WARN] Enter numbers between 1 and {n}.")
            continue
        return [transcripts[p] for p in sorted(picks)]


def _parse_member(
    zip_path: Path,
    member: str,
    kind: str,
    tmp: Path,
) -> tuple[list[dict], dict]:
    """Extract one member to a safe temp file and parse it.

    Returns (rows, counts) in the single-file parser contract shape. The
    extracted file lives at tmp/<basename> only.
    """
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(member)

    dst = tmp / Path(member).name
    dst.write_bytes(data)

    if kind == "whatsapp":
        from chat_analyzer.parser.whatsapp_parser import WhatsAppParser

        return WhatsAppParser().parse_file_with_report(str(dst))

    from chat_analyzer.parser.telegram_parser import parse_telegram_chat_with_report

    return parse_telegram_chat_with_report(str(dst))


def parse_zip_with_report(
    zip_path: Path, console
) -> tuple[list[dict], dict, str]:
    """Parse the chat transcripts inside a zip export, merged into one report.

    Returns (rows, counts, source) — the same contract run_pipeline expects
    from the single-file parsers, so the downstream pipeline is unchanged.
    counts is the SUM across chosen transcripts; source is "whatsapp",
    "telegram", or "mixed" (Item C: keep both formats).
    """
    try:
        transcripts = _list_transcripts(zip_path)
    except zipfile.BadZipFile:
        raise ValueError(
            f"Invalid or corrupted ZIP file: {zip_path}"
        ) from None

    if not transcripts:
        raise ValueError("No chat transcripts (.txt / .json) found inside the zip export")

    chosen = _select_transcripts(transcripts, console)

    rows: list[dict] = []
    counts = {
        "total_lines": 0,
        "parsed_messages": 0,
        "skipped_lines": 0,
        "system_messages": 0,
    }
    kinds = set()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for member, kind in chosen:
            try:
                member_rows, member_counts = _parse_member(zip_path, member, kind, tmp)
            except Exception:  # noqa: BLE001 - one bad transcript must not tank the batch
                console.print(f"[WARN] Skipped transcript: {member} (failed to parse).")
                continue
            rows.extend(member_rows)
            for key in counts:
                counts[key] += member_counts.get(key, 0)
            kinds.add(kind)

    if not rows:
        raise ValueError("No messages could be parsed from the selected transcripts")

    source = "mixed" if len(kinds) > 1 else next(iter(kinds))
    return rows, counts, source