#!/usr/bin/env python
"""benchmark.py - measure end-to-end chat_analyzer runtimes per NLP tier.

Times the REAL user-facing pipeline (`run_pipeline`) over one or more chat
exports at two tiers and writes a JSON results file the README's timing table
can consume:

  - tier "basic" (CHAT_ANALYZER_TIER=1): no NLP, VADER sentiment only.
  - tier "nlp"   (CHAT_ANALYZER_TIER=3): NLP on; over-cap chats (default cap
    50000, CHAT_ANALYZER_EMOTION_SAMPLE) AUTO-SAMPLE emotion scoring because
    the runs are non-interactive (piped stdout, never a prompt).

Design notes (why this is an honest benchmark):

  - Each (input, tier) runs in a FRESH subprocess that imports the package
    and calls `run_pipeline` exactly as a real `chat-analyzer <file>` user
    invocation would: interpreter + import startup and one-off model loading
    are all inside the measured wall time. No state bleeds between runs.
  - Non-tty runs are enforced (subprocess stdout is piped), which is exactly
    the branch where Option C auto-samples — the sampling prompt path is
    never exercised.
  - `run_pipeline` is timed directly instead of shelling out to the CLI so
    report HTML writing / browser opening are excluded (not the pipeline) and
    the sampling metadata is readable straight off the AnalysisResults dict.
  - Per-stage wall times are a best-effort bonus (the pipeline's off-tty
    `[OK] <stage>` narration is context-managed, so a wrapper can time each
    stage body); a failure there never affects the total time.
  - Missing input files are skipped with a warning and the script exits 0.

The `--run-one` mode is the internal per-(input, tier) worker; it prints a
single JSON line the orchestrator mode collects.

Usage:
  python scripts/benchmark.py [--inputs PATH...] [--tiers basic,nlp]
                              [--json C:\\temp\\benchmark_results.json]
  python scripts/benchmark.py --run-one <PATH> basic|nlp
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = (
    Path(os.environ.get("TEMP") or Path.home() / "AppData" / "Local" / "Temp")
    / "opencode"
    / "benchmark_results.json"
)

TIER_ENV = {"basic": "1", "nlp": "3"}
_TIERS = ("basic", "nlp")
_STAGE_LABELS = ("Parsing chat", "Computing insights", "Analyzing emotions", "Generating narrative")

# The author's real chat exports. Every entry is verified with Path.is_file()
# and skipped (with a warning) when absent, so a clone without these files can
# still run the harness against its own inputs via --inputs.
DEFAULT_INPUTS = (
    Path.home() / "Downloads" / "WhatsApp Chat with Suraj.txt",
    Path.home() / "Downloads" / "WhatsApp Chat with Anu.zip",
    Path.home() / "Downloads" / "WhatsApp Chat with Keblasss \u2615.zip",
    Path.home() / "Downloads" / "WhatsApp Chat with Anuradha Shaw.zip",
)

BUSY_NOTE = (
    "Machine NOT idle during the benchmark: VS Code and opencode were running "
    "throughout, so wall-clock times are realistic on a loaded dev machine and "
    "should read as upper bounds for a quiet machine."
)


def windows_cpu_load_percent() -> float | None:
    """Instantaneous CPU load average across all logical cores (Windows CIM)."""
    if os.name != "nt":
        return None
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "(Get-CimInstance Win32_Processor | Measure-Object "
                    "LoadPercentage -Average).Average"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        text = out.stdout.strip()
        return float(text.splitlines()[-1]) if text else None
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def free_gb_c() -> float | None:
    """Free space on C: in GiB, or None on a non-Windows host."""
    if os.name != "nt":
        return None
    try:
        return round(shutil.disk_usage("C:/").free / (1024**3), 2)
    except OSError:
        return None


def _timed_stage_hook() -> tuple[dict[str, float], object]:
    """Return (stage_times, unwrap) for timing run_pipeline's stages.

    The pipeline's off-tty narration calls the module-global `stage(...)`
    which returns a context manager wrapping each stage body. Swapping the
    module global for a timing wrapper records per-stage wall time without
    touching src/. `unwrap` restores the original for any later callers.
    """
    import chat_analyzer.cli.pipeline as pipeline_mod

    stage_times: dict[str, float] = {}

    class _TimedStage:
        def __init__(self, label: str, inner) -> None:
            self.label = label
            self.inner = inner
            self.seconds = 0.0

        def __enter__(self):
            self._t0 = time.perf_counter()
            self.inner.__enter__()

        def __exit__(self, *exc):
            try:
                self.inner.__exit__(*exc)
            finally:
                self.seconds = time.perf_counter() - self._t0
            stage_times[self.label] = (
                stage_times.get(self.label, 0.0) + self.seconds
            )

    original = pipeline_mod.stage

    def _timed_stage(console, progress, task_id, label):
        return _TimedStage(label, original(console, progress, task_id, label))

    pipeline_mod.stage = _timed_stage

    def unwrap() -> None:
        pipeline_mod.stage = original

    return stage_times, unwrap


def run_one(path: Path, tier: str) -> dict:
    """Run run_pipeline once for (path, tier) and return the measured record."""
    from rich.console import Console

    from chat_analyzer.cli.pipeline import run_pipeline

    stage_times, unwrap = _timed_stage_hook()
    try:
        console = Console()
        t0 = time.perf_counter()
        try:
            results = run_pipeline(path, console, nlp_enabled=(tier == "nlp"))
        except Exception as exc:  # noqa: BLE001 - report the failure honestly
            return {
                "ok": False,
                "file": str(path),
                "tier": tier,
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.perf_counter() - t0, 3),
                "stage_times": dict(stage_times),
            }
        total = time.perf_counter() - t0

        emotion = results.get("emotion") or {}
        sample = emotion.get("sample")
        return {
            "ok": True,
            "file": str(path),
            "source": results["source"],
            "messages": results["parse"]["parsed_messages"],
            "tier": tier,
            "mode": "sampled" if sample and sample.get("sampled") else "exact",
            "sample_scored": sample.get("scored") if sample else None,
            "sample_total": sample.get("total") if sample else None,
            "sample_cap": sample.get("cap") if sample else None,
            "seconds": round(total, 3),
            "stage_times": dict(stage_times),
        }
    finally:
        unwrap()


def _run_one_entry(argv: list[str]) -> int:
    path = Path(argv[0])
    tier = argv[1]
    if not path.is_file():
        print(
            json.dumps(
                {"ok": False, "file": str(path), "tier": tier, "error": "file not found"}
            )
        )
        return 0
    record = run_one(path, tier)
    print(json.dumps(record))
    return 0


def _spawn_and_run(
    path: Path,
    tier: str,
    script: Path,
    log_path: Path | None,
    worker_timeout: int,
) -> dict:
    """Run one (path, tier) in a fresh subprocess; return its record dict."""
    env = os.environ.copy()
    env["CHAT_ANALYZER_TIER"] = TIER_ENV[tier]
    env["CHAT_ANALYZER_NO_OPEN"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(script), "--run-one", str(path), tier],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=worker_timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "file": str(path),
            "tier": tier,
            "error": f"worker timed out after {worker_timeout}s",
            "seconds": float(worker_timeout),
            "wall_seconds": float(worker_timeout),
        }
    wall = round(time.perf_counter() - started, 3)

    record = None
    for line in reversed(proc.stdout.splitlines()):
        try:
            record = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if record is None:
        record = {
            "ok": False,
            "file": str(path),
            "tier": tier,
            "error": f"worker produced no JSON (exit {proc.returncode})",
            "seconds": wall,
        }
        if log_path is not None:
            log_path.write_text(
                "STDOUT:\n" + proc.stdout + "\nSTDERR:\n" + proc.stderr,
                encoding="utf-8",
            )
    record["wall_seconds"] = wall
    return record


def _record_fingerprint() -> dict:
    return {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "cores": os.cpu_count(),
        "cpu_load_before": windows_cpu_load_percent(),
        "free_gb_before": free_gb_c(),
    }


def _record_fingerprint_after() -> dict:
    return {
        "cpu_load_after": windows_cpu_load_percent(),
        "free_gb_after": free_gb_c(),
    }


def _write_results(
    summary: dict,
    results: list[dict],
    missing: list[str],
    started_wall: float,
    json_path: Path,
) -> None:
    """Write the current benchmark state to the JSON results file.

    Called after every completed run so a later interruption never loses the
    runs already finished (the final call after all runs is authoritative).
    """
    summary.update(_record_fingerprint_after())
    summary["busy_note"] = BUSY_NOTE
    summary["total_wall_seconds"] = round(time.perf_counter() - started_wall, 1)
    summary["inputs_missing_skipped"] = missing
    summary["results"] = results
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_table(results: list[dict]) -> str:
    """Plain ASCII table: file | messages | tier | mode | seconds."""
    headers = ("file", "messages", "tier", "mode", "seconds")
    rows: list[list[str]] = []
    for r in results:
        file_label = Path(r["file"]).name
        if r["ok"]:
            rows.append(
                [
                    file_label,
                    str(r["messages"]),
                    r["tier"],
                    r["mode"],
                    f"{r['seconds']:.2f}",
                ]
            )
        else:
            rows.append([file_label, "-", r["tier"], "FAILED", f"{r['seconds']:.2f}"])
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [line]
    out.append(
        "|"
        + "|".join(f" {h:<{w}} " for h, w in zip(headers, widths))
        + "|"
    )
    out.append(line)
    for row in rows:
        out.append(
            "|"
            + "|".join(f" {c:<{w}} " for c, w in zip(row, widths))
            + "|"
        )
    out.append(line)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark.py",
        description=(
            "Time the chat_analyzer pipeline per NLP tier over chat exports "
            "and write a JSON results file."
        ),
    )
    parser.add_argument(
        "--run-one",
        nargs=2,
        metavar=("PATH", "TIER"),
        help="Internal worker: run one (input, tier) and print a JSON line.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        help="Chat export paths to benchmark (default: the 4 author chats).",
    )
    parser.add_argument(
        "--tiers",
        default="basic,nlp",
        help="Comma-separated tiers to run (basic,nlp).",
    )
    parser.add_argument(
        "--json",
        default=str(DEFAULT_JSON),
        help="Path to write the JSON results file (default: TEMP\\opencode\\benchmark_results.json).",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="Append one progress line per completed run to this file.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=2400,
        help="Per-run worker timeout in seconds (default 2400).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load prior results from --json and skip runs already measured OK.",
    )
    args = parser.parse_args(argv)

    # Windows console encoding bootstrap (mirrors main.py): default CMD cp1252
    # cannot print the coffee-emoji filename or other non-Latin-1 exports.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if args.run_one is not None:
        return _run_one_entry(args.run_one)

    inputs: list[Path] = [Path(p) for p in args.inputs] if args.inputs else [
        Path(p) for p in DEFAULT_INPUTS
    ]
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip() in _TIERS]
    if not tiers:
        tiers = list(_TIERS)

    script = Path(__file__).resolve()
    json_path = Path(args.json).resolve()
    log_path = Path(args.log).resolve() if args.log else None
    json_path.parent.mkdir(parents=True, exist_ok=True)

    existing = [(p, t) for p in inputs for t in tiers if p.is_file()]
    missing = sorted({str(p) for p in inputs if not p.is_file()})
    for p in missing:
        print(f"[WARN] Skipping missing input (not found): {p}")

    if not existing:
        print("No input files found; nothing to benchmark. Exit 0.")
        return 0

    results: list[dict] = []
    if args.resume and json_path.exists():
        try:
            prior = json.loads(json_path.read_text(encoding="utf-8"))
            results = [r for r in prior.get("results") or [] if isinstance(r, dict)]
        except (json.JSONDecodeError, OSError):
            results = []
    done_pairs = {(r["file"], r["tier"]) for r in results if r.get("ok")}
    to_run = [(p, t) for p, t in existing if (str(p), t) not in done_pairs]
    for p, t in existing:
        if (str(p), t) in done_pairs:
            print(f"[SKIP] Already measured OK: {Path(p).name} [{t}]")

    summary = _record_fingerprint()
    started_wall = time.perf_counter()
    print(
        f"Benchmarking {len(to_run)} (input, tier) runs across "
        f"{len({p for p, _ in to_run})} files "
        f"({len(done_pairs)} already measured, skipped)."
    )
    print(
        f"CPU load before: {summary['cpu_load_before']}% | "
        f"cores: {summary['cores']} | free on C: {summary['free_gb_before']} GiB"
    )
    for path, tier in to_run:
        label = f"{Path(path).name} [{tier}]"
        print(f"\n=== START {label} ===")
        t0 = time.perf_counter()
        record = _spawn_and_run(path, tier, script, log_path, args.timeout)
        elapsed = time.perf_counter() - t0
        results = [
            r
            for r in results
            if not (r.get("file") == str(path) and r.get("tier") == tier)
        ]
        results.append(record)
        status = (
            "OK"
            if record["ok"]
            else f"FAILED ({record.get('error', 'unknown')})"
        )
        print(
            f"=== DONE {label} -> {status} in {elapsed:.1f}s "
            f"(worker {record['seconds']}s) ==="
        )
        if record.get("ok"):
            if record["mode"] == "sampled":
                print(
                    f"[INFO] Emotions sampled: {record['sample_scored']} of "
                    f"{record['sample_total']} messages (cap {record['sample_cap']})."
                )
            if record.get("stage_times"):
                parts = ", ".join(
                    f"{k}: {v:.1f}s" for k, v in record["stage_times"].items()
                )
                print(f"[INFO] Per-stage: {parts}")
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"{label}: {status} in {elapsed:.1f}s\n")
        _write_results(summary, results, sorted(missing), started_wall, json_path)

    summary.update(_record_fingerprint_after())
    summary["busy_note"] = BUSY_NOTE
    summary["total_wall_seconds"] = round(time.perf_counter() - started_wall, 1)
    summary["inputs_missing_skipped"] = sorted(missing)
    summary["results"] = results

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + build_table(results))
    print(f"\nCPU load after: {summary['cpu_load_after']}% | "
          f"free on C: {summary['free_gb_after']} GiB")
    print(f"Results JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())