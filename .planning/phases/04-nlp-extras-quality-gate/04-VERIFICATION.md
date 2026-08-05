# Verification — Phase 04 (NLP Extras & Quality Gate)

- **Verifier**: orchestrator (goal-backward, direct code spot-check)
- **Date**: 2026-08-05
- **Status**: PASS

## Goal vs. Implementation

Phase 04 goal (ROADMAP): gate the heavy NLP features (emotion, summarization)
behind a `[nlp]` extra while keeping relationship health + network graph
always-on, deliver a 3-option interactive NLP download menu that never freezes
the terminal, and close the QUAL-02 quality gate (legacy tests rewired to real
`chat_analyzer.*` modules, pytest + ruff both clean).

| Plan | Claimed deliverable | Verified |
|------|--------------------|----------|
| 04-01 | Health + network always-on in base install; report to cwd, no flags | `run_pipeline` runs `analyze_relationship_health` + `analyze_network` unconditionally; `report_html.py` renders health/network tabs with `data:image/png;base64,` charts; report written to `Path.cwd()/<stem>_report.html` (D-09), zero `--output`/`--no-report` flags (D-08) |
| 04-02 | Emotion `[0]`-parse fix; silent availability gate; locked model | `emotion.py` consumes the flat list-of-dicts from transformers 4.x `top_k=None`; `nlp_gate.nlp_available()` never raises (import failures → False); `MODEL_ID = bhadresh-savani/distilbert-base-uncased-emotion` |
| 04-03 | 3-option tty menu + guarded installer; degrade + hint | `main._nlp_menu` (line 83) with option 2 CPU-only default; gated by `sys.stdin.isatty()`; `nlp_gate.install_nlp` (arg list, no shell); RuntimeError → basic analysis + hint |
| 04-04 | Legacy tests rewired to real modules; QUAL-02 gate | `tests/test_analysis.py` + `test_parser.py` import `chat_analyzer.*` (commits 5a79e0c/252c636); pandas 2.x freq drift + cp1252 fixture fixed (97ad7e9/af89446) |
| 04-05 | Quickstart-first README, neutral options, traceability reconciled | README commands `chat-analyzer <path>`; REQUIREMENTS rows ANAL-07/09 always-on, OUT-04/05 NO FLAG (D-08), all Complete |

## Evidence

- **pytest**: `python -m pytest` = **175/175 pass** (verified post-04-04; prior-phase regression run re-confirmed 58/58 for phases 1+2 after review fixes).
- **ruff**: `python -m ruff check src/chat_analyzer tests` = **All checks passed** (0 errors), re-run after review fixes.
- **Code review** (`04-REVIEW.md`): 0 critical / 5 warnings / 2 info. WR-01 (CPU-only `--index-url` replacing PyPI → `transformers` unresolvable), WR-02 (no install timeout), WR-05 (always-on tests not pinning NLP gate) fixed in commit `8221ea2` and re-verified. WR-03 (positional tab-lead coupling in `report_html.py`), WR-04 (`weekly_digest` logging/HTML) logged to `deferred-items.md` as backlog items #4/#5 — accepted risk, safe while health/network remain always-on.
- **No XSS / no injection** verified empirically: Jinja autoescape active, chart URIs validated via `data:image/png;base64,` prefix, install uses arg list without `shell=True`.
- **QUAL-02 closure**: baseline legacy failures (test_end_to_end 11, test_reporting 15, ruff 380) were all pre-existing (verified via `git stash` baseline) and are now fully resolved — nothing defers Phase 4 completion.

## UAT mapping

- ROADMAP success criteria #1/#2 (emotion/summary gated, health/network always-on): PASS.
- #3 (NLP download question, no freeze): PASS (tty-only menu, guarded installer, timeout, degrade + hint).
- #4/#5 (report to cwd, always generated, no skip flag): PASS.

## Gaps & Risks

- WR-03: `report_html.py` renders `{{ insights[5] }}`/`{{ insights[6] }}` ungated — safe today only because health/network are always-on. Backlogged (#4).
- WR-04: `weekly_digest.py` not in CLI path but ships in package. Backlogged (#5).
- `torch is importable in base env` warning on this dev machine (pre-existing install) — pyproject structural confinement still asserted by `test_lean_base_structural`.
