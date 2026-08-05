---
phase: 04-nlp-extras-quality-gate
plan: 04
subsystem: testing
tags: [quality-gate, qual-02, d-16, d-17, unittest, pytest, ruff, legacy-rewire]

# Dependency graph
requires:
  - phase: 04-nlp-extras-quality-gate
    provides: 04-01/04-02/04-03 shipped feature set (always-on health+network, gated emotion+summary, interactive menu + hint + friendly errors) and the locked D-07/D-07b/D-08/D-09/D-16/D-17 decisions this terminal gate verifies
provides:
  - QUAL-02 closed: legacy duplicated-logic tests rewired to exercise the REAL chat_analyzer.* modules (test_analysis.py → ChatEDA/add_sentiment_analysis/EmotionAnalyzer/analyze_relationship_health/ChatVisualizer; test_parser.py → real WhatsAppParser/parse_telegram_chat_with_report on inline fixtures)
  - D-16 honored: unittest framework retained in legacy files, aggregators (run_analysis_tests) + `if __name__ == "__main__"` intact, tests/ becomes a package (tests/__init__.py) enabling `python -m unittest tests.test_analysis` module-path invocation
  - D-17 honored: only heavy callables mocked (transformers.pipeline / from_pretrained); real analyzer logic exercised; suite fast + offline-safe + cp1252-safe (MPLBACKEND=Agg, redirect_stdout on sentiment import, -X utf8, UTF-8 fixture writes)
  - Full-suite green in one command: pytest 175/175 pass; ruff src/chat_analyzer tests 380 → 0 errors (full-scope, user-approved)
  - Legacy pandas-2.x drift fixed in-scope: test_end_to_end.py + test_reporting.py freq strings (6H→h etc.) corrected; end-to-end emoji fixture written as UTF-8
affects: [phase verification (QUAL-02 UAT), v1 milestone close-out, legacy-lint debt closure]

# Tech tracking
tech-stack:
  added: []  # NO new packages; tests/__init__.py marks tests a package for unittest module-path runs
  patterns:
    - "Legacy in-place rewire: keep unittest + 7 Test* classes + run_analysis_tests aggregator; replace hand-rolled pandas math with real module calls on fixture DataFrames"
    - "cp1252 safety (Pitfall 5): MPLBACKEND=Agg before pyplot import; redirect_stdout around sentiment.py module import; explicit encoding='utf-8' on fixture file writes; -X utf8 for emoji-printing subprocesses"

key-files:
  created: [tests/__init__.py, .planning/phases/04-nlp-extras-quality-gate/04-04-SUMMARY.md]
  modified: [tests/test_analysis.py, tests/test_parser.py, tests/test_end_to_end.py, tests/test_reporting.py, src/chat_analyzer/** (ruff --fix, no behavior change)]

key-decisions:
  - "tests/__init__.py added (empty) so `python -m unittest tests.test_analysis` and `python -m unittest tests.test_parser` resolve — reconciles the plan's acceptance 'python -m unittest tests.test_analysis -v' which previously failed with ModuleNotFoundError; pytest collection unaffected"
  - "D-17 emotion mock patches ONLY chat_analyzer.analysis.emotion.transformers.pipeline with a faithful flat list-of-dicts, never EmotionAnalyzer; test asserts non-uniform scores (the [0]-parse regression trap)"
  - "User-approved FULL SCOPE for Task 3: pre-existing (baseline, not plan-caused) failures in test_end_to_end.py (11) + test_reporting.py (15), plus the documented 380 pre-existing ruff errors (deferred-items #1), were all fixed so BOTH Task 3 acceptance gates (pytest exit 0 + ruff 0 errors) genuinely close"
  - "end_to_end emoji fixture (😊/💪) written with encoding='utf-8' — the cp1252 default (Pitfall 5) crashed every TestCompletePipeline setUp on Windows"

patterns-established:
  - "Full-suite terminal gate = `python -m pytest` (unittest + pytest styles, one discovery) AND `python -m ruff check src/chat_analyzer tests` — both must exit 0 to close QUAL-02"