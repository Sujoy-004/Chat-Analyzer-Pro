# Deferred Items — Phase 2 Plan 02

Out-of-scope discoveries logged per the executor scope boundary (do not fix
during this plan unless a later task requires it).

## Pre-existing lint findings in src/chat_analyzer/analysis/eda.py (reused module, not in ruff gate)

Discovered while verifying Task 5 (run_pipeline consumes `ChatEDA.analyze_content`).

- `eda.py:1:1 I001` — import block un-sorted (pandas/numpy/matplotlib/seaborn/Counter/re/datetime)
- `eda.py:4:19 F401` — `seaborn` imported but unused (only `sns` palette would use it; `sns` never referenced)
- `eda.py:7:22 F401` — `datetime` imported but unused
- `eda.py:135:14 RUF059` — unpacked `axes` never used in `create_dashboard`

These predate Phase 2 and are outside the ruff gates (cli/, parser/+, ingestion.py
non-growth). eda.py is a "reuse, don't rewrite" module per AGENTS.md — leaving as-is.

## Fixed in Task 5 (NOT deferred) — for the record

- `eda.py:101` `_clean_text` regex `r'[^\\w\\s]'` over-escaped `\w`/`\s` → kept only
  `\`, `w`, `s` chars, making `word_frequency` empty for every chat. Fixed to
  `r'[^\w\s]'` (one-character Rule 1 bugfix, directly blocking Task 5's top_words
  done-criteria). Documented in the Task 5 GREEN commit.
