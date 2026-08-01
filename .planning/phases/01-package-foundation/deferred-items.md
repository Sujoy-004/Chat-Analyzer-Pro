# Deferred Items — Phase 01 (Package Foundation)

Items discovered during plan 01-02 execution that are out of scope for the
current task and carried forward.

| Item | Found During | Description | Recommended Action |
|------|-------------|-------------|---------------------|
| Stale codebase map | Plan 01-02 Task 3 close-out | `.planning/codebase/STRUCTURE.md` (and likely STACK.md/ARCHITECTURE.md) still describe the pre-restructure layout: `app/`, `deployment/`, `.streamlit/`, `src/parser/` without the `chat_analyzer/` package and without the new `cli/` subpackage. Plan 01 moved the whole tree and did not refresh the map; a piecemeal `cli/` addition to a map that lacks `chat_analyzer/` would be inconsistent. | Re-run `/gsd-map-codebase` (or a targeted STRUCTURE.md rewrite) to reflect `src/chat_analyzer/{analysis,ingest,parser,reporting,utils,cli}` and the deleted web-app dirs. |
| Pre-existing env plotly | Plan 01-02 Task 3 pip freeze sweep | `plotly==6.7.0` remains installed in the base Python env (old Streamlit-app era). It is NOT in pyproject dependencies, `pip install -e .` did not pull it, and the QUAL-04 package-tree scan is clean. It only affects the local env, not the wheel. | Uninstall `plotly` from the dev env (`pip uninstall plotly`) when convenient; not a packaging defect. |
| Legacy lint debt | Plan 01-02 Task 3 lint gate | Pre-existing F401/E-style violations in moved legacy modules (e.g. unused imports in `eda.py`/`sentiment.py`) were deliberately left untouched to preserve plan 01's reuse-not-rewrite constraint. | Tracked for a later quality phase (Phase 4 QUAL-02 rewiring/lint cleanup). |
