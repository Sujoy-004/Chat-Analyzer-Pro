# Pitfalls Research

**Domain:** Pip-installable chat-analysis CLI (WhatsApp `.txt` + Telegram `.json` → terminal insights + inline charts + HTML report), targeting non-technical users on Windows
**Researched:** 2026-07-31
**Confidence:** MEDIUM-HIGH (grounded in the existing codebase + official docs + multiple independent sources; per-item confidence listed)

> Phase references below (P1–P7) are *suggested roadmap phases* for the planner to adopt or rename:
> **P1** Package hygiene + CLI skeleton · **P2** Parser hardening · **P3** Terminal output + inline charts · **P4** HTML report · **P5** NLP/heavy-deps integration · **P6** Error handling + UX polish · **P7** CI + real tests

## Critical Pitfalls

### Pitfall 1: Silent `datetime.now()` fallback fabricates timestamps

**What goes wrong:**
Every WhatsApp line that fails `strptime` gets `timestamp = datetime.now()` — the existing parser does exactly this (`src/parser/whatsapp_parser.py:61,63,77,79`). Any unhandled date variant (a new locale, a 4-digit year in an unexpected position, a seconds-less iOS line) produces messages stamped "today" instead of their real date. Date-range stats, hourly activity, streaks, and response-time analysis all become garbage. The failure is **invisible** — the run "succeeds" with wrong numbers.

**Why it happens:**
Naive parsers treat timestamp parse failure as a recoverable edge case. Developers test only with their own phone's export (US or EU locale), so every other format falls into the fallback path silently.

**How to avoid:**
- Strict parsing: on `strptime` failure for a line that matched the message regex, raise with the offending line, OR collect it in a `skipped_lines` counter and surface `"⚠️ 37 lines could not be parsed"` at the end.
- Never fabricate a timestamp. A skipped line is honest; a fake timestamp is corrupt data.
- Sniff the format from the first 20 successfully parsed lines (sample-based detection), then parse the rest with one confirmed format rather than trying N formats per line.

**Warning signs:**
- `date_range.start` equals today or yesterday for an old chat.
- Hour-of-day histogram shows a suspicious spike at the current hour.
- Parsed-line count << raw-line count with no warning printed.

**Phase to address:** P2 (Parser hardening). Verification: unit test asserting a format the parser doesn't know *errors loudly*, never yields `now()` timestamps.

---

### Pitfall 2: WhatsApp regional date-format ambiguity (M/D vs D/M) — "the single biggest source of parser bugs"

**What goes wrong:**
US phones export `6/14/26, 9:07 PM - Alex: msg`; UK/EU phones export `14/06/2026, 21:07 - Alex: msg`. The existing parser tries `%m/%d` **before** `%d/%m` (`whatsapp_parser.py:52-54,68-70`), so `12/3/26` parses as December 3 for an EU user whose chat was March 12. Both formats succeed for day ≤ 12, so the format loop silently picks the wrong one. On top of that: 2- vs 4-digit years, seconds present (`2:30:45 PM`) or absent, and iOS bracket structure `[12/3/26, 2:30:45 PM] Maria: msg` vs Android `12/3/26, 2:30:45 PM - Maria: msg`.

**Why it happens:**
The export format mirrors the phone's regional settings and the platform (iOS vs Android). Developers test one locale and one platform, then assume one regex handles the world.

**How to avoid:**
- Detect format once from a sample: if any sample line has `AM`/`PM` → 12-hour; if year is 4 digits → `%Y`; for the ambiguous M/D-D/M choice use a locale hint (like `whatsapp-chat-analyzer`'s `ParserConfig(locale="it")`) or a majority-vote heuristic across the first 50 lines (a month field > 12 is impossible → resolves ambiguity).
- Support both structural variants (bracket-format iOS lines and dash-format Android lines) in the same run.
- Expose `--date-format` / `--locale` override for the pathological case; document it in the error message.

**Warning signs:**
- A one-year-old chat shows date range starting in the wrong month.
- Two exports from the same user parse with swapped day/month.
- GitHub issues on every WhatsApp parser repo contain "wrong date" complaints (known community pattern).

**Phase to address:** P2. Verification: fixture files for US 12h, EU 24h, iOS bracket, 4-digit-year — each asserting the *correct* month/day.

---

### Pitfall 3: System messages and localized media placeholders leak into message content

**What goes wrong:**
WhatsApp exports contain lines with **no sender and no colon**: `Messages and calls are end-to-end encrypted.` (always line 2 of the file), `Alex added Sam`, `Group name changed to "Trip 2026"`, `This message was deleted`. The existing regex `([^:]+):\s(.*)` cannot match these, so they fall through as "continuation lines" and get **appended to the previous message** — corrupting the last real message before them. Media placeholders are also localized: `<Media omitted>` (en), `<Média absent>` (fr), `<Medien ausgeschlossen>` (de), plus `audio omitted` / `sticker omitted` variants — so string-matching for English `"<Media omitted>"` misses non-English exports (existing `media_patterns`, `whatsapp_parser.py:151-159`, is English-only).

**Why it happens:**
Developers model the export as "every line is a message"; WhatsApp models it as "lines that match a header pattern are messages, everything else is either a continuation or a system notice."

**How to avoid:**
- Classify structurally, not by string: a line that matches the timestamp header but has **no `sender: ` part** is a system message → tag `type="system"` and *drop it* (or count it) — never append to the previous message.
- Detect media by *absence of content*, not by string: a matched line whose message body is empty or matches a **localized** placeholder regex (`<.*omitted.*>`, `.*absent.*`, etc.) → `type="media"`.
- Always skip the `Messages and calls are end-to-end encrypted` line explicitly (it's reliably line 2).

**Warning signs:**
- Top message by word count is a concatenation of unrelated texts.
- "Participants" list contains phrases like "Messages and calls are".
- Media-message count is 0 for a chat you know has photos (localized placeholder missed).

**Phase to address:** P2. Verification: fixtures with French/German exports and system messages asserting `type` classification and that no message body contains `end-to-end encrypted`.

---

### Pitfall 4: Telegram JSON shape drift — single-chat export ≠ full export; `text` is not a string

**What goes wrong:**
The existing parser does `data.get('messages', [])` (`telegram_parser.py:25`). That works for a **full export** (`result.json` → `chats.list[i].messages`) but a **single-chat export** (`result.json` → a bare Chat object with `messages` at top level) — actually that one works too, but the wrapper shape differs. The official schema (core.telegram.org/import-export) is explicit: *"If you exported a single chat, the result.json file will instead represent a single Chat object"*. Other drift points that break naive parsers:
- `text` is a **string OR an array** of strings and entity dicts (`{type: "mention", text: "..."}`). The existing parser drops any dict part without a `'text'` key (`telegram_parser.py:44-46`) — silent content loss.
- Service messages have `type: "service"` with `actor`/`action`/`members` and **no `from`**; channel posts may also lack `from`.
- `date` is usually naive ISO (`2019-05-21T17:10:55`) but sometimes has `Z` or an offset — and it is **UTC**, while WhatsApp exports are *local device time*.
- `id` can exceed 32 bits (schema: "may have more than 32 significant bits").
- `except: continue` in the existing parser (`telegram_parser.py:35-36`) silently drops any malformed message — a 10k-message chat can silently lose hundreds.

**Why it happens:**
Telegram's export format has grown organically (entities, polls, forwards, replies, service actions) and the top-level shape depends on what the user exported. Developers test one export.

**How to avoid:**
- Support all three top-level shapes: `messages` at root (single chat), `chats.list` array (full export), and missing `messages` (non-chat exports like contacts/stories) → friendly error.
- Walk `text` recursively: string → use; list → join str items and dict items' `text`; non-string/empty + media fields present → `<Media omitted>`.
- Map `sender = msg.get('from') or msg.get('actor') or 'Unknown'`; classify `service` messages as system and filter.
- Parse `date` with a helper that handles `Z`, `+HH:MM`, and naive — **and normalize to naive UTC** at parse time.
- Never `except: continue` silently — count dropped messages and report them.

**Warning signs:**
- Total message count differs hugely from what the user sees in the Telegram app.
- Sender column shows many "Unknown" for a channel or group with service messages.
- "0 messages parsed" when the user picks the wrong JSON (e.g., `personal_information` file).

**Phase to address:** P2. Verification: fixtures for single-chat JSON, full-export JSON, and one with entity-array `text` + service messages.

---

### Pitfall 5: Windows console encoding — `UnicodeEncodeError` crashes the CLI on cp1252, and it also kills error messages

**What goes wrong:**
Windows CMD/PowerShell default to code page cp1252 (or cp850). Any `print()` of emoji, box-drawing characters (`┌─┐│`), em-dashes, or arrows raises `UnicodeEncodeError: 'charmap' codec can't encode character`. Two aggravations documented in real-world incidents (caveman#152, AIPass#296, rich#2882/#3437):
1. It crashes even in the **error handler** — the fallback message itself prints `❌` and crashes, so the user sees *nothing*.
2. **Output redirection** (`analyze chat.txt > report.txt`) switches the encoding and crashes where interactive output worked fine.
The existing code already ships this landmine: `summarizer.py:59` prints `✅ {model_name} loaded successfully!`. Rich also auto-detects the "legacy Windows console" and routes through `_win32_console.py` which crashes the same way; setting `NO_COLOR=1` does **not** fix it.

**Why it happens:**
UTF-8 is the Python and file-system default, but Windows console default is still a legacy ANSI code page. PEP 686 (UTF-8 mode default) is targeted at Python 3.15 — not available to this project's 3.10-era support window.

**How to avoid:**
- At CLI entry, before any output: `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` and the same for `stderr` (exact pattern from caveman#152's accepted fix).
- Design the output layer to be **ASCII-first**: use `[OK]`/`[ERROR]`/`[WARN]`, `->`, `|`/`-`/`=` instead of `✅❌→┌─` (Medium article: "If it's production code, it should be ASCII-only" — that is what this audience needs).
- Never rely on the user setting `PYTHONUTF8=1`; document it as a troubleshooting step, not a requirement.
- Test on a real Windows console (CMD), with redirection, and with captured subprocess output — the three environments where crashes surface (VS Code/Windows Terminal hide the bug).

**Warning signs:**
- CLI works in VS Code terminal but crashes in plain CMD.
- `analyze chat.txt > out.txt` crashes while `analyze chat.txt` works.
- The tool's own error output fails to print.

**Phase to address:** P1 (encoding bootstrap belongs in the CLI skeleton from day one). Verification: run the CLI under `cmd /c` with `chcp 1252` and assert emoji-free, crash-free output; CI job with a Windows runner and `PYTHONIOENCODING` unset.

---

### Pitfall 6: Terminal charts fail on Windows and in narrow/piped terminals

**What goes wrong:**
Two chart libraries are candidates here, both with documented failure modes:
- **plotext**: known distortion ("black bars") in Windows Terminal; the high-res 3×2 unicode mosaic markers are explicitly *"not available in windows"* (README); width auto-detection degrades when piped or in narrow terminals. The author's README warns he's rewriting the project and may not respond to issues promptly — an active-maintenance risk for a tool that will be installed by non-technical users for years.
- **Rich**: on legacy Windows consoles or when stdout is redirected, Rich falls back to the legacy Windows renderer and crashes with `UnicodeEncodeError` on box-drawing characters (rich#2882, rich#3437 — the latter traces exactly through `legacy_windows_render → _win32_console.write_text → cp1252.encode`).
- Shared root cause: `shutil.get_terminal_size()` returns (80, 24) fallback when piped/non-interactive, so charts render squashed or wrap.

**Why it happens:**
Developers develop in a wide, UTF-8, ANSI-capable terminal (VS Code, Windows Terminal, macOS) and never test the three hostile environments: plain CMD, `|`-piped output, and redirected output.

**How to avoid:**
- Wrap every chart render in `try/except Exception` and degrade to a plain-text summary (tables via rich/plain `|` tables, ASCII bars) — a chart crash must never take down the whole run.
- Gate charts on interactivity: if `not sys.stdout.isatty()`, skip inline charts entirely (the HTML report still carries the full visualizations).
- Set an explicit chart width (`min(term_width, 100)`) and a `--no-charts` flag.
- Prefer simple ASCII bar/line markers over unicode mosaics; treat fancy glyphs as progressive enhancement.
- If plotext is chosen, pin it and budget for it being unmaintained; consider rich-only tables + ASCII bars for v1 and evaluate plotext again later (this is a decision for the STACK research, but the pitfall is real either way).

**Warning signs:**
- "It looked fine on my machine" and nothing else — nobody has tested CMD/pipes.
- Users report charts as "garbage", "boxes", or "distorted" — font/glyph problems, not logic bugs.
- Piped output (`analyze x.txt | less`) mangles the layout.

**Phase to address:** P3 (terminal output + inline charts). Verification: run with `| Out-Null` and in a 40-column window; assert the run completes and output is readable.

---

### Pitfall 7: Heavy install + first-run model download makes "pip install, run one command" a lie for non-technical users

**What goes wrong:**
- `pip install torch` on Windows pulls the **CUDA-bundled PyPI wheel** (~2–2.5 GB download) — per the PyTorch forums and lerobot docs, Windows PyPI default is a CUDA-Windows wheel; CPU-only requires `--index-url https://download.pytorch.org/whl/cpu` (~250 MB). A `transformers` + `torch` dependency makes the *install* take 10–30+ minutes and multi-GB of disk before the user has even run `analyze`.
- First real run then silently downloads the model: `t5-small` (~242 MB, used by `summarizer.py`) and the default sentiment pipeline (distilbert, ~260 MB) via `from_pretrained` into `~/.cache/huggingface/hub`. A non-technical user sees a frozen terminal for minutes — with zero explanation — or a failure if offline/corporate-proxied.
- This is compounded by the project's own constraints: `transformers` now requires **Python 3.10+ and torch 2.4+** (official install docs), while the repo claims Python 3.8+ — `pip` will either refuse or resolve an old, incompatible transformers.

**Why it happens:**
"Keep the heavy NLP features" was decided for a web app where the *developer* bears the install cost. A CLI shifts that cost onto every end user, and first-run model downloads are a second surprise after install.

**How to avoid:**
- Split deps with extras: `pip install chat-analyzer-pro` = fast core (pandas + vaderSentiment + rich); `pip install chat-analyzer-pro[ai]` = torch + transformers. The base install must be quick — this is the product's first impression.
- Lazy-import every heavy module (torch, transformers, sklearn, networkx) inside the function that needs it, with a clear message: "Installing AI features? Run `pip install chat-analyzer-pro[ai]`."
- On first AI use, print what is being downloaded and how big it is *before* calling `from_pretrained`; respect `HF_HUB_OFFLINE=1`.
- Declare `requires-python = ">=3.10"` honestly (matching transformers) or pin an older transformers line — do not claim 3.8.
- Consider `torch --index-url .../whl/cpu` documentation in README for Windows users; you cannot express per-platform torch variants in `pyproject.toml` dependencies (PyTorch forum: "Pretty much impossible to define dynamically" — a real, unsolved packaging limitation; document the CPU command instead).

**Warning signs:**
- README says "pip install" without mentioning a 2 GB download.
- Any `import` of `src.analysis.summarizer` at module load (see Pitfall 8) drags torch into every run.
- First-run UX untested with a cold `~/.cache/huggingface` and slow network.

**Phase to address:** P5 (NLP/heavy-deps integration) — and the extras split must be decided in P1 packaging because it shapes the dependency metadata. Verification: fresh-venv install of base extras completes in <1 min; `--no-ai` path never imports torch.

---

### Pitfall 8: Import-time crash in `summarizer` + same failure class in other heavy modules

**What goes wrong:**
`src/analysis/summarizer.py:12` does a top-level `from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration` with no `try/except` (documented in `.planning/codebase/CONCERNS.md`). Any `import src.analysis.summarizer` — even just to reach a function the user didn't ask for — raises `ModuleNotFoundError` in any environment without transformers. The parallel failure exists in `src/ingest/ingestion.py` (undeclared `pytesseract`, `pdfplumber`, `pdf2image` imports) and `src/analysis/sentiment.py` handles `textblob` but `vaderSentiment`/`nltk` data downloads (`vader_lexicon`, stopwords) can fail on first run with no guidance. In a CLI, an import-time crash is catastrophic: it happens *before* the CLI can print its own friendly error.

**Why it happens:**
Web-app code could always assume the full requirements.txt was installed (the Docker image shipped everything); a CLI must degrade gracefully when optional deps are absent.

**How to avoid:**
- Module-level imports of heavy/optional deps only inside functions, guarded with `try/except ImportError` that raises a *friendly, actionable* error (what to install, what feature is affected).
- Same rule for `nltk.download('vader_lexicon')` — wrap with offline/cached fallback and clear instructions.
- The CLI entry point module must import **only stdlib + light deps** so `analyze --help` always works even on a broken install.

**Warning signs:**
- Running the CLI without `[ai]` extras yields a raw traceback mentioning `transformers`.
- `analyze --help` is slow because it imports torch.
- Any new `import` at the top of a module under `src/` — code-review tripwire.

**Phase to address:** P5 (enforcement via review in P1). Verification: fresh venv without `[ai]` — CLI help works, and invoking an AI feature prints a guidance message instead of a traceback.

---

### Pitfall 9: Naive-vs-timezone-aware datetime mixing crashes mixed analysis

**What goes wrong:**
Already a live bug in the codebase (CONCERNS.md "Known Bugs"): Telegram parsing produces tz-aware datetimes (`datetime.fromisoformat(date.replace('Z','+00:00'))`, `telegram_parser.py:34`) while WhatsApp parsing produces naive ones. `df.sort_values('datetime')` or `(now - prev)` then raises `TypeError: can't compare offset-naive and offset-aware datetimes`. In the CLI, the failure mode is identical whenever a user analyzes any WhatsApp export (naive) — and becomes a *silent timezone skew* if Telegram data is mixed in, because Telegram exports are UTC while WhatsApp exports are local device time.

**Why it happens:**
Two parsers written independently normalize differently; pandas happily stores both in one column and only blows up at the first comparison.

**How to avoid:**
- Single normalization contract at the parser boundary: **all timestamps → naive UTC** (parse tz-aware, then `.astimezone(timezone.utc).replace(tzinfo=None)`; naive → assume already UTC-equivalent per source convention).
- Store tz metadata separately if you ever want to display local time.
- Add a schema test: assert `df['datetime'].dt.tz is None` for both parsers.

**Warning signs:**
- `TypeError: can't compare offset-naive and offset-aware` in CI or user runs.
- Hour-of-day stats shifted by a few hours for Telegram exports (UTC vs local).

**Phase to address:** P2. Verification: end-to-end test analyzing a WhatsApp fixture, a Telegram fixture, and both together; assert naive-UTC throughout.

---

### Pitfall 10: Packaging landmines — `_init_.py` typo, console_scripts wiring, and the generic `analyze` command name

**What goes wrong:**
The repo ships every package marker as `_init_.py` (missing underscores — CONCERNS.md). That alone breaks `pip install .`, pytest collection, and standard tooling; the CLI pivot *cannot* ship until every package gets a real `__init__.py`. Additional documented failure modes:
- `console_scripts` entry points silently fail when the target function is not importable — e.g., the module lives in a package not listed in `packages`, or the function is only defined under `if __name__ == "__main__":`, or a module-level heavy import crashes before `main()` is reached (see Pitfall 8). The classic symptom: command installed but `ImportError: Entry point ... not found` (SO, setuptools docs).
- **`analyze` is an extremely generic command name.** It will collide with existing PATH entries (there are already PyPI packages exposing `analyze`); on Windows the installed Scripts dir may not even be on the user's PATH. A non-technical user hitting "command not found" or a *different* tool's `analyze` is a fatal first-run experience.
- Wheels silently omit non-`.py` files (templates, CSS/JS, sample data) unless declared via `package-data`/`include-package-data`/MANIFEST.in — the HTML report feature dies in production if it reads a template relative to `os.getcwd()`.

**Why it happens:**
Developers verify with `python src/cli.py` (working dir on sys.path, typo'd markers irrelevant) instead of building a wheel and installing into a clean venv. Entry-point imports also bypass the `__main__` guard, so "it ran as a script" ≠ "it runs as a console command".

**How to avoid:**
- Fix `__init__.py` (all of `src/` + `tests/`) as the first commit of P1; delete the namespace-package fallback thinking entirely.
- Add `[project.scripts]` (`analyze = "src.cli:main"` or a dedicated `cli` package) and smoke-test in a **clean venv**: `pip install .` → `analyze --help` → `python -m src` parity.
- Reconsider the command name: prefer something collision-resistant (`chat-analyzer`, `chatlytics`, `cpro`) or ship BOTH the documented friendly alias and a namespaced `python -m` fallback that always works. Decide this in P1; renaming later breaks the README and user habits.
- Use `importlib.resources` (not `__file__`/cwd) for any shipped asset; declare `package-data` for HTML templates.
- Test the sdist as well as the wheel (`pip install` from `sdist`), since non-technical users may install from a GitHub archive.

**Warning signs:**
- `pip install .` fails or installs nothing (`pip show chat-analyzer-pro` says files but `import src` fails).
- `analyze` runs something that isn't your tool (name collision) or `analyze` is "not recognized".
- HTML report renders from the repo but 404s/missing-template from the installed wheel.

**Phase to address:** P1 (packaging) + P4 (report assets). Verification: CI job `pip install .` in a clean venv on Windows + Linux, run the console script, assert output.

---

### Pitfall 11: HTML report — encoding, escaping, and output-path failures

**What goes wrong:**
- Files written with the platform default encoding (`open(path, 'w')`) on Windows → cp1252 mojibake or `UnicodeEncodeError` for emoji/Devanagari/Chinese content — the entire point of the product is analyzing chats full of emoji and non-Latin scripts.
- Report HTML without `<meta charset="utf-8">` renders as mojibake in browsers.
- Chat content interpolated unescaped into HTML: a message containing `<script>` or stray `<b>` breaks the page layout (and is an injection vector if the report is shared — see Security table).
- Writing the report to the current working directory surprises users who ran `analyze C:\Users\me\Desktop\chat.txt` from a different folder; a filename derived from a non-ASCII chat name can hit filesystem encoding issues.
- Embedding matplotlib figures (existing `visualization.py` produces matplotlib objects) requires the `Agg` backend set *before* matplotlib import — otherwise headless runs fail with `no display` or spawn GUI windows.

**Why it happens:**
"Save an HTML file" looks trivial; encoding, escaping, and cwd semantics are invisible until a real user with a real filename and a real emoji-laden chat runs it.

**How to avoid:**
- Always `open(path, 'w', encoding='utf-8')` and emit `<meta charset="utf-8">` + `<!DOCTYPE html>`.
- `html.escape()` every piece of chat-derived content (messages, sender names, stats values) before interpolation.
- Default report path: `<input_stem>_report.html` in the same directory as the input; print the absolute path after writing.
- Set `matplotlib.use('Agg')` before any matplotlib import (or import `ChatVisualizer` lazily); keep the CLI import graph matplotlib-free.
- Sanitize the output filename (strip path separators, control chars, leading dots) even when derived from input.

**Warning signs:**
- Report opens as garbled text in a browser.
- A chat message containing `<3` or `<a href` visibly breaks the report layout.
- Report lands in an unexpected folder; user can't find it.
- `analyze` crashes with `TclError: no display name` on a headless box.

**Phase to address:** P4. Verification: fixture chat with emoji + `<script>` text + non-ASCII sender name → assert valid UTF-8 file, escaped content, expected path, and `Agg`-safe headless run.

---

### Pitfall 12: Streamlit-era dependency baggage bloats the CLI install

**What goes wrong:**
`requirements.txt` still lists `streamlit`, `plotly`, `seaborn`, `wordcloud`, `reportlab`, `pytz`, etc. — every one of these gets pulled into the CLI install as hard dependencies if copied as-is. The user's `pip install chat-analyzer-pro` silently installs a web framework they'll never use, adding hundreds of MB and a large attack surface. The root manifest also diverges from `deployment/requirements.txt` (CONCERNS.md), so "which requirements are real" is already ambiguous. Unpinned `>=` ranges everywhere make resolution non-reproducible and allow a new pandas/streamlit major to break the CLI.

**Why it happens:**
The pivot reuses `src/` (right call) but a natural shortcut is to reuse the *requirements* too (wrong call) — the app-layer deps are not the analysis-core deps.

**How to avoid:**
- Build a fresh `dependencies` list in pyproject from the modules the CLI actually imports at runtime (pandas, numpy, vaderSentiment, rich, maybe plotext, regex, tqdm) — verified by a script that scans `import` statements, then trimmed by hand.
- Prune `streamlit`, `plotly`, `seaborn`, `wordcloud`, `reportlab` unless a phase genuinely needs them (PDF report is explicitly out of scope).
- Pin with `~=` (compatible-release) or generate a lockfile; deliberately test upgrades (CONCERNS.md "unpinned ranges" risk).

**Warning signs:**
- `pip install chat-analyzer-pro` output shows `streamlit` being resolved.
- Fresh-venv install time > 2 minutes for the base (non-AI) extra.
- Two requirements files still exist and disagree.

**Phase to address:** P1. Verification: `pip install .` in a clean venv, then `pip freeze` contains no streamlit/plotly/seaborn.

---

### Pitfall 13: No progress feedback on long-running analysis (row loops + model loads)

**What goes wrong:**
The analysis core loops rows in Python (`relationship_health.py`, `emotion.py`, `summarizer.py` — all flagged in CONCERNS.md) and the summarizer's T5 load blocks for ~30s–minutes. A CLI with no progress output *looks hung*: the user's single `analyze chat.txt` returns nothing for 60+ seconds, so they Ctrl+C, conclude the tool is broken, and never come back.

**Why it happens:**
Web apps show spinners implicitly (Streamlit re-renders); a CLI must explicitly narrate progress, and it usually doesn't because the developer's own runs are fast on small chats.

**How to avoid:**
- Stage-by-stage narration: `Parsing chat…`, `Computing sentiment…`, `Loading AI models (first run downloads ~250 MB)…`, `Writing report → C:\…\_report.html`.
- `tqdm` over message iteration for parse + analysis passes (tqdm already in requirements).
- Print the parsed-message count immediately after parsing ("Parsed 12,483 messages from 3 participants") — it proves the file was understood and gives scale context.

**Warning signs:**
- Any wait >2s without stdout output.
- Users report the tool "hangs" or "does nothing".

**Phase to address:** P3 (output) + P6 (UX polish). Verification: run against a 50k-message fixture, assert progress output appears within 2s and every stage is narrated.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| `datetime.now()` fallback on parse failure | "It never crashes" | Fabricated timestamps corrupt all stats silently | **Never** — error loudly or skip-and-count |
| Single regex assumed to cover all WhatsApp locales | Fast to write | Every locale/device quirk is a new bug report | **Never** — sample-based format detection is cheap |
| `except: continue` when parsing Telegram messages | Skips malformed messages | Silent data loss; user gets wrong "total messages" | Only with a skipped-count warning surfaced to the user |
| Copy `requirements.txt` into the CLI unchanged | One-line reuse | Streamlit/plotly/seaborn baggage, multi-hundred-MB installs | **Never** for a CLI pivot — rebuild the dep list from imports |
| String-match English `"<Media omitted>"` | Simple | Non-English exports report 0 media messages | Only for v1 with a documented `--media-words` override |
| Import torch/transformers at module top | Simpler code | Every code path pays import cost; `--help` breaks without extras | **Never** — lazy import with friendly errors |
| Ship emoji/box-drawing in CLI output | Prettier demos | Crash on cp1252 Windows consoles | Only behind a detected-UTF-8 gate with ASCII fallback |
| Write report to cwd | No path logic | Users can't find the output; surprises | Never — write next to the input, print the path |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| WhatsApp `.txt` input | Assume one date/time format | Sniff format from first 20 lines; support 12h/24h, 2/4-digit year, seconds or not, iOS-bracket and Android-dash variants |
| WhatsApp `.txt` input | `open(path, 'r')` without encoding | `encoding='utf-8-sig'` (WhatsApp sometimes adds a BOM) then `errors='replace'` fallback; the BOM silently kills the first line's regex match otherwise |
| Telegram `.json` input | `data.get('messages')` only | Handle bare Chat (single-chat export) vs `chats.list[]` (full export); reject non-chat JSONs with a clear message |
| Telegram `.json` input | Treat `text` as a string | Recursively join str + dict-entity parts; empty text + media fields ⇒ media message |
| HuggingFace models (`t5-small`, sentiment) | `from_pretrained` with no forewarning | Print model name + approximate download size first; honor `HF_HUB_OFFLINE=1`; cache at `~/.cache/huggingface/hub` (documented by HF) |
| torch on Windows | Declare `torch` in dependencies and forget it | PyPI default Windows wheel bundles CUDA (~2 GB); document `pip install torch --index-url https://download.pytorch.org/whl/cpu` — per-platform index-urls can't be expressed in pyproject |
| matplotlib figures → HTML | `import matplotlib` at top level | `matplotlib.use('Agg')` before import; embed PNGs via base64 with `importlib.resources` for any template |
| Windows console stdout | `print("✅ …")` anywhere | ASCII markers `[OK]`/`[ERROR]`; if unicode is a must, gate on `sys.stdout.encoding` and use `errors='replace'` |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Row-wise `.iterrows()`/`.loc` loops in health/emotion/summarizer analysis (CONCERNS.md) | Analysis takes minutes on large chats; CPU pinned | Vectorize with `shift()`/`diff()`/`groupby`; at minimum wrap loops in `tqdm` so it looks alive | ~50k+ messages (typical WhatsApp export can hit 40k with media) |
| Whole file read into memory | RAM spikes on big exports | Read line-by-line for WhatsApp txt; cap input size with a warning; count + stream where possible | Multi-hundred-MB exports |
| First-run `from_pretrained` (T5-small ~242 MB, distilbert ~260 MB) | Frozen terminal for minutes on first AI run | Pre-download model weights at *install* time via a post-install script (optional extra), or print size + progress; cache is reused afterwards | First AI run on any machine; every machine on a fresh cache |
| Re-parsing the file for every chart/report | Duplicate O(n) passes | Parse once into a DataFrame, pass it through the pipeline (CLI is one-shot so this is mostly free — just don't re-read the file) | Any run with >1 output stage |
| Rich/plotext width detection in piped mode | Squashed/wrapped charts | Fix width explicitly (`min(term_width, 100)`); skip charts when `not sys.stdout.isatty()` | Any `analyze x.txt > file` or CI run |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Chat content interpolated into HTML report unescaped | Broken report; stored/injection vector when reports are shared | `html.escape()` every message/sender/stats string (the old app's `unsafe_allow_html` bug must not survive the pivot) |
| `exec()` of runtime-downloaded code carried over from the web app | Arbitrary code execution — CONCERNS.md calls this critical | The CLI imports `src/` directly; delete the GitHub-fetch loader entirely (already planned) |
| Full tracebacks printed to non-technical users | Confusing; leaks paths | Top-level `except Exception` → friendly message + `--verbose` for traceback (old app's `st.exception` pattern inverted) |
| Unbounded file reads (no size cap) | OOM on a 2 GB "chat" | Warn above ~100 MB, refuse above a cap, and count lines to bound memory |
| Secrets in code (weekly_digest SMTP/TG tokens) | Credential leaks — out of scope for v1, but `src/reporting/weekly_digest.py` already has plaintext patterns | Keep weekly_digest out of the CLI v1; if resurrected, load tokens from env only |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| No Python installed / no understanding of venvs ("I ran pip and it broke my Python") | Install fails or destroys another tool's env | README quickstart with a copy-paste `pip install --user chat-analyzer-pro` or a provided `install.bat`; explain what Python is in one line |
| `analyze` command name collision or missing Scripts dir on PATH | "Command not recognized" or runs the *wrong* tool | Collision-resistant name + always-documented `python -m chat_analyzer` fallback |
| File-picker-free CLI: user must type a path with spaces | Typing errors, quoting confusion | Accept the path as-is (no shell needed if quoted); print "file not found: <path>" with the exact path echoed back so they see the typo |
| Wrong-format input (e.g., a `.txt` that's actually a Telegram HTML export) | "0 messages parsed" dead end | Detect extension *and* content sniffing; error message tells them which export format each supported tool produces and links the README section |
| Silent long waits (model download, analysis) | User kills the process | Narrate every stage; print counts and progress bars (Pitfall 13) |
| Emoji/unicode output that crashes on CMD | Tool "crashes" for Windows-default users | ASCII-first output layer; test in real CMD (Pitfall 5) |
| Report file "lost" | User can't find output | Write next to input, print absolute path, and (v1 nice-to-have) offer `--out` flag |

## "Looks Done But Isn't" Checklist

- [ ] **Parser:** Tested only with your own phone's locale — verify US 12h, EU 24h, iOS bracket, 4-digit-year, and a non-English media placeholder fixture
- [ ] **Parser:** No `datetime.now()` fallback remains anywhere in `src/parser/` — grep for `datetime.now()` and delete those lines
- [ ] **Packaging:** `pip install .` into a *clean venv* (not `python src/cli.py`) and run the console script — this is the only test that catches `_init_.py`, entry-point, and PATH issues
- [ ] **Packaging:** All `src/**/_init_.py` renamed to `__init__.py`; `tests/` has one too; `packages`/`[tool.setuptools]` config lists everything
- [ ] **Console:** Run under `cmd /c chcp 1252 && analyze chat.txt` — if it prints anything non-ASCII without crashing, you're safe
- [ ] **Console:** Run `analyze chat.txt > out.txt` (redirection) — the encoding must not change
- [ ] **Charts:** Run in a 40-column window and with `| Out-Null` — chart code must degrade, not crash
- [ ] **HTML:** Open the report in a browser; check emoji, `<3` in a message, and a sender named `Alice <3 Bob`
- [ ] **Dependencies:** `pip freeze` after base install contains no `streamlit`/`plotly`/`seaborn`; `transformers`/`torch` only via `[ai]` extra
- [ ] **AI path:** Cold `~/.cache/huggingface` + `[ai]` extras — model download prints size/progress before freezing; `--no-ai` never imports torch
- [ ] **Errors:** Trigger "file not found", "wrong extension", "empty file", "0 messages parsed" — each prints a *friendly* message with next steps, not a traceback
- [ ] **Tests:** Tests import `src.parser.whatsapp_parser` and `src.parser.telegram_parser` (the production modules), not inline mock reimplementations (CONCERNS.md: current tests exercise nothing real)

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Fabricated timestamps (Pitfall 1) | HIGH — output data is wrong, but input file is intact | Re-run after fixing the parser; the fix is *informing* the user which lines were skipped; no data migration needed (no persistence in v1) |
| UnicodeEncodeError crash (Pitfall 5) | LOW — transient, environment-specific | Add `sys.stdout.reconfigure(...)` at CLI entry; advise user to set `PYTHONUTF8=1` as stopgap |
| tz-naive/aware TypeError (Pitfall 9) | LOW | Single normalization point in the parser layer; re-run |
| `analyze` name collision / PATH problem (Pitfall 10) | MEDIUM — user-visible confusion | Ship `python -m` fallback + document `pip install --user` Scripts dir; consider renaming the command in a later release with a deprecation alias |
| Torch/transformers install explosion (Pitfall 7) | MEDIUM — user already installed the giant version | Provide `[ai]` extras so future installs are small; document `pip uninstall torch` + CPU-index reinstall for those affected |
| Report mojibake/broken HTML (Pitfall 11) | LOW | Regenerate with `encoding='utf-8'` + `html.escape()`; always re-writeable from the parsed DF — keep the pipeline one-shot and idempotent |
| Charts rendering garbage (Pitfall 6) | LOW | Chart code is wrapped in try/except; tell user `--no-charts`; rely on HTML report for visuals |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Silent `datetime.now()` timestamps (1) | P2 Parser hardening | Unit test: unknown format → loud error/skip-count, never `now()` |
| WhatsApp regional formats (2) | P2 | 4 fixture exports (US/EU/iOS/4-digit-year) assert correct dates |
| System messages + localized media (3) | P2 | FR/DE fixtures; no message body contains "encrypted"; media counts > 0 |
| Telegram JSON shape drift (4) | P2 | Single-chat, full-export, entity-text fixtures; dropped-message counter |
| Windows console encoding (5) | P1 CLI skeleton | `cmd /c chcp 1252` + redirection test on a Windows CI runner |
| Terminal chart failures (6) | P3 Terminal output | 40-col window + piped run complete without crash; `--no-charts` |
| Heavy install/model download (7) | P1 packaging (extras) + P5 AI | Fresh-venv base install <1 min; cold-cache AI run prints size first |
| Import-time transformers crash (8) | P1 review rule + P5 | `analyze --help` works without `[ai]` extras; AI feature prints guidance |
| tz-naive/aware mixing (9) | P2 | Both parsers assert naive-UTC column; mixed-source e2e test |
| Packaging landmines + `analyze` name (10) | P1 | Clean-venv `pip install .` → console script runs on Windows + Linux CI |
| HTML report encoding/escaping/path (11) | P4 HTML report | Emoji + `<script>` fixture → valid UTF-8, escaped, correct path |
| Streamlit-era dependency baggage (12) | P1 | `pip freeze` after base install excludes streamlit/plotly/seaborn |
| No progress feedback (13) | P3 + P6 UX polish | 50k-message fixture: progress output within 2s, all stages narrated |

## Sources

- **Codebase (authoritative for existing bugs):** `.planning/codebase/CONCERNS.md`; `src/parser/whatsapp_parser.py` (`datetime.now()` fallbacks at :61,63,77,79; English-only media patterns :151-159); `src/parser/telegram_parser.py` (:34 tz-aware parse, :35 `except: continue`); `src/analysis/summarizer.py` (:12 top-level transformers import, :59 emoji print). [HIGH]
- **Telegram export schema (official):** https://core.telegram.org/import-export — single-chat export is a bare Chat object; `text` is String or Array of String/MessageEntity; `actor`/`members` for service messages; `id` > 32 bits. [HIGH]
- **WhatsApp export format quirks:** https://www.whatsquiz.com/blog/whatsapp-chat-export-file-format/ (regional date formats = "the single biggest source of parser bugs"; localized media placeholders; system events without sender; encryption notice always line 2; UTF-8 BOM) and https://chatanalyzer.syncori.net/en/blog/how-to-export-whatsapp-chat (Android vs iOS format differences; "date varies between Android and iOS and between regions"; 40k-message export cap). [MEDIUM — two independent secondary sources; WhatsApp FAQ (faq.whatsapp.com/1180414079177245) cited by both but not directly fetchable]
- **Windows console cp1252 crashes:** https://github.com/JuliusBrussee/caveman/issues/152 (crash in error handler; `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` fix); https://github.com/AIOSAI/AIPass/issues/296 (rich + cp1252); https://stackoverflow.com/questions/78817860 (redirection changes encoding → crash); https://github.com/Textualize/rich/issues/3437 and https://github.com/Textualize/rich/issues/2882 (rich legacy Windows renderer crashes on redirect; `NO_COLOR` doesn't help; PEP 686 UTF-8 default in 3.15). [HIGH]
- **plotext Windows/terminal limits:** https://github.com/piccolomo/plotext (README: 3×2 mosaic markers "not available in windows"; author maintenance notice) + https://news.ycombinator.com/item?id=27719759 (Windows Terminal distortion report). [MEDIUM]
- **torch/transformers install friction:** https://discuss.pytorch.org/t/index-url-to-install-pytorch/198253 and https://discuss.pytorch.org/t/torch-cuda-installation-on-cpu-only-machine/169962 (PyPI default Windows wheel ships CUDA runtime; "pretty much impossible to define dynamically" per-platform); https://huggingface.co/docs/lerobot/installation (Windows PyPI default = CUDA-Windows wheel; CPU wheel via `--index-url https://download.pytorch.org/whl/cpu`); https://pypi.org/project/transformers/ (Python 3.10+, torch 2.4+); https://huggingface.co/docs/transformers/installation (model download/cache at `~/.cache/huggingface/hub`, `HF_HUB_OFFLINE`). [MEDIUM-HIGH — sizes approximate]
- **Packaging:** https://setuptools.pypa.io/en/stable/userguide/entry_point.html (console_scripts must point to an importable function); https://packaging.python.org/en/latest/specifications/entry-points/ (console script = command name; case-insensitive filesystem caveats); https://stackoverflow.com/questions/39280326 (entry point not found when module not installed); https://www.w3reference.com/blog/python-why-setuptools-doesn-t-include-my-package-data/ (package-data omission). [HIGH]
- **Community practice (competitor parser):** https://pypi.org/project/whatsapp-chat-analyzer/ — validates the recommended pattern: automatic iOS/Android format detection, locale hint to disambiguate M/D vs D/M, multilingual media classification, BOM handling, `max_lines_per_message` guard. [MEDIUM]

---

*Pitfalls research for: Chat-Analyzer-Pro CLI pivot (WhatsApp/Telegram chat analysis CLI)*
*Researched: 2026-07-31*
