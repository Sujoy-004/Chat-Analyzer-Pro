# Testing Patterns

**Analysis Date:** 2026-07-31

## Test Framework

**Runner:**
- `unittest` (Python standard library) — the ONLY test framework in use
- pytest is NOT used: `pytest>=7.4.0` and `pytest-cov>=4.1.0` appear only as COMMENTED-OUT lines in `requirements.txt:86-87`
- No pytest config, no `pytest.ini`, `setup.cfg`, `pyproject.toml`, or `conftest.py` exists anywhere in the repo
- No CI configuration exists (no `.github/` directory, no Makefile)

**Assertion Library:**
- `unittest.TestCase` assertion methods: `assertEqual`, `assertIn`, `assertTrue`, `assertIsInstance`, `assertGreater`, `assertGreaterEqual`, `assertLess`, `assertLessEqual`, `assertAlmostEqual`, `assertIsNotNone`
- pandas assertions used occasionally: `pd.api.types.is_datetime64_any_dtype(df['datetime'])` (`tests/test_parser.py:57`, `tests/test_parser.py:219`)

**Run Commands:**
```bash
python -m unittest discover -s tests -v     # Discover & run all tests in tests/
python -m unittest tests.test_parser -v     # Run one test module
python tests/test_parser.py                 # Run via module __main__ block (unittest.main())
python -c "from test_parser import run_parser_tests; run_parser_tests()"  # Per-file suite runner
```
- Each test file defines a `run_*_tests()` function that builds a `TestSuite` and runs with `TextTestRunner(verbosity=2)` (e.g., `run_parser_tests` at `tests/test_parser.py:314-326`, `run_analysis_tests` at `tests/test_analysis.py:381-397`, `run_reporting_tests` at `tests/test_reporting.py:406-422`, `run_end_to_end_tests` at `tests/test_end_to_end.py:628-646`)
- There is NO single aggregator that runs all four test files together

## Test File Organization

**Location:**
- All tests live in the top-level `tests/` directory — NOT co-located with source. Production code is in `src/`; there is no `test` subpackage, no `tests/__init__.py`.

**Naming:**
- `test_<target_module>.py`:
  - `tests/test_parser.py` → covers `src/parser/whatsapp_parser.py` + `src/parser/telegram_parser.py`
  - `tests/test_analysis.py` → covers `src/analysis/` (EDA, sentiment, emotion, relationship health, gamification, rolling score, visualization)
  - `tests/test_reporting.py` → covers `src/reporting/` (PDF report, weekly digest, email/Telegram delivery, scheduling)
  - `tests/test_end_to_end.py` → cross-module pipeline tests

**Structure:**
```
tests/
├── test_parser.py          # TestWhatsAppParser, TestTelegramParser, TestParserEdgeCases
├── test_analysis.py        # TestEDAModule, TestSentimentAnalysis, TestEmotionClassification,
│                           # TestRelationshipHealth, TestGamificationFeatures,
│                           # TestRollingHealthScore, TestVisualizationIntegration
├── test_reporting.py       # TestPDFReportGeneration, TestWeeklyDigest, TestEmailDelivery,
│                           # TestTelegramDelivery, TestReportAttachments, TestScheduling,
│                           # TestReportingIntegration
└── test_end_to_end.py      # TestCompletePipeline, TestDataFlow, TestErrorHandling,
                            # TestScalability, TestModuleIntegration, TestOutputFormats,
                            # TestRobustness, TestPerformance, TestDataValidation
```

## Test Structure

**Suite Organization:**
- Each test class extends `unittest.TestCase` and is named `Test<ModuleOrArea>` (PascalCase)
- `setUp()` builds fixture data; `tearDown()` cleans up temp files
- Every test method is named `test_<behavior>` and carries a one-line docstring

```python
# tests/test_parser.py:13-41
class TestWhatsAppParser(unittest.TestCase):
    """Test cases for WhatsApp parser."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_whatsapp_text = """12/25/23, 9:30 AM - Alice: Hey! How are you?
12/25/23, 9:35 AM - Bob: I'm good, thanks! 😊"""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        self.temp_file.write(self.sample_whatsapp_text)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_basic_parsing(self):
        """Test basic WhatsApp message parsing."""
        df = self._parse_whatsapp_file(self.temp_file.name)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5, "Should parse 5 messages")
        self.assertIn('datetime', df.columns)
```

**Patterns:**
- **Setup:** inline fixture DataFrames built with `pd.DataFrame({...})` and `pd.date_range`/`np.random` (e.g., `tests/test_analysis.py:17-23`); sample chat text as multi-line strings; `tempfile.NamedTemporaryFile` for file I/O tests
- **Teardown:** `os.unlink()` on temp files (some tests inline temp creation and clean up at method end instead of `tearDown` — see `tests/test_parser.py:75-93`)
- **Assertion:** behavioral assertions with inline failure messages on the critical ones (`self.assertEqual(len(df), 5, "Should parse 5 messages")`)
- **Range checks** are the norm for score-like outputs: `assertGreaterEqual(score, 0)` + `assertLessEqual(score, 1)` (`tests/test_analysis.py:218-219`)
- **Column checks:** `assertIn('datetime', df.columns)` verifies schema after parsing/analysis steps

## Mocking

**Framework:** None. `unittest.mock` (MagicMock/patch) is NOT used anywhere in `tests/`.

**Patterns:**
- "Mocking" is done by **re-implementing the production function inline inside the test class** as a private helper with a `# Mock` or `# Helper method (mock implementation)` comment. The tests then assert against this duplicated logic, NOT the real `src/` module:

```python
# tests/test_parser.py:134-136
# Helper method (mock implementation)
def _parse_whatsapp_file(self, filepath):
    """Mock WhatsApp parser for testing."""
    import re
    ...
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[AP]M)\s-\s([^:]+):\s(.+?)(?=\d{1,2}/\d{1,2}/\d{2,4},|\Z)'
    ...
```

- The same duplicated regex appears in `_parse_whatsapp_file` (`tests/test_parser.py:143`) and `_mock_parse` (`tests/test_end_to_end.py:124`) — both copy parser logic instead of importing `src/parser/whatsapp_parser.py`
- Sentiment tests hardcode score lists rather than invoking `vaderSentiment`: `positive_scores = [0.8, 0.9, 0.7, 0.85]` (`tests/test_analysis.py:94`)
- Analysis tests operate on self-constructed DataFrames and re-derive metrics with pandas directly rather than calling `src/analysis/relationship_health.py` functions (compare `tests/test_analysis.py:161-170` with `identify_conversation_starters` in `src/analysis/relationship_health.py:28`)
- **Consequence:** no test in `tests/` imports anything from `src/` (verified: zero `import src` / `from src` matches in `tests/`). The suite currently validates test-authored logic, not production code.

**What to Mock:**
- Per current convention, external dependencies (VADER, transformers, network requests, email SMTP) are never invoked in tests — behavior is replaced by literal fixtures. If you keep this approach, isolate the duplicated logic so it does not drift from `src/`.
- `app/streamlit_app.py` downloads modules from GitHub at runtime (`app/streamlit_app.py:42-71`) — never exercise this in unit tests; stub the network layer instead.

**What NOT to Mock:**
- Do not duplicate production regex/parsing logic in test helpers (current practice, and the main reason tests can pass while production code is broken). Prefer importing the real functions from `src/` (`from src.parser.whatsapp_parser import WhatsAppParser`) and mocking only heavyweight external calls (transformers pipelines, `requests.get`, SMTP) — e.g., with `unittest.mock.patch`.

## Fixtures and Factories

**Test Data:**
- Fixtures are inline per test class via `setUp()` — no shared fixture files, no factory functions, no `conftest.py`
- DataFrame fixtures follow the canonical schema (`datetime`, `sender`, `message`, `message_length`, optional `sentiment`/`sentiment_score`) — see `tests/test_analysis.py:18-23`, `tests/test_reporting.py:20-27`
- Synthetic data uses `np.random.randint` / `np.random.choice` / `np.random.uniform` for filler columns (nondeterministic, but assertions are range-based so they hold)
- Sample chat files exist in the repo at `data/sample_chats/whatsapp_sample.txt` and `data/sample_chats/telegram_sample.json`, but the tests do NOT reference them — all fixtures are embedded strings/dicts

**Location:**
- Inline in test classes only. No `tests/fixtures/` directory.

## Coverage

**Requirements:** None enforced. No coverage config file exists; `pytest-cov>=4.1.0` is commented out in `requirements.txt:87`. `.gitignore:44-52` anticipates coverage artifacts (`.coverage`, `htmlcov/`, `coverage.xml`) but none are generated.

**View Coverage:**
```bash
pip install pytest pytest-cov
python -m pytest --cov=src tests/ --cov-report=term-missing   # Requires installing pytest first
```
(Not part of the current project setup — would need to be added.)

## Test Types

**Unit Tests:**
- `tests/test_parser.py` — parsing correctness and edge cases (sender extraction, datetime parsing, emoji preservation, multiline messages, system-message filtering, empty files, Unicode, long messages)
- `tests/test_analysis.py` — EDA stats, sentiment threshold logic, emotion categories, relationship health metrics, streaks, milestones, rolling windows, visualization data prep
- `tests/test_reporting.py` — PDF metadata, digest summaries, email/Telegram formatting and delivery config, attachments, scheduling
- Note: these are unit tests in the "single file per module" sense, but they test duplicated logic rather than the `src/` implementation (see Mocking section)

**Integration Tests:**
- `tests/test_end_to_end.py` — pipeline flow tests that chain parse → analyze → visualize/report steps using the in-class `_mock_parse` helper (`tests/test_end_to_end.py:117-139`), plus module-to-module data-flow tests (`TestDataFlow`, `TestModuleIntegration`)
- `TestErrorHandling` (`tests/test_end_to_end.py:230-284`) covers empty/malformed/missing-column/invalid-datetime scenarios; `TestScalability` covers 10k-row datasets and a 10MB memory bound; `TestPerformance` includes a wall-clock assertion

**E2E Tests:**
- No browser/UI E2E framework (no Selenium, Playwright, or Streamlit AppTest). The "end-to-end" suite is a simulated in-process pipeline.

## Common Patterns

**Async Testing:**
- No async code in the project (no `asyncio`, no `await`) — not applicable

**Error Testing:**
```python
# tests/test_end_to_end.py:272-284 — invalid datetime coercion
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
valid_rows = df[df['datetime'].notna()]
self.assertEqual(len(valid_rows), 1)
```
```python
# tests/test_end_to_end.py:246-256 — malformed input should not crash
try:
    result = []  # Mock: no valid messages found
    self.assertEqual(len(result), 0)
except Exception as e:
    self.assertIsInstance(e, (ValueError, KeyError))
```

**File I/O Testing (temp files):**
```python
# tests/test_reporting.py:292-301
pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
pdf_path.write(b'%PDF-1.4 fake pdf content')
pdf_path.close()
self.assertTrue(os.path.exists(pdf_path.name))
self.assertTrue(pdf_path.name.endswith('.pdf'))
os.unlink(pdf_path.name)
```

**Output Format Testing:**
- CSV round-trip (`tests/test_end_to_end.py:391-412`): write DataFrame to temp CSV, `pd.read_csv` it back, assert row counts equal
- JSON round-trip (`tests/test_end_to_end.py:414-434`): `json.dump` then `json.load`, assert values
- HTML content assertions with `assertIn('<html>', content)` (`tests/test_end_to_end.py:436-459`)

**Score Range Testing (idiomatic across all files):**
```python
self.assertGreaterEqual(health_score, 0)
self.assertLessEqual(health_score, 1)
```

---

*Testing analysis: 2026-07-31*
