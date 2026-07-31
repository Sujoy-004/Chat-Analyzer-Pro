# External Integrations

**Analysis Date:** 2026-07-31

## APIs & External Services

**Runtime module loading (GitHub raw content):**
- GitHub raw content API — `app/streamlit_app.py` lines 46-51 fetches four analysis modules at startup via `https://raw.githubusercontent.com/Sujoy-004/Chat-Analyzer-Pro/refs/heads/main/...`:
  - `src/ingest/ingestion.py` (ingestion)
  - `src/parser/whatsapp_parser.py` (whatsapp_parser)
  - `src/parser/telegram_parser.py` (telegram_parser)
  - `src/analysis/relationship_health.py` (relationship_health)
- Implementation: `requests.get(url, timeout=10)` with `@st.cache_data` (`load_github_modules`); code is executed via `exec(code, namespace)` in `load_and_execute_modules` (lines 73-111). Failure degrades to built-in fallback parsers (lines 165-241) or "basic mode".
- Auth: None (public repository). No GitHub token/env var.
- The `main` branch of the same repo is the source of truth; files are ALSO vendored locally under `src/`.

**Telegram Bot API:**
- `src/reporting/weekly_digest.py` line 453 — sends weekly digest messages to `https://api.telegram.org/bot{bot_token}/sendMessage` (JSON payload, `parse_mode: Markdown`)
- SDK/Client: `requests.post` (no official SDK)
- Auth: `bot_token` + `chat_id` passed via `telegram_config` dict (`create_digest_bot()` factory, lines 557-596)
- Not wired into the Streamlit app — standalone module with `if __name__ == "__main__"` examples

**Email (SMTP):**
- `src/reporting/weekly_digest.py` `send_email_digest()` (lines 347-418) — `smtplib.SMTP` + STARTTLS, HTML body via `MIMEMultipart`, PDF/attachment support
- Default server `smtp.gmail.com:587` (`create_digest_bot()` line 562-563)
- Auth: `sender_email` + `sender_password` (app-specific password for Gmail per docstring) passed via `email_config` dict

**HuggingFace model downloads (runtime, from HuggingFace Hub):**
- `cardiffnlp/twitter-roberta-base-sentiment-latest` — HF sentiment pipeline (`src/analysis/sentiment.py` line 33)
- `j-hartmann/emotion-english-distilroberta-base` — emotion classification pipeline (`src/analysis/emotion.py` line 39)
- `t5-small` — summarization pipeline via `T5Tokenizer.from_pretrained` / `T5ForConditionalGeneration.from_pretrained` (`src/analysis/summarizer.py` lines 52-53)
- Auth: None; models fetched on first use and cached at the process level (module-level singletons in `sentiment.py` / `emotion.py`)
- Requires `transformers` + `torch` — only in `deployment/requirements.txt`, so these features are offline on root-level installs

**NLTK data downloads:**
- `vader_lexicon`, `punkt`, `stopwords` downloaded at Docker build time (`deployment/Dockerfile` line 29)
- Used by `src/analysis/sentiment.py` (VADER) and preprocessing

## Data Storage

**Databases:**
- None — no SQL/NoSQL database in the stack. No ORM, no `db.*` files committed (only gitignore patterns for `*.db`, `*.sqlite*`)

**File Storage:**
- Local filesystem only
  - `data/sample_chats/` — committed sample exports (`whatsapp_sample.txt`, `telegram_sample.json`)
  - `data/processed/` — CSV output (`example_parsed.csv` committed)
  - User uploads are processed in-memory in the Streamlit session and discarded ("Files are processed temporarily and discarded after analysis" — `app/streamlit_app.py` line 1201); no persistent upload storage
  - `.gitignore` excludes `data/raw/*`, `data/processed/*.csv`, `uploads/`, `user_data/`, `outputs/`, `reports/`

**Caching:**
- None external — Streamlit in-memory caching only (`@st.cache_data` on `load_github_modules`, `app/streamlit_app.py` line 42) and module-level singletons for ML models (`_vader_analyzer`, `_hf_analyzer`, `_emotion_analyzer`)

## Authentication & Identity

**Auth Provider:**
- None — no user accounts, no login, no API keys required for the web app
- Streamlit's built-in `enableXsrfProtection = true` is on (`.streamlit/config.toml` line 8); `enableCORS = false` (line 7)
- The app makes an explicit privacy claim that chat data never leaves the browser session (`app/streamlit_app.py` lines 1197-1202) — though note the GitHub module fetch at startup does make outbound requests

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry/BugSnag etc.)

**Logs:**
- Python `logging` module in analysis modules (`src/analysis/relationship_health.py` line 24, `src/utils/visualization.py` lines 18-20, `src/ingest/ingestion.py` lines 30-31)
- Streamlit-level logging configured in `.streamlit/config.toml` `[logger]` section (`level = "info"`, `messageFormat = "%(asctime)s - %(levelname)s - %(message)s"`)
- UI warnings/errors via `st.warning`, `st.error`, `st.exception` in `app/streamlit_app.py`
- Note: `src/reporting/weekly_digest.py` line 23 calls `logging.basicConfig(level=logging.INFO)` at import time — reconfigures the root logger globally

## CI/CD & Deployment

**Hosting:**
- Streamlit Cloud — primary target (`README.md` → `https://chat-analyzer-pro-sujoy.streamlit.app/`); consumes `requirements.txt` (root), `apt.txt`, `packages.txt`, `.streamlit/config.toml`
- Docker — `deployment/Dockerfile` (multi-stage, HEALTHCHECK on `/_stcore/health`)
- Heroku — `deployment/Procfile` (web: `streamlit run ... --server.port=$PORT`)

**CI Pipeline:**
- None detected — no GitHub Actions, `.github/` directory, or other CI config present

## Environment Configuration

**Required env vars:**
- None hard-required by the web app
- Optional credentials for the digest bot (`src/reporting/weekly_digest.py`) — passed as constructor/factory args, documented as placeholders only:
  - SMTP: `smtp_server`, `smtp_port`, `sender_email`, `sender_password`
  - Telegram: `bot_token`, `chat_id`
- Docker build-time vars: `PYTHONPATH=/app`, `STREAMLIT_SERVER_PORT=8501`, `STREAMLIT_SERVER_ADDRESS=0.0.0.0`, `STREAMLIT_SERVER_HEADLESS=true` (`deployment/Dockerfile` lines 45-48)

**Secrets location:**
- No `.env` file committed; `.env` is gitignored (`.gitignore` line 91)
- `.streamlit/secrets.toml` gitignored (`.gitignore` line 189) — Streamlit secrets mechanism available but unused in code
- No `load_dotenv()` call anywhere — `python-dotenv` dependency is declared but never invoked

## Webhooks & Callbacks

**Incoming:**
- None — no webhook endpoints; the app is a single Streamlit page with no HTTP API surface

**Outgoing:**
- GitHub raw content GETs at app startup (`app/streamlit_app.py` lines 47-51)
- Telegram Bot API `sendMessage` POST (`src/reporting/weekly_digest.py` line 453)
- SMTP outbound email (`src/reporting/weekly_digest.py` `smtplib`)
- HuggingFace Hub model downloads (first use)

---

*Integration audit: 2026-07-31*
