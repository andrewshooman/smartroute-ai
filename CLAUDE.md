# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```powershell
# Create venv and install dependencies
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

# Run the app
.\.venv\Scripts\python -m streamlit run app.py

# Tail logs in real time
Get-Content .\logs\router.log -Wait

# Pin current installed versions back to requirements.txt
.\.venv\Scripts\pip freeze | Select-String "streamlit|langchain|dotenv|anthropic|openai|google"
```

There are no tests or linting configured yet.

## Architecture

The app is a Streamlit chat interface that routes each prompt to either a local or cloud LLM to minimize API cost. Every module is independently importable — `app.py` is UI-only.

### Data flow

1. User submits a prompt in `app.py`
2. `app.py` calls `router.judge_route()` using the local LLM as a routing judge (in Auto mode)
3. The judge returns a `RouteDecision(use_cloud, reason)`
4. `app.py` calls `providers.get_llm(provider_key, model)` to get the appropriate cached LLM
5. Response is streamed via `st.write_stream(llm.stream(messages))`
6. If local response contains weak-confidence markers, `router.quality_low()` triggers a cloud fallback
7. Both the user message and assistant response are persisted to SQLite via `memory.py`

### Module responsibilities

- **`config.py`** — all env var reads, path safety validation (`_safe_path` rejects traversal outside project root), `parse_options` helper. Import constants from here, never call `os.getenv` elsewhere.
- **`providers.py`** — `ProviderInfo` dataclass, provider discovery (`local_providers()`, `cloud_providers()`), and `@st.cache_resource`-wrapped LLM factories (`get_llm(key, model)`). A provider only appears if its API key/URL is set in `.env`.
- **`router.py`** — stateless routing logic. `judge_route()` asks the local LLM to return JSON `{use_cloud, reason}`; on parse failure it falls back to `heuristic_route()` (keyword + length scoring). `quality_low()` checks the completed local answer for uncertainty phrases. `to_lc_messages()` converts the flat history list to LangChain message objects.
- **`memory.py`** — SQLite read/write/clear. Schema is a single `messages` table (role, content, meta, created_at). Opens a new connection per call — fine for single-user local use.
- **`app.py`** — Streamlit UI only. Builds the sidebar provider/model selectors, enforces `MAX_PROMPT_CHARS`, calls modules above, and streams responses into `st.empty()` slots so fallback can replace a streamed local answer with a cloud answer.

### Provider system

Providers are detected at runtime from `.env`. Local providers (Ollama, LM Studio) are always shown; cloud providers (Gemini, OpenAI, Anthropic) only appear when their API key is present. LLM instances are cached per `(provider_key, model)` via `@st.cache_resource` so they survive Streamlit reruns. To add a new provider: add its env vars to `config.py`, add a cached factory function and a `ProviderInfo` entry in `providers.py`, and add a `get_llm` branch.

### Key env vars

See `.env.example` for the full list. The most important:
- `OLLAMA_BASE_URL` — defaults to `http://localhost:11434`
- `GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` — each enables its cloud provider
- `LMSTUDIO_BASE_URL` — set to enable LM Studio as a local provider
- `MAX_PROMPT_CHARS` — rejects prompts above this length (default 8000)
- `LOG_FILE` / `MEMORY_DB_PATH` — must stay within the project root (enforced by `config._safe_path`)

### Streamlit rerun behaviour

Streamlit reruns the entire script on every user interaction. LLM instances are protected by `@st.cache_resource`. Message history is loaded from SQLite once per session into `st.session_state.messages` and kept in sync manually on each turn.
