# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## AI assistant guidance

This file is used by Claude Code and is also intended as a shared reference for any AI coding tool (Cursor, Copilot, Windsurf, etc.) working on this repo. The conventions below apply regardless of the tool.

**Before claiming a tool or CLI is unavailable:** verify with `where <tool>` (Windows) or `which <tool>` (Unix), then `<tool> --version`. Do not assume absence from a single failed shell command — the tool may be on a PATH not visible to the current shell.

**Before stating an integration is unsupported:** check the vendor's current docs. Third-party tools update quickly (e.g. LM Studio added a native Anthropic-compatible API in 2025). Training-data knowledge may be stale.

**Before making multi-file edits:** confirm the scope — this codebase has 5 focused modules; most features touch 1–2 files. Ask one clarifying question if the scope is ambiguous rather than assuming.

---

## Commands

```powershell
# Create venv and install dependencies
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

# Run the app
.\.venv\Scripts\python -m streamlit run app.py

# Tail logs in real time
Get-Content .\logs\router.log -Wait

# Re-pin dependency versions after installing new packages
.\.venv\Scripts\pip freeze | Select-String "streamlit|langchain|dotenv|anthropic|openai|google"
```

No test runner or linter is configured yet. Smoke-test logic with plain `python` scripts against individual modules (they import without Streamlit).

---

## Architecture

SmartRoute AI is a Streamlit chat app that routes each prompt to either a local (Ollama / LM Studio) or cloud (Gemini / OpenAI / Anthropic) LLM to minimize API cost. Every module is independently importable — `app.py` is UI-only and imports from the other four.

### Data flow per prompt

1. User submits prompt → `app.py` enforces `MAX_PROMPT_CHARS`
2. `router.estimate_complexity()` scores the prompt (0–1)
3. If score < 0.2: skip LLM judge, go local immediately
4. Otherwise: `router.judge_route()` asks the local LLM to return `{use_cloud, reason}` JSON; falls back to `router.heuristic_route()` on parse failure (logs the error)
5. `providers.get_llm(key, model)` returns a cached `BaseChatModel` instance
6. `app._tracked_stream()` wraps the LangChain stream, capturing `usage_metadata` from chunks while yielding text to `st.write_stream()`
7. If local answer passes `router.quality_low()` check (scans first 40% of response for uncertainty phrases), cloud fallback fires
8. Token counts, latency, and estimated cost written to `meta` JSON; persisted to SQLite via `memory.save_message()`

### Module responsibilities

| Module | Owns | Must not |
|---|---|---|
| `config.py` | All `os.getenv()` calls, path safety, cost table, context limits | — |
| `providers.py` | `ProviderInfo` dataclass, provider discovery, `@st.cache_resource` LLM factories, timeout at constructor level | Call `os.getenv` directly |
| `router.py` | Complexity scoring, judge/heuristic routing, quality detection, `to_lc_messages()` | Import Streamlit |
| `memory.py` | SQLite init (WAL mode), read/write/clear | Import Streamlit |
| `health.py` | Ollama ping (`@st.cache_data(ttl=30)`), cloud key format checks | Make real API calls |
| `app.py` | Streamlit UI, sidebar, session state, calling all modules above | Contain business logic |

### Provider system

Providers are detected at runtime from `.env`. Local providers (Ollama, LM Studio) are always shown; cloud providers appear only when their API key is present. LLM instances are cached per `(provider, model)` via `@st.cache_resource` — survives Streamlit reruns. Timeout is set at constructor level in `providers.py` via `LLM_TIMEOUT` env var (default 120s).

**To add a new provider:**
1. Add its env vars to `config.py`
2. Add a `@st.cache_resource` factory and a `ProviderInfo` entry in `providers.py`
3. Add a `get_llm` branch in `providers.py`
4. Add cost-per-1M and context-limit entries in `config.py`
5. Add key format check in `health.py`

### Key env vars

See `.env.example` for the full list. The most impactful:

| Var | Effect |
|---|---|
| `OLLAMA_BASE_URL` | Local Ollama server (default `http://localhost:11434`) |
| `LMSTUDIO_BASE_URL` | Enable LM Studio as a second local provider (blank = disabled) |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Enable each cloud provider |
| `MAX_PROMPT_CHARS` | Hard cap on prompt length (default 8000) |
| `LLM_TIMEOUT` | Seconds before a model call times out (default 120) |
| `LOG_FILE` / `MEMORY_DB_PATH` | Must stay within project root (enforced by `config._safe_path`) |

### Routing logic details

`estimate_complexity()` scores on three signals: prompt length, keyword hits (design/architecture/debug/etc.), and code block presence. Score 0–1; threshold 0.65 → cloud via heuristic, 0.2 → skip judge call entirely.

`quality_low()` scans for uncertainty phrases. For responses < 100 chars it scans all; for longer responses it only scans the first 40% — a hedge buried at the end of a long detailed answer should not trigger a fallback.

### Streamlit rerun behaviour

Streamlit reruns the entire script on every user interaction. LLM instances are protected by `@st.cache_resource`. `init_db()` is guarded by a module-level flag. Message history loads from SQLite once per session into `st.session_state.messages` and is kept in sync manually each turn.

### Session state keys

| Key | Type | Purpose |
|---|---|---|
| `messages` | `list[dict]` | Full conversation history (role, content, meta) |
| `routing_history` | `list[dict]` | Per-request routing decisions for sidebar panel |
| `session_stats` | `dict` | Running totals: calls, tokens, cost |
| `system_prompt` | `str` | User-editable system instruction |
| `confirm_clear` | `bool` | Two-stage clear memory gate |

### Meta dict schema (per assistant message)

```json
{
  "label": "Local · Ollama/llama3.1 · trivial complexity (0.00)",
  "provider": "ollama",
  "model": "llama3.1:8b-instruct-q4_0",
  "latency_ms": 842,
  "input_tokens": 312,
  "output_tokens": 94,
  "cost_usd": 0.0,
  "fallback": false
}
```

---

## Session context (April 2026)

Key decisions and findings from the initial development sprint that are not obvious from the code:

**Why the module split happened mid-sprint:** the original `app.py` was a 360-line monolith. Splitting was done as a prerequisite to testing individual pieces — `router.py` and `memory.py` can now be imported and tested with plain Python without running Streamlit.

**Why `quality_low()` scans only the first 40% of longer responses:** the original 100-char length check was causing valid short confident answers ("Yes", "42", "O(n log n)") to escalate to cloud. Replaced with uncertainty-phrase scanning. Phrases near the end of long responses are hedges, not admissions of failure — the 40% cutoff prevents false fallbacks on detailed answers that include caveats.

**Why timeout is at the LLM constructor, not `stream()`:** `ChatOllama` and `ChatGoogleGenerativeAI` do not accept a `timeout=` kwarg on `.stream()`. Passing it there causes a `TypeError`. Timeout is set via `request_timeout` (Gemini) or `timeout` (Ollama, OpenAI, Anthropic) at construction time in `providers.py`.

**Why the judge skips on complexity < 0.2:** the LLM judge itself requires a full local model call (via `invoke()`). For trivially simple prompts (greetings, one-word questions), that call adds latency with no benefit. The 0.2 threshold was chosen so that "hi", "thanks", "yes" skip the judge; anything with a keyword hit or code block still goes through it.

**Why LM Studio uses `api_key="lm-studio"`:** LM Studio's OpenAI-compatible endpoint requires a non-empty API key string but ignores its value. The placeholder `"lm-studio"` satisfies the SDK validation without leaking anything real.

**gh CLI path on this machine:** `C:\Program Files\GitHub CLI`. The bash environment used by Claude Code does not include this on PATH — use PowerShell with `$env:PATH += ";C:\Program Files\GitHub CLI"` before `gh` commands.
