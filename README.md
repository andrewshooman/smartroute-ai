# SmartRoute AI

SmartRoute AI is a local-first chat router built to save cloud tokens.  
Easy prompts stay on your machine with Ollama, while harder prompts can be escalated to Gemini only when needed.

## Core Goal

- Reduce cloud token spend by handling simple requests locally
- Use your local hardware first for fast, private, low-cost responses
- Escalate to cloud only for tasks that need stronger reasoning or fallback confidence

## Current Features

- Hybrid routing modes: `Auto`, `Local only`, `Cloud only`
- Local-model routing judge in `Auto` mode
- Gemini fallback when local responses are uncertain or low quality
- Persistent SQLite memory across app restarts
- Router logs that show which model handled each request
- Sidebar controls for routing mode and memory reset

## Tech Stack

- Python 3.10+
- Streamlit
- LangChain
- Ollama for local inference
- Google Gemini via `langchain-google-genai`
- SQLite for persistent memory

## Quick Start

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

2. Create your environment file:

```powershell
Copy-Item .env.example .env
```

3. Update `.env`:

```env
GEMINI_API_KEY=your_key_here
LOCAL_MODEL=llama3.1:8b-instruct-q4_0
CLOUD_MODEL=gemini-2.5-flash
LOCAL_BASE_URL=http://localhost:11434
LOG_FILE=logs/router.log
MEMORY_DB_PATH=data/memory.db
```

4. Confirm your local model exists:

```powershell
ollama list
```

5. Run the app:

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

## Routing Behavior

- `Local only`: always uses Ollama
- `Cloud only`: always uses Gemini (when key is configured)
- `Auto`: local judge decides route first, then fallback can escalate to Gemini if local output is weak

## Memory

- Chat history is persisted to `data/memory.db`
- Use **Clear persistent memory** in the sidebar to wipe saved history

## Logging

- Router and model call logs are written to `logs/router.log`
- Live monitor:

```powershell
Get-Content .\logs\router.log -Wait
```

## Future Features

- Token usage + estimated cost dashboard per request/session
- Memory consolidation (rolling summaries to keep context efficient)
- Multi-profile memory spaces (work, personal, coding)
- Provider expansion (additional cloud/local backends)
- Prompt-level policy controls (force local/cloud by category)
