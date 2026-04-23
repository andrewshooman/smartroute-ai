# Hybrid Memory Router Chat

A Streamlit + LangChain chat app that prioritizes your local Ollama model and escalates to Gemini when needed.

## Features

- Hybrid routing modes: `Auto`, `Local only`, `Cloud only`
- Local-model-first routing judge in `Auto` mode
- Gemini fallback when local responses are weak or uncertain
- Persistent SQLite chat memory across restarts
- Router logs showing which model was called for each turn

## Stack

- Python 3.10+
- Streamlit
- LangChain
- Ollama (local model)
- Google Gemini (`langchain-google-genai`)
- SQLite (persistent memory)

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

2. Create your env file:

```powershell
Copy-Item .env.example .env
```

3. Edit `.env`:

```env
GEMINI_API_KEY=your_key_here
LOCAL_MODEL=llama3.1:8b-instruct-q4_0
CLOUD_MODEL=gemini-2.5-flash
LOCAL_BASE_URL=http://localhost:11434
LOG_FILE=logs/router.log
MEMORY_DB_PATH=data/memory.db
```

4. Make sure your local model exists in Ollama:

```powershell
ollama list
```

## Run

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

## How Routing Works

- `Local only`: always uses Ollama.
- `Cloud only`: always uses Gemini (if key is configured).
- `Auto`: local model first decides if cloud is needed; if local answers weakly, app escalates to Gemini.

## Memory

- Conversation messages are persisted in SQLite at `data/memory.db`.
- Use sidebar button **Clear persistent memory** to wipe stored messages.

## Logs

- Router/model logs are written to `logs/router.log`.
- Monitor live:

```powershell
Get-Content .\logs\router.log -Wait
```
