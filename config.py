import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Local providers ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", os.getenv("LOCAL_BASE_URL", "http://localhost:11434"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", os.getenv("LOCAL_MODEL", "llama3.1:8b-instruct-q4_0"))
OLLAMA_MODEL_OPTIONS = os.getenv("OLLAMA_MODEL_OPTIONS", os.getenv("LOCAL_MODEL_OPTIONS", "llama3.1:8b-instruct-q4_0,gemma4:latest"))

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "")
LMSTUDIO_MODEL_OPTIONS = os.getenv("LMSTUDIO_MODEL_OPTIONS", "")

# --- Cloud providers ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", os.getenv("CLOUD_MODEL", "gemini-2.5-flash"))
GEMINI_MODEL_OPTIONS = os.getenv("GEMINI_MODEL_OPTIONS", os.getenv("CLOUD_MODEL_OPTIONS", "gemini-2.5-flash,gemini-2.5-pro"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MODEL_OPTIONS = os.getenv("OPENAI_MODEL_OPTIONS", "gpt-4o-mini,gpt-4o,o3-mini")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
ANTHROPIC_MODEL_OPTIONS = os.getenv("ANTHROPIC_MODEL_OPTIONS", "claude-haiku-4-5-20251001,claude-sonnet-4-6,claude-opus-4-7")

# --- App behaviour ---
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "8000"))


def _safe_path(raw: str, default: str) -> str:
    """Resolve to absolute path; reject anything that escapes the project root."""
    base = Path(__file__).parent.resolve()
    resolved = (base / raw).resolve()
    if not str(resolved).startswith(str(base)):
        return str(base / default)
    return str(resolved)


LOG_FILE = _safe_path(os.getenv("LOG_FILE", "logs/router.log"), "logs/router.log")
MEMORY_DB_PATH = _safe_path(os.getenv("MEMORY_DB_PATH", "data/memory.db"), "data/memory.db")


def parse_options(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]
