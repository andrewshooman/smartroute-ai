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

# Cost per 1M tokens (input, output) in USD — used for session cost estimates.
# Update these when providers reprice.
COST_PER_1M: dict[str, dict[str, float]] = {
    "gemini/gemini-2.5-flash":           {"input": 0.15,  "output": 0.60},
    "gemini/gemini-2.5-pro":             {"input": 1.25,  "output": 5.00},
    "openai/gpt-4o-mini":                {"input": 0.15,  "output": 0.60},
    "openai/gpt-4o":                     {"input": 2.50,  "output": 10.00},
    "openai/o3-mini":                    {"input": 1.10,  "output": 4.40},
    "anthropic/claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "anthropic/claude-sonnet-4-6":       {"input": 3.00,  "output": 15.00},
    "anthropic/claude-opus-4-7":         {"input": 15.00, "output": 75.00},
}

# Approximate context window sizes in tokens per model name.
CONTEXT_LIMITS: dict[str, int] = {
    "llama3.1:8b-instruct-q4_0": 128_000,
    "gpt-oss:20b":               128_000,
    "gemma4:latest":             128_000,
    "mistral:latest":            32_000,
    "gemini-2.5-flash":          1_000_000,
    "gemini-2.5-pro":            1_000_000,
    "gpt-4o-mini":               128_000,
    "gpt-4o":                    128_000,
    "o3-mini":                   200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-sonnet-4-6":         200_000,
    "claude-opus-4-7":           200_000,
}
DEFAULT_CONTEXT_LIMIT = 128_000


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
