import json
import urllib.request
from typing import Optional

import streamlit as st


@st.cache_data(ttl=30)
def check_ollama(base_url: str) -> tuple[bool, list[str]]:
    """Ping Ollama and return (reachable, available_model_names). Cached 30s."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as r:
            data = json.loads(r.read())
            models = [m["name"] for m in data.get("models", [])]
            return True, models
    except Exception:
        return False, []


def validate_cloud_key(provider: str, key: str) -> bool:
    """Format-only check — no API call made."""
    if not key or len(key) < 8:
        return False
    checks: dict[str, object] = {
        "openai": lambda k: k.startswith("sk-") and len(k) > 20,
        "anthropic": lambda k: k.startswith("sk-ant-") and len(k) > 20,
        "gemini": lambda k: len(k) > 10,
    }
    fn = checks.get(provider)
    return fn(key) if callable(fn) else bool(key)


def status_icon(ok: bool) -> str:
    return "🟢" if ok else "🔴"
