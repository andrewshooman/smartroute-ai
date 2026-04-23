from dataclasses import dataclass
from typing import Optional

import streamlit as st
from langchain_core.language_models.chat_models import BaseChatModel

from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_MODEL_OPTIONS,
    LMSTUDIO_BASE_URL, LMSTUDIO_MODEL_OPTIONS,
    GEMINI_API_KEY, GEMINI_MODEL, GEMINI_MODEL_OPTIONS,
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MODEL_OPTIONS,
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_MODEL_OPTIONS,
    parse_options,
)


@dataclass
class ProviderInfo:
    key: str
    label: str
    is_local: bool
    default_model: str
    model_options: list[str]


# ---------------------------------------------------------------------------
# Cached LLM factories — one instance per (provider, model) pair per session
# ---------------------------------------------------------------------------

@st.cache_resource
def _ollama_llm(model: str, base_url: str) -> BaseChatModel:
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, base_url=base_url, temperature=0.2)


@st.cache_resource
def _lmstudio_llm(model: str, base_url: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, base_url=base_url, api_key="lm-studio", temperature=0.2)


@st.cache_resource
def _gemini_llm(model: str, api_key: str) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.2)


@st.cache_resource
def _openai_llm(model: str, api_key: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, api_key=api_key, temperature=0.2)


@st.cache_resource
def _anthropic_llm(model: str, api_key: str) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, api_key=api_key, temperature=0.2)


# ---------------------------------------------------------------------------
# Provider discovery
# ---------------------------------------------------------------------------

def local_providers() -> list[ProviderInfo]:
    providers = []

    providers.append(ProviderInfo(
        key="ollama",
        label="Ollama",
        is_local=True,
        default_model=OLLAMA_MODEL,
        model_options=parse_options(OLLAMA_MODEL_OPTIONS) or [OLLAMA_MODEL],
    ))

    if LMSTUDIO_BASE_URL:
        opts = parse_options(LMSTUDIO_MODEL_OPTIONS)
        providers.append(ProviderInfo(
            key="lmstudio",
            label="LM Studio",
            is_local=True,
            default_model=opts[0] if opts else "local-model",
            model_options=opts or ["local-model"],
        ))

    return providers


def cloud_providers() -> list[ProviderInfo]:
    providers = []

    if GEMINI_API_KEY:
        providers.append(ProviderInfo(
            key="gemini",
            label="Google Gemini",
            is_local=False,
            default_model=GEMINI_MODEL,
            model_options=parse_options(GEMINI_MODEL_OPTIONS) or [GEMINI_MODEL],
        ))

    if OPENAI_API_KEY:
        providers.append(ProviderInfo(
            key="openai",
            label="OpenAI",
            is_local=False,
            default_model=OPENAI_MODEL,
            model_options=parse_options(OPENAI_MODEL_OPTIONS) or [OPENAI_MODEL],
        ))

    if ANTHROPIC_API_KEY:
        providers.append(ProviderInfo(
            key="anthropic",
            label="Anthropic",
            is_local=False,
            default_model=ANTHROPIC_MODEL,
            model_options=parse_options(ANTHROPIC_MODEL_OPTIONS) or [ANTHROPIC_MODEL],
        ))

    return providers


def get_llm(provider_key: str, model: str) -> Optional[BaseChatModel]:
    if provider_key == "ollama":
        return _ollama_llm(model, OLLAMA_BASE_URL)
    if provider_key == "lmstudio":
        return _lmstudio_llm(model, LMSTUDIO_BASE_URL)
    if provider_key == "gemini":
        return _gemini_llm(model, GEMINI_API_KEY)
    if provider_key == "openai":
        return _openai_llm(model, OPENAI_API_KEY)
    if provider_key == "anthropic":
        return _anthropic_llm(model, ANTHROPIC_API_KEY)
    return None
