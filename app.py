import json
import logging
import os
import time
from typing import Generator

import streamlit as st
from dotenv import load_dotenv

from config import (
    CONTEXT_LIMITS, COST_PER_1M, DEFAULT_CONTEXT_LIMIT,
    LOG_FILE, MAX_PROMPT_CHARS,
    GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    OLLAMA_BASE_URL,
)
from health import check_ollama, status_icon, validate_cloud_key
from memory import clear_messages, init_db, load_messages, save_message
from providers import ProviderInfo, cloud_providers, get_llm, get_provider, local_providers
from router import heuristic_route, judge_route, quality_low, to_lc_messages

load_dotenv()


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("hybrid_router")
    if logger.handlers:
        return logger
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tracked_stream(llm_stream, usage_out: dict) -> Generator[str, None, None]:
    """Yield text chunks from a LangChain stream; capture usage metadata in usage_out."""
    for chunk in llm_stream:
        um = getattr(chunk, "usage_metadata", None)
        if um:
            usage_out["input_tokens"] = getattr(um, "input_tokens", usage_out.get("input_tokens", 0)) or usage_out.get("input_tokens", 0)
            usage_out["output_tokens"] = getattr(um, "output_tokens", usage_out.get("output_tokens", 0)) or usage_out.get("output_tokens", 0)
        content = getattr(chunk, "content", "") or ""
        if content:
            yield content


def _estimate_cost(provider: str, model: str, input_tok: int, output_tok: int) -> float:
    rates = COST_PER_1M.get(f"{provider}/{model}")
    if not rates or not input_tok:
        return 0.0
    return (input_tok * rates["input"] + output_tok * rates["output"]) / 1_000_000


def _estimate_context_tokens(messages: list[dict]) -> int:
    return sum(len(m.get("content", "")) for m in messages) // 4


def _provider_selectbox(label: str, providers: list[ProviderInfo], key: str) -> ProviderInfo | None:
    if not providers:
        return None
    keys = [p.key for p in providers]
    selected_key = st.selectbox(
        label,
        options=keys,
        format_func=lambda k: next((p.label for p in providers if p.key == k), k),
        key=key,
    )
    return get_provider(providers, selected_key)


def _format_cost(usd: float) -> str:
    if usd == 0:
        return "$0.00"
    if usd < 0.001:
        return f"${usd * 100:.4f}¢"
    return f"${usd:.4f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger = _setup_logger()
    init_db()

    st.set_page_config(page_title="SmartRoute AI", page_icon=":speech_balloon:", layout="wide")
    st.title("SmartRoute AI")
    st.caption("Routes prompts to local or cloud models to minimize cost.")

    # ── Session state init ──────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = load_messages()
        logger.info("memory_loaded | count=%s", len(st.session_state.messages))
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False
    if "routing_history" not in st.session_state:
        st.session_state.routing_history = []
    if "session_stats" not in st.session_state:
        st.session_state.session_stats = {
            "local_calls": 0, "cloud_calls": 0,
            "total_input_tokens": 0, "total_output_tokens": 0,
            "total_cost_usd": 0.0,
        }
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""

    local_pvds = local_providers()
    cloud_pvds = cloud_providers()

    # ── Health checks ───────────────────────────────────────────────────────
    ollama_ok, ollama_models = check_ollama(OLLAMA_BASE_URL)
    cloud_key_ok = {
        "gemini": validate_cloud_key("gemini", GEMINI_API_KEY),
        "openai": validate_cloud_key("openai", OPENAI_API_KEY),
        "anthropic": validate_cloud_key("anthropic", ANTHROPIC_API_KEY),
    }

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        # Routing mode
        st.subheader("Routing")
        mode = st.selectbox(
            "Mode", ["Auto", "Local only", "Cloud only"], key="routing_mode",
            help="Auto: local judge decides, with cloud fallback on weak answers.",
        )

        # Local provider
        st.subheader("Local provider")
        local_info = _provider_selectbox("Provider", local_pvds, key="local_provider")
        if local_info:
            icon = status_icon(ollama_ok) if local_info.key == "ollama" else "🟡"
            hint = f"{icon} {'Online' if ollama_ok else 'Offline'}" if local_info.key == "ollama" else f"{icon} Check server"
            st.caption(hint)
            local_model = st.selectbox("Model", local_info.model_options, key="local_model")
        else:
            st.info("No local providers available.")
            local_model = None

        # Cloud provider
        st.subheader("Cloud provider")
        cloud_info = _provider_selectbox("Provider", cloud_pvds, key="cloud_provider")
        if cloud_info:
            key_ok = cloud_key_ok.get(cloud_info.key, False)
            st.caption(f"{status_icon(key_ok)} Key {'looks valid' if key_ok else 'invalid or missing'}")
            cloud_model = st.selectbox("Model", cloud_info.model_options, key="cloud_model")
        else:
            st.info(
                "No cloud providers configured.  \n"
                "Add to `.env`: `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`"
            )
            cloud_model = None

        # System prompt
        with st.expander("System prompt", expanded=False):
            st.session_state.system_prompt = st.text_area(
                "Custom instruction",
                value=st.session_state.system_prompt,
                height=100,
                placeholder="You are a helpful assistant...",
                label_visibility="collapsed",
            )
            if st.session_state.system_prompt:
                st.caption("Active — injected into every request.")

        # Context window
        ctx_tokens = _estimate_context_tokens(st.session_state.messages)
        ctx_limit = CONTEXT_LIMITS.get(local_model or "", DEFAULT_CONTEXT_LIMIT) if local_model else DEFAULT_CONTEXT_LIMIT
        if cloud_model:
            ctx_limit = max(ctx_limit, CONTEXT_LIMITS.get(cloud_model, DEFAULT_CONTEXT_LIMIT))
        ctx_pct = min(ctx_tokens / ctx_limit, 1.0)
        with st.expander("Context window", expanded=False):
            st.progress(ctx_pct, text=f"~{ctx_tokens:,} / {ctx_limit:,} tokens ({ctx_pct:.0%})")
            if ctx_pct > 0.75:
                st.warning("Context window is nearly full. Consider clearing memory.")

        # Session stats
        stats = st.session_state.session_stats
        total_calls = stats["local_calls"] + stats["cloud_calls"]
        with st.expander("Session stats", expanded=total_calls > 0):
            if total_calls == 0:
                st.caption("No requests yet this session.")
            else:
                c1, c2 = st.columns(2)
                c1.metric("Local calls", stats["local_calls"])
                c2.metric("Cloud calls", stats["cloud_calls"])
                if stats["total_input_tokens"]:
                    st.caption(
                        f"Tokens in: {stats['total_input_tokens']:,} · "
                        f"out: {stats['total_output_tokens']:,}"
                    )
                if stats["total_cost_usd"] > 0:
                    st.caption(f"Est. cloud cost: {_format_cost(stats['total_cost_usd'])}")

        # Routing history
        if st.session_state.routing_history:
            with st.expander(f"Routing history ({len(st.session_state.routing_history)})", expanded=False):
                for entry in reversed(st.session_state.routing_history[-20:]):
                    fallback_badge = " ↩ fallback" if entry.get("fallback") else ""
                    st.caption(
                        f"**{entry['provider']}/{entry['model']}**{fallback_badge}  \n"
                        f"{entry['latency_ms']}ms · {entry['prompt']}"
                    )

        st.divider()

        # Clear memory (two-stage)
        if not st.session_state.confirm_clear:
            if st.button("Clear memory", type="secondary", use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("Permanently delete all conversation history?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, clear", type="primary", use_container_width=True):
                    clear_messages()
                    st.session_state.messages = []
                    st.session_state.routing_history = []
                    st.session_state.session_stats = {
                        "local_calls": 0, "cloud_calls": 0,
                        "total_input_tokens": 0, "total_output_tokens": 0,
                        "total_cost_usd": 0.0,
                    }
                    st.session_state.confirm_clear = False
                    logger.info("memory_cleared")
                    st.rerun()
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()

    cloud_available = cloud_info is not None and cloud_model is not None
    local_available = local_info is not None and local_model is not None

    # ── Chat history ─────────────────────────────────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            meta = message.get("meta", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {"label": meta} if meta else {}
            if meta.get("label"):
                parts = [meta["label"]]
                if meta.get("latency_ms"):
                    parts.append(f"{meta['latency_ms']}ms")
                if meta.get("cost_usd"):
                    parts.append(_format_cost(meta["cost_usd"]))
                st.caption(" · ".join(parts))

    # ── Prompt input ─────────────────────────────────────────────────────────
    prompt = st.chat_input("Ask anything...")
    if not prompt:
        return

    if len(prompt) > MAX_PROMPT_CHARS:
        st.warning(f"Prompt too long ({len(prompt):,} chars). Limit is {MAX_PROMPT_CHARS:,}.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt, "meta": {}})
    save_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Routing ──────────────────────────────────────────────────────────────
    local_llm = get_llm(local_info.key, local_model) if local_available else None
    cloud_llm = get_llm(cloud_info.key, cloud_model) if cloud_available else None

    if mode == "Local only" or not cloud_available:
        use_cloud = False
        route_reason = "manual: local only" if mode == "Local only" else "cloud unavailable"
    elif mode == "Cloud only":
        use_cloud = True
        route_reason = "manual: cloud only"
    else:
        decision = judge_route(local_llm, prompt, cloud_available) if local_llm else heuristic_route(prompt, cloud_available)
        use_cloud = decision.use_cloud
        route_reason = decision.reason

    lc_messages = to_lc_messages(
        st.session_state.messages,
        system_prompt=st.session_state.system_prompt,
    )
    logger.info(
        "request | chars=%s | local=%s/%s | cloud=%s/%s | mode=%s | route=%s",
        len(prompt),
        local_info.key if local_info else "none", local_model or "none",
        cloud_info.key if cloud_info else "none", cloud_model or "none",
        mode, "cloud" if use_cloud else "local",
    )

    # ── Generation ───────────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        answer = ""
        meta: dict = {}
        usage: dict = {}
        slot = st.empty()

        pending = (
            f"Routing to cloud ({cloud_info.label} / {cloud_model})…"
            if use_cloud and cloud_info
            else f"Routing to local ({local_info.label} / {local_model})…"
            if local_info
            else "No provider available."
        )
        status = st.status(pending, expanded=False)

        try:
            t0 = time.perf_counter()

            if use_cloud and cloud_llm is not None:
                logger.info("call_start | provider=%s | model=%s", cloud_info.key, cloud_model)
                with slot.container():
                    answer = st.write_stream(_tracked_stream(cloud_llm.stream(lc_messages), usage))
                ms = int((time.perf_counter() - t0) * 1000)
                cost = _estimate_cost(cloud_info.key, cloud_model, usage.get("input_tokens", 0), usage.get("output_tokens", 0))
                meta = {"label": f"Cloud · {cloud_info.label}/{cloud_model} · {route_reason}", "provider": cloud_info.key, "model": cloud_model, "latency_ms": ms, "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0), "cost_usd": cost}
                status.update(label=f"Cloud · {cloud_info.label}/{cloud_model} · {ms}ms", state="complete")
                logger.info("call_done | provider=%s | chars=%s | ms=%s | tokens_in=%s | tokens_out=%s", cloud_info.key, len(answer), ms, usage.get("input_tokens", 0), usage.get("output_tokens", 0))
                st.session_state.session_stats["cloud_calls"] += 1
                st.session_state.session_stats["total_cost_usd"] += cost

            elif local_llm is not None:
                logger.info("call_start | provider=%s | model=%s", local_info.key, local_model)
                with slot.container():
                    answer = st.write_stream(_tracked_stream(local_llm.stream(lc_messages), usage))
                ms = int((time.perf_counter() - t0) * 1000)
                meta = {"label": f"Local · {local_info.label}/{local_model} · {route_reason}", "provider": local_info.key, "model": local_model, "latency_ms": ms, "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0), "cost_usd": 0.0}
                logger.info("call_done | provider=%s | chars=%s | ms=%s", local_info.key, len(answer), ms)
                st.session_state.session_stats["local_calls"] += 1

                if cloud_llm is not None and quality_low(answer):
                    logger.info("fallback | %s → %s/%s", local_info.key, cloud_info.key, cloud_model)
                    status.update(label=f"Quality low — escalating to {cloud_info.label}…", state="running")
                    slot.empty()
                    usage_fb: dict = {}
                    t_fb = time.perf_counter()
                    with slot.container():
                        answer = st.write_stream(_tracked_stream(cloud_llm.stream(lc_messages), usage_fb))
                    ms_fb = int((time.perf_counter() - t_fb) * 1000)
                    cost_fb = _estimate_cost(cloud_info.key, cloud_model, usage_fb.get("input_tokens", 0), usage_fb.get("output_tokens", 0))
                    meta = {"label": f"Escalated · {cloud_info.label}/{cloud_model} · local quality low", "provider": cloud_info.key, "model": cloud_model, "latency_ms": ms_fb, "input_tokens": usage_fb.get("input_tokens", 0), "output_tokens": usage_fb.get("output_tokens", 0), "cost_usd": cost_fb, "fallback": True}
                    status.update(label=f"Escalated to {cloud_info.label}/{cloud_model} · {ms_fb}ms", state="complete")
                    st.session_state.session_stats["cloud_calls"] += 1
                    st.session_state.session_stats["total_cost_usd"] += cost_fb
                    usage = usage_fb
                else:
                    status.update(label=f"Local · {local_info.label}/{local_model} · {ms}ms", state="complete")

            else:
                answer = "No providers available. Configure a local or cloud provider in `.env`."
                meta = {"label": "No provider"}
                slot.warning(answer)
                status.update(label="No provider configured", state="error")

        except TimeoutError:
            logger.error("call_timeout | timeout=%ss", LLM_TIMEOUT)
            answer = f"Request timed out after {LLM_TIMEOUT}s. The model may be overloaded or unreachable."
            meta = {"label": "Timeout"}
            slot.error(answer)
            status.update(label="Timed out", state="error")

        except Exception as exc:
            logger.exception("call_error | %s", exc)
            answer = "Something went wrong generating a response. Check the log for details."
            meta = {"label": "Error"}
            slot.error(answer)
            status.update(label="Error", state="error")

        # Update aggregate token counters
        st.session_state.session_stats["total_input_tokens"] += usage.get("input_tokens", 0)
        st.session_state.session_stats["total_output_tokens"] += usage.get("output_tokens", 0)

        # Caption
        if meta.get("label"):
            parts = [meta["label"]]
            if meta.get("latency_ms"):
                parts.append(f"{meta['latency_ms']}ms")
            if meta.get("cost_usd"):
                parts.append(_format_cost(meta["cost_usd"]))
            if meta.get("input_tokens"):
                parts.append(f"{meta['input_tokens']+meta.get('output_tokens',0):,} tokens")
            st.caption(" · ".join(parts))

    # Record routing history entry
    st.session_state.routing_history.append({
        "prompt": prompt[:70] + ("…" if len(prompt) > 70 else ""),
        "provider": meta.get("provider", "?"),
        "model": meta.get("model", "?"),
        "latency_ms": meta.get("latency_ms", 0),
        "fallback": meta.get("fallback", False),
    })

    meta_str = json.dumps(meta)
    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
    save_message("assistant", answer, meta_str)


if __name__ == "__main__":
    main()
