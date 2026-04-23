import json
import logging
import os
import time

import streamlit as st
from dotenv import load_dotenv

from config import LOG_FILE, MAX_PROMPT_CHARS
from memory import clear_messages, init_db, load_messages, save_message
from providers import ProviderInfo, cloud_providers, get_llm, get_provider, local_providers
from router import heuristic_route, judge_route, quality_low, to_lc_messages

load_dotenv()

LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))


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


def main() -> None:
    logger = _setup_logger()
    init_db()

    st.set_page_config(page_title="SmartRoute AI", page_icon=":speech_balloon:")
    st.title("SmartRoute AI")
    st.caption("Routes prompts to local or cloud models to minimize cost.")

    if "messages" not in st.session_state:
        st.session_state.messages = load_messages()
        logger.info("memory_loaded | count=%s", len(st.session_state.messages))

    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False

    local_pvds = local_providers()
    cloud_pvds = cloud_providers()

    # ------------------------------------------------------------------ sidebar
    with st.sidebar:
        st.subheader("Routing")
        mode = st.selectbox(
            "Mode",
            ["Auto", "Local only", "Cloud only"],
            key="routing_mode",
            help="Auto: local judge decides, with cloud fallback on weak answers.",
        )

        st.subheader("Local provider")
        local_info = _provider_selectbox("Provider", local_pvds, key="local_provider")
        if local_info:
            local_model = st.selectbox(
                "Model", local_info.model_options, key="local_model"
            )
        else:
            st.info("No local providers available.")
            local_model = None

        st.subheader("Cloud provider")
        cloud_info = _provider_selectbox("Provider", cloud_pvds, key="cloud_provider")
        if cloud_info:
            cloud_model = st.selectbox(
                "Model", cloud_info.model_options, key="cloud_model"
            )
        else:
            st.info(
                "No cloud providers configured.  \n"
                "Add key(s) to `.env`:  \n"
                "`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`"
            )
            cloud_model = None

        st.divider()

        # Two-stage confirmation for destructive clear action
        if not st.session_state.confirm_clear:
            if st.button("Clear memory", type="secondary", use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("This will permanently delete all conversation history.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, clear", type="primary", use_container_width=True):
                    clear_messages()
                    st.session_state.messages = []
                    st.session_state.confirm_clear = False
                    logger.info("memory_cleared")
                    st.rerun()
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()

    cloud_available = cloud_info is not None and cloud_model is not None
    local_available = local_info is not None and local_model is not None

    # ---------------------------------------------------------- chat history
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
                st.caption(" · ".join(parts))

    # ---------------------------------------------------------- prompt input
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

    # ---------------------------------------------------------- routing
    local_llm = get_llm(local_info.key, local_model) if local_available else None
    cloud_llm = get_llm(cloud_info.key, cloud_model) if cloud_available else None

    if mode == "Local only" or not cloud_available:
        use_cloud = False
        route_reason = "manual: local only" if mode == "Local only" else "cloud unavailable"
    elif mode == "Cloud only":
        use_cloud = True
        route_reason = "manual: cloud only"
    else:
        if local_llm is not None:
            decision = judge_route(local_llm, prompt, cloud_available)
        else:
            decision = heuristic_route(prompt, cloud_available)
        use_cloud = decision.use_cloud
        route_reason = decision.reason

    lc_messages = to_lc_messages(st.session_state.messages)
    logger.info(
        "request | chars=%s | local=%s/%s | cloud=%s/%s | mode=%s | route=%s",
        len(prompt),
        local_info.key if local_info else "none", local_model or "none",
        cloud_info.key if cloud_info else "none", cloud_model or "none",
        mode, "cloud" if use_cloud else "local",
    )

    # ---------------------------------------------------------- generation
    with st.chat_message("assistant"):
        answer = ""
        meta: dict = {}
        slot = st.empty()

        # Show routing decision immediately, before the first token arrives.
        if use_cloud and cloud_info:
            pending_label = f"Routing to cloud ({cloud_info.label} / {cloud_model})…"
        elif local_info:
            pending_label = f"Routing to local ({local_info.label} / {local_model})…"
        else:
            pending_label = "No provider available."
        status = st.status(pending_label, expanded=False)

        try:
            t_start = time.perf_counter()

            if use_cloud and cloud_llm is not None:
                logger.info("call_start | provider=%s | model=%s", cloud_info.key, cloud_model)
                with slot.container():
                    answer = st.write_stream(cloud_llm.stream(lc_messages, timeout=LLM_TIMEOUT))
                latency_ms = int((time.perf_counter() - t_start) * 1000)
                meta = {
                    "label": f"Cloud · {cloud_info.label} / {cloud_model} · {route_reason}",
                    "provider": cloud_info.key,
                    "model": cloud_model,
                    "latency_ms": latency_ms,
                }
                status.update(label=f"Cloud ({cloud_info.label} / {cloud_model}) · {latency_ms}ms", state="complete")
                logger.info("call_done | provider=%s | chars=%s | ms=%s", cloud_info.key, len(answer), latency_ms)

            elif local_llm is not None:
                logger.info("call_start | provider=%s | model=%s", local_info.key, local_model)
                with slot.container():
                    answer = st.write_stream(local_llm.stream(lc_messages, timeout=LLM_TIMEOUT))
                latency_ms = int((time.perf_counter() - t_start) * 1000)
                meta = {
                    "label": f"Local · {local_info.label} / {local_model} · {route_reason}",
                    "provider": local_info.key,
                    "model": local_model,
                    "latency_ms": latency_ms,
                }
                logger.info("call_done | provider=%s | chars=%s | ms=%s", local_info.key, len(answer), latency_ms)

                if cloud_llm is not None and quality_low(answer):
                    logger.info("fallback | %s → %s/%s", local_info.key, cloud_info.key, cloud_model)
                    status.update(label=f"Quality low — escalating to {cloud_info.label}…", state="running")
                    slot.empty()
                    t_fallback = time.perf_counter()
                    with slot.container():
                        answer = st.write_stream(cloud_llm.stream(lc_messages, timeout=LLM_TIMEOUT))
                    fallback_ms = int((time.perf_counter() - t_fallback) * 1000)
                    meta = {
                        "label": f"Escalated to cloud · {cloud_info.label} / {cloud_model} · local quality low",
                        "provider": cloud_info.key,
                        "model": cloud_model,
                        "latency_ms": fallback_ms,
                        "fallback": True,
                    }
                    status.update(label=f"Escalated to {cloud_info.label} ({cloud_model}) · {fallback_ms}ms", state="complete")
                    logger.info("call_done | provider=%s | chars=%s | ms=%s", cloud_info.key, len(answer), fallback_ms)
                else:
                    status.update(label=meta["label"], state="complete")

            else:
                answer = "No providers available. Configure a local or cloud provider in `.env`."
                meta = {"label": "No provider"}
                slot.warning(answer)
                status.update(label="No provider configured", state="error")

        except TimeoutError:
            logger.error("call_timeout | provider=%s | timeout=%ss", "cloud" if use_cloud else "local", LLM_TIMEOUT)
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

        if meta.get("label"):
            parts = [meta["label"]]
            if meta.get("latency_ms"):
                parts.append(f"{meta['latency_ms']}ms")
            st.caption(" · ".join(parts))

    meta_str = json.dumps(meta)
    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
    save_message("assistant", answer, meta_str)


if __name__ == "__main__":
    main()
