import logging
import os

import streamlit as st
from dotenv import load_dotenv

from config import LOG_FILE, MAX_PROMPT_CHARS
from memory import clear_messages, init_db, load_messages, save_message
from providers import ProviderInfo, cloud_providers, get_llm, local_providers
from router import judge_route, quality_low, to_lc_messages

load_dotenv()


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


def _provider_selectbox(label: str, providers: list[ProviderInfo]) -> ProviderInfo | None:
    if not providers:
        return None
    keys = [p.key for p in providers]
    selected_key = st.selectbox(
        label,
        options=keys,
        format_func=lambda k: next(p.label for p in providers if p.key == k),
    )
    return next(p for p in providers if p.key == selected_key)


def main() -> None:
    logger = _setup_logger()
    init_db()

    st.set_page_config(page_title="SmartRoute AI", page_icon=":speech_balloon:")
    st.title("SmartRoute AI")
    st.caption("Routes prompts to local or cloud models to minimize cost.")

    if "messages" not in st.session_state:
        st.session_state.messages = load_messages()
        logger.info("memory_loaded | count=%s", len(st.session_state.messages))

    local_pvds = local_providers()
    cloud_pvds = cloud_providers()

    # ------------------------------------------------------------------ sidebar
    with st.sidebar:
        st.subheader("Routing")
        mode = st.selectbox(
            "Mode",
            ["Auto", "Local only", "Cloud only"],
            help="Auto: local judge decides, with cloud fallback on weak answers.",
        )

        st.subheader("Local provider")
        local_info = _provider_selectbox("Provider", local_pvds)
        if local_info:
            local_model = st.selectbox("Model##local", local_info.model_options)
        else:
            st.info("No local providers available.")
            local_model = None

        st.subheader("Cloud provider")
        cloud_info = _provider_selectbox("Provider##cloud", cloud_pvds)
        if cloud_info:
            cloud_model = st.selectbox("Model##cloud", cloud_info.model_options)
        else:
            st.info("No cloud providers configured.\nAdd API key(s) to `.env`:\n`GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`.")
            cloud_model = None

        st.divider()
        if st.button("Clear memory", type="secondary"):
            clear_messages()
            st.session_state.messages = []
            logger.info("memory_cleared")
            st.rerun()

    cloud_available = cloud_info is not None and cloud_model is not None
    local_available = local_info is not None and local_model is not None

    # ---------------------------------------------------------- chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("meta"):
                st.caption(message["meta"])

    # ---------------------------------------------------------- prompt input
    prompt = st.chat_input("Ask anything...")
    if not prompt:
        return

    if len(prompt) > MAX_PROMPT_CHARS:
        st.warning(f"Prompt too long ({len(prompt):,} chars). Limit is {MAX_PROMPT_CHARS:,}.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
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
            from router import heuristic_route
            decision = heuristic_route(prompt, cloud_available)
        use_cloud = decision.use_cloud
        route_reason = decision.reason

    lc_messages = to_lc_messages(st.session_state.messages)
    logger.info(
        "request | chars=%s | local=%s/%s | cloud=%s/%s | mode=%s | route=%s",
        len(prompt),
        local_info.key if local_info else "none",
        local_model or "none",
        cloud_info.key if cloud_info else "none",
        cloud_model or "none",
        mode,
        "cloud" if use_cloud else "local",
    )

    # ---------------------------------------------------------- generation
    with st.chat_message("assistant"):
        answer = ""
        route_meta = ""
        slot = st.empty()

        try:
            if use_cloud and cloud_llm is not None:
                logger.info("call_start | provider=%s | model=%s", cloud_info.key, cloud_model)
                with slot.container():
                    answer = st.write_stream(cloud_llm.stream(lc_messages))
                route_meta = f"Cloud ({cloud_info.label} / {cloud_model}): {route_reason}"
                logger.info("call_done | provider=%s | chars=%s", cloud_info.key, len(answer))

            elif local_llm is not None:
                logger.info("call_start | provider=%s | model=%s", local_info.key, local_model)
                with slot.container():
                    answer = st.write_stream(local_llm.stream(lc_messages))
                route_meta = f"Local ({local_info.label} / {local_model}): {route_reason}"
                logger.info("call_done | provider=%s | chars=%s", local_info.key, len(answer))

                if cloud_llm is not None and quality_low(answer):
                    logger.info(
                        "fallback | %s → %s/%s",
                        local_info.key, cloud_info.key, cloud_model,
                    )
                    slot.empty()
                    with slot.container():
                        answer = st.write_stream(cloud_llm.stream(lc_messages))
                    route_meta = f"Escalated to cloud ({cloud_info.label} / {cloud_model}): local quality low"
                    logger.info("call_done | provider=%s | chars=%s", cloud_info.key, len(answer))

            else:
                answer = "No providers available. Configure a local or cloud provider in `.env`."
                route_meta = "No provider"
                slot.warning(answer)

        except Exception as exc:
            logger.exception("call_error | %s", exc)
            answer = "Something went wrong generating a response. Check the log for details."
            route_meta = "Error"
            slot.error(answer)

        st.caption(route_meta)

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": route_meta})
    save_message("assistant", answer, route_meta)


if __name__ == "__main__":
    main()
