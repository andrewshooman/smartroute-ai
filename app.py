import os
import logging
import sqlite3
import json
from dataclasses import dataclass
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama


load_dotenv()

LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3.1:8b-instruct-q4_0")
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "gemini-2.5-flash")
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:11434")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LOG_FILE = os.getenv("LOG_FILE", "logs/router.log")
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "data/memory.db")


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("hybrid_router")
    if logger.handlers:
        return logger

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _get_db_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(MEMORY_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_memory_db() -> None:
    with _get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                meta TEXT DEFAULT '',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def _load_memory() -> List[Dict[str, str]]:
    with _get_db_connection() as conn:
        rows = conn.execute(
            "SELECT role, content, COALESCE(meta, '') AS meta FROM messages ORDER BY id ASC"
        ).fetchall()
    return [
        {"role": row["role"], "content": row["content"], "meta": row["meta"]} for row in rows
    ]


def _save_message(role: str, content: str, meta: str = "") -> None:
    with _get_db_connection() as conn:
        conn.execute(
            "INSERT INTO messages (role, content, meta) VALUES (?, ?, ?)",
            (role, content, meta),
        )


def _clear_memory() -> None:
    with _get_db_connection() as conn:
        conn.execute("DELETE FROM messages")


@dataclass
class RouteDecision:
    use_cloud: bool
    reason: str


def _estimate_complexity(user_text: str) -> float:
    text = user_text.strip()
    lower = text.lower()
    score = 0.0

    if len(text) > 450:
        score += 0.4
    elif len(text) > 220:
        score += 0.2

    complex_keywords = [
        "design",
        "architecture",
        "tradeoff",
        "compare",
        "analyze",
        "debug",
        "production",
        "refactor",
        "optimize",
        "strategy",
        "multi-step",
        "step by step",
        "mathematical proof",
    ]
    hits = sum(1 for kw in complex_keywords if kw in lower)
    score += min(0.5, hits * 0.12)

    if "```" in text or "error" in lower or "stack trace" in lower:
        score += 0.25

    return min(score, 1.0)


def _route(user_text: str, cloud_available: bool) -> RouteDecision:
    complexity = _estimate_complexity(user_text)
    if complexity >= 0.65 and cloud_available:
        return RouteDecision(True, f"high complexity ({complexity:.2f})")
    return RouteDecision(False, f"low/medium complexity ({complexity:.2f})")


def _local_route_decision(
    local_llm: ChatOllama, user_text: str, cloud_available: bool
) -> RouteDecision:
    if not cloud_available:
        return RouteDecision(False, "cloud unavailable")

    router_prompt = (
        "You are a routing judge. Decide if a small local model can fully answer "
        "the user request at high quality.\n"
        "Return ONLY JSON with keys: use_cloud (boolean), reason (string).\n"
        "Rules:\n"
        "- use_cloud=true when task needs deep reasoning, complex coding/debugging, "
        "or high confidence beyond local capability.\n"
        "- use_cloud=false for simple Q&A, rewriting, summarization, extraction, "
        "light brainstorming.\n"
        "User request:\n"
        f"{user_text}"
    )

    try:
        judge_response = local_llm.invoke([HumanMessage(content=router_prompt)])
        raw = str(judge_response.content).strip()
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        use_cloud = bool(parsed.get("use_cloud", False))
        reason = str(parsed.get("reason", "local judge decision"))[:180]
        return RouteDecision(use_cloud, f"local-judge: {reason}")
    except Exception:
        return _route(user_text, cloud_available=cloud_available)


def _local_quality_low(answer: str) -> bool:
    lower = answer.lower().strip()
    weak_markers = [
        "i don't know",
        "i do not know",
        "not sure",
        "cannot help with that",
        "can't help with that",
        "couldn't find information",
        "can't find information",
        "cannot find information",
        "i don't have access",
        "i do not have access",
        "cannot access",
        "can't access",
        "cannot browse",
        "can't browse",
        "need to look up",
        "needs to be looked up",
        "insufficient context",
        "i may be wrong",
        "might be wrong",
    ]
    if len(answer.strip()) < 100:
        return True
    return any(marker in lower for marker in weak_markers)


def _to_lc_messages(history: List[Dict[str, str]]) -> List[BaseMessage]:
    messages: List[BaseMessage] = [
        SystemMessage(
            content=(
                "You are a helpful assistant. Keep answers concise unless the user asks for detail."
            )
        )
    ]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


def main() -> None:
    logger = _setup_logger()
    _init_memory_db()
    st.set_page_config(page_title="SmartRoute AI", page_icon=":speech_balloon:")
    st.title("SmartRoute AI")
    st.caption("Routes easy prompts to local model and escalates hard prompts to cloud.")

    if "messages" not in st.session_state:
        st.session_state.messages = _load_memory()
        logger.info("memory_loaded | messages=%s", len(st.session_state.messages))

    cloud_available = bool(GEMINI_API_KEY)
    local_llm = ChatOllama(model=LOCAL_MODEL, base_url=LOCAL_BASE_URL, temperature=0.2)
    cloud_llm = None
    if cloud_available:
        cloud_llm = ChatGoogleGenerativeAI(
            model=CLOUD_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.2,
        )

    with st.sidebar:
        st.subheader("Config")
        mode = st.selectbox(
            "Routing mode",
            options=["Auto", "Local only", "Cloud only"],
            index=0,
            help="Auto uses heuristics, Local only always uses Ollama, Cloud only always uses Gemini.",
        )
        st.write(f"Local model: `{LOCAL_MODEL}`")
        st.write(f"Cloud model: `{CLOUD_MODEL}`")
        st.write(f"Cloud available: `{'yes' if cloud_available else 'no'}`")
        st.write(f"Log file: `{LOG_FILE}`")
        st.write(f"Memory DB: `{MEMORY_DB_PATH}`")
        st.markdown("Cloud requires `GEMINI_API_KEY` in your `.env` file.")
        if st.button("Clear persistent memory", type="secondary"):
            _clear_memory()
            st.session_state.messages = []
            logger.info("memory_cleared")
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("meta"):
                st.caption(message["meta"])

    prompt = st.chat_input("Ask anything...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    _save_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    if mode == "Local only":
        decision = RouteDecision(False, "manual mode: local only")
    elif mode == "Cloud only":
        if cloud_available:
            decision = RouteDecision(True, "manual mode: cloud only")
        else:
            decision = RouteDecision(False, "cloud-only requested but cloud unavailable")
    else:
        decision = _local_route_decision(
            local_llm=local_llm, user_text=prompt, cloud_available=cloud_available
        )
    lc_messages = _to_lc_messages(st.session_state.messages)
    logger.info(
        "request_received | chars=%s | cloud_available=%s | mode=%s | route_decision=%s",
        len(prompt),
        cloud_available,
        mode,
        "cloud" if decision.use_cloud else "local",
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ""
            route_meta = ""
            try:
                if decision.use_cloud and cloud_llm is not None:
                    logger.info("model_call_start | provider=gemini | model=%s", CLOUD_MODEL)
                    response = cloud_llm.invoke(lc_messages)
                    answer = response.content
                    route_meta = f"Routed to cloud: {decision.reason}"
                    logger.info("model_call_done | provider=gemini | model=%s", CLOUD_MODEL)
                else:
                    logger.info("model_call_start | provider=ollama | model=%s", LOCAL_MODEL)
                    response = local_llm.invoke(lc_messages)
                    answer = response.content
                    route_meta = f"Routed to local: {decision.reason}"
                    logger.info("model_call_done | provider=ollama | model=%s", LOCAL_MODEL)

                    if cloud_llm is not None and _local_quality_low(answer):
                        logger.info(
                            "fallback_triggered | from=ollama:%s | to=gemini:%s",
                            LOCAL_MODEL,
                            CLOUD_MODEL,
                        )
                        cloud_response = cloud_llm.invoke(lc_messages)
                        answer = cloud_response.content
                        route_meta = "Local answer quality low, escalated to cloud fallback."
                        logger.info("model_call_done | provider=gemini | model=%s", CLOUD_MODEL)
            except Exception as exc:
                answer = f"Error while generating response: {exc}"
                route_meta = "Model call failed."
                logger.exception("model_call_error | error=%s", exc)

        st.markdown(answer)
        st.caption(route_meta)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "meta": route_meta}
    )
    _save_message("assistant", answer, route_meta)


if __name__ == "__main__":
    main()
