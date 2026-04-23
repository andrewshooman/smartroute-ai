import json
import logging
from dataclasses import dataclass
from typing import Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger("hybrid_router")

# Prompts scoring below this skip the LLM judge entirely and go local immediately.
_JUDGE_SKIP_THRESHOLD = 0.2


@dataclass
class RouteDecision:
    use_cloud: bool
    reason: str


def estimate_complexity(user_text: str) -> float:
    text = user_text.strip()
    lower = text.lower()
    score = 0.0

    if len(text) > 450:
        score += 0.4
    elif len(text) > 220:
        score += 0.2

    complex_keywords = [
        "design", "architecture", "tradeoff", "compare", "analyze",
        "debug", "production", "refactor", "optimize", "strategy",
        "multi-step", "step by step", "mathematical proof",
    ]
    hits = sum(1 for kw in complex_keywords if kw in lower)
    score += min(0.5, hits * 0.12)

    if "```" in text or "error" in lower or "stack trace" in lower:
        score += 0.25

    return min(score, 1.0)


def heuristic_route(user_text: str, cloud_available: bool) -> RouteDecision:
    complexity = estimate_complexity(user_text)
    if complexity >= 0.65 and cloud_available:
        return RouteDecision(True, f"high complexity ({complexity:.2f})")
    return RouteDecision(False, f"low/medium complexity ({complexity:.2f})")


def judge_route(local_llm: BaseChatModel, user_text: str, cloud_available: bool) -> RouteDecision:
    if not cloud_available:
        return RouteDecision(False, "cloud unavailable")

    # Skip the judge call entirely for obviously simple prompts.
    complexity = estimate_complexity(user_text)
    if complexity < _JUDGE_SKIP_THRESHOLD:
        return RouteDecision(False, f"trivial complexity ({complexity:.2f}), skipped judge")

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
        response = local_llm.invoke([HumanMessage(content=router_prompt)])
        raw = str(response.content).strip()
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        use_cloud = bool(parsed.get("use_cloud", False))
        reason = str(parsed.get("reason", "local judge decision"))[:180]
        return RouteDecision(use_cloud, f"local-judge: {reason}")
    except Exception as exc:
        logger.warning("judge_route_failed | falling back to heuristic | error=%s", exc)
        return heuristic_route(user_text, cloud_available=cloud_available)


def quality_low(answer: str) -> bool:
    """Return True only if weak-confidence markers appear in the first third of the response."""
    stripped = answer.strip()
    lower = stripped.lower()

    # Short responses: scan all (a hedge anywhere in a short answer = low quality).
    # Longer responses: only scan the first 40% — hedges buried at the end don't count.
    scan_zone = lower if len(lower) < 100 else lower[: int(len(lower) * 0.4)]

    weak_markers = [
        "i don't know", "i do not know", "not sure",
        "cannot help with that", "can't help with that",
        "couldn't find information", "can't find information",
        "cannot find information", "i don't have access",
        "i do not have access", "cannot access", "can't access",
        "cannot browse", "can't browse", "need to look up",
        "needs to be looked up", "insufficient context",
        "i may be wrong", "might be wrong",
    ]
    return any(marker in scan_zone for marker in weak_markers)


def to_lc_messages(
    history: List[Dict[str, str]], system_prompt: str = ""
) -> List[BaseMessage]:
    system = system_prompt or "You are a helpful assistant. Keep answers concise unless the user asks for detail."
    messages: List[BaseMessage] = [SystemMessage(content=system)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages
