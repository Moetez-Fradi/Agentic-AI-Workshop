import json

from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from config import (
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

_active_label = None 


def active_model_label() -> str:
    """Human-readable label for whichever backend most recently answered
    (or, before the first call, whichever one will be tried first)."""
    if _active_label:
        return _active_label
    if GEMINI_API_KEY:
        return f"{GEMINI_MODEL} (Gemini, primary)"
    return f"{OLLAMA_MODEL} (Ollama - no GEMINI_API_KEY set)"


def _gemini_model() -> ChatOpenAI:
    return ChatOpenAI(model=GEMINI_MODEL, api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)


def _ollama_model() -> ChatOpenAI:
    base_url = OLLAMA_BASE_URL.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return ChatOpenAI(model=OLLAMA_MODEL, api_key="ollama", base_url=base_url)


def _json_objects(text: str) -> list[str]:
    """Extract top-level {...} substrings from text via brace matching (so
    a nested object, like a tool call's own "arguments", doesn't break it)."""
    objects, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : i + 1])
                start = None
    return objects


def _recover_inline_tool_calls(tools: list) -> RunnableLambda:
    valid_names = {t.name for t in tools}

    def _fn(message):
        if message.tool_calls or not isinstance(message.content, str):
            return message
        for candidate in _json_objects(message.content):
            try:
                data = json.loads(candidate)
            except ValueError:
                continue
            if isinstance(data, dict) and data.get("name") in valid_names and "arguments" in data:
                message.tool_calls = [
                    {"name": data["name"], "args": data["arguments"], "id": "call_0", "type": "tool_call"}
                ]
                message.content = ""
                break
        return message

    return RunnableLambda(_fn)


def _tag_label(label: str) -> RunnableLambda:
    def _fn(message):
        global _active_label
        _active_label = label
        return message

    return RunnableLambda(_fn)


def get_model(tools: list):
    """Return a tool-bound chat model: Gemini first, Ollama as an automatic
    fallback if a Gemini call fails and GEMINI_API_KEY is set; Ollama alone
    otherwise."""
    ollama = (
        _ollama_model().bind_tools(tools)
        | _recover_inline_tool_calls(tools)
        | _tag_label(OLLAMA_MODEL)
    )

    if not GEMINI_API_KEY:
        return ollama

    gemini = _gemini_model().bind_tools(tools) | _tag_label(f"{GEMINI_MODEL} (Gemini)")
    return gemini.with_fallbacks([ollama])
