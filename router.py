from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Literal, TypedDict

from openai import OpenAI
from dotenv import load_dotenv


class ToolStep(TypedDict):
    tool: Literal["weather", "exchange", "math", "chat"]
    input: Any


class Plan(TypedDict):
    steps: List[ToolStep]


_PLANNER_SYSTEM_PROMPT = """You are an AI Orchestrator planner.

Your job is to plan which tools to call (and in what order) to answer the user's
latest question, based on the full conversation history.

Tools:
1) weather  – input: a city name string (e.g. "Dubai")
2) exchange – input: a currency code string (e.g. "USD", "EUR")
3) math     – input: a math expression string (do NOT compute; just build expression)
4) chat     – input: the user's question (a string). This must be the final step.

Rules:
- Return ONLY valid JSON, exactly one JSON object, nothing else.
- Output format MUST be: {"steps": [{"tool": "...", "input": ...}, ...]}
- The final step MUST always be {"tool":"chat","input":"<original user question>"}.
- Do NOT answer the question. Do NOT calculate anything.
- For multi-step queries, include all required tool calls before the final chat.

Token rules for building math expressions:
- You may reference prior tool outputs using tokens that will be substituted later:
  - {{exchange:USD}} means "ILS per 1 USD"
  - {{exchange:EUR}} means "ILS per 1 EUR"
  - {{weather:Dubai}} means "current temperature (°C) in Dubai"
- Example (Euros from 100 dollars):
  steps: exchange USD, exchange EUR, math "100 * ({{exchange:USD}} / {{exchange:EUR}})", chat
- Example (how many times hotter Dubai vs Stockholm):
  steps: weather Dubai, weather Stockholm, math "({{weather:Dubai}} / {{weather:Stockholm}})", chat

If ambiguous or not needing tools, plan only the final chat step.
"""


def route(messages: List[Dict[str, str]]) -> Plan:
    """
    Create a tool execution plan based on the full conversation history.
    """
    load_dotenv()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI()

    raw_text = _call_openai_planner(client=client, model=model, messages=messages)
    return _parse_plan(raw_text, messages)


def _call_openai_planner(*, client: OpenAI, model: str, messages: List[Dict[str, str]]) -> str:
    """
    Calls OpenAI and returns the raw text content.

    We keep this small and defensive because SDK surfaces can vary by version.
    """
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
                *messages,
            ],
            # If supported by the model/SDK, this nudges JSON-only output.
            response_format={"type": "json_object"},
        )
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:
        # Fall back to chat.completions for older SDK usage.
        pass

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            *messages,
        ],
    )
    return resp.choices[0].message.content.strip()


def _parse_plan(raw_text: str, messages: List[Dict[str, str]]) -> Plan:
    obj = _loads_json_object(raw_text)
    steps = obj.get("steps")
    if not isinstance(steps, list):
        steps = []

    normalized: List[ToolStep] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        tool = str(step.get("tool", "")).strip()
        tool_input = step.get("input", "")
        if tool in {"weather", "exchange", "math", "chat"}:
            normalized.append({"tool": tool, "input": tool_input})  # type: ignore[typeddict-item]

    user_question = _latest_user_question(messages)
    if not normalized or normalized[-1]["tool"] != "chat":
        normalized.append({"tool": "chat", "input": user_question})
    else:
        normalized[-1] = {"tool": "chat", "input": user_question}

    return {"steps": normalized}


def _latest_user_question(messages: List[Dict[str, str]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content", "")).strip()
    return ""


def _loads_json_object(raw_text: str) -> Dict[str, Any]:
    raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Last-resort: extract a JSON object substring.
    match = re.search(r"\{[\s\S]*\}", raw_text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Safe fallback: route to chat with the original text as input.
    return {"tool": "chat", "input": raw_text}

