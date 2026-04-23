from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

import requests
from dotenv import load_dotenv
from openai import OpenAI
from simpleeval import SimpleEval


class ToolStep(TypedDict):
    tool: Literal["weather", "exchange", "math", "chat"]
    input: Any


class WeatherResult(TypedDict):
    city: str
    temp_c: float
    description: str


class ExchangeResult(TypedDict):
    currency: str
    ils_per_unit: float


class MathResult(TypedDict):
    expression: str
    result: float


ToolResult = WeatherResult | ExchangeResult | MathResult | str


def getWeather(city: str) -> WeatherResult:
    """
    Fetch current temperature for a city using OpenWeatherMap.
    Requires OPENWEATHER_API_KEY in `.env`.
    """
    load_dotenv()
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENWEATHER_API_KEY in .env")

    resp = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "appid": api_key, "units": "metric"},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()

    temp_c = float(data["main"]["temp"])
    description = str((data.get("weather") or [{}])[0].get("description", "")).strip()
    return {"city": city, "temp_c": temp_c, "description": description}


def getExchangeRate(currencyCode: str) -> ExchangeResult:
    """
    Fetch FX rate for `currencyCode` relative to ILS (shekels).

    Prefers ExchangeRate-API (requires EXCHANGERATE_API_KEY). If missing, falls back
    to the free open.er-api.com endpoint (no key) to keep the exercise runnable.
    """
    load_dotenv()
    code = currencyCode.strip().upper()
    if not code:
        raise ValueError("currencyCode is empty")

    api_key = os.getenv("EXCHANGERATE_API_KEY")
    if api_key:
        url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{code}"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        ils = float(data["conversion_rates"]["ILS"])
        return {"currency": code, "ils_per_unit": ils}

    # Fallback (no key)
    resp = requests.get(f"https://open.er-api.com/v6/latest/{code}", timeout=20)
    resp.raise_for_status()
    data = resp.json()
    ils = float(data["rates"]["ILS"])
    return {"currency": code, "ils_per_unit": ils}


def calculateMath(expression: str) -> MathResult:
    """
    Deterministic math evaluation. The LLM must NOT perform calculations.
    """
    expr = expression.strip()
    if not expr:
        raise ValueError("expression is empty")

    s = SimpleEval()
    s.functions = {"abs": abs, "round": round}
    s.names = {}

    value = s.eval(expr)
    try:
        num = float(value)
    except Exception as e:
        raise ValueError(f"Expression did not evaluate to a number: {value!r}") from e

    return {"expression": expr, "result": num}


def generalChat(
    *,
    context: List[Dict[str, str]],
    userInput: str,
    toolResults: List[Dict[str, Any]],
) -> str:
    """
    Fallback/general assistant response. Uses the OpenAI API and includes the full
    conversation history + tool results so the answer can reference computations.
    """
    load_dotenv()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI()

    tool_blob = json.dumps(toolResults, ensure_ascii=False)
    system = (
        "ענה בעברית, קצר וטבעי.\n"
        "אל תבצע חישובים בעצמך.\n"
        "אל תסביר שלבים.\n"
        "אם חסרים נתונים, אמור זאת בקצרה.\n"
        "אל תשתמש בניסוחים כמו: \"כדי לחשב\", \"אז\", \"כלומר\"."
    )

    cleaned_context = list(context)
    if cleaned_context:
        last = cleaned_context[-1]
        if last.get("role") == "user" and str(last.get("content", "")).strip() == userInput.strip():
            cleaned_context = cleaned_context[:-1]

    messages = [
        {"role": "system", "content": system},
        *cleaned_context,
        {"role": "system", "content": f"TOOL_RESULTS_JSON: {tool_blob}"},
        {"role": "user", "content": userInput},
    ]

    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content.strip()


def _substitute_tokens(expression: str, scratch: Dict[str, Any]) -> str:
    """
    Replaces tokens like {{weather:Dubai}} or {{exchange:USD}} using `scratch`.
    """
    out = expression
    for kind in ("weather", "exchange"):
        kind_map = scratch.get(kind, {})
        if isinstance(kind_map, dict):
            for key, value in kind_map.items():
                token = f"{{{{{kind}:{key}}}}}"
                out = out.replace(token, str(value))
    return out


def tool_weather(*, history: List[Dict[str, str]], tool_input: Any, scratch: Dict[str, Any]) -> ToolResult:
    city_raw = str(tool_input)
    city = " ".join(city_raw.strip().split()).title()
    try:
        result = getWeather(city)
    except Exception:
        return "לא ניתן להביא נתוני מזג אוויר כרגע."
    scratch.setdefault("weather", {})[city] = result["temp_c"]
    temp = int(round(float(result["temp_c"])))
    desc_map = {
        "scattered clouds": "מעונן חלקית",
        "clear sky": "שמיים בהירים",
        "few clouds": "מעט עננים",
        "broken clouds": "עננים מפוזרים",
        "overcast clouds": "מעונן",
        "light rain": "גשם קל",
        "moderate rain": "גשם",
        "heavy intensity rain": "גשם חזק",
        "thunderstorm": "סופת רעמים",
        "mist": "ערפל",
        "haze": "אובך",
        "fog": "ערפל",
    }
    raw_desc = str(result.get("description", "")).strip()
    desc = desc_map.get(raw_desc, raw_desc)
    if any("a" <= ch.lower() <= "z" for ch in desc):
        desc = "מעונן"
    return f"ב{city} יש {temp} מעלות, {desc}."


def tool_exchange(*, history: List[Dict[str, str]], tool_input: Any, scratch: Dict[str, Any]) -> ToolResult:
    code = str(tool_input).strip().upper()
    result = getExchangeRate(code)
    scratch.setdefault("exchange", {})[code] = result["ils_per_unit"]
    rate = float(result["ils_per_unit"])
    s = f"{rate:.2f}".rstrip("0").rstrip(".")
    return f"שער {code} הוא {s} ש\"ח"


def tool_math(*, history: List[Dict[str, str]], tool_input: Any, scratch: Dict[str, Any]) -> ToolResult:
    expr = str(tool_input)
    substituted = _substitute_tokens(expr, scratch)
    result = calculateMath(substituted)
    scratch.setdefault("math", []).append(result)
    num = float(result["result"])
    s = f"{num:.2f}"
    s = s.rstrip("0").rstrip(".")
    return s


def tool_chat(*, history: List[Dict[str, str]], tool_input: Any, scratch: Dict[str, Any]) -> ToolResult:
    user_input = str(tool_input)
    if scratch.get("math"):
        num = float(scratch["math"][-1]["result"])
        s = f"{num:.2f}"
        s = s.rstrip("0").rstrip(".")
        return s

    text = user_input.strip()
    text_cf = text.casefold()

    is_name_q = any(q in text_cf for q in ["איך קוראים לי", "מה השם שלי", "what is my name"])
    is_loc_q = any(q in text_cf for q in ["איפה אני גרה", "איפה אני גר", "where do i live"])
    is_study_q = any(q in text_cf for q in ["מה אני לומדת", "מה אני לומד", "what do i study"])

    if is_name_q or is_loc_q or is_study_q:
        for msg in reversed(history):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = str(msg.get("content", ""))
            c_cf = content.casefold()

            if is_name_q:
                for prefix in ["קוראים לי ", "השם שלי הוא ", "my name is "]:
                    idx = c_cf.find(prefix)
                    if idx != -1:
                        val = content[idx + len(prefix):].split('.')[0].split('!')[0].split('?')[0].split('\n')[0].strip()
                        if val: return f"השם שלך הוא {val}."
            
            if is_loc_q:
                for prefix in ["אני גרה ב", "אני גר ב", "i live in "]:
                    idx = c_cf.find(prefix)
                    if idx != -1:
                        val = content[idx + len(prefix):].split('.')[0].split('!')[0].split('?')[0].split('\n')[0].strip()
                        if val: return f"את גרה ב{val}."

            if is_study_q:
                for prefix in ["אני לומדת ", "אני לומד ", "i study "]:
                    idx = c_cf.find(prefix)
                    if idx != -1:
                        val = content[idx + len(prefix):].split('.')[0].split('!')[0].split('?')[0].split('\n')[0].strip()
                        if val: return f"את לומדת {val}."

        return "אין לי מידע על זה."

    tool_results = scratch.get("_results", [])
    if isinstance(tool_results, list) and tool_results:
        last = tool_results[-1]
        if isinstance(last, dict) and "output" in last:
            return str(last["output"])
    if not isinstance(tool_results, list):
        tool_results = []
    return generalChat(context=history, userInput=user_input, toolResults=tool_results)


TOOL_DISPATCH: Dict[str, Callable[..., ToolResult]] = {
    "weather": tool_weather,
    "exchange": tool_exchange,
    "math": tool_math,
    "chat": tool_chat,
}

