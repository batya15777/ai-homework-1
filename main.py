from __future__ import annotations

from tools import TOOL_DISPATCH
from router import route

import json
from pathlib import Path


HISTORY_PATH = Path("history.json")


def _load_history() -> list[dict[str, str]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            print("ברוך שובך")
            return [
                m for m in data if isinstance(m, dict) and m.get("role") in {"user", "assistant", "system"}
            ]
    except Exception:
        pass
    return []


def _save_history(messages: list[dict[str, str]]) -> None:
    HISTORY_PATH.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")


def _reset_history() -> list[dict[str, str]]:
    try:
        HISTORY_PATH.unlink(missing_ok=True)
    except TypeError:
        # Python < 3.8 compat (not expected here, but harmless)
        if HISTORY_PATH.exists():
            HISTORY_PATH.unlink()
    return []


def main() -> None:
    messages: list[dict[str, str]] = _load_history()

    while True:
        user_question = input("Ask something ('/reset' to clear): ").strip()
        if not user_question:
            continue

        if user_question == "/reset":
            messages = _reset_history()
            print("שיחה חדשה התחילה.")
            continue

        messages.append({"role": "user", "content": user_question})

        plan = route(messages)
        steps = plan["steps"]

        scratch: dict[str, object] = {"_results": []}
        assistant_text: str | None = None

        for step in steps:
            tool_name = step["tool"]
            tool_input = step["input"]

            tool_fn = TOOL_DISPATCH.get(tool_name)
            if tool_fn is None:
                assistant_text = f"Unknown tool planned: {tool_name!r}"
                break

            try:
                result = tool_fn(history=messages, tool_input=tool_input, scratch=scratch)
            except Exception as e:
                assistant_text = f"Tool error in {tool_name!r}: {e}"
                break

            # Keep a structured record of tool outputs for the final chat step.
            if tool_name != "chat":
                scratch_results = scratch.get("_results")
                if isinstance(scratch_results, list):
                    scratch_results.append({"tool": tool_name, "input": tool_input, "output": result})
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"[TOOL {tool_name}] input={tool_input} output={result}",
                    }
                )

            if tool_name == "chat":
                assistant_text = str(result)

        if assistant_text is None:
            assistant_text = "Sorry, I couldn't complete that request."

        print(assistant_text)
        messages.append({"role": "assistant", "content": assistant_text})
        _save_history(messages)


if __name__ == "__main__":
    main()