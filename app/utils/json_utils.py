import json
import re
from typing import Any


def parse_json_response(response: str) -> dict[str, Any]:
    """
    Parses JSON returned by an LLM.
    Handles both clean JSON and accidental text-wrapped JSON.
    """
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", response, re.DOTALL)

    if not match:
        raise ValueError("No valid JSON object found in LLM response.")

    return json.loads(match.group())


def pretty_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)