import json
import re


def safe_json_loads(text: str) -> dict:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("JSON not found")

        return json.loads(match.group())
    except Exception:
        return {
            "is_relevant": False,
            "category": "other",
            "reason": "Некорректный формат ответа модели"
        }
