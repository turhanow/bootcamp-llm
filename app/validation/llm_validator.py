# app/validation/llm_validator.py

from app.client import client
from app.config import VALIDATION_MODEL, MAX_VALIDATION_TOKENS
from app.json_utils import safe_json_loads
from app.validation.prompts import VALIDATION_SYSTEM_PROMPT


def llm_validate(text: str) -> dict:
    response = client.chat.completions.create(
        model=VALIDATION_MODEL,
        messages=[
            {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=MAX_VALIDATION_TOKENS,
    )

    content = response.choices[0].message.content
    return safe_json_loads(content)
