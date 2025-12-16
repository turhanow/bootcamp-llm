# app/validation/pre_llm_validator.py

from typing import Dict, Any

from .preprocessing.preprocessing import build_synthetic_model, validate_query_v2

_NB_MODEL = build_synthetic_model()


def pre_llm_validate(query: str) -> Dict[str, Any]:
    text, accepted, reason = validate_query_v2(
        query=query,
        model=_NB_MODEL,
        decline_unsafe=0.85,
        decline_out_of_domain=0.92,
        hard_rules=True,
    )

    return {
        "accepted": accepted,
        "text": text,
        "reason": reason,
        "layer": "pre_llm",
    }
