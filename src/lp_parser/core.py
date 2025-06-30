"""
Natural-language ➜ structured JSON extractor for small LP word problems.
Scope (v0): max-profit LPs with ‘≤’ constraints and integer, non-negative variables.
"""

from __future__ import annotations
import os, json
from openai import OpenAI

_JSON_SCHEMA = """
{
  "objective": {"sense": "max", "coeff": {"<var>": <float>, ...}},
  "vars": {"<var>": {"ub": <int>}, ...},
  "constraints": [
    {"name": "<string>", "coeff": {"<var>": <float>, ...}, "rhs": <float>},
    ...
  ]
}
"""

def _llm_extract(problem_text: str, model: str = "gpt-4o") -> str:
    """Call OpenAI with JSON mode and return raw JSON string."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the linear-program parameters ONLY as valid JSON "
                    f"matching this schema (no extra keys): {_JSON_SCHEMA}"
                ),
            },
            {"role": "user", "content": problem_text},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content

def parse_word_problem(problem_text: str, model: str = "gpt-4o") -> dict:
    """Return a Python dict describing the LP or raise ValueError on failure."""
    raw = _llm_extract(problem_text, model=model)
    try:
        lp = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}") from None
    # ⬇️ very light validation (students can improve)
    if "objective" not in lp or "vars" not in lp or "constraints" not in lp:
        raise ValueError("Parsed JSON missing required keys.")
    return lp