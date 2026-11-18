"""The system under test: compare vendor documents against the customer's standard terms."""

import json
import time

import anthropic

from data import CLASSIFICATIONS, CLAUSE_TYPES

MODEL = "claude-opus-5-5"

# Structured outputs: the response is guaranteed to parse into this shape.
# That makes format failures rare, so the interesting checks are coverage and correctness.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "clauses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clause_type": {"type": "string", "enum": CLAUSE_TYPES},
                    "classification": {"type": "string", "enum": CLASSIFICATIONS},
                    "source_document": {"type": "string"},
                    "evidence": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["clause_type", "classification", "source_document", "evidence", "explanation"],
                "additionalProperties": False,
            },
        },
        "documents_reviewed": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["clauses", "documents_reviewed"],
    "additionalProperties": False,
}


class ComparatorError(Exception):
    """A run that produced no gradable answer (refusal, truncation, wrong model). Never scored as a wrong answer."""


def build_user_message(standard_terms: dict[str, str], documents: dict[str, str]) -> str:
    terms = "\n".join(f'<clause type="{k}">{v}</clause>' for k, v in standard_terms.items())
    docs = "\n".join(f'<document name="{name}">\n{text}\n</document>' for name, text in documents.items())
    return f"<standard_terms>\n{terms}\n</standard_terms>\n\n<vendor_documents>\n{docs}\n</vendor_documents>"


def compare(client: anthropic.Anthropic, system_prompt: str, standard_terms: dict[str, str],
            documents: dict[str, str], model: str = MODEL) -> tuple[dict, dict]:
    """Returns (output, meta). meta has model, usage, latency, stop_reason."""
    start = time.monotonic()
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=system_prompt,
        messages=[{"role": "user", "content": build_user_message(standard_terms, documents)}],
        output_config={
            "effort": "medium",  # set explicitly so a default change can't silently move the baseline
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
    )
    meta = {
        "model": response.model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_s": round(time.monotonic() - start, 1),
        "stop_reason": response.stop_reason,
    }
    if not response.model.startswith(model):
        raise ComparatorError(f"asked for {model}, got {response.model}")
    if response.stop_reason in ("max_tokens", "refusal"):
        raise ComparatorError(f"stop_reason={response.stop_reason}")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text), meta
