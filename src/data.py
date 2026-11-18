"""Loading the eval inputs: standard terms, cases with gold labels, and contract documents."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

CLAUSE_TYPES = [
    "limitation_of_liability",
    "auto_renewal",
    "payment_terms",
    "termination_for_convenience",
]
CLASSIFICATIONS = ["matches", "partially_matches", "does_not_match", "not_found"]


def load_standard_terms() -> dict[str, str]:
    return json.loads((DATA / "standard_terms.json").read_text())


def load_cases() -> list[dict]:
    return json.loads((DATA / "cases.json").read_text())["cases"]


def load_documents(case: dict) -> dict[str, str]:
    """Document name -> text, in the order the case lists them."""
    folder = DATA / "contracts" / case["id"]
    return {name: (folder / f"{name}.md").read_text() for name in case["documents"]}


def load_prompt(variant: str) -> str:
    return (ROOT / "prompts" / f"{variant}.md").read_text()
