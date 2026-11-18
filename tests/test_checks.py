"""Grader tests. No API calls. Proves the checks catch what they claim to before any money is spent."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from checks import grade_case, ship_gate, summarize  # noqa: E402
from data import CLAUSE_TYPES, load_cases, load_documents  # noqa: E402

CASES = load_cases()


def oracle_output(case):
    """A perfect answer built from the gold labels."""
    return {
        "clauses": [
            {"clause_type": t, "classification": g["classification"], "source_document": g["source_document"] or "",
             "evidence": g["evidence"], "explanation": "gold"}
            for t, g in case["labels"].items()
        ],
        "documents_reviewed": list(case["documents"]),
    }


def graded(case, output):
    return grade_case(output, case, load_documents(case))


def test_gold_labels_cover_every_clause_type():
    for case in CASES:
        assert set(case["labels"]) == set(CLAUSE_TYPES), case["id"]


def test_oracle_scores_perfectly():
    results = [graded(c, oracle_output(c)) for c in CASES]
    metrics = summarize(results, errors=[])
    for name in ("coverage", "accuracy", "mismatch_recall", "evidence_grounded",
                 "source_correct_multi_doc", "injection_safe"):
        assert metrics[name] == 1.0, name
    assert metrics["false_alarm_rate"] == 0.0


def test_null_output_fails_everything():
    result = graded(CASES[0], {"clauses": [], "documents_reviewed": []})
    assert not result["coverage"]
    assert not any(r["correct"] for r in result["clauses"])


def test_unparseable_output_fails_schema():
    assert not graded(CASES[0], "not json")["schema_ok"]


def test_skipped_amendment_is_caught():
    """The real production bug: the model ignores the amendment and grades the main agreement only."""
    case = next(c for c in CASES if "multi_doc" in c["tags"])
    out = oracle_output(case)
    out["documents_reviewed"] = ["main"]
    for c in out["clauses"]:
        c["source_document"] = "main"
    result = graded(case, out)
    assert not result["coverage"]
    assert result["missing_documents"] == ["amendment_1"]
    assert not all(r["source_correct"] for r in result["clauses"])


def test_invented_quote_is_not_grounded():
    case = CASES[0]
    out = oracle_output(case)
    out["clauses"][0]["evidence"] = "Liability is capped at twelve months of fees."  # plausible, not in the text
    assert not graded(case, out)["clauses"][0]["grounded"]


def test_quote_matching_ignores_case_whitespace_and_curly_quotes():
    case = CASES[0]
    out = oracle_output(case)
    out["clauses"][0]["evidence"] = "  NEITHER PARTY’S   aggregate liability  "
    assert graded(case, out)["clauses"][0]["grounded"]


def test_injection_that_works_fails_the_gate():
    case = next(c for c in CASES if "injection" in c["tags"])
    out = oracle_output(case)
    for c in out["clauses"]:
        c["classification"] = "matches"
    metrics = summarize([graded(case, out)], errors=[])
    assert metrics["injection_safe"] == 0.0
    assert not ship_gate(metrics)[0]


def test_missing_clause_must_not_be_stretched():
    """case_02 has termination for breach, not for convenience. Calling it a match is wrong."""
    case = next(c for c in CASES if "missing_clause" in c["tags"])
    out = copy.deepcopy(oracle_output(case))
    term = next(c for c in out["clauses"] if c["clause_type"] == "termination_for_convenience")
    term.update(classification="partially_matches", source_document="main",
                evidence="Either party may terminate this Agreement if the other party materially breaches it")
    row = next(r for r in graded(case, out)["clauses"] if r["clause_type"] == "termination_for_convenience")
    assert row["grounded"] and not row["correct"]


def test_errors_block_shipping():
    metrics = summarize([graded(c, oracle_output(c)) for c in CASES], errors=[{"case_id": "x", "error": "timeout"}])
    metrics["judge_pass"] = 1.0
    assert not ship_gate(metrics)[0]
