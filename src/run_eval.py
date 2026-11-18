"""Run the eval for one prompt variant, or compare two saved runs.

    python src/run_eval.py --variant v1 --cases case_01 --no-judge   # pilot: 1 case, ~1 API call
    python src/run_eval.py --variant v1                              # full run
    python src/run_eval.py --variant v2
    python src/run_eval.py --compare v1 v2                           # no API calls
"""

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from checks import clause_passes, grade_case, ship_gate, summarize
from comparator import MODEL, compare
from data import ROOT, load_cases, load_documents, load_prompt, load_standard_terms
from judge import JUDGE_MODEL, judge_explanation

RESULTS = ROOT / "results"
PRICE_PER_MTOK = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0)}  # (input, output) USD


def cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICE_PER_MTOK[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def run_case(client, case, system_prompt, standard_terms, use_judge):
    documents = load_documents(case)
    output, meta = compare(client, system_prompt, standard_terms, documents)
    graded = grade_case(output, case, documents)
    graded["meta"] = meta
    graded["raw_output"] = output
    judge_in = judge_out = 0
    if use_judge:
        for row in graded["clauses"]:
            if row["predicted"] in (None, "not_found"):
                continue  # nothing to explain
            try:
                row["judge"], usage = judge_explanation(client, standard_terms[row["clause_type"]], documents, row)
                judge_in += usage["input_tokens"]
                judge_out += usage["output_tokens"]
            except Exception as e:  # a judge error is recorded, never counted as a FAIL
                row["judge"] = {"verdict": "ERROR", "reason": f"{type(e).__name__}: {e}"}
    graded["judge_usage"] = {"input_tokens": judge_in, "output_tokens": judge_out}
    return graded


def run(variant: str, case_ids: list[str] | None, use_judge: bool) -> dict:
    client = anthropic.Anthropic(max_retries=4)
    system_prompt = load_prompt(variant)
    standard_terms = load_standard_terms()
    cases = [c for c in load_cases() if not case_ids or c["id"] in case_ids]

    graded, errors = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {c["id"]: pool.submit(run_case, client, c, system_prompt, standard_terms, use_judge) for c in cases}
        for case_id, future in futures.items():
            try:
                graded.append(future.result())
            except Exception as e:  # API errors, refusals, truncation: recorded, never scored as wrong answers
                errors.append({"case_id": case_id, "error": f"{type(e).__name__}: {e}"})

    model_cost = sum(cost(MODEL, g["meta"]["input_tokens"], g["meta"]["output_tokens"]) for g in graded)
    judge_cost = sum(cost(JUDGE_MODEL, g["judge_usage"]["input_tokens"], g["judge_usage"]["output_tokens"])
                     for g in graded)
    metrics = summarize(graded, errors)
    passed, _ = ship_gate(metrics)
    result = {
        "variant": variant,
        "model": MODEL,
        "judge_model": JUDGE_MODEL if use_judge else None,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
        "ship": passed,
        "cost_usd": {"model": round(model_cost, 4), "judge": round(judge_cost, 4)},
        "latency_s": {g["case_id"]: g["meta"]["latency_s"] for g in graded},
        "errors": errors,
        "cases": graded,
    }
    save(result)
    return result


def save(result: dict) -> None:
    out = RESULTS / result["variant"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(result, indent=2))
    # For judge calibration: fill in your_grade (PASS/FAIL) on some rows, then run calibrate.py.
    with open(out / "calibration.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "clause_type", "predicted", "explanation", "judge_verdict", "judge_reason", "your_grade"])
        for g in result["cases"]:
            for r in g["clauses"]:
                if r.get("judge"):
                    w.writerow([g["case_id"], r["clause_type"], r["predicted"], r["explanation"],
                                r["judge"]["verdict"], r["judge"].get("reason", ""), ""])


def mark(ok) -> str:
    return "ok" if ok else "FAIL"


def print_run(result: dict) -> None:
    print(f"\n=== {result['variant']}  (model {result['model']}, judge {result['judge_model']}) ===\n")
    print(f"{'case':8} {'clause':28} {'gold':18} {'predicted':18} {'right':6} {'quote':6} {'source':7} {'judge':6} result")
    for g in result["cases"]:
        for r in g["clauses"]:
            judge = r.get("judge", {}).get("verdict", "-")
            print(f"{g['case_id']:8} {r['clause_type']:28} {r['gold']:18} {str(r['predicted']):18} "
                  f"{mark(r['correct']):6} {mark(r['grounded']):6} {mark(r['source_correct']):7} {judge:6} "
                  f"{'PASS' if clause_passes(r) else 'FAIL'}")
        if not g["coverage"]:
            print(f"{'':8} COVERAGE FAIL: missing clauses {g['missing_clauses']}, "
                  f"documents not reviewed {g['missing_documents']}, duplicates {g['duplicate_clauses']}")
    for e in result["errors"]:
        print(f"ERROR {e['case_id']}: {e['error']}")
    print_gate([result])
    c = result["cost_usd"]
    print(f"\ncost: model ${c['model']:.3f}, judge ${c['judge']:.3f}   latency per case (s): {result['latency_s']}")


def print_gate(results: list[dict]) -> None:
    gates = [ship_gate(r["metrics"]) for r in results]
    print(f"\n{'ship gate':34}" + "".join(f"{r['variant']:>16}" for r in results))
    for i, (name, _, threshold, _) in enumerate(gates[0][1]):
        cells = ""
        for _, lines in gates:
            _, value, _, ok = lines[i]
            cells += f"{str(value) + ('' if ok else ' FAIL'):>16}"
        rule = "== 0" if name == "errors" else f">= {threshold}"
        print(f"  {name + ' ' + rule:34}{cells}")
    for name in ("false_alarm_rate", "clause_pass"):  # reported, not gated
        print(f"  {name + ' (info)':34}" + "".join(f"{str(r['metrics'][name]):>16}" for r in results))
    print(f"{'SHIP?':34}" + "".join(f"{'YES' if ok else 'NO':>16}" for ok, _ in gates))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", help="prompt file in prompts/, e.g. v1")
    parser.add_argument("--cases", nargs="*", help="only these case ids (for a pilot)")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge (cheaper pilot)")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"), help="compare two saved runs")
    args = parser.parse_args()

    if args.compare:
        runs = [json.loads((RESULTS / v / "run.json").read_text()) for v in args.compare]
        print_gate(runs)
        return
    if not args.variant:
        sys.exit("pass --variant or --compare")
    print_run(run(args.variant, args.cases, not args.no_judge))


if __name__ == "__main__":
    main()
