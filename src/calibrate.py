"""Does the judge agree with a human? Fill your_grade (PASS/FAIL) in results/<variant>/calibration.csv, then:

    python src/calibrate.py v1
"""

import csv
import sys

from data import ROOT


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "v1"
    with open(ROOT / "results" / variant / "calibration.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["your_grade"].strip().upper() in ("PASS", "FAIL")]
    if not rows:
        sys.exit("No rows graded yet: write PASS or FAIL in the your_grade column first.")
    agree = [r for r in rows if r["your_grade"].strip().upper() == r["judge_verdict"]]
    print(f"Judge agrees with you on {len(agree)}/{len(rows)} ({len(agree) / len(rows):.0%}). Target: 90%.")
    for r in rows:
        if r not in agree:
            print(f"  disagree: {r['case_id']} {r['clause_type']}: you {r['your_grade']}, judge {r['judge_verdict']}"
                  f" ({r['judge_reason']})")


if __name__ == "__main__":
    main()
