# Contract Terms Evals

A small eval harness for an LLM feature that compares a vendor's contract against a customer's standard terms, clause by clause, and says whether each clause matches.

It answers two questions: **is the output right**, and **is a prompt or model change safe to ship**.

## Why

At Rivio I worked on a feature like this and we tested it by hand. That's how I found a real bug: when several documents were compared at once, the model silently skipped one. No error, plausible output, wrong answer. This project is the eval I wish we'd had, rebuilt from scratch on short made-up contracts (no Rivio code or data).

## What's tested

4 contracts x 4 clause types = 16 clause judgments, each with a hand-checked gold label in `data/cases.json`.

| Case | Tag | What it tests |
|---|---|---|
| case_01 | single_doc | Every clause matches, reworded. No false alarms. |
| case_02 | single_doc, missing_clause | Weaker terms; a missing clause that must be `not_found`, not stretched from "termination for breach". |
| case_03 | multi_doc | An amendment overrides the main agreement. **The real production bug.** |
| case_04 | injection | The contract tells AI reviewers to mark everything as matching. |

## How it's graded

**Code checks first** (`src/checks.py`, unit-tested, free):

- `coverage`: every clause answered, every document listed as reviewed. Fail-on-any. Catches the silent skip.
- `accuracy`: classification equals the gold label.
- `mismatch_recall`: of the bad clauses, how many were flagged. The costly error is calling a bad clause fine.
- `evidence_grounded`: the quoted evidence really appears in the cited document.
- `source_correct_multi_doc`: when an amendment overrides, the amendment is cited.
- `injection_safe`: no bad clause waved through on the injection case.

**Then an LLM judge** (`src/judge.py`, `claude-sonnet-5-5`, a different model from the one under test) grades only what code can't: does the explanation name the concrete difference, and does it avoid claims the text doesn't support. The verdict is computed in code from those two properties. A code failure overrides a judge PASS.

**Ship gate** (all must hold): coverage, injection and multi-doc citation at 100%, mismatch recall >= 95%, accuracy >= 90%, grounding >= 95%, judge pass >= 80%, zero errors. API errors are recorded separately, never scored as wrong answers.

**Judge calibration:** I grade a sample of explanations myself (`results/<variant>/calibration.csv`) and `src/calibrate.py` reports agreement. Target 90%.

## The regression demo

`prompts/v1.md` is the baseline. `prompts/v2.md` is a plausible "optimization": shorter, cheaper, "focus on the main agreement". The eval shows whether that change is safe to ship.

## Results (first run)

| | v1 (baseline) | v2 ("cheaper, focus on main agreement") |
|---|---|---|
| Correct classifications | 16/16 | 16/16 |
| Read the amendment and cited it | yes | yes |
| Judge pass rate | 100% | 93% |
| Model cost | $0.096 | $0.077 |
| Gate | ship | ship |

What I learned:

1. **The expected regression didn't happen.** v2 still read the amendment. The eval answered that instead of a guess.
2. **The judge caught a quality drop code couldn't.** One v2 explanation named the vendor's terms but not the standard it was compared to.
3. **The judge was too lenient.** Against my own grades on 8 v2 explanations it agreed 5/8 (62%, target 90%), and all 3 misses were false approvals: it credited comparisons the explanation never made, filling in the standard terms itself. So v2's 93% judge score isn't trustworthy yet. Next: a stricter rubric, re-checked on explanations I grade fresh.
4. **A perfect baseline means the test set is too easy.** Next: harder cases (long contracts, conflicting amendments).

Full outputs: `results/<variant>/run.json`. My grades: `results/v2/calibration.csv`.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests                      # grader tests, no API calls
export ANTHROPIC_API_KEY=...
.venv/bin/python src/run_eval.py --variant v1 --cases case_03 --no-judge   # pilot, 1 call
.venv/bin/python src/run_eval.py --variant v1
.venv/bin/python src/run_eval.py --variant v2
.venv/bin/python src/run_eval.py --compare v1 v2
.venv/bin/python src/calibrate.py v1                     # after filling your_grade
```

## Limits (honest)

- 16 judgments is small: one miss moves accuracy about 6 points. It's a demo of the method, not a benchmark.
- One run per case. Next: 3 reps per case to measure run-to-run noise.
- Next: more cases from real traffic, running it in CI on every prompt change.
