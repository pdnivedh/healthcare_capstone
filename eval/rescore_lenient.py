"""Supplementary re-scoring of saved Text-to-SQL results. No LLM and no LM Studio needed.

Why: two kinds of failure in the strict benchmark are not about the model's SQL logic:
  1. Guardrail v1 rejected harmless CASE expressions ("Unauthorized function: CASE").
  2. The strict scorer compares columns by position, so the right answer with
     columns in a different order (e.g. avg first, GENDER second) is marked wrong.

This script re-runs the SAVED generated SQL for those rows only (read-only, no external
access) and reports a lenient accuracy next to the strict one. Raw files are never edited.
The strict result remains the primary benchmark figure.

Run from the project root:
    python eval/rescore_lenient.py --model "Qwen 2.5 7B=eval/results/qwen_7b_results.csv" --model "Qwen3-VL 4B=eval/results/qwen3vl_4b_results.csv"
"""
import argparse
import itertools
import math
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("schema_only", "with_semantic_layer")


# same comparison rules as eval/evaluate_llm.py (copied so this script is standalone)
def same_cell(a, b):
    if pd.isna(a) or pd.isna(b):
        return bool(pd.isna(a) and pd.isna(b))
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    return str(a) == str(b)


def same_results(expected, actual):
    if expected.shape != actual.shape:
        return False
    unmatched = list(actual.itertuples(index=False, name=None))
    for gold_row in expected.itertuples(index=False, name=None):
        idx = next((i for i, row in enumerate(unmatched)
                    if all(same_cell(x, y) for x, y in zip(gold_row, row))), None)
        if idx is None:
            return False
        unmatched.pop(idx)
    return not unmatched


def same_ignoring_column_order(expected, actual):
    if expected.shape != actual.shape:
        return False
    n = actual.shape[1]
    if n > 6:
        return same_results(expected, actual)
    return any(same_results(expected, actual.iloc[:, list(p)]) for p in itertools.permutations(range(n)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help='"Label=path/to/results.csv"')
    ap.add_argument("--output-dir", type=Path, default=ROOT / "eval" / "results" / "analysis_comparison")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(ROOT / "hospital.duckdb"), read_only=True)
    con.execute("SET enable_external_access = false")

    def run(sql):
        text = str(sql).strip().rstrip(";")
        if not text.lower().startswith(("select", "with")):
            raise ValueError("not a SELECT statement")
        return con.execute(text).fetchdf()

    detail, summary = [], []
    for item in args.model:
        label, path = item.split("=", 1)
        df = pd.read_csv(path.strip(), keep_default_na=False, encoding="utf-8-sig")
        for cond in CONDITIONS:
            sub = df[df["condition"] == cond]
            strict = int((sub["status"] == "correct").sum())
            gained_case = gained_order = 0
            for _, r in sub[sub["status"] != "correct"].iterrows():
                case_block = r["status"] == "blocked_sql" and "Unauthorized function: CASE" in r["error"]
                wrong = r["status"] == "wrong_result"
                if not (case_block or wrong):
                    continue
                try:
                    gold, got = run(r["gold_sql"]), run(r["generated_sql"])
                except Exception as exc:
                    detail.append({"model": label, "id": r["id"], "condition": cond,
                                   "original_status": r["status"], "rescored": f"error: {str(exc)[:80]}"})
                    continue
                if same_results(gold, got):
                    outcome = "correct_if_CASE_allowed" if case_block else "correct"
                    gained_case += int(case_block)
                elif same_ignoring_column_order(gold, got):
                    outcome = "correct_if_column_order_ignored"
                    gained_order += 1
                else:
                    outcome = "still_wrong_after_CASE_allowed" if case_block else "genuinely_wrong"
                detail.append({"model": label, "id": r["id"], "condition": cond,
                               "original_status": r["status"], "rescored": outcome})
            lenient = strict + gained_case + gained_order
            n = len(sub)
            summary.append({"model": label, "condition": cond, "n": n, "strict_correct": strict,
                            "strict_pct": round(100 * strict / n, 1),
                            "plus_CASE_allowed": gained_case, "plus_column_order_ignored": gained_order,
                            "lenient_correct": lenient, "lenient_pct": round(100 * lenient / n, 1)})
    con.close()

    summary = pd.DataFrame(summary)
    detail = pd.DataFrame(detail)
    summary.to_csv(args.output_dir / "lenient_rescoring_summary.csv", index=False)
    detail.to_csv(args.output_dir / "lenient_rescoring_detail.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nSTRICT vs LENIENT ACCURACY (strict is the primary benchmark figure)")
    print(summary.to_string(index=False))
    print("\nROWS RE-SCORED")
    print(detail.to_string(index=False) if not detail.empty else "none")
    print("\nSaved to:", args.output_dir)
    print("Raw result files were not modified. Timeouts and invalid SQL are not re-scored here.")


if __name__ == "__main__":
    main()
