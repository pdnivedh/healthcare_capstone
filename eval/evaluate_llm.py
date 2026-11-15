"""Incremental, resumable Text-to-SQL benchmark for the hospital capstone.

From the project root:
    python eval/evaluate_llm.py --dry-run
    python eval/evaluate_llm.py --batch-size 1
    python eval/evaluate_llm.py --batch-size 3

Default model: whatever MODEL_NAME specifies in src/assistant/llm_client.py.
Runs the exact same questions under two prompt conditions (schema-only and semantic).
Writes local results to eval/results/; never modifies hospital.duckdb.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import pandas as pd
import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assistant.guardrails import run_sql, validate_sql
from src.assistant.llm_client import (
    MODEL_NAME,
    build_prompt,
    client,
    generate_sql,
)

QUESTIONS = ROOT / "eval" / "questions.csv"
RESULTS = ROOT / "eval" / "results" / "qwen_7b_results.csv"
FIELDS = [
    "id", "difficulty", "question", "condition", "model",
    "generated_sql", "gold_sql", "status", "error",
    "gold_rows", "generated_rows", "gold_preview", "generated_preview",
]
CONDITIONS = (("schema_only", False), ("with_semantic_layer", True))
COMPLETED = {"correct", "wrong_result", "blocked_sql", "invalid_sql", "execution_error"}


def same_cell(a, b):
    """Compare SQL result values with tight tolerance for floating-point outputs."""
    if pd.isna(a) or pd.isna(b):
        return bool(pd.isna(a) and pd.isna(b))
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    return str(a) == str(b)


def same_results(expected, actual):
    """Match result multisets; ignore output aliases and row order, not values."""
    if expected.shape != actual.shape:
        return False
    unmatched = list(actual.itertuples(index=False, name=None))
    for gold_row in expected.itertuples(index=False, name=None):
        match_index = next(
            (
                i for i, result_row in enumerate(unmatched)
                if all(same_cell(x, y) for x, y in zip(gold_row, result_row))
            ),
            None,
        )
        if match_index is None:
            return False
        unmatched.pop(match_index)
    return not unmatched


def preview(df):
    if df is None:
        return ""
    # For review only; no patient-level values need to appear in console.
    return df.head(5).to_json(orient="values", date_format="iso")


def load_saved():
    if not RESULTS.exists():
        return {}
    with RESULTS.open(encoding="utf-8-sig", newline="") as handle:
        return {
            (row["id"], row["condition"]): row
            for row in csv.DictReader(handle)
        }


def save_saved(rows):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    temp = RESULTS.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows.values())
    temp.replace(RESULTS)


def run_reference_sql(sql):
    """Execute prevalidated reference SQL read-only, in-process.

    Reference answers are not model outputs.  Avoid starting the
    12-second query worker while the local LLM is consuming RAM.
    The generated SQL still uses the original guarded worker and timeout.
    """
    safe_sql = validate_sql(sql)
    connection = duckdb.connect(str(ROOT / "hospital.duckdb"), read_only=True)
    try:
        connection.execute("SET enable_external_access = false")
        connection.execute("SET threads = 1")
        connection.execute("SET memory_limit = '512MB'")
        return connection.execute(safe_sql).fetchdf()
    finally:
        connection.close()


def evaluate_one(question, condition, use_semantics, gold_df):
    generated_sql = ""
    got = None
    status = ""
    error = ""
    try:
        generated_sql = generate_sql(
            question["question"], use_semantic_layer=use_semantics
        )
    except Exception as exc:
        status, error = "llm_error", str(exc)[:500]
    else:
        try:
            validate_sql(generated_sql)
        except ValueError as exc:
            status = "invalid_sql" if "Invalid SQL" in str(exc) else "blocked_sql"
            error = str(exc)[:500]
        else:
            try:
                got = run_sql(generated_sql)
            except Exception as exc:
                status, error = "execution_error", str(exc)[:500]
            else:
                status = "correct" if same_results(gold_df, got) else "wrong_result"

    return {
        "id": question["id"],
        "difficulty": question["difficulty"],
        "question": question["question"],
        "condition": condition,
        "model": MODEL_NAME,
        "generated_sql": generated_sql,
        "gold_sql": question["gold_sql"],
        "status": status,
        "error": error,
        "gold_rows": len(gold_df),
        "generated_rows": len(got) if got is not None else "",
        "gold_preview": preview(gold_df),
        "generated_preview": preview(got),
    }


def main():
    global RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Check CSV and prompts; does not need LM Studio")
    parser.add_argument("--batch-size", type=int, default=1, help="How many not-yet-complete questions to process (default 1)")
    parser.add_argument("--check-gold", action="store_true", help="Check all reference SQL in-process without LM Studio")
    parser.add_argument("--results", default="qwen_7b_results.csv", help="Results file name inside eval/results/")
    args = parser.parse_args()
    RESULTS = ROOT / "eval" / "results" / args.results

    if not QUESTIONS.exists():
        sys.exit(f"File not found: {QUESTIONS}")
    with QUESTIONS.open(encoding="utf-8-sig", newline="") as handle:
        questions = list(csv.DictReader(handle))
    if len(questions) != 50 or len({q["id"] for q in questions}) != 50:
        sys.exit("Expected 50 questions with unique IDs.")
    if not all(q.get("gold_sql") and q.get("question") for q in questions):
        sys.exit("Found an empty question or reference SQL.")

    if args.dry_run:
        p0, p1 = build_prompt(False), build_prompt(True)
        assert "INNER JOIN is permitted" in p0 and "INNER JOIN is permitted" in p1
        assert "30-DAY READMISSION" in p1 and "30-DAY READMISSION" not in p0
        print(f"CSV: {len(questions)} questions with unique IDs")
        print(f"Model: {MODEL_NAME}")
        print(f"Schema-only prompt: {len(p0)} characters")
        print(f"Semantic prompt: {len(p1)} characters")
        print("Prompt conditions verified. LM Studio not required.")
        return

    if args.check_gold:
        for q in questions:
            try:
                gold = run_reference_sql(q["gold_sql"])
            except Exception as exc:
                sys.exit(f"{q['id']} reference SQL FAILED: {exc}")
            print(f"{q['id']} PASS ({len(gold)} row(s))", flush=True)
        print(f"Reference-query checks: {len(questions)}/{len(questions)} passed.")
        print("Manual semantic review of reference answers is still required.")
        return

    if args.batch_size < 1:
        sys.exit("--batch-size must be at least 1.")

    # Fail early instead of filling the benchmark with connection-error rows.
    try:
        models = [item.id for item in client.models.list().data]
    except Exception as exc:
        sys.exit(f"LM Studio local server unavailable. Start/load the model first. {exc}")
    print("Local LLM server reachable. Listed models:", ", ".join(models) or "none")
    if MODEL_NAME not in models:
        sys.exit(f"Model '{MODEL_NAME}' is not listed. Load it in LM Studio or confirm its identifier.")

    saved = load_saved()
    pending = [
        q for q in questions
        if any(
            (q["id"], cond) not in saved
            or saved[(q["id"], cond)]["status"] not in COMPLETED
            for cond, _ in CONDITIONS
        )
    ]
    if not pending:
        print("All 50 questions already evaluated under both conditions.")
        return

    for question in pending[:args.batch_size]:
        # Gold uses the same SQL validation, but executes in-process so that
        # Python worker startup under LM Studio load cannot abort a batch.
        try:
            gold_df = run_reference_sql(question["gold_sql"])
        except Exception as exc:
            sys.exit(
                f"Reference SQL for {question['id']} failed: {exc}. "
                "Existing saved model results are preserved."
            )
        for condition, use_semantics in CONDITIONS:
            key = (question["id"], condition)
            if key in saved and saved[key]["status"] in COMPLETED:
                continue
            print(f"{question['id']} / {condition}: generating...", flush=True)
            record = evaluate_one(question, condition, use_semantics, gold_df)
            saved[key] = record
            save_saved(saved)
            print(f"  {record['status']}", flush=True)
            if record["status"] == "llm_error":
                sys.exit("LLM request failed; the result was saved. Fix the issue and rerun this command.")

    total = len(saved)
    correct = sum(r["status"] == "correct" for r in saved.values())
    print(f"Recorded {total}/100 question-condition results; {correct} correct so far.")
    print("Saved to:", RESULTS)
    print("Review incorrect outputs manually; automatic error categories are preliminary.")


if __name__ == "__main__":
    main()
