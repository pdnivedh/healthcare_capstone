"""Run every gold SQL query in eval/questions.csv and write a sheet for manual review.

Run from the project root:  python scripts/review_gold.py
Needs hospital.duckdb in the project root.
"""
import os
import duckdb
import pandas as pd

q = pd.read_csv("eval/questions.csv")
print("Columns found:", list(q.columns))


def find(*keys):
    """Return the first column whose lowercase name contains one of the keys."""
    for c in q.columns:
        if any(k in c.lower() for k in keys):
            return c
    return None


col_id = find("id")
col_diff = find("difficulty", "level")
col_q = find("question")
col_sql = find("gold", "sql")

if col_q is None or col_sql is None:
    raise SystemExit("Could not find the question and gold SQL columns. Columns are: " + str(list(q.columns)))

con = duckdb.connect("hospital.duckdb", read_only=True)
rows = []
for i, r in q.iterrows():
    qid = r[col_id] if col_id else i + 1
    diff = r[col_diff] if col_diff else ""
    try:
        df = con.execute(str(r[col_sql])).df()
        preview = df.head(8).to_string(index=False)
        n = len(df)
        err = ""
    except Exception as e:
        preview, n, err = "", -1, str(e)[:200]
    rows.append({
        "id": qid, "difficulty": diff, "question": r[col_q], "gold_sql": r[col_sql],
        "n_rows": n, "result_preview": preview, "error": err,
        "A_verdict": "", "A_notes": "",
    })
    print(f"\n[{qid}] ({diff}) {r[col_q]}")
    print(("ERROR: " + err) if err else preview)

os.makedirs("eval/review", exist_ok=True)
pd.DataFrame(rows).to_csv("eval/review/gold_review.csv", index=False)
print("\nSaved eval/review/gold_review.csv")
con.close()
