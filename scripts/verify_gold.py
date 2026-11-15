"""Independently re-derive selected gold answers with pandas and compare with the gold SQL.

Run from the project root:  python scripts/verify_gold.py
Needs hospital.duckdb and eval/questions.csv.
Writes eval/review/gold_verification.csv
"""
import os
import duckdb
import pandas as pd

q = pd.read_csv("eval/questions.csv")
q["id"] = q["id"].astype(str).str.strip()
sql_by_id = dict(zip(q["id"], q["gold_sql"]))

con = duckdb.connect("hospital.duckdb", read_only=True)
s = con.execute("SELECT * FROM syn_stays").df()
r = con.execute("SELECT * FROM syn_risk_scores").df()
tz = con.execute("SELECT current_setting('TimeZone')").fetchone()[0]
print("DuckDB session TimeZone:", tz)
print("Rows: stays", len(s), "| risk scores", len(r), "\n")

s["year_utc"] = pd.to_datetime(s["START"], utc=True).dt.year


def gold(qid):
    return con.execute(sql_by_id[qid]).df()


def as_scalar(df):
    return float(df.iloc[0, 0])


def as_dict(df):
    return {str(k): float(v) for k, v in zip(df.iloc[:, 0], df.iloc[:, 1])}


def close(a, b, tol=1e-4):
    return abs(a - b) <= tol * max(1.0, abs(b))


def dict_close(a, b, tol=1e-4):
    return set(a) == set(b) and all(close(a[k], b[k], tol) for k in a)


def merged():
    return s.merge(r[["PATIENT", "START", "risk_lr", "risk_gb"]], on=["PATIENT", "START"], how="inner")


checks = {
    "Q01": ("scalar", lambda: float(len(s))),
    "Q02": ("scalar", lambda: float(s["PATIENT"].nunique())),
    "Q03": ("scalar", lambda: float(s["readmit_30d"].sum())),
    "Q06": ("scalar", lambda: float(s["los_days"].mean())),
    "Q09": ("scalar", lambda: float(s["age"].mean())),
    "Q13": ("scalar", lambda: float((s["age"] > 65).sum())),
    "Q19": ("scalar", lambda: float(((s["age"] > 65) & s["readmit_30d"]).sum())),
    "Q20": ("scalar", lambda: float((s["los_days"] > 7).sum())),
    "Q23": ("scalar", lambda: float((r["risk_lr"] >= 0.2).sum())),
    "Q27": ("dict", lambda: {str(k): float(v) for k, v in s.groupby("GENDER")["readmit_30d"].mean().items()}),
    "Q29": ("dict", lambda: {str(k): float(v) for k, v in s.groupby("RACE")["readmit_30d"].mean().items()}),
    "Q32": ("dict", lambda: {str(k): float(v) for k, v in s.groupby("readmit_30d")["los_days"].mean().items()}),
    "Q33": ("scalar", lambda: float((s.groupby("PATIENT").size() > 1).sum())),
    "Q38": ("dict", lambda: {str(k): float(v) for k, v in s["year_utc"].value_counts().items()}),
    "Q42": ("scalar", lambda: float(((s["year_utc"] == 2020) & s["readmit_30d"]).sum())),
    "Q43": ("scalar", lambda: float(s.loc[s["year_utc"] == 2021, "PATIENT"].nunique())),
    "Q44": ("scalar", lambda: float(len(merged()))),
    "Q49": ("dict", lambda: {str(k): float(v) for k, v in
                             merged()[merged()["readmit_30d"]].groupby("GENDER")["risk_lr"].mean().items()}),
    "Q50": ("scalar", lambda: float(((r["risk_lr"] >= 0.2) & r["readmit_30d"]).sum())),
}

rows = []
for qid, (kind, fn) in checks.items():
    if qid not in sql_by_id:
        print(f"{qid}: not found in questions.csv, skipped")
        continue
    try:
        g = gold(qid)
        mine = fn()
        if kind == "scalar":
            gv = as_scalar(g)
            ok = close(gv, mine)
            detail = f"gold={gv:.6g} pandas={mine:.6g}"
        else:
            gv = as_dict(g)
            ok = dict_close(gv, mine)
            detail = f"gold={ {k: round(v, 4) for k, v in gv.items()} } pandas={ {k: round(v, 4) for k, v in mine.items()} }"
        status = "PASS" if ok else "MISMATCH"
    except Exception as e:
        status, detail = "ERROR", str(e)[:200]
    rows.append({"id": qid, "status": status, "detail": detail})
    print(f"{qid}: {status}  {detail if status != 'PASS' else ''}")

os.makedirs("eval/review", exist_ok=True)
out = pd.DataFrame(rows)
out.to_csv("eval/review/gold_verification.csv", index=False)
print("\nPassed", (out["status"] == "PASS").sum(), "of", len(out), "| saved eval/review/gold_verification.csv")
con.close()
