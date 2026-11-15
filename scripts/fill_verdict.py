"""Fill the A_verdict column of eval/review/gold_review.csv.

Run from the project root:  python scripts/fill_verdict.py
- verified_pandas: recomputed independently with pandas (scripts/verify_gold.py, all PASS)
- verified_consistency: not recomputed, but its result agrees with other verified answers
- TO_READ: you still need to read the SQL once and write your own verdict
"""
import pandas as pd

PATH = "eval/review/gold_review.csv"
df = pd.read_csv(PATH, keep_default_na=False)

pandas_ids = {"Q01", "Q02", "Q03", "Q06", "Q09", "Q13", "Q19", "Q20", "Q23", "Q27",
              "Q29", "Q32", "Q33", "Q38", "Q42", "Q43", "Q44", "Q49", "Q50"}
notes = {
    "Q04": "equals Q01 minus Q03",
    "Q05": "equals Q01 (one score per stay)",
    "Q10": "matches the number of rows in Q25",
    "Q11": "matches the number of rows in Q28",
    "Q12": "weighted mean of Q35 reproduces it",
    "Q34": "larger value equals Q08",
    "Q14": "Q13 + Q14 = Q01",
    "Q15": "Q15 + Q16 = Q01", "Q16": "Q15 + Q16 = Q01",
    "Q17": "Q17 + Q18 = Q03", "Q18": "Q17 + Q18 = Q03",
    "Q25": "matches Q15 and Q16", "Q26": "matches Q17 and Q18",
    "Q28": "counts sum to Q01",
    "Q30": "weighted mean reproduces Q06", "Q31": "weighted mean reproduces Q09",
    "Q35": "weighted mean reproduces Q12",
    "Q39": "year rows consistent with Q42",
    "Q41": "matches the 2019 row of Q38",
    "Q48": "counts sum to Q23",
}

verdicts, why = [], []
for qid in df["id"].astype(str).str.strip():
    if qid in pandas_ids:
        verdicts.append("verified_pandas"); why.append("recomputed in pandas, matched gold")
    elif qid in notes:
        verdicts.append("verified_consistency"); why.append(notes[qid])
    else:
        verdicts.append("TO_READ"); why.append("read the SQL and the question once")

df["A_verdict"] = verdicts
df["A_notes"] = why
df.to_csv(PATH, index=False)
print(df["A_verdict"].value_counts().to_string())
print("\nStill to read:", ", ".join(df.loc[df["A_verdict"] == "TO_READ", "id"].astype(str)))
