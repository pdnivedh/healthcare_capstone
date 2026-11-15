"""Validate eval/questions.csv against the local hospital.duckdb.

Run from D:\\healthcare_capstone using: python eval/validate_questions.py
The database is opened read-only. This script does not use LM Studio.
"""
import csv
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.assistant.guardrails import validate_sql

CSV_PATH = ROOT / 'eval' / 'questions.csv'
DB_PATH = ROOT / 'hospital.duckdb'

if not CSV_PATH.exists() or not DB_PATH.exists():
    raise SystemExit('Missing eval/questions.csv or hospital.duckdb in the project root.')

with CSV_PATH.open('r', encoding='utf-8-sig', newline='') as f:
    questions = list(csv.DictReader(f))

print(f'Checking {len(questions)} reference SQL queries against DuckDB...')
if len(questions) != 50 or len({q['id'] for q in questions}) != 50:
    raise SystemExit('Expected exactly 50 unique question IDs.')

failures = []
con = duckdb.connect(str(DB_PATH), read_only=True)
try:
    con.execute('SET enable_external_access=false')
    con.execute('SET threads=1')
    con.execute("SET memory_limit='512MB'")
    for q in questions:
        try:
            safe_sql = validate_sql(q['gold_sql'])
            rows = con.execute(safe_sql).fetchall()
            print(f"{q['id']} PASS ({len(rows)} result row(s))")
        except Exception as exc:
            failures.append((q['id'], str(exc)))
            print(f"{q['id']} FAIL: {exc}")
finally:
    con.close()

print(f'\nReference-query checks: {len(questions)-len(failures)}/{len(questions)} passed')
if failures:
    print('These queries need to be corrected and rechecked before LLM evaluation.')
    raise SystemExit(1)
print('SQL execution checks passed. The reference answers still require manual review.')
