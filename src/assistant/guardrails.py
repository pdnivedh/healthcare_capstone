
from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "hospital.duckdb"

ALLOWED_TABLES = {
    "syn_stays",
    "syn_risk_scores",
}

SAFE_FUNCTIONS = {
    "COUNT", "SUM", "AVG", "MIN", "MAX",
    "ROUND", "ABS", "COALESCE", "NULLIF",
    "EXTRACT", "DATE_TRUNC", "LOWER", "UPPER",
    "CAST", "YEAR", "MONTH",
}

MAX_ROWS = 1000
QUERY_TIMEOUT = 12

FORBIDDEN = (
    exp.Insert, exp.Update, exp.Delete,
    exp.Drop, exp.Create, exp.Alter,
    exp.Merge, exp.Command,
)


def validate_sql(sql: str) -> str:
    """Validate a SQL query before executing it."""

    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL cannot be empty.")

    if len(sql) > 5000:
        raise ValueError("SQL query is too long.")

    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.ParseError as exc:
        raise ValueError(f"Invalid SQL: {exc}") from exc

    if len(statements) != 1 or statements[0] is None:
        raise ValueError("Only one SQL statement is allowed.")

    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.Union)):
        raise ValueError("Only SELECT queries are allowed.")

    for node in tree.walk():
        if isinstance(node, FORBIDDEN) or node.key == "into":
            raise ValueError("Unsafe SQL operation detected.")

    # Restrict the first version to single-table queries.
    if any(tree.find_all(exp.Join)):
        raise ValueError("Joins are disabled in this first version.")

    cte_names = {
        cte.alias_or_name.lower()
        for cte in tree.find_all(exp.CTE)
    }

    for clause in tree.find_all(exp.With):
        if clause.args.get("recursive"):
            raise ValueError("Recursive CTEs are not allowed.")

    if cte_names & ALLOWED_TABLES:
        raise ValueError("CTE cannot shadow a hospital table.")

    real_table_found = False

    for table in tree.find_all(exp.Table):
        name = table.name.lower()

        if table.db or table.catalog:
            raise ValueError("Qualified table names are blocked.")

        if name in ALLOWED_TABLES:
            real_table_found = True
        elif name not in cte_names:
            raise ValueError(f"Unauthorized table: {name}")

    if not real_table_found:
        raise ValueError("No approved hospital table referenced.")

    # Allow only approved analytical SQL functions.
    for func in tree.find_all(exp.Func):
        name = (
            func.name if isinstance(func, exp.Anonymous)
            else func.sql_name()
        ).upper()

        if name not in SAFE_FUNCTIONS:
            raise ValueError(f"Unauthorized function: {name}")

    # Enforce a final result limit on SELECT, CTE, and UNION.
    validated = tree.sql(dialect="duckdb")

    return (
        f"SELECT * FROM ({validated}) AS safe_result "
        f"LIMIT {MAX_ROWS}"
    )


def run_sql(sql: str) -> pd.DataFrame:
    """Run validated SQL in a time-limited subprocess."""

    validate_sql(sql)

    try:
        process = subprocess.run(
            [sys.executable, "-m", "src.assistant.query_worker"],
            input=sql,
            text=True,
            capture_output=True,
            timeout=QUERY_TIMEOUT,
            cwd=str(ROOT),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"SQL execution exceeded {QUERY_TIMEOUT} seconds."
        ) from exc

    if process.returncode != 0:
        raise ValueError(
            "Query execution failed: " + process.stderr[:500]
        )

    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "SQL worker did not return valid results."
        ) from exc

    return pd.DataFrame(
        payload["rows"],
        columns=payload["columns"],
    )
