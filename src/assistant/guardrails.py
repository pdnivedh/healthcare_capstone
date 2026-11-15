
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
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Merge,
    exp.Command,
)


def validate_approved_join(tree):
    """
    Allow only a direct INNER JOIN between syn_stays and
    syn_risk_scores using PATIENT, START, and STOP.

    Queries without JOINs remain unaffected.
    """

    joins = list(tree.find_all(exp.Join))

    if not joins:
        return

    # Only one direct JOIN is permitted.
    # JOINs inside CTEs, subqueries, or UNIONs are blocked.
    if (
        not isinstance(tree, exp.Select)
        or len(joins) != 1
        or tree.find(exp.CTE)
        or tree.find(exp.Subquery)
        or tree.find(exp.Union)
    ):
        raise ValueError(
            "Only one direct, approved INNER JOIN is allowed."
        )

    top_level_joins = tree.args.get("joins") or []

    if (
        len(top_level_joins) != 1
        or top_level_joins[0] is not joins[0]
    ):
        raise ValueError("Nested JOINs are not allowed.")

    join = joins[0]
    from_clause = tree.args.get("from_")

    if from_clause is None:
        raise ValueError("JOIN must have a FROM table.")

    left = from_clause.this
    right = join.this

    if (
        not isinstance(left, exp.Table)
        or not isinstance(right, exp.Table)
    ):
        raise ValueError(
            "JOIN requires two direct hospital tables."
        )

    # Both approved tables must be present exactly once.
    tables = list(tree.find_all(exp.Table))

    if (
        len(tables) != 2
        or {left.name.lower(), right.name.lower()}
        != ALLOWED_TABLES
    ):
        raise ValueError(
            "JOIN must use syn_stays and syn_risk_scores."
        )

    # Reject LEFT, RIGHT, FULL, CROSS,
    # NATURAL, and USING joins.
    kind = (join.args.get("kind") or "").upper()

    if (
        kind not in ("", "INNER")
        or join.args.get("side")
        or join.args.get("method")
        or join.args.get("using")
    ):
        raise ValueError(
            "Only INNER JOIN ... ON is allowed."
        )

    # Retrieve table names or aliases.
    aliases = {
        left.name.lower(): left.alias_or_name.lower(),
        right.name.lower(): right.alias_or_name.lower(),
    }

    stay_alias = aliases["syn_stays"]
    risk_alias = aliases["syn_risk_scores"]

    if stay_alias == risk_alias:
        raise ValueError(
            "JOIN table aliases must be different."
        )

    # Split an ON expression into AND conditions.
    def split_and(expression):
        if isinstance(expression, exp.Paren):
            return split_and(expression.this)

        if isinstance(expression, exp.And):
            return (
                split_and(expression.this)
                + split_and(expression.expression)
            )

        return [expression]

    conditions = split_and(join.args.get("on"))
    actual_pairs = set()

    for condition in conditions:

        if not isinstance(condition, exp.EQ):
            raise ValueError(
                "JOIN requires equality conditions."
            )

        a = condition.this
        b = condition.expression

        if (
            not isinstance(a, exp.Column)
            or not isinstance(b, exp.Column)
        ):
            raise ValueError(
                "JOIN conditions must compare columns."
            )

        pair = frozenset({
            (a.table.lower(), a.name.lower()),
            (b.table.lower(), b.name.lower()),
        })

        if len(pair) != 2:
            raise ValueError(
                "Invalid JOIN comparison."
            )

        actual_pairs.add(pair)

    # The three verified matching keys.
    required_pairs = {
        frozenset({
            (stay_alias, key),
            (risk_alias, key),
        })
        for key in ("patient", "start", "stop")
    }

    if (
        len(conditions) != 3
        or actual_pairs != required_pairs
    ):
        raise ValueError(
            "JOIN must match PATIENT, START, and STOP exactly."
        )


def validate_sql(sql: str) -> str:
    """
    Validate SQL before execution.

    Allowed:
        - SELECT queries
        - Approved hospital tables
        - Approved analytical functions
        - One restricted INNER JOIN

    Blocked:
        - Data modification
        - Unauthorized tables
        - Multiple SQL statements
        - Unsafe JOIN patterns
        - Unauthorized SQL functions
    """

    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL cannot be empty.")

    if len(sql) > 5000:
        raise ValueError("SQL query is too long.")

    try:
        statements = sqlglot.parse(
            sql,
            read="duckdb",
        )
    except sqlglot.errors.ParseError as exc:
        raise ValueError(
            f"Invalid SQL: {exc}"
        ) from exc

    if len(statements) != 1 or statements[0] is None:
        raise ValueError(
            "Only one SQL statement is allowed."
        )

    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.Union)):
        raise ValueError(
            "Only SELECT queries are allowed."
        )

    # Block destructive SQL operations.
    for node in tree.walk():
        if (
            isinstance(node, FORBIDDEN)
            or node.key == "into"
        ):
            raise ValueError(
                "Unsafe SQL operation detected."
            )

    # NEW: validate restricted hospital-table JOINs.
    validate_approved_join(tree)

    # Collect CTE names.
    cte_names = {
        cte.alias_or_name.lower()
        for cte in tree.find_all(exp.CTE)
    }

    # Recursive CTEs are not allowed.
    for clause in tree.find_all(exp.With):
        if clause.args.get("recursive"):
            raise ValueError(
                "Recursive CTEs are not allowed."
            )

    # CTEs cannot impersonate approved tables.
    if cte_names & ALLOWED_TABLES:
        raise ValueError(
            "CTE cannot shadow a hospital table."
        )

    real_table_found = False

    # Validate referenced tables.
    for table in tree.find_all(exp.Table):

        name = table.name.lower()

        if table.db or table.catalog:
            raise ValueError(
                "Qualified table names are blocked."
            )

        if name in ALLOWED_TABLES:
            real_table_found = True

        elif name not in cte_names:
            raise ValueError(
                f"Unauthorized table: {name}"
            )

    if not real_table_found:
        raise ValueError(
            "No approved hospital table referenced."
        )

    
    # Allow only approved analytical SQL functions.
    for func in tree.find_all(exp.Func):

        # AND, OR, and XOR are logical connectors,
        # not database functions.
        if isinstance(func, exp.Connector):
            continue

        name = (
            func.name
            if isinstance(func, exp.Anonymous)
            else func.sql_name()
        ).upper()

        if name not in SAFE_FUNCTIONS:
            raise ValueError(
                f"Unauthorized function: {name}"
            )


    # Normalize SQL using the DuckDB dialect.
    validated = tree.sql(
        dialect="duckdb"
    )

    # Restrict the final number of returned rows.
    return (
        f"SELECT * FROM ({validated}) AS safe_result "
        f"LIMIT {MAX_ROWS}"
    )


def run_sql(sql: str) -> pd.DataFrame:
    """
    Validate and execute SQL in a time-limited subprocess.
    """

    validate_sql(sql)

    try:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.assistant.query_worker",
            ],
            input=sql,
            text=True,
            capture_output=True,
            timeout=QUERY_TIMEOUT,
            cwd=str(ROOT),
            check=False,
        )

    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"SQL execution exceeded "
            f"{QUERY_TIMEOUT} seconds."
        ) from exc

    if process.returncode != 0:
        raise ValueError(
            "Query execution failed: "
            + process.stderr[:500]
        )

    try:
        payload = json.loads(
            process.stdout
        )

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "SQL worker did not return valid results."
        ) from exc

    return pd.DataFrame(
        payload["rows"],
        columns=payload["columns"],
    )
