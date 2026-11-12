
import json
import sys

import duckdb

from .guardrails import DB_PATH, validate_sql


def main():
    sql = sys.stdin.read()
    safe_sql = validate_sql(sql)

    con = duckdb.connect(
        str(DB_PATH),
        read_only=True,
        config={
            "enable_external_access": "false",
            "memory_limit": "512MB",
            "threads": "1",
        },
    )

    try:
        result = con.execute(safe_sql)

        columns = [
            col[0] for col in result.description
        ]

        rows = result.fetchall()

        print(json.dumps({
            "columns": columns,
            "rows": rows,
        }, default=str))

    finally:
        con.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
