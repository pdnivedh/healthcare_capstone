
from pathlib import Path
import duckdb

DB_PATH = Path(__file__).resolve().parents[2] / "hospital.duckdb"

# Tables the assistant is allowed to know about
ALLOWED_TABLES = [
    "syn_stays",
    "syn_risk_scores",
]

def get_schema_text():
    """Extract table structures for the LLM prompt."""

    con = duckdb.connect(str(DB_PATH), read_only=True)

    schema_parts = []

    try:
        for table in ALLOWED_TABLES:
            columns = con.execute(
                f'DESCRIBE "{table}"'
            ).fetchall()

            column_definitions = []

            for column in columns:
                name = column[0]
                data_type = column[1]

                column_definitions.append(
                    f'    "{name}" {data_type}'
                )

            create_statement = (
                f'CREATE TABLE "{table}" (\n'
                + ",\n".join(column_definitions)
                + "\n);"
            )

            schema_parts.append(create_statement)

    finally:
        con.close()

    return "\n\n".join(schema_parts)


if __name__ == "__main__":
    print("=== HOSPITAL DATABASE SCHEMA ===")
    print(get_schema_text())
