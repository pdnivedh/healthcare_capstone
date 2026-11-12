
import re
from openai import OpenAI

# Connect to LM Studio running on this laptop
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    timeout=120.0
)

MODEL_NAME = "qwen2.5-7b-instruct"

# Start with a small, verified database schema
SYSTEM_PROMPT = """
You are a healthcare data analyst specializing in DuckDB SQL.

Your task is to convert English questions into valid DuckDB SQL.

Available table:

syn_stays(
    PATIENT VARCHAR,
    adm_id BIGINT,
    START TIMESTAMP WITH TIME ZONE,
    STOP TIMESTAMP WITH TIME ZONE,
    reason VARCHAR,
    los_days DOUBLE,
    readmit_30d BOOLEAN,
    GENDER VARCHAR,
    RACE VARCHAR,
    age DOUBLE
)

Rules:
1. Generate DuckDB-compatible SQL.
2. Use only the provided table and columns.
3. Generate SELECT queries only.
4. Do not invent columns.
5. Return only SQL, without explanations.
6. Never generate INSERT, UPDATE, DELETE, or DROP.
"""

def clean_sql(response_text: str) -> str:
    """Remove Markdown code fences from LLM-generated SQL."""

    sql = response_text.strip()

    # Remove opening ```sql or ```duckdb
    sql = re.sub(
        r"^```(?:sql|duckdb)?\s*",
        "",
        sql,
        flags=re.IGNORECASE,
    )

    # Remove closing ```
    sql = re.sub(r"\s*```$", "", sql)

    return sql.strip()


def generate_sql(question: str) -> str:
    """Generate SQL using the local Qwen model."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=256,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    sql = response.choices[0].message.content

    return clean_sql(sql) if sql else ""


if __name__ == "__main__":
    question = "How many patients were readmitted within 30 days?"

    print("Question:", question)
    print("\nGenerated SQL:")
    print(generate_sql(question))
