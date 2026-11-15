
import re
from pathlib import Path

from openai import OpenAI

from .schema import get_schema_text


MODEL_NAME = "qwen2.5-7b-instruct"

SEMANTIC_PATH = (
    Path(__file__).resolve().parent / "semantic_layer.txt"
)

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    timeout=120.0,
    max_retries=0,
)


def build_prompt(use_semantic_layer=True):
    """
    Build the Text-to-SQL system prompt.

    Both evaluation conditions receive the same schema
    and SQL safety requirements.

    Only the semantic-layer condition receives
    additional healthcare definitions.
    """

    prompt = """
You are a data analyst specializing in DuckDB SQL.

Convert the user's natural-language question into
one valid DuckDB SELECT query.

SQL REQUIREMENTS:
1. Return only SQL, without explanations or markdown.
2. Use only the supplied tables and columns.
3. Never invent table or column names.
4. Generate only read-only SELECT queries.
5. Do not access external files or databases.
6. Use DuckDB-compatible SQL syntax.
7. Use only approved analytical SQL functions.
8. Do not generate destructive SQL statements.

JOIN REQUIREMENTS:
- The available tables are syn_stays and syn_risk_scores.
- INNER JOIN is permitted between these two tables.
- When joining, match all three columns:
  PATIENT, START, and STOP.
- Never join using PATIENT alone.
- Do not use LEFT, RIGHT, FULL, or CROSS JOIN.
- Do not generate nested JOINs.

DATABASE SCHEMA:
"""

    prompt += "\n" + get_schema_text()

    if use_semantic_layer:
        semantic_text = SEMANTIC_PATH.read_text(
            encoding="utf-8"
        )

        prompt += "\n\nHEALTHCARE SEMANTIC LAYER:\n"
        prompt += semantic_text

    return prompt


def clean_sql(response_text: str) -> str:
    """Remove markdown formatting from generated SQL."""

    sql = response_text.strip()

    sql = re.sub(
        r"^```(?:sql|duckdb)?\s*",
        "",
        sql,
        flags=re.IGNORECASE,
    )

    sql = re.sub(
        r"\s*```$",
        "",
        sql,
    )

    return sql.strip()


def generate_sql(
    question: str,
    use_semantic_layer: bool = True,
) -> str:
    """Generate DuckDB SQL using Qwen through LM Studio."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=384,
        messages=[
            {
                "role": "system",
                "content": build_prompt(
                    use_semantic_layer=use_semantic_layer
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            "LLM returned an empty response."
        )

    return clean_sql(content)
