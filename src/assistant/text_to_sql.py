
from .llm_client import generate_sql
from .guardrails import run_sql


def ask(
    question: str,
    use_semantic_layer: bool = True,
) -> dict:
    """
    Convert a hospital question into SQL,
    validate it, execute it, and return results.
    """

    response = {
        "sql": None,
        "rows": None,
        "ok": False,
        "error": None,
    }

    if not isinstance(question, str) or not question.strip():
        response["error"] = "Please enter a question."
        return response

    try:
        # Step 1: Ask Qwen to generate SQL.
        sql = generate_sql(
            question,
            use_semantic_layer=use_semantic_layer,
        )
        response["sql"] = sql

        # Step 2: Validate and execute SQL.
        # run_sql() includes guardrails and a timeout.
        result_df = run_sql(sql)

        # Step 3: Return the verified result.
        response["rows"] = result_df
        response["ok"] = True

    except Exception as exc:
        response["error"] = str(exc)

    return response


if __name__ == "__main__":
    question = (
        "How many inpatient stays were followed "
        "by a 30-day readmission?"
    )

    result = ask(question)

    print("\nQuestion:", question)
    print("\nGenerated SQL:", result["sql"])
    print("\nGuardrails passed:", result["ok"])

    if result["ok"]:
        print("\nDatabase result:")
        print(result["rows"].to_string(index=False))
    else:
        print("\nError:", result["error"])
