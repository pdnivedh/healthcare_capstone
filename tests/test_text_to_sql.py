
import subprocess
import pytest

from src.assistant import text_to_sql
from src.assistant import guardrails
from src.assistant.llm_client import clean_sql


def test_valid_question_without_llm(monkeypatch):
    """Test the pipeline without calling Qwen."""

    def fake_generate_sql(question, use_semantic_layer=True):
        return """
        SELECT COUNT(*) AS readmitted_stays
        FROM syn_stays
        WHERE readmit_30d = TRUE
        """

    monkeypatch.setattr(
        text_to_sql,
        "generate_sql",
        fake_generate_sql,
    )

    result = text_to_sql.ask(
        "How many inpatient stays had a readmission?"
    )

    assert result["ok"] is True
    assert result["error"] is None
    assert result["rows"].iloc[0, 0] == 255


def test_unsafe_llm_response(monkeypatch):
    """Check that dangerous SQL is rejected."""

    monkeypatch.setattr(
        text_to_sql,
        "generate_sql",
        lambda *args, **kwargs: "DROP TABLE syn_stays",
    )

    result = text_to_sql.ask("Delete everything")

    assert result["ok"] is False
    assert result["rows"] is None


def test_sensitive_table_blocked():
    """Block access to sensitive patient columns."""

    with pytest.raises(ValueError):
        guardrails.validate_sql(
            "SELECT SSN FROM syn_patients"
        )


def test_sql_code_fence_cleaning():
    raw = "```sql\nSELECT COUNT(*) FROM syn_stays;\n```"

    cleaned = clean_sql(raw)

    assert cleaned == "SELECT COUNT(*) FROM syn_stays;"


def test_empty_question():
    result = text_to_sql.ask("")

    assert result["ok"] is False
    assert result["rows"] is None


def test_query_timeout(monkeypatch):
    """Simulate an SQL execution timeout."""

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd="query_worker",
            timeout=12,
        )

    monkeypatch.setattr(
        guardrails.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(TimeoutError):
        guardrails.run_sql(
            "SELECT COUNT(*) FROM syn_stays"
        )
