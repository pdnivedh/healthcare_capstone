
import pytest

from src.assistant.guardrails import validate_sql, run_sql


def test_valid_select():
    sql = "SELECT COUNT(*) FROM syn_stays"
    result = run_sql(sql)
    assert result.iloc[0, 0] == 2023


def test_valid_with():
    sql = """
    WITH admissions AS (
        SELECT PATIENT FROM syn_stays
    )
    SELECT COUNT(*) FROM admissions
    """
    result = run_sql(sql)
    assert result.iloc[0, 0] == 2023


def test_valid_union():
    sql = """
    SELECT GENDER FROM syn_stays
    UNION
    SELECT GENDER FROM syn_stays
    """
    result = run_sql(sql)
    assert set(result.iloc[:, 0]) == {"M", "F"}


@pytest.mark.parametrize("sql", [
    "DROP TABLE syn_stays",
    "DELETE FROM syn_stays",
    "UPDATE syn_stays SET age = 50",
    "SELECT * FROM syn_stays; DROP TABLE syn_stays",
    "SELECT * FROM uci_diabetes",
    "SELECT * FROM read_csv('private.csv')",
    "SELECT read_text('private.txt') FROM syn_stays",
])
def test_unsafe_queries_blocked(sql):
    with pytest.raises(ValueError):
        validate_sql(sql)


def test_row_limit():
    sql = validate_sql("SELECT * FROM syn_stays")
    assert "LIMIT 1000" in sql.upper()
    assert len(run_sql("SELECT * FROM syn_stays")) <= 1000

def test_case_expression_allowed():
    sql = """
    SELECT GENDER,
           SUM(CASE WHEN readmit_30d = TRUE THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS rate
    FROM syn_stays GROUP BY GENDER
    """
    validate_sql(sql)                      # must not raise
    result = run_sql(sql)
    assert len(result) == 2
    assert all(0 <= float(v) <= 1 for v in result.iloc[:, 1])


def test_case_cannot_hide_unsafe_function():
    sql = "SELECT SUM(CASE WHEN readmit_30d THEN read_text('private.txt') ELSE 0 END) FROM syn_stays"
    with pytest.raises(ValueError):
        validate_sql(sql)
