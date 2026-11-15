
import pytest

from src.assistant.guardrails import validate_sql, run_sql


VALID_JOIN = """
SELECT COUNT(*) AS matched_rows
FROM syn_stays s
INNER JOIN syn_risk_scores r
    ON s.PATIENT = r.PATIENT
   AND s.START = r.START
   AND s.STOP = r.STOP
"""


def test_approved_join_validation():
    validated = validate_sql(VALID_JOIN)
    assert "LIMIT 1000" in validated.upper()


def test_approved_join_execution():
    result = run_sql(VALID_JOIN)
    assert int(result.iloc[0]["matched_rows"]) == 2023


def test_patient_only_join_blocked():
    sql = """
    SELECT COUNT(*)
    FROM syn_stays s
    JOIN syn_risk_scores r
        ON s.PATIENT = r.PATIENT
    """

    with pytest.raises(ValueError):
        validate_sql(sql)


def test_left_join_blocked():
    sql = """
    SELECT COUNT(*)
    FROM syn_stays s
    LEFT JOIN syn_risk_scores r
        ON s.PATIENT = r.PATIENT
       AND s.START = r.START
       AND s.STOP = r.STOP
    """

    with pytest.raises(ValueError):
        validate_sql(sql)


def test_cross_join_blocked():
    sql = """
    SELECT COUNT(*)
    FROM syn_stays s
    CROSS JOIN syn_risk_scores r
    """

    with pytest.raises(ValueError):
        validate_sql(sql)


def test_extra_join_condition_blocked():
    sql = """
    SELECT COUNT(*)
    FROM syn_stays s
    JOIN syn_risk_scores r
        ON s.PATIENT = r.PATIENT
       AND s.START = r.START
       AND s.STOP = r.STOP
       AND s.age = r.age
    """

    with pytest.raises(ValueError):
        validate_sql(sql)
