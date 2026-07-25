"""
Day 2 — pytest unit tests for src/utils.py.

Covers happy-path behaviour plus edge cases for each utility function.
Run with: pytest tests/test_utils.py -v
"""

import pytest

from src.utils import (
    calculate_average,
    classify_grade,
    count_passing_students,
    categorize_study_hours,
    study_hours_summary,
)


def test_calculate_average_happy_path():
    assert calculate_average([70, 80, 90]) == 80.0


def test_calculate_average_empty_raises():
    with pytest.raises(ValueError):
        calculate_average([])


@pytest.mark.parametrize(
    "score,expected",
    [
        (95, "A"),
        (90, "A"),  # lower boundary of A
        (89, "B"),  # just below A
        (60, "C"),
        (40, "D"),
        (0, "F"),
    ],
)
def test_classify_grade_boundaries(score, expected):
    assert classify_grade(score) == expected


def test_classify_grade_out_of_range_raises():
    with pytest.raises(ValueError):
        classify_grade(150)


def test_count_passing_students():
    scores = [35, 40, 55, 90, 20]
    assert count_passing_students(scores) == 3  # 40, 55, 90 pass at default mark


def test_count_passing_students_custom_passing_mark():
    scores = [35, 40, 55, 90, 20]
    assert count_passing_students(scores, passing_mark=50) == 2  # 55, 90


@pytest.mark.parametrize(
    "hours,expected",
    [
        (1, "Low"),
        (1.9, "Low"),
        (2, "Medium"),
        (4, "Medium"),
        (4.1, "High"),
    ],
)
def test_categorize_study_hours_boundaries(hours, expected):
    assert categorize_study_hours(hours) == expected


def test_categorize_study_hours_negative_raises():
    with pytest.raises(ValueError):
        categorize_study_hours(-1)


def test_study_hours_summary():
    result = study_hours_summary([1.5, 3, 4.5, 2, 5])
    assert result == {"min": 1.5, "max": 5, "average": 3.2}


def test_study_hours_summary_empty_raises():
    with pytest.raises(ValueError):
        study_hours_summary([])
