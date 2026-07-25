"""
Day 2 — Python Basics utilities for the Student Score Prediction System.

Small, tested helper functions that cover core Python fundamentals
(variables, data types, operators, loops, functions) while staying
directly relevant to the project domain: student study hours and
exam scores.

Implementation added in Session 2.
"""

from __future__ import annotations


def calculate_average(scores: list[float]) -> float:
    """Return the arithmetic mean of a list of exam scores.

    Args:
        scores: A non-empty list of numeric scores (e.g. [72.5, 88, 91]).

    Returns:
        The average score as a float.

    Raises:
        ValueError: If `scores` is empty.
    """
    if not scores:
        raise ValueError("scores must not be empty")

    total = 0.0
    for score in scores:
        total += score

    return total / len(scores)


def classify_grade(score: float) -> str:
    """Map a numeric exam score to a letter grade.

    Grading scale:
        90-100 -> "A"
        75-89  -> "B"
        60-74  -> "C"
        40-59  -> "D"
        0-39   -> "F"

    Args:
        score: A numeric score between 0 and 100.

    Returns:
        The letter grade as a string.

    Raises:
        ValueError: If `score` is outside the 0-100 range.
    """
    if score < 0 or score > 100:
        raise ValueError("score must be between 0 and 100")

    if score >= 90:
        return "A"
    elif score >= 75:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "F"


def count_passing_students(scores: list[float], passing_mark: float = 40.0) -> int:
    """Count how many scores meet or exceed the passing mark.

    Args:
        scores: A list of numeric exam scores.
        passing_mark: The minimum score considered a pass (default 40.0).

    Returns:
        The number of scores >= passing_mark.
    """
    passing_count = 0
    for score in scores:
        if score >= passing_mark:
            passing_count += 1

    return passing_count


def categorize_study_hours(hours: float) -> str:
    """Bucket a daily study-hours value into a category.

    Categories:
        < 2 hours   -> "Low"
        2-4 hours   -> "Medium"
        > 4 hours   -> "High"

    Args:
        hours: Number of study hours (must be >= 0).

    Returns:
        One of "Low", "Medium", "High".

    Raises:
        ValueError: If `hours` is negative.
    """
    if hours < 0:
        raise ValueError("hours must not be negative")

    if hours < 2:
        return "Low"
    elif hours <= 4:
        return "Medium"
    else:
        return "High"


def study_hours_summary(hours_list: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of study-hour values.

    Args:
        hours_list: A non-empty list of daily study-hour values.

    Returns:
        A dict with keys "min", "max", and "average" mapping to floats.

    Raises:
        ValueError: If `hours_list` is empty.
    """
    if not hours_list:
        raise ValueError("hours_list must not be empty")

    lowest = hours_list[0]
    highest = hours_list[0]
    total = 0.0

    for hours in hours_list:
        if hours < lowest:
            lowest = hours
        if hours > highest:
            highest = hours
        total += hours

    return {
        "min": lowest,
        "max": highest,
        "average": total / len(hours_list),
    }


if __name__ == "__main__":
    # Quick manual smoke test — will run once functions are implemented
    # in Session 2. Left here so `python src/utils.py` is runnable end to end.
    demo_scores = [72.5, 88, 91, 35, 60]
    demo_hours = [1.5, 3, 4.5, 2, 5]

    print("Average score:", calculate_average(demo_scores))
    print("Grades:", [classify_grade(s) for s in demo_scores])
    print("Passing students:", count_passing_students(demo_scores))
    print("Study categories:", [categorize_study_hours(h) for h in demo_hours])
    print("Study hours summary:", study_hours_summary(demo_hours))
