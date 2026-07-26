"""
Day 3 — NumPy Fundamentals for the Student Score Prediction System.

Covers array creation, indexing/slicing, and vectorized mathematical
operations — the required Day 3 task — applied directly to the project
domain: arrays of study hours and exam scores.

Implementation added in Session 1.
"""

from __future__ import annotations

import numpy as np


def make_score_array(scores: list[float]) -> np.ndarray:
    """Convert a plain Python list of scores into a NumPy array.

    Args:
        scores: A list of numeric exam scores.

    Returns:
        A 1-D NumPy array of dtype float64.

    Raises:
        ValueError: If `scores` is empty.
    """
    if not scores:
        raise ValueError("scores must not be empty")

    return np.array(scores, dtype=np.float64)


def top_n_scores(scores: np.ndarray, n: int) -> np.ndarray:
    """Return the top `n` scores in descending order using NumPy indexing.

    Args:
        scores: A 1-D array of exam scores.
        n: How many top scores to return.

    Returns:
        A 1-D array of the `n` highest scores, sorted descending.

    Raises:
        ValueError: If `n` is not between 1 and len(scores).
    """
    if n < 1 or n > scores.size:
        raise ValueError("n must be between 1 and the number of scores")

    # np.argsort gives ascending indices; reverse with slicing, then slice top n.
    sorted_desc = np.sort(scores)[::-1]
    return sorted_desc[:n]


def scores_above_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean-mask indexing: select every score at or above a threshold.

    Args:
        scores: A 1-D array of exam scores.
        threshold: The cutoff value (inclusive).

    Returns:
        A 1-D array containing only scores >= threshold.
    """
    return scores[scores >= threshold]


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores into the [0, 1] range using vectorized ops.

    Args:
        scores: A 1-D array of exam scores.

    Returns:
        A 1-D array of the same shape, scaled to [0, 1].

    Raises:
        ValueError: If all scores are identical (division by zero range).
    """
    lowest = np.min(scores)
    highest = np.max(scores)

    if highest == lowest:
        raise ValueError("cannot normalize scores that are all identical")

    return (scores - lowest) / (highest - lowest)


def study_score_correlation(study_hours: np.ndarray, scores: np.ndarray) -> float:
    """Compute the Pearson correlation coefficient between study hours and scores.

    Demonstrates array-based statistical calculation with NumPy
    (element-wise operations, mean, and standard deviation) rather than
    a loop.

    Args:
        study_hours: A 1-D array of study-hour values.
        scores: A 1-D array of exam scores, same length as `study_hours`.

    Returns:
        The Pearson correlation coefficient as a float, in [-1, 1].

    Raises:
        ValueError: If the two arrays don't have matching, non-zero length.
    """
    if study_hours.shape != scores.shape:
        raise ValueError("study_hours and scores must have the same shape")
    if study_hours.size == 0:
        raise ValueError("arrays must not be empty")

    correlation_matrix = np.corrcoef(study_hours, scores)
    return float(correlation_matrix[0, 1])


def weighted_final_scores(
    component_scores: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Compute weighted final scores for a batch of students via matrix math.

    Each row of `component_scores` is one student's scores across several
    components (e.g. quiz, assignment, exam); `weights` gives the weight
    of each component. Uses NumPy's dot product instead of nested loops.

    Args:
        component_scores: A 2-D array of shape (num_students, num_components).
        weights: A 1-D array of shape (num_components,) that sums to 1.0.

    Returns:
        A 1-D array of shape (num_students,) with each student's weighted
        final score.

    Raises:
        ValueError: If the number of components doesn't match the number
            of weights, or if the weights don't sum to ~1.0.
    """
    if component_scores.shape[1] != weights.shape[0]:
        raise ValueError("number of weight values must match number of components")
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("weights must sum to 1.0")

    return component_scores.dot(weights)


if __name__ == "__main__":
    # Quick manual smoke test
    demo_scores = make_score_array([72.5, 88, 91, 35, 60, 79])
    demo_hours = make_score_array([1.5, 4.0, 4.5, 1.0, 2.5, 3.5])

    print("Score array:", demo_scores)
    print("Top 3 scores:", top_n_scores(demo_scores, 3))
    print("Scores >= 70:", scores_above_threshold(demo_scores, 70))
    print("Normalized scores:", normalize_scores(demo_scores))
    print("Study/score correlation:", study_score_correlation(demo_hours, demo_scores))

    components = np.array([[80, 70, 90], [60, 65, 55]])
    component_weights = np.array([0.3, 0.3, 0.4])
    print("Weighted finals:", weighted_final_scores(components, component_weights))
