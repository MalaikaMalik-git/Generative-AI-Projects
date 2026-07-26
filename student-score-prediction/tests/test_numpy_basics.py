"""
Day 3 — pytest unit tests for src/numpy_basics.py.

Covers happy-path behaviour plus edge cases for each NumPy utility.
Run with: pytest tests/test_numpy_basics.py -v
"""

import numpy as np
import pytest

from src.numpy_basics import (
    make_score_array,
    top_n_scores,
    scores_above_threshold,
    normalize_scores,
    study_score_correlation,
    weighted_final_scores,
)


def test_make_score_array_happy_path():
    result = make_score_array([70, 80, 90])
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, np.array([70.0, 80.0, 90.0]))


def test_make_score_array_empty_raises():
    with pytest.raises(ValueError):
        make_score_array([])


def test_top_n_scores_returns_descending():
    scores = make_score_array([50, 90, 70, 60, 100])
    result = top_n_scores(scores, 3)
    np.testing.assert_array_equal(result, np.array([100.0, 90.0, 70.0]))


@pytest.mark.parametrize("n", [0, 6])
def test_top_n_scores_invalid_n_raises(n):
    scores = make_score_array([50, 90, 70, 60, 100])
    with pytest.raises(ValueError):
        top_n_scores(scores, n)


def test_scores_above_threshold_filters_correctly():
    scores = make_score_array([45, 60, 75, 90])
    result = scores_above_threshold(scores, 60)
    np.testing.assert_array_equal(result, np.array([60.0, 75.0, 90.0]))


def test_scores_above_threshold_none_match():
    scores = make_score_array([10, 20, 30])
    result = scores_above_threshold(scores, 99)
    assert result.size == 0


def test_normalize_scores_min_max_bounds():
    scores = make_score_array([50, 75, 100])
    result = normalize_scores(scores)
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.5)


def test_normalize_scores_identical_raises():
    scores = make_score_array([80, 80, 80])
    with pytest.raises(ValueError):
        normalize_scores(scores)


def test_study_score_correlation_perfect_positive():
    hours = make_score_array([1, 2, 3, 4, 5])
    scores = make_score_array([50, 60, 70, 80, 90])
    result = study_score_correlation(hours, scores)
    assert result == pytest.approx(1.0)


def test_study_score_correlation_shape_mismatch_raises():
    hours = make_score_array([1, 2, 3])
    scores = make_score_array([50, 60])
    with pytest.raises(ValueError):
        study_score_correlation(hours, scores)


def test_weighted_final_scores_computes_correctly():
    components = np.array([[80.0, 70.0, 90.0], [60.0, 65.0, 55.0]])
    weights = np.array([0.3, 0.3, 0.4])
    result = weighted_final_scores(components, weights)

    expected = np.array(
        [80 * 0.3 + 70 * 0.3 + 90 * 0.4, 60 * 0.3 + 65 * 0.3 + 55 * 0.4]
    )
    np.testing.assert_allclose(result, expected)


def test_weighted_final_scores_bad_weights_raises():
    components = np.array([[80.0, 70.0, 90.0]])
    bad_weights = np.array([0.5, 0.5, 0.5])  # sums to 1.5
    with pytest.raises(ValueError):
        weighted_final_scores(components, bad_weights)
