"""
Day 3 — pytest unit tests for src/linear_regression.py.

Verifies the from-scratch gradient descent implementation converges
correctly and behaves sensibly on edge cases.
Run with: pytest tests/test_linear_regression.py -v
"""

import numpy as np
import pytest

from src.linear_regression import LinearRegressionScratch


def test_init_default_hyperparameters():
    model = LinearRegressionScratch()
    assert model.learning_rate == 0.01
    assert model.n_iterations == 1000
    assert model.weight == 0.0
    assert model.bias == 0.0


@pytest.mark.parametrize(
    "learning_rate,n_iterations", [(0, 100), (-0.1, 100), (0.01, 0), (0.01, -5)]
)
def test_init_invalid_hyperparameters_raise(learning_rate, n_iterations):
    with pytest.raises(ValueError):
        LinearRegressionScratch(learning_rate=learning_rate, n_iterations=n_iterations)


def test_fit_mismatched_shapes_raise():
    model = LinearRegressionScratch()
    with pytest.raises(ValueError):
        model.fit(np.array([1.0, 2.0]), np.array([1.0]))


def test_fit_empty_data_raises():
    model = LinearRegressionScratch()
    with pytest.raises(ValueError):
        model.fit(np.array([]), np.array([]))


def test_cost_decreases_over_training():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # perfectly linear: y = 2x

    model = LinearRegressionScratch(learning_rate=0.05, n_iterations=200)
    model.fit(x, y)

    assert len(model.cost_history) == 200
    # Cost should trend downward: last value much smaller than first.
    assert model.cost_history[-1] < model.cost_history[0]
    # Cost should be (near) monotonically non-increasing for this simple,
    # well-conditioned case with a small learning rate.
    assert all(
        model.cost_history[i + 1] <= model.cost_history[i] + 1e-6
        for i in range(len(model.cost_history) - 1)
    )


def test_converges_to_known_line():
    # y = 3x + 1, noise-free, should converge close to w=3, b=1.
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = 3 * x + 1

    model = LinearRegressionScratch(learning_rate=0.05, n_iterations=3000)
    model.fit(x, y)

    assert model.weight == pytest.approx(3.0, abs=0.05)
    assert model.bias == pytest.approx(1.0, abs=0.05)
    assert model.cost_history[-1] < 0.01


def test_predict_uses_learned_parameters():
    model = LinearRegressionScratch()
    model.weight = 2.0
    model.bias = 1.0

    result = model.predict(np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(result, np.array([1.0, 3.0, 5.0]))


def test_fit_returns_self_for_chaining():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    model = LinearRegressionScratch(n_iterations=10)

    result = model.fit(x, y)
    assert result is model
