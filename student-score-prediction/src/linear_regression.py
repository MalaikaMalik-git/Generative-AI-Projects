"""
Day 3 — Linear Regression from scratch, built only with NumPy.

Advanced-layer task: implement gradient descent, the cost function, and
weight updates by hand, before ever touching scikit-learn's
LinearRegression as a black box. Applied to the project's core problem —
predicting an exam score from study hours.

Implementation added in Session 2.
"""

from __future__ import annotations

import numpy as np


class LinearRegressionScratch:
    """A single-feature linear regression model trained via batch gradient descent.

    Fits the line y = w * x + b by minimizing Mean Squared Error (MSE)
    using gradient descent, entirely with NumPy array operations.

    Attributes:
        learning_rate: Step size used for each gradient descent update.
        n_iterations: Number of gradient descent iterations to run.
        weight: The learned slope (w), set after calling `fit`.
        bias: The learned intercept (b), set after calling `fit`.
        cost_history: MSE cost recorded at every iteration of `fit`,
            used to plot the convergence curve.
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000) -> None:
        """Initialize the model with training hyperparameters.

        Args:
            learning_rate: Step size for gradient descent (must be > 0).
            n_iterations: Number of training iterations (must be > 0).

        Raises:
            ValueError: If `learning_rate` or `n_iterations` is not positive.
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weight: float = 0.0
        self.bias: float = 0.0
        self.cost_history: list[float] = []

    def _compute_cost(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Mean Squared Error for the current weight and bias.

        Args:
            x: 1-D array of feature values (study hours).
            y: 1-D array of true target values (exam scores).

        Returns:
            The MSE cost as a float.
        """
        n = x.shape[0]
        predictions = self.weight * x + self.bias
        errors = predictions - y
        return float(np.sum(errors**2) / n)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearRegressionScratch":
        """Train the model on data using batch gradient descent.

        At each iteration:
            1. Compute predictions: y_hat = w*x + b
            2. Compute the MSE cost and record it in `cost_history`
            3. Compute gradients of the cost w.r.t. w and b
            4. Update w and b by stepping against the gradient

        Args:
            x: 1-D array of feature values (study hours).
            y: 1-D array of true target values (exam scores), same
                length as `x`.

        Returns:
            self, so `fit` can be chained, e.g.
            `model = LinearRegressionScratch().fit(x, y)`.

        Raises:
            ValueError: If `x` and `y` don't have matching, non-zero length.
        """
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        if x.size == 0:
            raise ValueError("x and y must not be empty")

        n = x.shape[0]
        self.weight = 0.0
        self.bias = 0.0
        self.cost_history = []

        for _ in range(self.n_iterations):
            predictions = self.weight * x + self.bias
            errors = predictions - y

            # Record cost before this iteration's update, for the convergence curve.
            self.cost_history.append(float(np.sum(errors**2) / n))

            # Gradients of MSE w.r.t. weight and bias.
            weight_gradient = (2 / n) * np.sum(errors * x)
            bias_gradient = (2 / n) * np.sum(errors)

            # Gradient descent update step.
            self.weight -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict target values for new feature data using the learned line.

        Args:
            x: 1-D array of feature values (study hours).

        Returns:
            A 1-D array of predicted values.
        """
        return self.weight * x + self.bias


if __name__ == "__main__":
    # Quick manual smoke test on a simple, near-linear dataset.
    demo_hours = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    demo_scores = np.array([42.0, 51.0, 62.0, 71.0, 80.0, 90.0])

    model = LinearRegressionScratch(learning_rate=0.03, n_iterations=2000)
    model.fit(demo_hours, demo_scores)

    print(f"Learned weight (slope): {model.weight:.4f}")
    print(f"Learned bias (intercept): {model.bias:.4f}")
    print(f"Final cost: {model.cost_history[-1]:.4f}")
    print("Predictions for [2.5, 5.5]:", model.predict(np.array([2.5, 5.5])))
