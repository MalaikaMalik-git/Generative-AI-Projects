"""
Day 3 — Training demo: from-scratch Linear Regression + convergence plot.

Advanced-layer task: train the LinearRegressionScratch model on a
synthetic study-hours-to-exam-score dataset and plot the cost-function
convergence curve to verify gradient descent is learning correctly.

Implementation added in Session 3.

Run with: python src/train_demo.py
Output: data/processed/day3_cost_convergence.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.linear_regression import LinearRegressionScratch


def generate_synthetic_data(
    n_samples: int = 50, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic study-hours vs. exam-score dataset.

    Scores follow a linear trend (roughly 9 points per study hour, plus
    a base score) with a bit of Gaussian noise added, so the model has
    something realistic — but still clearly linear — to learn from.

    Args:
        n_samples: How many synthetic students to generate.
        seed: Random seed for reproducibility.

    Returns:
        A tuple (study_hours, scores), each a 1-D NumPy array of length
        `n_samples`.
    """
    rng = np.random.default_rng(seed)

    study_hours = rng.uniform(0.5, 8.0, size=n_samples)
    noise = rng.normal(loc=0.0, scale=4.0, size=n_samples)
    scores = 9.0 * study_hours + 25.0 + noise

    # Clip to a realistic 0-100 score range.
    scores = np.clip(scores, 0, 100)

    return study_hours, scores


def plot_cost_convergence(cost_history: list[float], output_path: Path) -> None:
    """Plot the cost-function value at each gradient descent iteration.

    Args:
        cost_history: MSE cost recorded at every training iteration.
        output_path: File path to save the resulting PNG plot to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(cost_history)), cost_history, color="#2563eb", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Cost (MSE)")
    plt.title("Day 3 — Linear Regression Cost Convergence")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    """Train the from-scratch model on synthetic data and plot convergence."""
    study_hours, scores = generate_synthetic_data()

    model = LinearRegressionScratch(learning_rate=0.02, n_iterations=1000)
    model.fit(study_hours, scores)

    print(f"Learned weight (slope): {model.weight:.4f}")
    print(f"Learned bias (intercept): {model.bias:.4f}")
    print(f"Initial cost: {model.cost_history[0]:.4f}")
    print(f"Final cost:   {model.cost_history[-1]:.4f}")

    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "processed" / "day3_cost_convergence.png"
    plot_cost_convergence(model.cost_history, output_path)

    print(f"Convergence plot saved to: {output_path}")


if __name__ == "__main__":
    main()
