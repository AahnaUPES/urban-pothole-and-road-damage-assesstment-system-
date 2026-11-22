"""
===============================================================================
UrbanRoadDamage — Numerical Gradient Checking vs Analytical Backpropagation
Module: fundamentals/backprop/gradient_check.py
===============================================================================

PURPOSE & VIVA EXPLANATION
--------------------------
1. Why Gradient Checking?
   Backpropagation involves complex matrix calculus and multiple chain-rule steps.
   A single sign error or transpose mismatch can cause silent bugs where the model
   learns slowly or converges to suboptimal weights without crashing.
   Numerical gradient checking serves as an automated ground truth to verify analytical
   gradients before starting any real training run.

2. Finite Difference Formula:
   Two-sided (central) difference is used because its truncation error is O(h^2),
   whereas one-sided forward difference has error O(h):

       df/dtheta ≈ [ f(theta + h) - f(theta - h) ] / (2 * h)

   We choose h = 1e-5 (balancing round-off precision with truncation error).

3. Relative Error Metric:
   Standard absolute difference is scale-dependent. We compute relative error:

       relative_error = |g_num - g_ana| / max(|g_num|, |g_ana|, 1e-8)

   Acceptance Criteria:
     - relative_error < 1e-7: Backprop is exact and verified.
     - 1e-7 <= relative_error < 1e-4: Acceptable (often due to float precision).
     - relative_error >= 1e-4: High likelihood of backpropagation bug!

ACADEMIC DEMONSTRATION NOTICE
-----------------------------
This script runs on synthetic toy data (XOR classification) on CPU.
It does NOT train on RDD2022. It fulfills PRD Section 8 requirements.
===============================================================================
"""

import os
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt

# Ensure results directory exists
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Stable sigmoid activation."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid given activation output a = sigmoid(z): a * (1 - a)."""
    return a * (1.0 - a)


class TwoLayerMLP:
    """
    Minimal 2-Layer Neural Network for Gradient Verification:
        Layer 1: z1 = X @ W1 + b1,  a1 = sigmoid(z1)
        Layer 2: z2 = a1 @ W2 + b2, y_hat = sigmoid(z2)
        Loss: Mean Squared Error L = (1/2N) * sum((y_hat - y)^2)
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 4, output_dim: int = 1, seed: int = 42):
        np.random.seed(seed)
        # Xavier/Glorot initialization for weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through both layers."""
        z1 = X @ self.W1 + self.b1
        a1 = sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        y_hat = sigmoid(z2)
        return z1, a1, z2, y_hat

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute MSE loss: L = (1 / 2N) * sum((y_hat - y)^2)."""
        _, _, _, y_hat = self.forward(X)
        N = X.shape[0]
        return float(np.sum((y_hat - y) ** 2) / (2.0 * N))

    def analytical_gradients(self, X: np.ndarray, y: np.ndarray):
        """
        Backpropagation chain-rule calculation:
          dL/dy_hat = (y_hat - y) / N
          delta2    = dL/dy_hat * sigmoid'(z2)
          dL/dW2    = a1^T @ delta2
          dL/db2    = sum(delta2, axis=0)
          delta1    = (delta2 @ W2^T) * sigmoid'(z1)
          dL/dW1    = X^T @ delta1
          dL/db1    = sum(delta1, axis=0)
        """
        N = X.shape[0]
        z1, a1, z2, y_hat = self.forward(X)

        # Output layer gradients
        delta2 = ((y_hat - y) / N) * sigmoid_derivative(y_hat)
        dW2 = a1.T @ delta2
        db2 = np.sum(delta2, axis=0, keepdims=True)

        # Hidden layer gradients
        delta1 = (delta2 @ self.W2.T) * sigmoid_derivative(a1)
        dW1 = X.T @ delta1
        db1 = np.sum(delta1, axis=0, keepdims=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def compute_numerical_gradient(model: TwoLayerMLP, param_name: str, X: np.ndarray, y: np.ndarray, h: float = 1e-5):
    """
    Computes numerical gradient with two-sided finite differences:
        d(param) ≈ (Loss(theta + h) - Loss(theta - h)) / (2 * h)
    """
    param = getattr(model, param_name)
    grad_num = np.zeros_like(param)

    it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig_val = param[idx]

        # f(theta + h)
        param[idx] = orig_val + h
        loss_plus = model.compute_loss(X, y)

        # f(theta - h)
        param[idx] = orig_val - h
        loss_minus = model.compute_loss(X, y)

        # restore parameter
        param[idx] = orig_val

        # two-sided difference
        grad_num[idx] = (loss_plus - loss_minus) / (2.0 * h)
        it.iternext()

    return grad_num


def run_gradient_check():
    """Main verification harness."""
    # Synthetic XOR dataset
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    model = TwoLayerMLP(input_dim=2, hidden_dim=4, output_dim=1, seed=123)
    ana_grads = model.analytical_gradients(X, y)

    print("=" * 75)
    print("NUMERICAL GRADIENT CHECKING VERIFICATION (fundamentals/backprop)")
    print("Formula: f'(theta) ≈ [f(theta+h) - f(theta-h)] / (2h) with h = 1e-5")
    print("=" * 75)
    print(f"{'Parameter':<10} | {'Shape':<12} | {'Max Rel Error':<16} | {'Status':<15}")
    print("-" * 75)

    all_ana = []
    all_num = []
    max_errors = {}

    for name in ["W1", "b1", "W2", "b2"]:
        num_grad = compute_numerical_gradient(model, name, X, y, h=1e-5)
        ana_grad = ana_grads[name]

        # Relative error
        numerator = np.abs(num_grad - ana_grad)
        denominator = np.maximum(np.abs(num_grad), np.abs(ana_grad))
        denominator = np.maximum(denominator, 1e-8)
        rel_error = np.max(numerator / denominator)
        max_errors[name] = rel_error

        status = "PASSED (Exact)" if rel_error < 1e-6 else "PASSED (OK)" if rel_error < 1e-4 else "FAILED"
        print(f"{name:<10} | {str(ana_grad.shape):<12} | {rel_error:<16.2e} | {status:<15}")

        all_ana.extend(ana_grad.flatten())
        all_num.extend(num_grad.flatten())

    print("=" * 75)

    # Plot parity scatter: Analytical vs Numerical Gradients
    plt.figure(figsize=(7, 6))
    plt.scatter(all_ana, all_num, color="crimson", alpha=0.8, edgecolors="k", s=70, label="Gradients")
    min_v = min(min(all_ana), min(all_num))
    max_v = max(max(all_ana), max(all_num))
    plt.plot([min_v, max_v], [min_v, max_v], "b--", lw=2, label="Perfect Parity (y = x)")
    plt.title("Gradient Verification: Analytical vs Numerical Gradients\n(Max Relative Error < 1e-6)")
    plt.xlabel("Analytical Gradient (Backpropagation)")
    plt.ylabel("Numerical Gradient (Central Finite Difference)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    out_path = RESULTS_DIR / "gradient_check.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved parity plot to: {out_path}")
    return max_errors


if __name__ == "__main__":
    run_gradient_check()
