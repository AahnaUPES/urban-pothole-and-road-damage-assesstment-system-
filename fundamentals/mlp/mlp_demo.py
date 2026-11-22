"""
===============================================================================
UrbanRoadDamage — Multilayer Perceptron (MLP) from Scratch (NumPy Only)
Module: fundamentals/mlp/mlp_demo.py
===============================================================================

PURPOSE & VIVA EXPLANATION
--------------------------
1. Why implement an MLP from scratch without PyTorch / TensorFlow?
   - To demonstrate thorough command over the core mathematics of feed-forward
     neural networks: forward activation, loss evaluation, backward sensitivity
     propagation via the chain rule, and parameter optimization.

2. Problem Formulation:
   - Non-linear 2D binary classification ("Two Moons" / circular boundary).
   - Single-layer Perceptrons fail on non-linearly separable data (Minsky & Papert 1969).
   - A Multilayer Perceptron with non-linear activation functions (ReLU / Tanh / Sigmoid)
     acts as a universal function approximator (Cybenko 1989; Hornik 1991).

3. Mathematical Formulation:
   - Architecture: Input (2) -> Hidden 1 (16, ReLU) -> Hidden 2 (8, ReLU) -> Output (1, Sigmoid)
   - Binary Cross-Entropy Loss:
       L = - (1/N) * sum( y * log(y_hat) + (1 - y) * log(1 - y_hat) )
   - Chain Rule Gradients:
       dL/dz_out = y_hat - y  (elegantly simplified cross-entropy + sigmoid derivative)
       dL/dW_out = a_h2.T @ dL/dz_out
       dL/db_out = sum(dL/dz_out)
       dL/da_h2  = dL/dz_out @ W_out.T
       dL/dz_h2  = dL/da_h2 * (z_h2 > 0)  [ReLU derivative]
       ... repeated recursively for each preceding layer.

ACADEMIC DEMONSTRATION NOTICE
-----------------------------
Demonstration on synthetic toy 2D data.
Plots decision boundary and learning curve to results/mlp_decision_boundary.png.
===============================================================================
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def relu(z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation: max(0, z)."""
    return np.maximum(0.0, z)


def relu_backward(dA: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if z > 0 else 0."""
    dZ = np.array(dA, copy=True)
    dZ[z <= 0] = 0.0
    return dZ


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Stable sigmoid activation function."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


class ScratchMLP:
    """
    3-layer Fully Connected Neural Network implemented entirely in NumPy.
    Input (2) -> Hidden 1 (16, ReLU) -> Hidden 2 (8, ReLU) -> Output (1, Sigmoid)
    """

    def __init__(self, layer_dims=(2, 16, 8, 1), seed=42):
        np.random.seed(seed)
        self.params = {}
        # He (Kaiming) initialization for ReLU layers
        for l in range(1, len(layer_dims)):
            scale = np.sqrt(2.0 / layer_dims[l - 1])
            self.params[f"W{l}"] = np.random.randn(layer_dims[l - 1], layer_dims[l]) * scale
            self.params[f"b{l}"] = np.zeros((1, layer_dims[l]))

        self.num_layers = len(layer_dims) - 1

    def forward(self, X: np.ndarray):
        """
        Forward propagation caching linear activations (z) and post-activations (a).
        """
        cache = {"a0": X}
        a = X

        # Hidden layers with ReLU
        for l in range(1, self.num_layers):
            z = a @ self.params[f"W{l}"] + self.params[f"b{l}"]
            a = relu(z)
            cache[f"z{l}"] = z
            cache[f"a{l}"] = a

        # Output layer with Sigmoid
        z_out = a @ self.params[f"W{self.num_layers}"] + self.params[f"b{self.num_layers}"]
        y_hat = sigmoid(z_out)
        cache[f"z{self.num_layers}"] = z_out
        cache[f"a{self.num_layers}"] = y_hat

        return y_hat, cache

    def compute_loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Binary cross-entropy loss with numerical clipping."""
        eps = 1e-12
        y_hat_clipped = np.clip(y_hat, eps, 1.0 - eps)
        N = y.shape[0]
        loss = -np.mean(y * np.log(y_hat_clipped) + (1.0 - y) * np.log(1.0 - y_hat_clipped))
        return float(loss)

    def backward(self, cache: dict, y: np.ndarray):
        """
        Analytical backpropagation using standard vectorised chain rule.
        """
        grads = {}
        N = y.shape[0]
        L = self.num_layers

        # Gradient at output layer (Cross-Entropy combined with Sigmoid simplifies directly):
        # dL/dz = y_hat - y
        dZ = (cache[f"a{L}"] - y) / N
        grads[f"W{L}"] = cache[f"a{L-1}"].T @ dZ
        grads[f"b{L}"] = np.sum(dZ, axis=0, keepdims=True)

        # Backprop through hidden layers
        for l in range(L - 1, 0, -1):
            dA = dZ @ self.params[f"W{l+1}"].T
            dZ = relu_backward(dA, cache[f"z{l}"])
            grads[f"W{l}"] = cache[f"a{l-1}"].T @ dZ
            grads[f"b{l}"] = np.sum(dZ, axis=0, keepdims=True)

        return grads

    def update_params(self, grads: dict, lr: float):
        """Standard Stochastic Gradient Descent weight update."""
        for l in range(1, self.num_layers + 1):
            self.params[f"W{l}"] -= lr * grads[f"W{l}"]
            self.params[f"b{l}"] -= lr * grads[f"b{l}"]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction from model probabilities."""
        y_hat, _ = self.forward(X)
        return (y_hat >= threshold).astype(int)


def generate_synthetic_data(n_samples: int = 400, seed: int = 42):
    """Generates concentric circles (non-linearly separable dataset)."""
    np.random.seed(seed)
    n = n_samples // 2
    # Inner circle (Class 0)
    r1 = np.random.uniform(0.0, 0.45, n)
    theta1 = np.random.uniform(0, 2 * np.pi, n)
    X1 = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)
    y1 = np.zeros((n, 1))

    # Outer ring (Class 1)
    r2 = np.random.uniform(0.65, 1.0, n)
    theta2 = np.random.uniform(0, 2 * np.pi, n)
    X2 = np.stack([r2 * np.cos(theta2), r2 * np.sin(theta2)], axis=1)
    y2 = np.ones((n, 1))

    X = np.vstack([X1, X2])
    y = np.vstack([y1, y2])
    # Shuffle
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def train_mlp_demo():
    """Trains the scratch MLP and plots loss + decision boundary."""
    X, y = generate_synthetic_data(n_samples=400, seed=42)
    mlp = ScratchMLP(layer_dims=(2, 16, 8, 1), seed=42)

    epochs = 1200
    lr = 0.15
    loss_history = []

    print("=" * 70)
    print("SCRATCH MLP NON-LINEAR CLASSIFICATION DEMO (fundamentals/mlp)")
    print(f"Dataset: Concentric Circles (N=400) | Architecture: [2 -> 16 -> 8 -> 1]")
    print(f"Optimizer: Gradient Descent | Learning Rate: {lr} | Epochs: {epochs}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        y_hat, cache = mlp.forward(X)
        loss = mlp.compute_loss(y_hat, y)
        loss_history.append(loss)

        grads = mlp.backward(cache, y)
        mlp.update_params(grads, lr=lr)

        if epoch % 200 == 0 or epoch == epochs:
            preds = mlp.predict(X)
            acc = np.mean(preds == y)
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss:.5f} | Accuracy: {acc:.2%}")

    # Plot 1: Decision Boundary, Plot 2: Loss Curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Decision Boundary
    ax1 = axes[0]
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs, _ = mlp.forward(grid)
    probs = probs.reshape(xx.shape)

    contour = ax1.contourf(xx, yy, probs, levels=25, cmap="RdYlBu", alpha=0.7)
    ax1.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2.5, linestyles="--")
    plt.colorbar(contour, ax=ax1, label="Predicted Probability P(y=1)")

    # Scatter points
    ax1.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], c="crimson", edgecolors="k", s=40, label="Class 0 (Inner)")
    ax1.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], c="royalblue", edgecolors="k", s=40, label="Class 1 (Outer)")
    ax1.set_title("Learned Non-Linear Decision Boundary\n(Universal Approximation in 2D)")
    ax1.set_xlabel("Feature x1")
    ax1.set_ylabel("Feature x2")
    ax1.legend(loc="upper right")

    # Loss Curve
    ax2 = axes[1]
    ax2.plot(loss_history, color="navy", lw=2, label="BCE Loss")
    ax2.set_title("Training Loss Convergence (Binary Cross-Entropy)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    out_file = RESULTS_DIR / "mlp_decision_boundary.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved visualization to: {out_file}")


if __name__ == "__main__":
    train_mlp_demo()
