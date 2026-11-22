"""
===============================================================================
UrbanRoadDamage — Artificial Neurons & Activation Dynamics
Module: fundamentals/neurons/mcculloch_pitts.py (PRD Section 8 & 12.3)
===============================================================================

PURPOSE & VIVA EXPLANATION
--------------------------
1. McCulloch-Pitts (M-P) Neuron (1943):
   - First computational model of a biological neuron.
   - Binary threshold logic: y = 1 if sum(w_i * x_i) >= theta else 0 (with excitatory & inhibitory inputs).
   - Capable of implementing basic boolean logic gates (AND, OR, NOT), but incapable of learning weights.

2. Rosenblatt Perceptron (1958):
   - Introduced learnable continuous weights and biases: y = step(w^T x + b).
   - Proven to converge on linearly separable patterns (Perceptron Convergence Theorem).

3. Smooth Differentiable Activations (PRD Section 12.3):
   - Step function: Gradient is 0 everywhere (non-differentiable at 0) -> Backpropagation is impossible!
   - Sigmoid: sigma(z) = 1 / (1 + e^-z), smooth derivative sigma'(z) = sigma(z)*(1 - sigma(z)).
     Issue: Vanishing gradients for |z| > 4; outputs are non-zero-centered.
   - Tanh: tanh(z) = (e^z - e^-z) / (e^z + e^-z), zero-centered, range [-1, 1], steeper gradient than sigmoid.
   - ReLU: f(z) = max(0, z), constant gradient 1 for z > 0, prevents vanishing gradients in deep networks.

ACADEMIC DEMONSTRATION NOTICE
-----------------------------
Runs on synthetic logic gates & activation range [-5, 5]. Saves results/activation_functions.png.
===============================================================================
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class McCullochPittsNeuron:
    """Boolean logic gates using threshold logic."""

    def __init__(self, weights: np.ndarray, threshold: float):
        self.weights = np.array(weights)
        self.threshold = threshold

    def forward(self, x: np.ndarray) -> int:
        return 1 if np.dot(self.weights, x) >= self.threshold else 0


def demonstrate_neurons_and_activations():
    print("=" * 70)
    print("ARTIFICIAL NEURONS & ACTIVATION DYNAMICS (fundamentals/neurons)")
    print("=" * 70)

    # 1. McCulloch-Pitts Boolean Gates
    mp_and = McCullochPittsNeuron(weights=[1, 1], threshold=2)
    mp_or  = McCullochPittsNeuron(weights=[1, 1], threshold=1)

    print("McCulloch-Pitts Logic Gates:")
    for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print(f"  Input: {x} -> AND: {mp_and.forward(x)} | OR: {mp_or.forward(x)}")

    # 2. Activation Function Comparison
    z = np.linspace(-5.0, 5.0, 400)
    step = np.heaviside(z, 0.5)
    sigmoid_act = 1.0 / (1.0 + np.exp(-z))
    tanh_act = np.tanh(z)
    relu_act = np.maximum(0.0, z)

    # Derivatives
    sigmoid_grad = sigmoid_act * (1.0 - sigmoid_act)
    tanh_grad = 1.0 - (tanh_act ** 2)
    relu_grad = (z > 0).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Activations
    axes[0].plot(z, step, "k--", label="Heaviside Step", lw=1.8)
    axes[0].plot(z, sigmoid_act, "r-", label="Sigmoid", lw=2)
    axes[0].plot(z, tanh_act, "b-", label="Tanh (Zero-centered)", lw=2)
    axes[0].plot(z, relu_act, "g-", label="ReLU (Unsaturated)", lw=2)
    axes[0].set_title("Activation Functions f(z)")
    axes[0].set_xlabel("Input Activation z")
    axes[0].set_ylabel("f(z)")
    axes[0].set_ylim([-1.2, 3.0])
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Derivatives
    axes[1].plot(z, np.zeros_like(z), "k--", label="Step Grad (0 everywhere)", lw=1.5)
    axes[1].plot(z, sigmoid_grad, "r-", label="Sigmoid Grad (Max 0.25)", lw=2)
    axes[1].plot(z, tanh_grad, "b-", label="Tanh Grad (Max 1.0)", lw=2)
    axes[1].plot(z, relu_grad, "g-", label="ReLU Grad (1.0 for z>0)", lw=2)
    axes[1].set_title("Gradients df/dz (Gradient Flow during Backpropagation)")
    axes[1].set_xlabel("Input Activation z")
    axes[1].set_ylabel("df/dz")
    axes[1].set_ylim([-0.1, 1.2])
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    out_file = RESULTS_DIR / "activation_functions.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"\nSaved activation function comparison to: {out_file}")


if __name__ == "__main__":
    demonstrate_neurons_and_activations()
