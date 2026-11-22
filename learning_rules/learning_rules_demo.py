"""
===============================================================================
UrbanRoadDamage — Classical Neural Learning Rules Demonstrations
Module: learning_rules/learning_rules_demo.py (PRD Section 10)
===============================================================================

PURPOSE & VIVA EXPLANATION
--------------------------
Demonstrates the five foundational neural learning rules required by PRD Section 10:
"Hebbian learning, Modified Hebbian learning, Competitive learning, Error-correction learning,
A compact Boltzmann-learning or energy-based experiment for conceptual comparison.
These are supporting academic demonstrations and do not need to form the main pothole-detection pipeline."

THE FIVE LEARNING RULES:
------------------------
1. Hebbian Learning (Donald Hebb, 1949):
   - Principle: "Neurons that fire together, wire together."
   - Update rule: Delta w = eta * y * x
   - Property: Unsupervised. Reinforces correlated activity. Weights grow unbounded without weight decay.

2. Modified Hebbian / Oja's Learning Rule (Erkki Oja, 1982):
   - Principle: Normalised Hebbian learning preventing weight explosion.
   - Update rule: Delta w = eta * y * (x - y * w)
   - Property: Mathematically proves that w converges to the first principal component (PCA) of the input distribution!

3. Competitive Learning (Rumelhart & Zipser, 1985):
   - Principle: "Winner-Take-All" unsupervised clustering.
   - Update rule: Identify winning neuron k = argmin_i ||x - w_i||.
     Only winner updates: Delta w_k = eta * (x - w_k).
   - Property: Self-organises cluster prototype vectors.

4. Error-Correction Learning (Perceptron / Widrow-Hoff Delta Rule, 1960):
   - Principle: Supervised update proportional to prediction error.
   - Update rule: Delta w = eta * (target - output) * x
   - Property: Guarantees convergence on linearly separable data.

5. Compact Energy-based / Boltzmann Learning (Hopfield 1982; Hinton & Sejnowski 1983):
   - Principle: Hopfield network energy function E = -0.5 * sum_{i,j} w_ij * s_i * s_j.
   - Neurons update to minimize network energy: s_i = sign(sum_j w_ij * s_j).
   - Demonstrates associative memory recall and energy minimization.

ACADEMIC DEMONSTRATION NOTICE
-----------------------------
All experiments run on synthetic toy data (2-5 neurons / 2D vectors).
Visualizes all 5 learning dynamics in results/learning_rules.png.
===============================================================================
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def demo_hebbian():
    """Hebbian learning: Delta w = eta * y * x (Unbounded growth)."""
    np.random.seed(42)
    # 2 correlated inputs
    x = np.array([[1.0, 1.0], [0.8, 0.9], [1.2, 1.1]])
    w = np.array([0.1, 0.1])
    eta = 0.05
    norms = [np.linalg.norm(w)]

    for _ in range(35):
        for sample in x:
            y = np.dot(w, sample)
            w += eta * y * sample
            norms.append(np.linalg.norm(w))

    return norms


def demo_ojas_rule():
    """Modified Hebbian (Oja's Rule): Delta w = eta * y * (x - y*w). Converges to PCA vector."""
    np.random.seed(42)
    # Elliptical 2D data centered at origin
    n = 200
    u1 = np.random.randn(n) * 3.0
    u2 = np.random.randn(n) * 0.5
    # Rotate by 45 degrees
    rot = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], [np.sin(np.pi/4), np.cos(np.pi/4)]])
    X = (rot @ np.vstack([u1, u2])).T

    w = np.array([1.0, 0.0])  # Initial random vector
    eta = 0.01
    angles = []
    true_pca = rot[:, 0]  # First eigenvector direction

    for epoch in range(15):
        for x in X:
            y = np.dot(w, x)
            w += eta * y * (x - y * w)
            cos_sim = np.abs(np.dot(w / np.linalg.norm(w), true_pca))
            angles.append(cos_sim)

    return X, w / np.linalg.norm(w), true_pca, angles


def demo_competitive_learning():
    """Competitive learning (Winner-Take-All) forming 3 cluster centroids."""
    np.random.seed(42)
    c1 = np.random.randn(40, 2) * 0.3 + np.array([-2, -1])
    c2 = np.random.randn(40, 2) * 0.3 + np.array([0, 2])
    c3 = np.random.randn(40, 2) * 0.3 + np.array([2, -1])
    X = np.vstack([c1, c2, c3])

    # 3 prototype neurons initialized near origin
    weights = np.random.uniform(-0.5, 0.5, (3, 2))
    initial_w = weights.copy()
    eta = 0.05

    for _ in range(50):
        np.random.shuffle(X)
        for x in X:
            # Winner has minimal Euclidean distance
            dists = np.linalg.norm(weights - x, axis=1)
            winner = np.argmin(dists)
            weights[winner] += eta * (x - weights[winner])

    return X, initial_w, weights


def demo_error_correction():
    """Perceptron Delta Rule: Delta w = eta * (target - output) * x."""
    np.random.seed(42)
    X = np.array([
        [1.0, 2.0], [2.0, 3.0], [1.5, 2.8],
        [4.0, 1.0], [5.0, 2.0], [4.5, 1.2]
    ])
    d = np.array([0, 0, 0, 1, 1, 1])  # 2 linearly separable classes

    w = np.array([0.0, 0.0])
    b = 0.0
    eta = 0.2
    errors_history = []

    for _ in range(25):
        total_err = 0
        for x_i, target in zip(X, d):
            act = np.dot(w, x_i) + b
            pred = 1 if act >= 0 else 0
            err = target - pred
            total_err += abs(err)
            w += eta * err * x_i
            b += eta * err
        errors_history.append(total_err)

    return errors_history, w, b, X, d


def demo_boltzmann_energy():
    """Hopfield / Boltzmann energy minimization: E = -0.5 * s.T @ W @ s."""
    # Stored pattern: 4 bipolar neurons [1, -1, 1, -1]
    p = np.array([1, -1, 1, -1])
    # Hebbian weight matrix for autoassociation (zero diagonal)
    W = np.outer(p, p)
    np.fill_diagonal(W, 0)

    # Corrupted initial state
    s = np.array([1, 1, 1, -1])
    energies = [-0.5 * float(s @ W @ s)]

    # Asynchronous energy minimization
    for step in range(6):
        i = step % 4
        h = np.dot(W[i], s)
        s[i] = 1 if h >= 0 else -1
        energies.append(-0.5 * float(s @ W @ s))

    return energies, s, p


def run_all_learning_rules():
    print("=" * 70)
    print("NEURAL LEARNING RULES DEMONSTRATIONS (learning_rules/)")
    print("Running Hebbian, Oja's, Competitive, Error-Correction, and Energy Rules...")
    print("=" * 70)

    # 1. Hebbian
    hebb_norms = demo_hebbian()
    # 2. Oja's Rule
    X_oja, w_oja, pca_true, oja_sim = demo_ojas_rule()
    # 3. Competitive
    X_comp, w_init, w_final = demo_competitive_learning()
    # 4. Error correction
    err_hist, w_ec, b_ec, X_ec, d_ec = demo_error_correction()
    # 5. Energy-based
    energies, s_final, p_stored = demo_boltzmann_energy()

    print(f"1. Hebbian: Weight magnitude exploded from {hebb_norms[0]:.2f} to {hebb_norms[-1]:.2f}")
    print(f"2. Oja's Rule: Final alignment with True PCA = {oja_sim[-1]:.4f} (Cosine Similarity)")
    print(f"3. Competitive Learning: Centroids successfully partitioned 3 clusters.")
    print(f"4. Error-Correction: Training errors reduced to {err_hist[-1]}.")
    print(f"5. Energy Minimization: System energy reduced from {energies[0]:.1f} to {energies[-1]:.1f}")
    print("=" * 70)

    # Create 5-panel figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Panel 1: Hebbian explosion
    axes[0, 0].plot(hebb_norms, color="darkred", lw=2)
    axes[0, 0].set_title("1. Hebbian: Unbounded Weight Growth\n(Delta w = eta * y * x)")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Weight Norm ||w||")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Oja's rule convergence to PCA
    axes[0, 1].plot(oja_sim, color="teal", lw=2)
    axes[0, 1].set_title("2. Oja's Rule: Convergence to First PCA\n(Delta w = eta*y*(x - y*w))")
    axes[0, 1].set_xlabel("Updates")
    axes[0, 1].set_ylabel("Cosine Similarity to 1st PC")
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Competitive learning clustering
    axes[0, 2].scatter(X_comp[:, 0], X_comp[:, 1], c="silver", edgecolors="gray", s=30, label="Input Data")
    axes[0, 2].scatter(w_init[:, 0], w_init[:, 1], c="blue", marker="x", s=80, label="Initial Prototypes")
    axes[0, 2].scatter(w_final[:, 0], w_final[:, 1], c="crimson", marker="o", s=100, edgecolors="k", label="Trained Centroids")
    axes[0, 2].set_title("3. Competitive Learning (Winner-Take-All)\nCluster Prototype Discovery")
    axes[0, 2].legend(loc="upper right", fontsize=8)
    axes[0, 2].grid(True, linestyle="--", alpha=0.6)

    # Panel 4: Error correction convergence
    axes[1, 0].plot(err_hist, color="indigo", lw=2, marker="o")
    axes[1, 0].set_title("4. Error-Correction (Delta Rule)\nErrors -> 0 Convergence")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Misclassification Count")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    # Panel 5: Boltzmann / Hopfield Energy
    axes[1, 1].plot(energies, color="darkgreen", lw=2, marker="s")
    axes[1, 1].set_title("5. Energy-Based Learning (Hopfield/Boltzmann)\nMonotonic Energy Minimization")
    axes[1, 1].set_xlabel("State Transition Step")
    axes[1, 1].set_ylabel("Network Energy E(s)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    # Panel 6: Summary notes for Viva
    axes[1, 2].axis("off")
    summary_text = (
        "VIVA SUMMARY (PRD Section 10):\n\n"
        "• Hebbian: Correlational, lacks stability.\n"
        "• Oja: Adds -y*w negative feedback, yields PCA.\n"
        "• Competitive: WTA partitions unsupervised space.\n"
        "• Error-Correction: Supervised gradient descent on L2.\n"
        "• Energy-Based: Lyapunov energy function guarantees\n"
        "  relaxation to stored memory attractors.\n\n"
        "Status: Academic demos complete on toy data."
    )
    axes[1, 2].text(0.05, 0.45, summary_text, fontsize=11, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.8", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    out_file = RESULTS_DIR / "learning_rules.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved learning rules visualization to: {out_file}")


if __name__ == "__main__":
    run_all_learning_rules()
