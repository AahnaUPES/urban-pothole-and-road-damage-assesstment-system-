"""
===============================================================================
UrbanRoadDamage — Tensor Mathematics & SVD Low-Rank Approximation
Module: fundamentals/tensor/tensor_math.py (PRD Section 8)
===============================================================================

PURPOSE & VIVA EXPLANATION
--------------------------
Demonstrates the core tensor algebra required by PRD Section 8:
  - NumPy broadcasting rules across arbitrary dimensional axes.
  - Dot products, matrix-vector multiplication, outer products.
  - Reshaping, transposing, flattening, and stride manipulation.
  - Singular Value Decomposition (SVD): A = U @ Sigma @ V^T
    Demonstrates low-rank matrix approximation for image/feature compression:
      A_k = sum_{i=1}^k sigma_i * u_i * v_i^T (Eckart-Young-Mirsky Theorem).

ACADEMIC DEMONSTRATION NOTICE
-----------------------------
Runs on synthetic matrices / toy road patch. Saves figure to results/svd_approximation.png.
===============================================================================
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_tensor_math_demonstrations():
    print("=" * 70)
    print("TENSOR MATHEMATICS & LINEAR ALGEBRA FOUNDATIONS (PRD Section 8)")
    print("=" * 70)

    # 1. Broadcasting
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape (2, 3)
    b = np.array([10.0, 20.0, 30.0])                  # Shape (3,)
    C = A + b                                         # b broadcast across rows -> (2, 3)
    print("1. Broadcasting Demo:")
    print(f"   Shape A: {A.shape} + Shape b: {b.shape} -> Result Shape: {C.shape}")

    # 2. Outer Product & Inner Product
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0])
    inner_prod = np.dot(u, v)
    outer_prod = np.outer(u, v)
    print("\n2. Vector Products:")
    print(f"   Inner Product <u, v> = {inner_prod:.1f}")
    print(f"   Outer Product u (x) v =\n{outer_prod}")

    # 3. SVD Low-Rank Approximation (Eckart-Young Theorem)
    print("\n3. Singular Value Decomposition (SVD):")
    # Generate a synthetic 64x64 patterned road-like matrix with rank 10 + noise
    np.random.seed(42)
    X = np.zeros((64, 64))
    # Base structure: diagonal crack + horizontal boundary
    for i in range(64):
        X[i, min(63, int(i * 0.9))] = 1.0
        X[min(63, int(i * 0.9)), i] = 0.8
        X[32, :] = 0.6
    X += np.random.randn(64, 64) * 0.05

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    print(f"   Matrix Shape: {X.shape} | Singular values top 5: {S[:5].round(2)}")

    # Low-rank reconstructions for k = 1, 3, 10, 30
    ranks = [1, 3, 10, 30]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))

    axes[0].imshow(X, cmap="gray")
    axes[0].set_title(f"Original Matrix\n(Full Rank {len(S)})")
    axes[0].axis("off")

    for idx, k in enumerate(ranks):
        # Truncated reconstruction: A_k = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
        Ak = np.dot(U[:, :k] * S[:k], Vt[:k, :])
        err = np.linalg.norm(X - Ak, "fro") / np.linalg.norm(X, "fro")
        axes[idx + 1].imshow(Ak, cmap="gray")
        axes[idx + 1].set_title(f"Rank k = {k}\nRel Error: {err:.1%}")
        axes[idx + 1].axis("off")

    plt.tight_layout()
    out_file = RESULTS_DIR / "svd_approximation.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved SVD low-rank approximation plot to: {out_file}")


if __name__ == "__main__":
    run_tensor_math_demonstrations()
