"""
===============================================================================
UrbanRoadDamage — Optimizer Comparison: Batch GD, SGD, Momentum, Nesterov
Module: fundamentals/optimizers/optimizer_comparison.py
===============================================================================

PURPOSE & VIVA EXPLANATION
--------------------------
1. Why compare these four classical optimizers?
   PRD Section 9 requires: "Compare Batch Gradient Descent, SGD, Momentum and Nesterov.
   Plot or record convergence/loss behavior for the optimizer comparison."

2. Theoretical Update Rules:
   Let theta be the parameter vector, eta be the learning rate, and L(theta) be loss:

   a) Batch Gradient Descent (BGD):
      theta_{t+1} = theta_t - eta * nabla L(theta_t)
      Characteristic: Deterministic, slow on ill-conditioned surfaces with severe ravines.

   b) Stochastic Gradient Descent (SGD with noise):
      theta_{t+1} = theta_t - eta * (nabla L(theta_t) + xi)   [xi ~ N(0, sigma^2)]
      Characteristic: High variance, zig-zags across ravines, can escape saddle points.

   c) Classical Momentum (Polyak 1964):
      v_{t+1} = beta * v_t + eta * nabla L(theta_t)
      theta_{t+1} = theta_t - v_{t+1}
      Characteristic: Accelerates along consistent gradient directions, dampens oscillations.

   d) Nesterov Accelerated Gradient (NAG / Nesterov 1983):
      theta_lookahead = theta_t - beta * v_t
      v_{t+1} = beta * v_t + eta * nabla L(theta_lookahead)
      theta_{t+1} = theta_t - v_{t+1}
      Characteristic: Computes the gradient at a lookahead point. Acts as an anticipatory
      braking mechanism, reducing overshoot near local minima.

3. Loss Surface:
   Ill-conditioned anisotropic quadratic valley (standard benchmark for ravine navigation):
       f(x, y) = 0.5 * (x^2 + 20 * y^2)
   Condition number kappa = 20. The curvature along y is 20x higher than along x,
   causing standard gradient descent to oscillate heavily along the y axis.

ACADEMIC DEMONSTRATION NOTICE
-----------------------------
Runs on synthetic 2D loss surface on CPU. Saves convergence plot to results/optimizer_comparison.png.
===============================================================================
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def loss_fn(w: np.ndarray) -> float:
    """Anisotropic quadratic bowl: f(x, y) = 0.5 * (x^2 + 20 * y^2)."""
    return float(0.5 * (w[0] ** 2 + 20.0 * (w[1] ** 2)))


def grad_fn(w: np.ndarray) -> np.ndarray:
    """Analytical gradient: df/dx = x, df/dy = 20*y."""
    return np.array([w[0], 20.0 * w[1]])


def run_batch_gd(w_init: np.ndarray, lr: float = 0.08, steps: int = 50):
    w = w_init.copy().astype(float)
    traj = [w.copy()]
    losses = [loss_fn(w)]
    for _ in range(steps):
        g = grad_fn(w)
        w -= lr * g
        traj.append(w.copy())
        losses.append(loss_fn(w))
    return np.array(traj), np.array(losses)


def run_sgd_noisy(w_init: np.ndarray, lr: float = 0.08, steps: int = 50, noise_std: float = 0.35, seed: int = 42):
    np.random.seed(seed)
    w = w_init.copy().astype(float)
    traj = [w.copy()]
    losses = [loss_fn(w)]
    for _ in range(steps):
        g = grad_fn(w) + np.random.randn(2) * noise_std
        w -= lr * g
        traj.append(w.copy())
        losses.append(loss_fn(w))
    return np.array(traj), np.array(losses)


def run_momentum(w_init: np.ndarray, lr: float = 0.04, beta: float = 0.85, steps: int = 50):
    w = w_init.copy().astype(float)
    v = np.zeros(2)
    traj = [w.copy()]
    losses = [loss_fn(w)]
    for _ in range(steps):
        g = grad_fn(w)
        v = beta * v + lr * g
        w -= v
        traj.append(w.copy())
        losses.append(loss_fn(w))
    return np.array(traj), np.array(losses)


def run_nesterov(w_init: np.ndarray, lr: float = 0.04, beta: float = 0.85, steps: int = 50):
    w = w_init.copy().astype(float)
    v = np.zeros(2)
    traj = [w.copy()]
    losses = [loss_fn(w)]
    for _ in range(steps):
        # Lookahead position: theta - beta * v
        w_lookahead = w - beta * v
        g = grad_fn(w_lookahead)
        v = beta * v + lr * g
        w -= v
        traj.append(w.copy())
        losses.append(loss_fn(w))
    return np.array(traj), np.array(losses)


def compare_optimizers():
    w_start = np.array([-4.5, 2.0])
    steps = 45

    traj_bgd, loss_bgd = run_batch_gd(w_start, lr=0.08, steps=steps)
    traj_sgd, loss_sgd = run_sgd_noisy(w_start, lr=0.08, steps=steps)
    traj_mom, loss_mom = run_momentum(w_start, lr=0.04, beta=0.82, steps=steps)
    traj_nag, loss_nag = run_nesterov(w_start, lr=0.04, beta=0.82, steps=steps)

    print("=" * 70)
    print("OPTIMIZER CONVERGENCE COMPARISON (fundamentals/optimizers)")
    print(f"Surface: f(x, y) = 0.5 * (x^2 + 20*y^2) | Start: {w_start} | Steps: {steps}")
    print("=" * 70)
    print(f"Batch GD Final Loss : {loss_bgd[-1]:.6f}")
    print(f"SGD (Noisy) Final Loss: {loss_sgd[-1]:.6f}")
    print(f"Momentum Final Loss : {loss_mom[-1]:.6f}")
    print(f"Nesterov Final Loss : {loss_nag[-1]:.6f}")
    print("=" * 70)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 2D Contour with Trajectories
    ax1 = axes[0]
    x_vals = np.linspace(-5.0, 5.0, 300)
    y_vals = np.linspace(-3.0, 3.0, 300)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = 0.5 * (X ** 2 + 20.0 * (Y ** 2))

    ax1.contour(X, Y, Z, levels=np.logspace(-1, 2.2, 25), cmap="viridis", alpha=0.6)
    ax1.plot(traj_bgd[:, 0], traj_bgd[:, 1], "r.-", label="Batch GD (Oscillates)", lw=1.5, markersize=5)
    ax1.plot(traj_sgd[:, 0], traj_sgd[:, 1], "g.--", label="SGD (Noisy)", lw=1.2, markersize=4, alpha=0.7)
    ax1.plot(traj_mom[:, 0], traj_mom[:, 1], "b.-", label="Momentum (Accelerated)", lw=1.8, markersize=5)
    ax1.plot(traj_nag[:, 0], traj_nag[:, 1], "m.-", label="Nesterov (Damped/Anticipatory)", lw=2.0, markersize=5)

    ax1.scatter([0], [0], color="gold", s=140, marker="*", edgecolors="k", zorder=10, label="Global Min (0, 0)")
    ax1.scatter([w_start[0]], [w_start[1]], color="black", s=90, marker="o", zorder=10, label="Start Point")
    ax1.set_title("Optimization Trajectories in Ill-Conditioned Ravine")
    ax1.set_xlabel("Parameter x (Low Curvature)")
    ax1.set_ylabel("Parameter y (High Curvature)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    # 2. Convergence Curve: Loss vs Step
    ax2 = axes[1]
    ax2.plot(loss_bgd, "r-", label="Batch GD", lw=2)
    ax2.plot(loss_sgd, "g--", label="SGD", lw=1.5, alpha=0.8)
    ax2.plot(loss_mom, "b-", label="Momentum", lw=2)
    ax2.plot(loss_nag, "m-", label="Nesterov (NAG)", lw=2.2)

    ax2.set_yscale("log")
    ax2.set_title("Log-Loss Convergence per Step")
    ax2.set_xlabel("Optimization Step")
    ax2.set_ylabel("Loss L(theta) [Log Scale]")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    out_path = RESULTS_DIR / "optimizer_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved optimizer comparison plot to: {out_path}")


if __name__ == "__main__":
    compare_optimizers()
