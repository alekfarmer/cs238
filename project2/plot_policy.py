"""Draw a grid problem's policy as a PNG. Drawing only: nothing is computed here.

- With ``--values`` (small): a heatmap of V(s) with an arrow for each state's action.
- Without (medium): a map of the chosen action over (position, velocity).

The policy and values come from ``project2.py``; this script only lays them
out on the grid the course defines (state ``s`` at column ``(s - 1) % width``,
row ``(s - 1) // width``).

Usage:
    python plot_policy.py <small|medium> <name>.policy <out>.png [--values <name>.values]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from numpy.typing import NDArray

from project2 import PROBLEMS, read_policy

ARROWS: dict[str, list[tuple[float, float]]] = {
    # small: 1 left, 2 right, 3 up (+10 = next row), 4 down.
    "small": [(-1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, -1.0)],
}
"""Arrow direction (dx, dy) for each 0-based action, per problem."""

AXIS_LABELS: dict[str, tuple[str, str]] = {
    "small": ("x", "y"),
    "medium": ("position (bin)", "velocity (bin)"),
}


def read_values(path: Path, n_states: int) -> NDArray[np.float64]:
    """Read one value per line and check the count."""
    values = np.loadtxt(path, dtype=np.float64, ndmin=1)
    if values.shape != (n_states,):
        raise ValueError(f"{path}: expected {n_states} values, got {values.shape[0]}")
    return values


def plot(problem_name: str, policy_path: Path, out_path: Path, values_path: Path | None) -> None:
    """Render the policy (and optionally the values) of a grid problem to ``out_path``."""
    problem = PROBLEMS[problem_name]
    if problem.grid is None:
        raise ValueError(f"{problem_name} has no grid layout to draw")
    width, height = problem.grid
    policy = read_policy(policy_path, problem).reshape(height, width)
    xlabel, ylabel = AXIS_LABELS.get(problem_name, ("column", "row"))

    fig, ax = plt.subplots(figsize=(7, 6) if width == height else (11, 4))
    if values_path is not None:
        values = read_values(values_path, problem.n_states).reshape(height, width)
        image = ax.imshow(values, origin="lower", cmap="viridis")
        fig.colorbar(image, ax=ax, label="V(s) under the learned model")
        if problem_name in ARROWS:
            directions = np.array(ARROWS[problem_name])[policy]
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            ax.quiver(
                cols, rows, directions[..., 0], directions[..., 1],
                color="white", pivot="middle", scale=1.6, scale_units="xy", width=0.006,
            )  # fmt: skip
        ax.set_xticks(range(width), [str(i + 1) for i in range(width)])
        ax.set_yticks(range(height), [str(i + 1) for i in range(height)])
    else:
        n = problem.n_actions
        cmap = ListedColormap(plt.get_cmap("coolwarm")(np.linspace(0, 1, n)))
        image = ax.imshow(
            policy, origin="lower", aspect="auto", cmap=cmap,
            norm=BoundaryNorm(np.arange(n + 1) - 0.5, n),
        )  # fmt: skip
        bar = fig.colorbar(image, ax=ax, ticks=range(n), label="action")
        bar.ax.set_yticklabels([str(i + 1) for i in range(n)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{problem_name}.policy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and draw; return the exit code."""
    parser = argparse.ArgumentParser(description="Draw a grid problem's policy.")
    parser.add_argument("problem", choices=sorted(p for p, v in PROBLEMS.items() if v.grid))
    parser.add_argument("policy", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--values", type=Path)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        plot(args.problem, args.policy, args.out, args.values)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
