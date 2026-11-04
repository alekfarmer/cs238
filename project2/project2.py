"""Offline reinforcement learning for AA228/CS238 Project 2.

Each dataset is a batch of logged transitions ``(s, a, r, sp)`` from an
unknown MDP. We never get to interact with the environment, so we:

    1. Estimate a model: maximum-likelihood transition probabilities
       T(sp | s, a) from counts, and the mean reward R(s, a).
    2. Solve that model with value iteration, giving Q(s, a) and a greedy
       policy for every state that appears in the data.
    3. Fill in states that never appear: the nearest observed state on the
       grid (medium), or the most common action (small and large, where every
       reachable state is observed).

Usage:
    python project2.py <small|medium|large> <infile>.csv <outfile>.policy
        [--summary out.json] [--values out.values] [options]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

HEADER = ("s", "a", "r", "sp")
"""Required CSV header, in this order."""


# ------------------------------------------------------------------- problems


@dataclass(frozen=True)
class Problem:
    """What the course tells us about one dataset's MDP.

    Attributes:
        name: Dataset name (``small``, ``medium`` or ``large``).
        n_states: Number of states; states are numbered ``1 .. n_states``.
        n_actions: Number of actions; actions are numbered ``1 .. n_actions``.
        gamma: Discount factor the policy is evaluated with.
        grid: ``(width, height)`` if state ``s`` sits at column
            ``(s - 1) % width`` and row ``(s - 1) // width`` of a grid, else None.
            Used to find the nearest observed state and for plotting.
        goal_reward: Transitions with at least this reward end the episode
            (no future value), or None if the problem has no terminal states.
    """

    name: str
    n_states: int
    n_actions: int
    gamma: float
    grid: tuple[int, int] | None = None
    goal_reward: float | None = None


PROBLEMS: dict[str, Problem] = {
    # 10x10 grid world; state = x + 10 * (y - 1), actions left/right/up/down.
    "small": Problem("small", 100, 4, 0.95, grid=(10, 10)),
    # Mountain car; state = 1 + pos + 500 * vel, 7 accelerations, ends at the flag.
    "medium": Problem("medium", 50_000, 7, 1.0, grid=(500, 100), goal_reward=50_000.0),
    # Secret problem; the course gives only its size and discount.
    "large": Problem("large", 302_020, 9, 0.95),
}


# ----------------------------------------------------------------------- data


@dataclass(frozen=True)
class Transitions:
    """Logged experience, converted to 0-based indices.

    Attributes:
        s: State of each sample, ``0 .. n_states - 1``.
        a: Action of each sample, ``0 .. n_actions - 1``.
        r: Reward of each sample.
        sp: Next state of each sample, ``0 .. n_states - 1``.
    """

    s: IntArray
    a: IntArray
    r: FloatArray
    sp: IntArray

    def __len__(self) -> int:
        """Return the number of samples."""
        return int(self.s.shape[0])


def load_csv(path: str | Path, problem: Problem) -> Transitions:
    """Read an ``s,a,r,sp`` CSV and check every value against ``problem``.

    Args:
        path: CSV file with header ``s,a,r,sp`` and one integer sample per row.
        problem: The problem whose state and action ranges the data must respect.

    Returns:
        The samples with 0-based states and actions.

    Raises:
        ValueError: If the header, a value or a range is wrong, or the file has
            no samples.
        OSError: If the file cannot be read.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        header = tuple(f.readline().strip().split(","))
    if header != HEADER:
        raise ValueError(f"{path}: header must be {','.join(HEADER)}, got {','.join(header)}")
    try:
        raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64, ndmin=2)
    except ValueError as exc:
        raise ValueError(f"{path}: every value must be an integer ({exc})") from exc
    if raw.shape[0] == 0 or raw.shape[1] != len(HEADER):
        raise ValueError(f"{path}: expected rows of {len(HEADER)} integers, got shape {raw.shape}")
    s, a, r, sp = raw.T
    for label, column, upper in (
        ("s", s, problem.n_states),
        ("sp", sp, problem.n_states),
        ("a", a, problem.n_actions),
    ):
        if column.min() < 1 or column.max() > upper:
            raise ValueError(f"{path}: {label} must lie in 1..{upper} for {problem.name}")
    return Transitions(s - 1, a - 1, r.astype(np.float64), sp - 1)


def write_policy(policy: IntArray, path: str | Path) -> None:
    """Write one 1-based action per line; line ``i`` is the action for state ``i``.

    Args:
        policy: 0-based action for every state.
        path: Output file (conventionally ``<name>.policy``).
    """
    Path(path).write_text("".join(f"{a + 1}\n" for a in policy.tolist()), encoding="utf-8")


def read_policy(path: str | Path, problem: Problem) -> IntArray:
    """Read a ``.policy`` file back into 0-based actions, checking its shape.

    Args:
        path: Policy file with one action per line.
        problem: The problem the policy is for.

    Returns:
        0-based action for every state.

    Raises:
        ValueError: If the line count or an action is out of range.
    """
    actions = np.loadtxt(path, dtype=np.int64, ndmin=1)
    if actions.shape != (problem.n_states,):
        raise ValueError(f"{path}: expected {problem.n_states} lines, got {actions.shape[0]}")
    if actions.min() < 1 or actions.max() > problem.n_actions:
        raise ValueError(f"{path}: actions must lie in 1..{problem.n_actions}")
    return actions - 1


# ---------------------------------------------------------------------- model


@dataclass(frozen=True)
class Model:
    """A maximum-likelihood MDP estimated from transitions, stored sparsely.

    The transition function is a list of edges ``(s, a) -> sp`` with
    probabilities; each (s, a) row sums to 1 over its edges.

    Attributes:
        n_states: Number of states.
        n_actions: Number of actions.
        edge_sa: Flat index ``s * n_actions + a`` of each edge's (s, a).
        edge_sp: Next state of each edge.
        edge_prob: T(sp | s, a) of each edge.
        edge_continues: 0.0 if the edge ends the episode, else 1.0.
        reward: Mean observed reward R(s, a), shape (n_states, n_actions);
            0 where (s, a) was never tried.
        observed: Whether (s, a) was ever tried, shape (n_states, n_actions).
    """

    n_states: int
    n_actions: int
    edge_sa: IntArray
    edge_sp: IntArray
    edge_prob: FloatArray
    edge_continues: FloatArray
    reward: FloatArray
    observed: BoolArray

    @property
    def observed_states(self) -> BoolArray:
        """Return whether each state has at least one tried action."""
        return np.asarray(self.observed.any(axis=1))


def grid_coordinates(states: IntArray, grid: tuple[int, int]) -> FloatArray:
    """Map 0-based states to (column, row) scaled to [0, 1] on each axis.

    Scaling makes one step along the short axis count as much as the same
    fraction of the long axis (on medium: 1/100 of velocity vs 1/500 of position).
    """
    width, height = grid
    return np.stack([(states % width) / width, (states // width) / height], axis=1)


def nearest_states(
    targets: IntArray, candidates: IntArray, grid: tuple[int, int], chunk: int = 2048
) -> IntArray:
    """For each target state, the candidate state nearest to it on the grid.

    Args:
        targets: 0-based states to look up.
        candidates: 0-based states to choose from (must be non-empty).
        grid: ``(width, height)`` of the grid.
        chunk: Targets processed per batch, to bound memory.

    Returns:
        The nearest candidate (Euclidean, scaled coordinates) for each target;
        ties go to the lowest-numbered candidate.
    """
    if candidates.size == 0:
        raise ValueError("no candidate states to choose from")
    cand_xy = grid_coordinates(candidates, grid)
    result = np.empty(targets.shape[0], dtype=np.int64)
    for start in range(0, targets.shape[0], chunk):
        xy = grid_coordinates(targets[start : start + chunk], grid)
        dist = ((xy[:, None, :] - cand_xy[None, :, :]) ** 2).sum(axis=2)
        result[start : start + chunk] = candidates[dist.argmin(axis=1)]
    return result


def estimate_model(data: Transitions, problem: Problem) -> Model:
    """Estimate T and R by maximum likelihood (counting).

    A transition with reward at least ``problem.goal_reward`` ends the
    episode. On a grid problem, a non-terminal next state that is never
    observed as a state is redirected to the nearest observed state, so value
    flows through it instead of treating it as a dead end worth 0.

    Args:
        data: The logged transitions.
        problem: Sizes, grid and goal rule.

    Returns:
        The estimated model.
    """
    n_s, n_a = problem.n_states, problem.n_actions
    terminal = (
        data.r >= problem.goal_reward
        if problem.goal_reward is not None
        else np.zeros(len(data), dtype=bool)
    )
    sp = data.sp.copy()
    if problem.grid is not None:
        seen = np.unique(data.s)
        dangling = ~terminal & ~np.isin(sp, seen)
        if dangling.any():
            unique_dangling, inverse = np.unique(sp[dangling], return_inverse=True)
            sp[dangling] = nearest_states(unique_dangling, seen, problem.grid)[inverse]

    sa = data.s * n_a + data.a
    sa_counts = np.bincount(sa, minlength=n_s * n_a).astype(np.float64)
    reward_sums = np.bincount(sa, weights=data.r, minlength=n_s * n_a)
    observed = sa_counts > 0
    reward = np.divide(reward_sums, sa_counts, out=np.zeros_like(reward_sums), where=observed)

    # One edge per distinct (s, a, sp, terminal); its probability is its share of (s, a).
    keys = np.stack([sa, sp, terminal.astype(np.int64)], axis=1)
    edges, edge_counts = np.unique(keys, axis=0, return_counts=True)
    edge_sa = edges[:, 0]
    return Model(
        n_states=n_s,
        n_actions=n_a,
        edge_sa=edge_sa,
        edge_sp=edges[:, 1],
        edge_prob=edge_counts / sa_counts[edge_sa],
        edge_continues=1.0 - edges[:, 2].astype(np.float64),
        reward=reward.reshape(n_s, n_a),
        observed=observed.reshape(n_s, n_a),
    )


# ------------------------------------------------------------------- planning


def expected_next_value(model: Model, values: FloatArray) -> FloatArray:
    """Return sum over sp of T(sp | s, a) * V(sp) (0 past a terminal edge), per (s, a)."""
    weights = model.edge_prob * model.edge_continues * values[model.edge_sp]
    size = model.n_states * model.n_actions
    flat = np.bincount(model.edge_sa, weights=weights, minlength=size).astype(np.float64)
    return flat.reshape(model.n_states, model.n_actions)


@dataclass(frozen=True)
class Solution:
    """The result of value iteration.

    Attributes:
        values: V(s); 0 for states with no observed action.
        q: Q(s, a); -inf where (s, a) was never tried.
        iterations: Bellman backups performed.
        residual: Largest change in V on the last backup.
    """

    values: FloatArray
    q: FloatArray
    iterations: int
    residual: float


def value_iteration(model: Model, gamma: float, tol: float, max_iter: int) -> Solution:
    """Solve the model by repeated Bellman backups, starting from V = 0.

    Only tried actions are considered. States with no tried action keep
    V = 0; that is reached only through next states nobody explored from.

    Args:
        model: The estimated MDP.
        gamma: Discount factor in (0, 1].
        tol: Stop when no value changes by more than this.
        max_iter: Stop after this many backups regardless.

    Returns:
        The values, Q table and convergence information.
    """
    has_action = model.observed_states
    values = np.zeros(model.n_states)
    q = np.full((model.n_states, model.n_actions), -np.inf)
    residual = np.inf
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        backup = model.reward + gamma * expected_next_value(model, values)
        q = np.where(model.observed, backup, -np.inf)
        new_values = np.where(has_action, q.max(axis=1, initial=-np.inf), 0.0)
        residual = float(np.abs(new_values - values).max())
        values = new_values
        if residual <= tol:
            break
    return Solution(values, q, iterations, residual)


def greedy_policy(model: Model, q: FloatArray) -> IntArray:
    """Return argmax_a Q(s, a) per state; -1 for states with no tried action."""
    policy: IntArray = np.asarray(q.argmax(axis=1), dtype=np.int64)
    policy[~model.observed_states] = -1
    return policy


def evaluate_policy(
    model: Model, action_probs: FloatArray, gamma: float, tol: float, max_iter: int
) -> FloatArray:
    """Iterative policy evaluation of a (possibly random) policy on the model.

    Args:
        model: The estimated MDP.
        action_probs: pi(a | s), shape (n_states, n_actions); must be zero on
            untried actions.
        gamma: Discount factor.
        tol: Stop when no value changes by more than this.
        max_iter: Stop after this many sweeps regardless.

    Returns:
        U^pi(s) under the model.
    """
    values = np.zeros(model.n_states)
    for _ in range(max_iter):
        q = model.reward + gamma * expected_next_value(model, values)
        new_values = (action_probs * q).sum(axis=1)
        residual = float(np.abs(new_values - values).max())
        values = new_values
        if residual <= tol:
            break
    return values


def one_hot(model: Model, policy: IntArray) -> FloatArray:
    """Turn a deterministic policy into pi(a | s); rows for unobserved states are 0."""
    probs = np.zeros((model.n_states, model.n_actions))
    rows = np.flatnonzero(policy >= 0)
    probs[rows, policy[rows]] = 1.0
    return probs


def uniform_random(model: Model) -> FloatArray:
    """Return pi(a | s) that picks uniformly among the tried actions of each state."""
    tried = model.observed.astype(np.float64)
    totals = tried.sum(axis=1, keepdims=True)
    return np.divide(tried, totals, out=np.zeros_like(tried), where=totals > 0)


# ---------------------------------------------------------------- unseen states


def fill_unobserved(policy: IntArray, problem: Problem) -> tuple[IntArray, str]:
    """Give every state with no data (``policy == -1``) an action.

    Args:
        policy: 0-based action per state, -1 where unknown.
        problem: Used for its grid, if any.

    Returns:
        The completed policy and a short description of the rule used.
    """
    known = np.flatnonzero(policy >= 0)
    unknown = np.flatnonzero(policy < 0)
    filled = policy.copy()
    if unknown.size == 0:
        return filled, "none needed"
    if problem.grid is not None:
        filled[unknown] = policy[nearest_states(unknown, known, problem.grid)]
        return filled, "action of nearest observed state on the grid"
    default = int(np.bincount(policy[known], minlength=problem.n_actions).argmax())
    filled[unknown] = default
    return filled, f"most common learned action ({default + 1})"


# ------------------------------------------------------------------------ CLI


@dataclass(frozen=True)
class SolveConfig:
    """Settings for one run.

    Attributes:
        gamma: Discount used for planning (defaults to the problem's).
        tol: Value-iteration stopping tolerance.
        max_iter: Value-iteration iteration cap.
    """

    gamma: float
    tol: float = 1e-6
    max_iter: int = 10_000

    def __post_init__(self) -> None:
        """Validate the settings."""
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        if self.tol <= 0.0:
            raise ValueError("tol must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")


@dataclass(frozen=True)
class RunResult:
    """Everything a run produces.

    Attributes:
        policy: 0-based action for every state.
        values: V(s) from value iteration (0 for unobserved states).
        summary: JSON-serialisable description of the run.
    """

    policy: IntArray
    values: FloatArray
    summary: dict[str, object]


def solve(data: Transitions, problem: Problem, config: SolveConfig) -> RunResult:
    """Estimate a model, solve it, and complete the policy for all states.

    Args:
        data: Logged transitions.
        problem: The problem definition.
        config: Planning settings.

    Returns:
        The policy, values and a summary (timings, convergence, and the
        model's estimate of the learned policy versus a random one).
    """
    start = time.perf_counter()
    model = estimate_model(data, problem)
    solution = value_iteration(model, config.gamma, config.tol, config.max_iter)
    learned = greedy_policy(model, solution.q)
    policy, fill_rule = fill_unobserved(learned, problem)
    runtime = time.perf_counter() - start

    # Model-based sanity check: how much better than random, under our own model?
    seen = model.observed_states
    random_values = evaluate_policy(
        model, uniform_random(model), config.gamma, config.tol, config.max_iter
    )
    greedy_values = evaluate_policy(
        model, one_hot(model, learned), config.gamma, config.tol, config.max_iter
    )
    summary: dict[str, object] = {
        "problem": asdict(problem),
        "settings": asdict(config),
        "samples": len(data),
        "observed_states": int(seen.sum()),
        "filled_states": int((learned < 0).sum()),
        "fill_rule": fill_rule,
        "iterations": solution.iterations,
        "final_residual": solution.residual,
        "converged": solution.residual <= config.tol,
        "runtime_seconds": round(runtime, 3),
        "model_mean_value_learned": float(greedy_values[seen].mean()),
        "model_mean_value_random": float(random_values[seen].mean()),
        "action_counts": np.bincount(policy, minlength=problem.n_actions).tolist(),
    }
    return RunResult(policy, solution.values, summary)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("problem", choices=sorted(PROBLEMS), help="which dataset this is")
    parser.add_argument("infile", type=Path, help="CSV of s,a,r,sp")
    parser.add_argument("outfile", type=Path, help="where to write the .policy file")
    parser.add_argument("--summary", type=Path, help="write a JSON run summary here")
    parser.add_argument("--values", type=Path, help="write V(s), one per line, here")
    parser.add_argument("--gamma", type=float, help="planning discount (default: the problem's)")
    parser.add_argument("--tol", type=float, default=1e-6, help="value-iteration tolerance")
    parser.add_argument("--max-iter", type=int, default=10_000, help="value-iteration cap")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline from the command line; return the exit code."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    problem = PROBLEMS[args.problem]
    try:
        config = SolveConfig(
            gamma=problem.gamma if args.gamma is None else args.gamma,
            tol=args.tol,
            max_iter=args.max_iter,
        )
        data = load_csv(args.infile, problem)
        result = solve(data, problem, config)
        write_policy(result.policy, args.outfile)
        if args.values is not None:
            args.values.write_text(
                "".join(f"{v:.6f}\n" for v in result.values.tolist()), encoding="utf-8"
            )
        if args.summary is not None:
            args.summary.write_text(json.dumps(result.summary, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    s = result.summary
    print(
        f"{args.problem}: {s['iterations']} iterations, {s['runtime_seconds']} s, "
        f"model value learned {s['model_mean_value_learned']:.2f} "
        f"vs random {s['model_mean_value_random']:.2f} -> {args.outfile}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
