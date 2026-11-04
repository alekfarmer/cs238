"""Tests for model estimation, value iteration and the full solve pipeline."""

from collections.abc import Sequence

import numpy as np
import pytest

from project2 import (
    Problem,
    SolveConfig,
    Transitions,
    estimate_model,
    evaluate_policy,
    greedy_policy,
    one_hot,
    solve,
    uniform_random,
    value_iteration,
)


def make_transitions(rows: Sequence[tuple[float, float, float, float]]) -> Transitions:
    """Build transitions from 1-based (s, a, r, sp) rows, as in the CSVs."""
    arr = np.array(rows, dtype=np.float64)
    return Transitions(
        s=arr[:, 0].astype(np.int64) - 1,
        a=arr[:, 1].astype(np.int64) - 1,
        r=arr[:, 2],
        sp=arr[:, 3].astype(np.int64) - 1,
    )


# Two states, two actions, deterministic:
#   state 1: action 1 stays (reward 0); action 2 moves to state 2 (reward 0)
#   state 2: action 1 stays (reward 1); action 2 moves to state 1 (reward 0)
# With gamma = 0.9 the optimum is V(2) = 1 / (1 - 0.9) = 10, V(1) = 0.9 * 10 = 9,
# policy: state 1 -> action 2, state 2 -> action 1.
CHAIN = Problem("chain", n_states=2, n_actions=2, gamma=0.9)
CHAIN_ROWS = [(1, 1, 0, 1), (1, 2, 0, 2), (2, 1, 1, 2), (2, 2, 0, 1)]


def test_value_iteration_matches_closed_form() -> None:
    model = estimate_model(make_transitions(CHAIN_ROWS), CHAIN)
    solution = value_iteration(model, gamma=0.9, tol=1e-10, max_iter=10_000)
    np.testing.assert_allclose(solution.values, [9.0, 10.0], atol=1e-8)
    assert greedy_policy(model, solution.q).tolist() == [1, 0]
    assert solution.residual <= 1e-10


def test_maximum_likelihood_probabilities_and_rewards() -> None:
    # (1, a=1) goes to state 1 three times and to state 2 once; rewards average to 2.
    rows = [(1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 5, 2)]
    model = estimate_model(make_transitions(rows), CHAIN)
    probs = {int(sp): float(p) for sp, p in zip(model.edge_sp, model.edge_prob, strict=True)}
    assert probs == pytest.approx({0: 0.75, 1: 0.25})
    assert model.reward[0, 0] == pytest.approx(2.0)
    assert model.observed.tolist() == [[True, False], [False, False]]


def test_stochastic_value_matches_linear_solve() -> None:
    # One state, one action: reward 1, stays with p = 0.5, else goes to an unexplored
    # state (value 0). V = 1 + 0.9 * 0.5 * V  =>  V = 1 / 0.55.
    rows = [(1, 1, 1, 1), (1, 1, 1, 2)]
    model = estimate_model(make_transitions(rows), CHAIN)
    solution = value_iteration(model, gamma=0.9, tol=1e-12, max_iter=10_000)
    assert solution.values[0] == pytest.approx(1 / 0.55)
    assert solution.values[1] == 0.0


def test_goal_reward_ends_the_episode() -> None:
    # Undiscounted: state 1 action 1 earns 100 and the episode ends (the logged next
    # state is a restart back in state 1); action 2 loops at cost 1. Without the
    # terminal rule, V(1) would grow by 100 every backup and never converge.
    problem = Problem("goal", n_states=2, n_actions=2, gamma=1.0, goal_reward=50.0)
    rows = [(1, 1, 100, 1), (1, 2, -1, 1)]
    model = estimate_model(make_transitions(rows), problem)
    solution = value_iteration(model, gamma=1.0, tol=1e-9, max_iter=1000)
    assert solution.values[0] == pytest.approx(100.0)
    assert solution.iterations < 1000


def test_policy_evaluation_agrees_with_value_iteration() -> None:
    model = estimate_model(make_transitions(CHAIN_ROWS), CHAIN)
    solution = value_iteration(model, gamma=0.9, tol=1e-10, max_iter=10_000)
    policy = greedy_policy(model, solution.q)
    values = evaluate_policy(model, one_hot(model, policy), 0.9, 1e-10, 10_000)
    np.testing.assert_allclose(values, solution.values, atol=1e-8)
    # Random is strictly worse in both states.
    random_values = evaluate_policy(model, uniform_random(model), 0.9, 1e-10, 10_000)
    assert (random_values < values - 1).all()


def test_solve_beats_random_on_a_noisy_corridor() -> None:
    # A 1x5 corridor with reward 10 for acting in the rightmost cell. Action 2 moves
    # right 80% of the time; action 1 moves left. The learned policy must go right.
    rng = np.random.default_rng(0)
    problem = Problem("corridor", n_states=5, n_actions=2, gamma=0.95, grid=(5, 1))
    rows = []
    for _ in range(2000):
        s, a = int(rng.integers(1, 6)), int(rng.integers(1, 3))
        step = 1 if a == 2 else -1
        if rng.random() < 0.2:
            step = -step
        rows.append((s, a, 10.0 if s == 5 else 0.0, min(max(s + step, 1), 5)))
    result = solve(make_transitions(rows), problem, SolveConfig(gamma=0.95))
    assert result.policy[:4].tolist() == [1, 1, 1, 1]
    learned = result.summary["model_mean_value_learned"]
    random = result.summary["model_mean_value_random"]
    assert isinstance(learned, float) and isinstance(random, float)
    assert learned > random


def test_config_rejects_bad_gamma() -> None:
    with pytest.raises(ValueError, match="gamma"):
        SolveConfig(gamma=1.5)
