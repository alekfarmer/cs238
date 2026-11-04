"""Tests for CSV loading, .policy I/O and filling unobserved states."""

from pathlib import Path

import numpy as np
import pytest

from project2 import (
    PROBLEMS,
    Problem,
    fill_unobserved,
    load_csv,
    nearest_states,
    read_policy,
    write_policy,
)

TINY = Problem("tiny", n_states=3, n_actions=2, gamma=0.9)


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "data.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_csv_converts_to_zero_based(tmp_path: Path) -> None:
    data = load_csv(write(tmp_path, "s,a,r,sp\n1,2,5,3\n3,1,-1,1\n"), TINY)
    assert data.s.tolist() == [0, 2]
    assert data.a.tolist() == [1, 0]
    assert data.r.tolist() == [5.0, -1.0]
    assert data.sp.tolist() == [2, 0]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("x,a,r,sp\n1,1,0,1\n", "header"),
        ("s,a,r,sp\n1,1,zero,1\n", "integer"),
        ("s,a,r,sp\n4,1,0,1\n", "s must lie"),
        ("s,a,r,sp\n1,3,0,1\n", "a must lie"),
        ("s,a,r,sp\n1,1,0,0\n", "sp must lie"),
    ],
)
def test_load_csv_rejects_bad_input(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_csv(write(tmp_path, text), TINY)


def test_policy_round_trip_is_one_based_with_one_line_per_state(tmp_path: Path) -> None:
    path = tmp_path / "tiny.policy"
    write_policy(np.array([1, 0, 1]), path)
    assert path.read_text(encoding="utf-8") == "2\n1\n2\n"
    assert read_policy(path, TINY).tolist() == [1, 0, 1]


def test_read_policy_rejects_wrong_length(tmp_path: Path) -> None:
    path = tmp_path / "short.policy"
    path.write_text("1\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 3 lines"):
        read_policy(path, TINY)


def test_nearest_states_uses_scaled_grid_distance() -> None:
    # On a 10-wide, 2-high grid, one row step is half the height (0.5) while one
    # column step is 0.1, so state 1 (col 1, row 0) is nearer to 4 (col 4) than to
    # 11 (col 1, row 1).
    assert nearest_states(np.array([1]), np.array([4, 11]), (10, 2)).tolist() == [4]


def test_fill_unobserved_on_grid_copies_nearest_action() -> None:
    problem = Problem("g", n_states=5, n_actions=3, gamma=0.9, grid=(5, 1))
    filled, rule = fill_unobserved(np.array([0, -1, -1, -1, 2]), problem)
    assert filled.tolist() == [0, 0, 0, 2, 2]  # state 2 ties -> lower-numbered state 0
    assert "nearest" in rule


def test_fill_unobserved_without_grid_uses_most_common_action() -> None:
    filled, _ = fill_unobserved(np.array([1, -1, 1]), TINY)
    assert filled.tolist() == [1, 1, 1]


def test_course_problem_sizes() -> None:
    sizes = {k: (p.n_states, p.n_actions, p.gamma) for k, p in PROBLEMS.items()}
    assert sizes == {
        "small": (100, 4, 0.95),
        "medium": (50_000, 7, 1.0),
        "large": (302_020, 9, 0.95),
    }
