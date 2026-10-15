"""Tests for data loading and the Bayesian score."""

import math
from pathlib import Path

import numpy as np
import pytest

from project1 import Dataset, FamilyScorer, bayesian_score, load_csv, read_gph

EXAMPLE = Path(__file__).resolve().parents[1] / "example"


def make_dataset(columns: dict[str, list[int]]) -> Dataset:
    raw = np.array(list(columns.values()), dtype=np.int64).T
    return Dataset(tuple(columns), raw - 1, raw.max(axis=0))


def test_example_score_matches_course_reference() -> None:
    data = load_csv(EXAMPLE / "example.csv")
    graph = read_gph(EXAMPLE / "example.gph", data.names)
    expected = float((EXAMPLE / "example.score").read_text())
    assert bayesian_score(data, graph) == pytest.approx(expected, abs=1e-8)


def test_single_variable_closed_form() -> None:
    # r = 2, counts (2, 1): lgamma(2) - lgamma(2 + 3) + lgamma(1 + 2) + lgamma(1 + 1)
    data = make_dataset({"a": [1, 1, 2]})
    expected = math.lgamma(2) - math.lgamma(5) + math.lgamma(3) + math.lgamma(2)
    assert bayesian_score(data, [frozenset()]) == pytest.approx(expected)
    assert expected == pytest.approx(math.log(1 / 12))


def test_unobserved_parent_configurations_contribute_nothing() -> None:
    # b has 3 outcomes but value 2 never appears; r_a = 2.
    data = make_dataset({"a": [1, 2, 2, 1], "b": [1, 1, 3, 3]})
    observed = 0.0
    for a_values in ([1, 2], [2, 1]):  # a given b=1, a given b=3
        counts = [a_values.count(k) for k in (1, 2)]
        observed += math.lgamma(2) - math.lgamma(2 + sum(counts))
        observed += sum(math.lgamma(1 + c) for c in counts)
    scorer = FamilyScorer(data)
    assert scorer.family(0, frozenset({1})) == pytest.approx(observed)


def test_dense_and_sparse_counting_agree() -> None:
    data = load_csv(EXAMPLE / "example.csv")
    dense, sparse = FamilyScorer(data), FamilyScorer(data, dense_limit=0)
    for child in range(data.n_vars):
        for parents in (frozenset(), frozenset({(child + 1) % 6, (child + 2) % 6})):
            assert dense.family(child, parents) == pytest.approx(sparse.family(child, parents))


def test_family_scores_are_cached() -> None:
    data = load_csv(EXAMPLE / "example.csv")
    scorer = FamilyScorer(data)
    scorer.family(0, frozenset({1}))
    scorer.family(0, frozenset({1}))
    assert scorer.evaluations == 1


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "empty"),
        ("a,b\n", "no data rows"),
        ("a,a\n1,2\n", "duplicate"),
        ("a,\n1,2\n", "empty variable name"),
        ("a,b\n1\n", "expected 2 values"),
        ("a,b\n1,x\n", "integers"),
        ("a,b\n0,1\n", "values must be in"),
    ],
)
def test_load_csv_rejects_bad_input(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        load_csv(path)


def test_load_csv_handles_quoted_header(tmp_path: Path) -> None:
    path = tmp_path / "q.csv"
    path.write_text('"x","y"\n1,3\n2,1\n')
    data = load_csv(path)
    assert data.names == ("x", "y")
    assert data.cardinalities.tolist() == [2, 3]
    assert data.values.tolist() == [[0, 2], [1, 0]]
