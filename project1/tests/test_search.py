"""Tests for the structure search pipeline."""

import json
import random
from pathlib import Path

import networkx
import numpy as np
import pytest

from project1 import (
    Dataset,
    FamilyScorer,
    SearchConfig,
    hill_climb,
    is_legal,
    k2,
    learn_structure,
    main,
    perturb,
    read_gph,
    to_digraph,
)

# Planted network: a -> c <- b, c -> d (a v-structure, so every edge is identifiable).
PLANTED = [frozenset(), frozenset(), frozenset({0, 1}), frozenset({2})]


def sample_planted(n: int = 3000, seed: int = 1) -> Dataset:
    rng = np.random.default_rng(seed)
    a = rng.random(n) < 0.5
    b = rng.random(n) < 0.4
    c = rng.random(n) < 0.1 + 0.4 * a + 0.4 * b
    d = np.where(c, rng.random(n) < 0.85, rng.random(n) < 0.2)
    values = np.column_stack([a, b, c, d]).astype(np.int64)
    return Dataset(("a", "b", "c", "d"), values, np.full(4, 2, dtype=np.int64))


def random_dataset(n_vars: int = 8, n: int = 300, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    cards = rng.integers(2, 4, size=n_vars)
    values = np.column_stack([rng.integers(0, r, size=n) for r in cards]).astype(np.int64)
    # Add some dependence so the search has edges to find.
    values[:, 1] = values[:, 0] % cards[1]
    return Dataset(tuple(f"v{i}" for i in range(n_vars)), values, cards.astype(np.int64))


def test_learn_structure_recovers_planted_network() -> None:
    data = sample_planted()
    result = learn_structure(data, SearchConfig(k2_orderings=5, perturbations=5))
    planted_score = FamilyScorer(data).total(PLANTED)
    assert result.parents == PLANTED
    assert result.score == pytest.approx(planted_score)


def test_learn_structure_phases_never_lose_score() -> None:
    data = random_dataset()
    result = learn_structure(data, SearchConfig(k2_orderings=3, perturbations=3))
    assert result.empty_graph_score <= result.k2_score <= result.hill_climb_score <= result.score
    assert result.score > result.empty_graph_score


def test_k2_with_true_order_finds_true_parents() -> None:
    data = sample_planted()
    assert k2(FamilyScorer(data), [0, 1, 2, 3], max_parents=2) == PLANTED


def test_hill_climb_from_empty_graph_matches_planted_score() -> None:
    data = sample_planted()
    scorer = FamilyScorer(data)
    learned = hill_climb(scorer, [frozenset()] * 4, max_parents=3)
    assert scorer.total(learned) >= scorer.total(PLANTED) - 1e-9


@pytest.mark.parametrize("max_parents", [1, 2, 3])
def test_search_output_is_acyclic_and_respects_max_parents(max_parents: int) -> None:
    data = random_dataset(seed=max_parents)
    result = learn_structure(
        data, SearchConfig(max_parents=max_parents, k2_orderings=2, perturbations=4)
    )
    assert networkx.is_directed_acyclic_graph(to_digraph(result.parents))
    assert max(len(p) for p in result.parents) <= max_parents


def test_perturb_keeps_graph_acyclic() -> None:
    rng = random.Random(0)
    parents: list[frozenset[int]] = [frozenset() for _ in range(10)]
    for _ in range(50):
        parents = perturb(parents, 5, 3, rng)
        assert networkx.is_directed_acyclic_graph(to_digraph(parents))


def test_is_legal_rejects_cycles() -> None:
    chain = [frozenset(), frozenset({0}), frozenset({1})]  # 0 -> 1 -> 2
    assert not is_legal(chain, ("add", 2, 0), 3)
    assert not is_legal(chain, ("add", 1, 0), 3)
    assert is_legal(chain, ("add", 0, 2), 3)
    assert is_legal(chain, ("reverse", 1, 2), 3)
    triangle = [frozenset(), frozenset({0}), frozenset({0, 1})]  # 0->1, 0->2, 1->2
    assert not is_legal(triangle, ("reverse", 0, 2), 3)  # 2->0 would close 0->1->2->0


def test_search_config_validation() -> None:
    with pytest.raises(ValueError):
        SearchConfig(k2_orderings=0)
    with pytest.raises(ValueError):
        SearchConfig(max_parents=-1)


def test_cli_writes_graph_and_summary(tmp_path: Path) -> None:
    csv = tmp_path / "d.csv"
    data = sample_planted(n=500)
    rows = "\n".join(",".join(str(v + 1) for v in row) for row in data.values)
    csv.write_text("a,b,c,d\n" + rows + "\n")
    gph, summary = tmp_path / "d.gph", tmp_path / "d.json"
    code = main([str(csv), str(gph), "--summary", str(summary), "--k2-orderings", "2"])
    assert code == 0
    assert networkx.is_directed_acyclic_graph(to_digraph(read_gph(gph, data.names)))
    info = json.loads(summary.read_text())
    assert info["score"] > info["empty_graph_score"]
    assert info["runtime_seconds"] >= 0


def test_cli_reports_bad_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv = tmp_path / "bad.csv"
    csv.write_text("a,b\n1,x\n")
    assert main([str(csv), str(tmp_path / "out.gph")]) == 2
    assert "integers" in capsys.readouterr().err
