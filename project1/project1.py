"""Bayesian network structure learning for AA228/CS238 Project 1.

Given a CSV of discrete variables, find a directed acyclic graph (DAG) that
maximises the Bayesian score with a uniform Dirichlet prior (every pseudo-count
alpha_ijk = 1), as in *Algorithms for Decision Making* section 5.1. The ln P(G)
term is omitted, as in the course's scoring.

Search strategy:
    1. K2 from many random variable orderings; keep the best graph.
    2. Steepest-ascent hill climbing over single-edge additions, removals and
       reversals, starting from the best K2 graph.
    3. Iterated local search: randomly perturb the best graph, hill-climb
       again, and keep the result whenever it improves.

Usage:
    python project1.py <infile>.csv <outfile>.gph [--summary out.json] [options]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx
import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
ParentSets = list[frozenset[int]]
"""A DAG as one parent set per variable: ``parents[i]`` holds the parents of ``i``."""

MAX_VALUE = 2**31 - 1
"""Largest value accepted in a CSV cell; keeps all count indices inside int64."""


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Dataset:
    """A table of discrete observations.

    Attributes:
        names: Variable names, in column order.
        values: Array of shape (n_samples, n_vars) with 0-based outcomes; column
            ``i`` takes values in ``0 .. cardinalities[i] - 1``.
        cardinalities: Number of outcomes r_i of each variable (its maximum value
            in the CSV, as the project specifies).
    """

    names: tuple[str, ...]
    values: IntArray
    cardinalities: IntArray

    @property
    def n_samples(self) -> int:
        """Number of rows."""
        return int(self.values.shape[0])

    @property
    def n_vars(self) -> int:
        """Number of variables (columns)."""
        return len(self.names)


def load_csv(path: str | Path) -> Dataset:
    """Read and validate a project CSV: a header row, then positive integers.

    Raises:
        ValueError: If the file is empty, the header is malformed, a row has the
            wrong length, or a value is not an integer in ``1 .. MAX_VALUE``.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path}: file is empty")
        names = tuple(name.strip() for name in header)
        if any(not name for name in names):
            raise ValueError(f"{path}: header has an empty variable name")
        if len(set(names)) != len(names):
            raise ValueError(f"{path}: header has duplicate variable names")

        rows: list[list[int]] = []
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(names):
                raise ValueError(
                    f"{path}:{line_no}: expected {len(names)} values, found {len(row)}"
                )
            try:
                parsed = [int(cell) for cell in row]
            except ValueError:
                raise ValueError(f"{path}:{line_no}: values must be integers") from None
            if min(parsed) < 1 or max(parsed) > MAX_VALUE:
                raise ValueError(f"{path}:{line_no}: values must be in 1..{MAX_VALUE}")
            rows.append(parsed)

    if not rows:
        raise ValueError(f"{path}: no data rows")
    raw = np.array(rows, dtype=np.int64)
    return Dataset(names=names, values=raw - 1, cardinalities=raw.max(axis=0))


def write_gph(dag: networkx.DiGraph, idx2names: Sequence[str], filename: str | Path) -> None:
    """Write the edges of ``dag`` as ``parent, child`` lines (course starter format)."""
    with open(filename, "w") as f:
        for edge in sorted(dag.edges()):
            f.write(f"{idx2names[edge[0]]}, {idx2names[edge[1]]}\n")


def read_gph(path: str | Path, names: Sequence[str]) -> ParentSets:
    """Read a ``.gph`` file into parent sets indexed like ``names``.

    Raises:
        ValueError: If a line is malformed or names an unknown variable.
    """
    index = {name: i for i, name in enumerate(names)}
    parents: list[set[int]] = [set() for _ in names]
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_no}: expected 'parent, child'")
            parent, child = fields
            for name in fields:
                if name not in index:
                    raise ValueError(f"{path}:{line_no}: unknown variable {name!r}")
            parents[index[child]].add(index[parent])
    return [frozenset(p) for p in parents]


def to_digraph(parents: ParentSets) -> networkx.DiGraph:
    """Convert parent sets to a networkx DiGraph on nodes ``0 .. n-1``."""
    dag = networkx.DiGraph()
    dag.add_nodes_from(range(len(parents)))
    dag.add_edges_from((p, child) for child, ps in enumerate(parents) for p in ps)
    return dag


# -------------------------------------------------------------------------- score


def _lgamma_range(start: int, length: int) -> FloatArray:
    """Return ``[lgamma(start), lgamma(start + 1), ..., lgamma(start + length - 1)]``."""
    return np.array([math.lgamma(start + m) for m in range(length)], dtype=np.float64)


class FamilyScorer:
    """Computes and caches the Bayesian score of each family (a child and its parents).

    The Bayesian score of a graph is a sum of one term per variable, so a
    change to one variable's parents only changes that variable's term. Caching
    terms by ``(child, parents)`` makes the search cheap.

    With alpha_ijk = 1 (so alpha_ij0 = r_i), the term for variable i is::

        sum_j [ lgamma(r_i) - lgamma(r_i + m_ij0) + sum_k lgamma(1 + m_ijk) ]

    where j ranges over parental configurations and m_ijk counts rows with
    parents in configuration j and variable i equal to k. Configurations that
    never occur contribute exactly zero, so only observed ones matter.
    """

    def __init__(self, data: Dataset, dense_limit: int = 1 << 20) -> None:
        """Prepare lookup tables for ``data``.

        Args:
            data: The observations to score against.
            dense_limit: Use a dense count table when (configurations x r_i) is at
                most this; otherwise count only observed configurations.
        """
        self.data = data
        self.dense_limit = dense_limit
        self.evaluations = 0
        self._cache: dict[tuple[int, frozenset[int]], float] = {}
        n = data.n_samples
        self._lg_one_plus = _lgamma_range(1, n + 1)  # lgamma(1 + m)
        self._lg_r_plus = [_lgamma_range(int(r), n + 1) for r in data.cardinalities]

    def family(self, child: int, parents: frozenset[int]) -> float:
        """Score contribution of ``child`` given ``parents`` (cached)."""
        key = (child, parents)
        score = self._cache.get(key)
        if score is None:
            score = self._compute(child, parents)
            self._cache[key] = score
        return score

    def total(self, parents: ParentSets) -> float:
        """Bayesian score of the whole graph."""
        return sum(self.family(i, ps) for i, ps in enumerate(parents))

    def _parent_configs(self, parents: frozenset[int]) -> tuple[IntArray, int]:
        """Index each row's parental configuration; return (indices, number of configs)."""
        x, cards = self.data.values, self.data.cardinalities
        config = np.zeros(self.data.n_samples, dtype=np.int64)
        n_configs = 1
        for p in sorted(parents):
            config = config * cards[p] + x[:, p]
            n_configs *= int(cards[p])
            if n_configs > MAX_VALUE:
                # Relabel to observed configurations only, so indices stay small.
                _, config = np.unique(config, return_inverse=True)
                n_configs = int(config.max()) + 1
        return config, n_configs

    def _compute(self, child: int, parents: frozenset[int]) -> float:
        self.evaluations += 1
        r = int(self.data.cardinalities[child])
        config, n_configs = self._parent_configs(parents)
        joint = config * r + self.data.values[:, child]
        if n_configs * r <= self.dense_limit:
            m_ijk = np.bincount(joint, minlength=n_configs * r)
            m_ij0 = m_ijk.reshape(n_configs, r).sum(axis=1)
        else:
            m_ijk = np.unique(joint, return_counts=True)[1]
            m_ij0 = np.unique(config, return_counts=True)[1]
        lg_r = self._lg_r_plus[child]
        return float(len(m_ij0) * lg_r[0] - lg_r[m_ij0].sum() + self._lg_one_plus[m_ijk].sum())


def bayesian_score(data: Dataset, parents: ParentSets) -> float:
    """Bayesian score of the DAG ``parents`` on ``data`` (uniform Dirichlet prior)."""
    return FamilyScorer(data).total(parents)


# ------------------------------------------------------------------------- search


def is_ancestor(
    parents: ParentSets, ancestor: int, node: int, ignore: tuple[int, int] | None = None
) -> bool:
    """True if there is a directed path ``ancestor -> ... -> node``.

    Args:
        parents: The graph.
        ancestor: Start of the path.
        node: End of the path.
        ignore: An edge ``(parent, child)`` to treat as absent.
    """
    stack, seen = [node], {node}
    while stack:
        current = stack.pop()
        for p in parents[current]:
            if ignore == (p, current):
                continue
            if p == ancestor:
                return True
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return False


def k2(scorer: FamilyScorer, order: Sequence[int], max_parents: int) -> ParentSets:
    """K2: each variable greedily adds parents from earlier in ``order``.

    The result is acyclic by construction, because edges only point forward in
    ``order``.
    """
    parents: ParentSets = [frozenset() for _ in order]
    for position, child in enumerate(order):
        chosen: frozenset[int] = frozenset()
        current = scorer.family(child, chosen)
        while len(chosen) < max_parents:
            best_score, best_parent = current, None
            for candidate in order[:position]:
                if candidate in chosen:
                    continue
                score = scorer.family(child, chosen | {candidate})
                if score > best_score:
                    best_score, best_parent = score, candidate
            if best_parent is None:
                break
            chosen, current = chosen | {best_parent}, best_score
        parents[child] = chosen
    return parents


Move = tuple[str, int, int]
"""``(kind, u, v)`` acting on edge ``u -> v``; kind is "add", "remove" or "reverse"."""


def apply_move(parents: ParentSets, move: Move) -> ParentSets:
    """Return a copy of ``parents`` with ``move`` applied."""
    kind, u, v = move
    result = list(parents)
    if kind == "add":
        result[v] = parents[v] | {u}
    elif kind == "remove":
        result[v] = parents[v] - {u}
    elif kind == "reverse":
        result[v] = parents[v] - {u}
        result[u] = parents[u] | {v}
    else:
        raise ValueError(f"unknown move kind {kind!r}")
    return result


def is_legal(parents: ParentSets, move: Move, max_parents: int) -> bool:
    """True if ``move`` keeps the graph acyclic and within ``max_parents``."""
    kind, u, v = move
    if u == v:
        return False
    if kind == "add":
        return (
            u not in parents[v] and len(parents[v]) < max_parents and not is_ancestor(parents, v, u)
        )
    if kind == "remove":
        return u in parents[v]
    if kind == "reverse":
        return (
            u in parents[v]
            and len(parents[u]) < max_parents
            and not is_ancestor(parents, u, v, ignore=(u, v))
        )
    return False


def hill_climb(
    scorer: FamilyScorer, parents: ParentSets, max_parents: int, tol: float = 1e-9
) -> ParentSets:
    """Steepest-ascent hill climbing over single-edge moves until no move helps.

    Every candidate's score change is computed from cached family scores; the
    (more expensive) cycle check runs only for a candidate that would beat the
    best move found so far.
    """
    parents = list(parents)
    n = len(parents)
    family = [scorer.family(i, parents[i]) for i in range(n)]
    while True:
        best_delta, best_move = tol, None
        for v in range(n):
            for u in range(n):
                if u == v:
                    continue
                if u in parents[v]:
                    removed = scorer.family(v, parents[v] - {u}) - family[v]
                    if removed > best_delta:
                        best_delta, best_move = removed, ("remove", u, v)
                    if len(parents[u]) < max_parents:
                        reversed_ = removed + scorer.family(u, parents[u] | {v}) - family[u]
                        move = ("reverse", u, v)
                        if reversed_ > best_delta and is_legal(parents, move, max_parents):
                            best_delta, best_move = reversed_, move
                elif len(parents[v]) < max_parents:
                    added = scorer.family(v, parents[v] | {u}) - family[v]
                    move = ("add", u, v)
                    if added > best_delta and is_legal(parents, move, max_parents):
                        best_delta, best_move = added, move
        if best_move is None:
            return parents
        parents = apply_move(parents, best_move)
        _, u, v = best_move
        family[u] = scorer.family(u, parents[u])
        family[v] = scorer.family(v, parents[v])


def perturb(parents: ParentSets, n_moves: int, max_parents: int, rng: random.Random) -> ParentSets:
    """Apply ``n_moves`` random legal moves (used to escape local optima)."""
    n = len(parents)
    applied, attempts = 0, 0
    while applied < n_moves and attempts < 100 * n_moves:
        attempts += 1
        u, v = rng.sample(range(n), 2)
        kind = "add" if u not in parents[v] else rng.choice(["remove", "reverse"])
        move = (kind, u, v)
        if is_legal(parents, move, max_parents):
            parents = apply_move(parents, move)
            applied += 1
    return parents


@dataclass(frozen=True)
class SearchConfig:
    """Search settings; iteration counts (not wall-clock limits) keep runs reproducible."""

    max_parents: int = 4
    k2_orderings: int = 50
    perturbations: int = 50
    perturbation_moves: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject settings that make no sense."""
        if self.max_parents < 0:
            raise ValueError("max_parents must be >= 0")
        if self.k2_orderings < 1:
            raise ValueError("k2_orderings must be >= 1")
        if self.perturbations < 0 or self.perturbation_moves < 1:
            raise ValueError("perturbations must be >= 0 and perturbation_moves >= 1")


@dataclass(frozen=True)
class SearchResult:
    """Outcome of :func:`learn_structure`, with the score after each phase."""

    parents: ParentSets
    score: float
    k2_score: float
    hill_climb_score: float
    empty_graph_score: float
    family_evaluations: int


def learn_structure(data: Dataset, config: SearchConfig) -> SearchResult:
    """Find a high-scoring DAG for ``data`` (K2 restarts, then iterated local search)."""
    rng = random.Random(config.seed)
    scorer = FamilyScorer(data)
    n = data.n_vars
    empty: ParentSets = [frozenset() for _ in range(n)]

    # Phase 1: K2 from random orderings.
    best, best_score = empty, scorer.total(empty)
    for _ in range(config.k2_orderings):
        order = rng.sample(range(n), n)
        candidate = k2(scorer, order, config.max_parents)
        score = scorer.total(candidate)
        if score > best_score:
            best, best_score = candidate, score
    k2_score = best_score

    # Phase 2: hill climbing from the best K2 graph.
    best = hill_climb(scorer, best, config.max_parents)
    best_score = scorer.total(best)
    hill_climb_score = best_score

    # Phase 3: iterated local search (perturb, re-climb, keep improvements).
    for _ in range(config.perturbations):
        kicked = perturb(best, config.perturbation_moves, config.max_parents, rng)
        candidate = hill_climb(scorer, kicked, config.max_parents)
        score = scorer.total(candidate)
        if score > best_score:
            best, best_score = candidate, score

    return SearchResult(
        parents=best,
        score=best_score,
        k2_score=k2_score,
        hill_climb_score=hill_climb_score,
        empty_graph_score=scorer.total(empty),
        family_evaluations=scorer.evaluations,
    )


# ---------------------------------------------------------------------------- CLI


def compute(
    infile: str | Path, outfile: str | Path, config: SearchConfig | None = None
) -> dict[str, object]:
    """Learn a structure for ``infile``, write it to ``outfile``, and return a summary.

    The reported runtime covers the search only, not reading data or writing
    output, as the project asks.
    """
    config = config or SearchConfig()
    data = load_csv(infile)

    start = time.perf_counter()
    result = learn_structure(data, config)
    runtime = time.perf_counter() - start

    dag = to_digraph(result.parents)
    if not networkx.is_directed_acyclic_graph(dag):  # independent check of the search
        raise RuntimeError("search produced a cyclic graph")
    write_gph(dag, data.names, outfile)

    return {
        "dataset": Path(infile).name,
        "n_vars": data.n_vars,
        "n_samples": data.n_samples,
        "n_edges": dag.number_of_edges(),
        "score": result.score,
        "k2_score": result.k2_score,
        "hill_climb_score": result.hill_climb_score,
        "empty_graph_score": result.empty_graph_score,
        "family_evaluations": result.family_evaluations,
        "runtime_seconds": round(runtime, 2),
        "config": asdict(config),
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    defaults = SearchConfig()
    parser = argparse.ArgumentParser(
        description="Learn a Bayesian network structure from a CSV and write it as .gph."
    )
    parser.add_argument("infile", type=Path, help="input CSV")
    parser.add_argument("outfile", type=Path, help="output .gph")
    parser.add_argument("--summary", type=Path, help="also write a JSON summary here")
    parser.add_argument("--max-parents", type=int, default=defaults.max_parents)
    parser.add_argument("--k2-orderings", type=int, default=defaults.k2_orderings)
    parser.add_argument("--perturbations", type=int, default=defaults.perturbations)
    parser.add_argument("--perturbation-moves", type=int, default=defaults.perturbation_moves)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        config = SearchConfig(
            max_parents=args.max_parents,
            k2_orderings=args.k2_orderings,
            perturbations=args.perturbations,
            perturbation_moves=args.perturbation_moves,
            seed=args.seed,
        )
        summary = compute(args.infile, args.outfile, config)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.summary is not None:
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"{summary['dataset']}: score {summary['score']:.2f} "
        f"({summary['n_edges']} edges) in {summary['runtime_seconds']:.2f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
