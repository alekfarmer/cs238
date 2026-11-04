# Project 1: how it works

The goal: given a table of discrete variables, find the directed acyclic graph
(Bayesian network structure) that best explains the data, measured by the
**Bayesian score**. All code is in `project1.py`.

## 1. The Bayesian score

For a graph G and data D, the score is ln P(D | G) with a uniform Dirichlet
prior over every conditional distribution (all pseudo-counts α = 1). For
variable i with r_i outcomes and parents in configuration j:

- m_ijk = number of rows where the parents are in configuration j and i = k
- m_ij0 = Σ_k m_ijk

    score(G) = Σ_i Σ_j [ lnΓ(r_i) − lnΓ(r_i + m_ij0) + Σ_k lnΓ(1 + m_ijk) ]

Things worth noticing:

- **It decomposes**: the score is a sum of one term per variable, and each
  term depends only on that variable and its parents. Changing one edge only
  changes one or two terms. `FamilyScorer` caches each term by
  `(child, parents)`, so most search steps are dictionary lookups.
- **It penalises complexity by itself.** Adding a parent multiplies the
  number of configurations j, which spreads the counts thinner. The Γ terms
  reward a parent only if the data really become more predictable.
- **Unseen configurations contribute 0**, so we only count rows (`np.bincount`
  for small tables, `np.unique` when there are too many configurations).
- Checked against the course's `example/example.score` (−132.5769) in
  `tests/test_score.py`.

## 2. The search

There are more DAGs than can ever be enumerated (about 10^18 for just 10
variables), so we search heuristically in three phases:

1. **K2 with random orderings.** Pick a variable order. Each variable
   greedily adds whichever earlier variable improves its term most, until
   nothing helps or it reaches `max_parents`. Edges only point forward in
   the order, so the result is always acyclic. Repeat for many random orders
   and keep the best.
2. **Hill climbing.** From the best K2 graph, consider every single-edge
   change (add, remove, or reverse) and make the best one. Repeat until no
   change helps. Reversals let it fix edges K2 pointed the wrong way, and it
   is not limited to one ordering. A change is rejected if it would create a
   cycle. That check walks ancestors, and it only runs for a candidate that
   would beat the current best move.
3. **Iterated local search.** Hill climbing stops at a local optimum. To
   escape it, randomly apply a few legal edge changes to the best graph,
   hill-climb again, and keep the result if it scores higher. Repeat.

## 3. Results

From `make learn-all` (seed 0; runtimes are search only, measured in Docker
on a laptop, and vary by tens of percent between runs):

| Dataset | Vars | Rows | Empty graph | After K2 | After hill climb | Final | Edges | Runtime |
|---|---|---|---|---|---|---|---|---|
| small (Titanic) | 8 | 889 | −4166.23 | −3797.88 | −3794.86 | **−3794.86** | 14 | 0.07 s |
| medium (Wine) | 13 | 6497 | −117669.00 | −96654.92 | −96473.49 | **−96285.22** | 29 | 1.12 s |
| large (secret) | 50 | 10000 | −524395.40 | −469823.39 | −455331.30 | **−450057.13** | 142 | 22.81 s |

The plots are in `output/*.png`, and the per-phase numbers are in `output/*.json`.

What we learned from experimenting:

- **The parent limit matters most.** On large, `max_parents` 4 → 6 → 8
  gave −466,565 → −459,544 → −453,828 (with the same other settings). Above
  8, nothing changed. The score's built-in complexity penalty already
  stops parent sets from growing, so a tight cap only hurts.
- **Each phase pays off more as the problem grows.** On small, K2 is
  nearly optimal. On large, hill climbing adds about 14,500 and iterated
  local search about 5,300 more.
- Results are reproducible: the same seed and settings give the same graph.

## 4. Try it

    make verify                                   # lint, types, tests
    make learn-small                              # writes output/small.{gph,json,png}
    make learn-small SEARCH_small="--max-parents 2 --k2-orderings 10 --perturbations 0"

Ideas to explore next: simulated annealing instead of hill climbing, searching
over orderings rather than graphs (K2 inside an ordering search), or tabu
search.

## A note on course policy

The course requires students to write and type their own code. This
implementation is for understanding while auditing, and should not be
submitted as a student's own work.
