# Architecture

## What this is

Course data and starter code for AA228/CS238 (from `sisl/AA228-CS238-Student`),
plus implementations of Project 1 (Bayesian network structure learning) and
Project 2 (offline reinforcement learning). Both are **command-line
pipelines**, not apps: there is no server, API or frontend, so the AGENTS.md
invariants about those (backend computes results, `127.0.0.1` publishing) have
nothing to apply to. Computation and drawing are still separated:
`project1.py` and `project2.py` compute, while `plot_graph.py` and
`plot_policy.py` only draw what those wrote to disk.

## Layout

    Makefile                 every command (build, lint, test, verify, learn-*, policy-*, plot-*, push, pr)
    docker/
      Dockerfile             images: runtime -> dev (adds test tools); tools (GitHub CLI)
      requirements.txt       pinned runtime deps: numpy, networkx, matplotlib
      requirements-dev.txt   pinned dev deps: pytest, ruff, mypy
    pyproject.toml           pytest / ruff / mypy configuration for both projects
    project1/
      project1.py            the whole algorithm: loading, score, search, .gph I/O, CLI
      plot_graph.py          .gph -> PNG (layered layout); no scoring
      tests/                 pytest suite (score vs. course reference, search, I/O)
      output/                learned results: <name>.gph, <name>.json (scores, runtime), <name>.png
      data/, example/        course-provided, unchanged
      project1.jl            course Julia starter, unchanged
      HOW_IT_WORKS.md        explanation of the method and results
    project2/
      project2.py            the whole pipeline: loading, model estimate, value iteration,
                             filling unseen states, .policy I/O, CLI
      plot_policy.py         .policy (+ .values) -> PNG for grid problems; no computation
      tests/                 pytest suite (closed-form MDPs, terminal rule, fill, I/O)
      output/                <name>.policy, <name>.json (convergence, runtime, model values),
                             small.values, small.png, medium.png
      data/                  course-provided, unchanged
      HOW_IT_WORKS.md        explanation of the method, the datasets and results

## Data flow

Project 1:

    data/<name>.csv --load_csv--> Dataset --learn_structure--> parent sets
        --write_gph--> output/<name>.gph --plot_graph.py--> output/<name>.png
                     \-> output/<name>.json (scores per phase, runtime, settings)

Project 2:

    data/<name>.csv --load_csv--> Transitions --estimate_model--> Model (sparse T, R)
        --value_iteration--> Q --greedy_policy--> policy (-1 where unseen)
        --fill_unobserved--> output/<name>.policy --plot_policy.py--> output/<name>.png
                           \-> output/<name>.json, output/<name>.values (V, optional)

## Design decisions

- **One module per project.** The course starter for Project 1 asks for
  `project1.py`; Project 2 mirrors it. Each is organised in sections and
  would be split into a package only if it grew substantially.
- **Project 1: graph = list of parent sets** (`list[frozenset[int]]`). The
  score decomposes by family, so parent sets are the natural unit and make cache
  keys. networkx is used only at the edges (final DAG check, writing, plotting).
- **Project 1: family-score cache** keyed by `(child, parents)`; all search
  phases share one `FamilyScorer`.
- **Project 1: reproducible search.** It uses a fixed seed and iteration
  counts rather than a time budget, so the same settings give the same graph
  on any machine. Runtime is measured and reported, not used as a stopping
  rule.
- **Project 2: model-based, not model-free.** Planning on the
  maximum-likelihood model (certainty equivalence) uses each sample exactly
  once and has no learning rate. Q-learning over the same fixed batch would
  converge to the same answer, more slowly.
- **Project 2: sparse model with dense value arrays.** T is a list of
  `(s·A + a, sp, prob, continues)` edges, and a Bellman backup is a single
  `np.bincount`. V and Q are dense over all states (at most 302,020 × 9), which
  keeps indexing trivial. Only numpy is needed; scipy would not earn its place.
- **Project 2: problem facts live in one table** (`PROBLEMS`): sizes, γ,
  grid layout and the goal-reward terminal rule, taken from the course page
  and the data. The algorithm itself is the same for all three datasets.
- **Input validation** happens in `load_csv`, `read_gph`, `read_policy`,
  `SearchConfig` and `SolveConfig`. The CLIs turn `ValueError`/`OSError` into
  a message and exit code 2.
- **No structure-learning or RL libraries**, as the course rules require.

## Containers

All commands run in Docker as the host user, with the repository mounted at
`/work`. Dependencies are installed when the image is built; the image is
named `cs238` (`:dev`, `:tools`). The `tools` image (Debian's `gh` package)
exists only for `make pr`, which reads `GH_TOKEN` from the gitignored `.env`.
`make push` and `make pr` use the `BRANCH` and `PR_TITLE` variables at the
top of the Makefile.
