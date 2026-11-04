# Architecture

## What this is

Course data and starter code for AA228/CS238 (from `sisl/AA228-CS238-Student`),
plus an implementation of Project 1 (Bayesian network structure learning).
Project 1 is a **command-line pipeline**, not an app: there is no server,
API or frontend, so the AGENTS.md invariants about those (backend computes
results, `127.0.0.1` publishing) have nothing to apply to. Computation and
drawing are still separated: `project1.py` computes, `plot_graph.py` only draws.

## Layout

    Makefile                 every command (build, lint, test, verify, learn-*, push, pr)
    docker/
      Dockerfile             images: runtime -> dev (adds test tools); tools (GitHub CLI)
      requirements.txt       pinned runtime deps: numpy, networkx, matplotlib
      requirements-dev.txt   pinned dev deps: pytest, ruff, mypy
    pyproject.toml           pytest / ruff / mypy configuration
    project1/
      project1.py            the whole algorithm: loading, score, search, .gph I/O, CLI
      plot_graph.py          .gph -> PNG (layered layout); no scoring
      tests/                 pytest suite (score vs. course reference, search, I/O)
      output/                learned results: <name>.gph, <name>.json (scores, runtime), <name>.png
      data/, example/        course-provided, unchanged
      project1.jl            course Julia starter, unchanged
      HOW_IT_WORKS.md        explanation of the method and results
    project2/                course data, untouched

## Data flow

    data/<name>.csv --load_csv--> Dataset --learn_structure--> parent sets
        --write_gph--> output/<name>.gph --plot_graph.py--> output/<name>.png
                     \-> output/<name>.json (scores per phase, runtime, settings)

## Design decisions

- **One module for the algorithm.** The course starter asks for code in
  `project1.py`; it is ~500 lines, organised in sections (data, score,
  search, CLI). Split into a package only if it grows substantially.
- **Graph = list of parent sets** (`list[frozenset[int]]`). The score
  decomposes by family, so parent sets are the natural unit and make cache
  keys. networkx is used only at the edges (final DAG check, writing, plotting).
- **Family-score cache** keyed by `(child, parents)`; all search phases share
  one `FamilyScorer`.
- **Reproducible search**: fixed seed and iteration counts rather than a time
  budget, so the same settings give the same graph on any machine. Runtime
  is measured and reported, not used as a stopping rule.
- **Input validation** happens in `load_csv`, `read_gph` and `SearchConfig`;
  the CLI turns `ValueError`/`OSError` into a message and exit code 2.
- **No structure-learning libraries**, as the course rules require.

## Containers

All commands run in Docker as the host user, with the repository mounted at
`/work`. Dependencies are installed when the image is built. The `tools`
image (Debian's `gh` package) exists only for `make pr`, which reads
`GH_TOKEN` from the gitignored `.env`.
