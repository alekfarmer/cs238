"""Draw a learned ``.gph`` structure as a PNG.

Nodes are laid out in layers by topological generation, so every edge points
downward-to-right (parents before children). This module only draws: the
structure and scores come from ``project1.py``.

Usage:
    python plot_graph.py <data>.csv <graph>.gph <out>.png [--title TITLE]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen first)
import networkx  # noqa: E402

from project1 import read_gph, to_digraph  # noqa: E402


def read_header(path: Path) -> list[str]:
    """Return the variable names from the first line of a CSV."""
    with open(path) as f:
        return [name.strip().strip('"') for name in f.readline().split(",")]


def draw(names: Sequence[str], gph: Path, out: Path, title: str) -> None:
    """Render the graph in ``gph`` (over variables ``names``) to ``out``."""
    dag = to_digraph(read_gph(gph, names))
    for layer, nodes in enumerate(networkx.topological_generations(dag)):
        for node in nodes:
            dag.nodes[node]["layer"] = layer
    pos = networkx.multipartite_layout(dag, subset_key="layer", align="vertical")
    # Children below parents: flip so layer 0 is at the top.
    pos = {node: (x, -y) for node, (y, x) in pos.items()}

    size = max(6.0, 0.35 * len(names))
    fig, ax = plt.subplots(figsize=(size * 1.4, size))
    networkx.draw_networkx(
        dag,
        pos,
        ax=ax,
        labels=dict(enumerate(names)),
        node_color="#dbe8f5",
        edgecolors="#2f5d8a",
        node_size=1400 if len(names) <= 15 else 700,
        font_size=8 if len(names) <= 15 else 6,
        arrowsize=12,
        edge_color="#555555",
    )
    ax.set_title(f"{title}: {dag.number_of_edges()} edges")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Draw a .gph structure as a PNG.")
    parser.add_argument("csv", type=Path, help="data CSV (for variable names)")
    parser.add_argument("gph", type=Path, help="graph to draw")
    parser.add_argument("png", type=Path, help="output image")
    parser.add_argument("--title", default="", help="plot title")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        draw(read_header(args.csv), args.gph, args.png, args.title or args.gph.stem)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
