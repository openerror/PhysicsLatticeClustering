# Identifying Clusters on a Discrete Periodic Lattice via Machine Learning

Research code from my 2019 methods paper in [*Computer Physics Communications*](https://doi.org/10.1016/j.cpc.2019.05.004) ([arXiv:1901.00091](https://arxiv.org/abs/1901.00091)), rebuilt and modernized in about 90 minutes for about 1 USD by an AI agent.

The paper solves a specific problem in computational biophysics: identifying clusters of simulated molecules on a discretized cell membrane, where lattice simulations use periodic boundary conditions to stand in for a much larger system. The code pairs hierarchical clustering with breadth-first search to handle those boundaries correctly.

## AI-assisted rebuild (November 2026)

The whole revision was one agent session: [GLM-5.3-flash](https://ollama.com/library/glm-5.3-flash) orchestrated through VS Code Copilot [fleet mode](https://docs.github.com/en/copilot/how-tos/copilot-sdk/features/fleet-mode), with the published paper fed in as Markdown via [Docling](https://docling-project.github.io/).

- **About 1 USD and 90 minutes**, start to finish.
- **A hidden bug found and fixed** that had gone unnoticed since 2019.
- **Jupyter notebooks replaced with Quarto Markdown** — the source is diffable and the output is one self-contained HTML file.
- **19 unit tests added** (4 for BFS, 15 for clustering) covering the full pipeline: `19 passed in 0.15s`.
- **PEP 8 naming and style conventions** applied throughout, following the [Ponytail](https://github.com/DietrichGebert/ponytail/blob/main/.github/copilot-instructions.md) style guide.
- **Leaner install**: the library needs only NumPy and SciPy; notebook and plotting tooling moved to an optional `docs` group.

## The problem

Simulating a cell membrane means discretizing it into a grid of patches. To analyze the result, you need to count and measure the groups of occupied patches — but the grid is usually periodic, so opposite edges are neighbours.

1. A 2D grid of 0s and 1s: count the connected groups of 1s, where only non-diagonal neighbours count.
2. The grid edges wrap around, as in compute-limited simulations standing in for a much larger lattice.
3. Distance-based clustering cannot see across that wrap, so the pipeline clusters first, then merges the clusters that meet at a seam with breadth-first search.

This is what the paper formalizes and what this repository implements.

## Quick start

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

``` sh
uv sync                                  # NumPy + SciPy + pytest
uv run python -m pytest src/ -v          # run the unit tests
```

``` sh
uv run --group docs quarto render ClusteringConcept-Showcase.qmd --to html
```

The second command renders the showcase: it needs the `docs` group and [Quarto](https://quarto.org/docs/get-started/) (`brew install --cask quarto`, v1.11.9 here). A pre-rendered `ClusteringConcept-Showcase.html` is checked in, but GitHub shows it as source: download and open it in a browser.

## Usage

``` python
import numpy as np
from src.clustering import (
    detect_clusters,
    extract_cluster_coordinates,
    find_cluster_grouping,
)

# Paper Figure 1: six occupied sites on a 5x5 lattice, as (row, col)
lattice = np.zeros((5, 5), dtype=np.int32)
for row, col in [(0, 4), (1, 1), (2, 3), (2, 1), (3, 1), (4, 4)]:
    lattice[row, col] = 1

params = {"sizeX": 5, "sizeY": 5}

linkage_matrix, hc_stats = detect_clusters(
    lattice, dendrogram_cutoff=1.1, method="single"
)
coords = list(extract_cluster_coordinates(hc_stats, params))
groups = find_cluster_grouping(coords, params)

print("cluster areas:", [len(x) for x, _, _, _ in coords])
print("merged groups:", [[int(c) for c in g] for g in groups])
```

``` text
cluster areas: [3, 1, 1, 1]
merged groups: [[0], [1], [2, 3]]
```

Four Euclidean clusters become three physical clusters: ids 2 and 3 meet across the periodic boundary.

## The code

| File | Role |
|---|---|
| `clustering.py` | SciPy `linkage`/`fcluster` wrappers plus periodic-boundary reconciliation |
| `bfs.py` | BFS connected components over the cluster-adjacency matrix |
| `test_clustering.py`, `test_bfs.py` | 19 pytest unit tests, one file per module |

## Citation and license

If you use this code, cite: E. Law, *Identifying clusters on a discrete periodic lattice via machine learning*, Computer Physics Communications **243** (2019) 106–109. [doi:10.1016/j.cpc.2019.05.004](https://doi.org/10.1016/j.cpc.2019.05.004) · [arXiv:1901.00091](https://arxiv.org/abs/1901.00091)

[`CITATION.cff`](CITATION.cff) gives GitHub a "Cite this repository" button. The applied biophysics follow-up is Law, Li, Kahraman & Haselwandter, *Phys. Rev. E* **104**, 014403 (2021), [doi:10.1103/PhysRevE.104.014403](https://doi.org/10.1103/PhysRevE.104.014403). Released under [CC BY 4.0](LICENSE).
