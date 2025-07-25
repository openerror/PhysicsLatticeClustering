# Identifying Clusters on a Discrete Periodic Lattice via Machine Learning

**July 2025: revamped repository with LLM coding assist.**

This repository contains Python code for my [peer-reviewed publication](https://www.sciencedirect.com/science/article/pii/S0010465519301535#aep-article-footnote-id1) published in *Computer Physics Communications*.

During my doctoral studies, I creatively combined machine learning with "traditional" computer science techniques, in order to automatically conduct a biophysical analysis. In particular, my amalgam of [hierarchical clustering](https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/#Visualizing-Your-Clusters) and [Breath-First Search (BFS)](https://en.wikipedia.org/wiki/Breadth-first_search) allowed me to quickly analyze simulated synaptic molecular clusters, the organization of which are critical to brain functions. More details can be found in [my peer-reviewed publication in *Physical Review E*](https://badge.dimensions.ai/details/id/pub.1139571861).

*Technical problem statement*: 
1. We have a 2D grid of 0s and 1s. We want to count how many connected clusters of 1s there are.
2. The catch is that there are periodic boundary conditions: grid edges wrap around and connect to the opposite side. This is a common method for running compute-limited simulations of a very large grid.
3. And *that* is why we need both BFS and hierarchical clustering! 

## Software Requirements
Tested on Python ~~3.5, 3.6, 3.7,~~ 3.12. Please see [requirements.txt](requirements.txt) file for dependencies.

## Usage Directions
All code files are located under `src`. For a visual demonstration of hierarchical clustering
see `ClusteringConcept-Showcase.ipynb`. As for a possible workflow using all code files,
please see `CompletWorkflow-Showcase.ipynb`.

Here are their purposes, briefly:

`clustering.py`
* Wrappers for hierarchical clustering routines provided by `SciPy`; output feeds into functions defined in `grouping.py`

`grouping.py`
* Handles periodic boundaries conditions; invokes `BFS.py`

`BFS.py`
* Executes breadth-first search to identify clusters touching across periodic boundaries'

`tests.py`
For now: basic unit tests on `BFS.py`. More to be implemented.
