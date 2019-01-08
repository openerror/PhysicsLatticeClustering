# Identifying Clusters on a Discrete Periodic Lattice via Machine Learning

Published as part of [insert link]

Python code for identifying clusters on a 2D periodic lattice, inspired by biophysical
studies and using **[hierarchical clustering](https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/#Visualizing-Your-Clusters)**. Utilizes Breadth-First Search ([BFS](https://en.wikipedia.org/wiki/Breadth-first_search)) to connect together clusters that touch each other across boundaries.

The code assumes that the input lattice is 'binarized': contains only 0s and 1s, and would attempt to identify clusters of 1s.

## Software Requirements
- Tested on Python 3.5, 3.6 and 3.7
- [NumPy](https://www.numpy.org/), library for creating and manipulating numerical arrays
- [SciPy](https://www.scipy.org/about.html), for performing hierarchical clustering on NumPy arrays
- [Matplotlib](https://matplotlib.org/), for generating dendrograms and other visualizations
- [IPython](https://ipython.org/) + [JupyterLab/Jupyter Notebook](https://jupyter.org/), for interacting with the supplied demos

Assuming you have already set up Python, you can run `pip install -r requirements.txt` in the console to install all necessary libraries; add the `--user` flag if you don't have the necessary filesystem permissions.
The file `requirements.txt` is supplied with this repository. Alternatively, use a Python distribution such as [Anaconda](https://anaconda.org/), [WinPython](https://winpython.github.io/) or [Pyzo](https://pyzo.org/), which provides a graphical interface for managing Python libraries. For more information on the libraries and their ecosystem, refer to [SciPy official website](https://www.scipy.org/install.html).

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
* Executes breadth-first search to identify clusters touching across periodic boundaries

Unit tests pending!
