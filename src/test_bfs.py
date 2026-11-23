"""Unit tests for src/bfs.py."""

import numpy as np

from src.bfs import connected_components


def test_invalid_inputs():
    # Should refuse to process and return None
    invalid_inputs = [None, np.zeros(1), np.zeros((2, 2, 2))]
    for data in invalid_inputs:
        assert connected_components(data) is None


def test_paper_example():
    # Paper Fig. 2: nodes 0, 2, 3 form one component; 1, 4, 5 are singletons
    C = np.zeros((6, 6))
    C[0, 2] = C[2, 0] = 1
    C[0, 3] = C[3, 0] = 1
    components = connected_components(C)
    assert sorted(sorted(c) for c in components) == [[0, 2, 3], [1], [4], [5]]


def test_path_graph_350():
    # Chained 0-1-2-...-349: one component; exercises deep recursion
    N = 350
    C = np.zeros((N, N))
    idx = np.arange(1, N)
    C[idx, idx - 1] = C[idx - 1, idx] = 1
    components = connected_components(C)
    assert len(components) == 1
    assert sorted(components[0]) == list(range(N))


def test_star_graph_350():
    # Hub connected to 349 leaves: >300 pending queue entries would be
    # silently dropped by the old deque(maxlen=300)
    N = 350
    C = np.zeros((N, N))
    C[0, 1:] = C[1:, 0] = 1
    components = connected_components(C)
    assert len(components) == 1
    assert sorted(components[0]) == list(range(N))
