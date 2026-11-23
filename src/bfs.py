'''Breadth-first search module.
Hierarchical clustering may detect clusters that are "actually" connected to each
other across the periodic boundaries. Given a connectivity matrix of these clusters,
find the connected components --- clusters that are together directly, or transitively
'''

import sys
from collections import deque

import numpy as np
from numpy.typing import NDArray


def connected_components(connectivity_matrix: NDArray) -> list[list[int]] | None:
    '''
        Assume an undirected graph, return all connected components using BFS
        connectivity_matrix: NxN matrix corresponding to graph with N nodes

        Returns a list, containing lists of connected nodes
    '''

    # Handles invalid edge cases
    if (connectivity_matrix is None or connectivity_matrix.size <= 1 or len(connectivity_matrix.shape) != 2):
        return None

    # The BFS recurses once per node in _bfs_helper, so keep the recursion limit above the component size
    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(connectivity_matrix) + 100))

    # Initialize queue for BFS, and result list
    visited = np.zeros((connectivity_matrix.shape[0], 1), dtype='int16')
    queue = deque()
    result, single_node_result = [], []

    for node in range(0, len(visited)):
        if visited[node] == 0:
            queue.append(node)
            _bfs_helper(queue, visited, connectivity_matrix, single_node_result)
            queue.clear()
            result.append(single_node_result.copy())
            single_node_result.clear()
    return result


def _bfs_helper(queue: deque[int], visited: NDArray, connectivity_matrix: NDArray, result: list[int]) -> None:
    '''
        Helper function carrying out one iteration of BFS; implemented with recursion.
        It's assumed that queue contains the starting node at the very beginning.
    '''

    if not queue:
        return

    # Pop and record
    current_node = queue.popleft()
    visited[current_node] = 1
    result.append(current_node)

    # Append neighbours of current_node
    neighbors = np.nonzero(connectivity_matrix[current_node, :])[0]
    for neighbor_idx in neighbors:
        if visited[neighbor_idx] == 0:
            queue.append(neighbor_idx)

    _bfs_helper(queue, visited, connectivity_matrix, result)
