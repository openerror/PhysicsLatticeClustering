import numpy as np
from collections import deque
from copy import copy

'''
Breadth-first search module to be used with singleStep.py
Hierarchical clustering may detect clusters that are "actually" connected to each
other across the periodic boundaries. Given a connectivity matrix of these clusters,
find the connected components --- clusters that are together directly, or transitively
'''

#Test matrix
#connectivity = np.array(
#        [[0,1,1,0,0],
#         [1,0,0,0,0],
#         [1,0,0,0,0],
#         [0,0,0,0,1],
#         [0,0,0,1,0]])

def connectedComponents(connectivityMatrix):
    '''
        Assume an undirected graph, return all connected components using BFS
        connectivityMatrix: NxN matrix corresponding to graph with N nodes

        Returns a list, containing lists of connected nodes
    '''

    # Short names cut down clutter
    cMatrix = connectivityMatrix

    # Handles invaid edge cases
    if (cMatrix is None or cMatrix.size <= 1 or len(cMatrix.shape) != 2):
        print("Invalid/Missing connectivity matrix. Returning NONE.")
        return None

    # Initialize queue for BFS, and result list
    # Limit queue to 300 items; should be enough and saves memory
    visited = np.zeros((cMatrix.shape[0],1), dtype='int16')
    queue = deque()
    result, singleNodeResult = [], []

    for node in range(0, len(visited)):
        if visited[node] == 0:
            queue.append(node)
            _bfsHelper(queue, visited, cMatrix, singleNodeResult)
            queue.clear()
            result.append(copy(singleNodeResult))
            singleNodeResult[:] = []

        # DEBUG; Comment out to avoid redundant comparison op
        # elif visited[node] == 1:
        #     print(f"Skipping over node {node}, visited alrdy")
        #     pass #Node already visited, no need to start BFS

    return result, cMatrix

def _bfsHelper(queue, visited, connectivityMatrix, result):
    '''
        Helper function carrying out one iteration of BFS; implemented with recursion.
        It's assumed that queue contains the starting node at the very beginning.
    '''
    cMatrix = connectivityMatrix

    if not queue:
        return result

    # Pop and record
    topNode = queue.popleft()
    visited[topNode] = 1
    result.append(topNode)

    # Append neighbours of topNode
    neighbours = np.nonzero(cMatrix[topNode,:])[0]
    for neg in neighbours:

        if visited[neg]==0:
            queue.append(neg)

    _bfsHelper(queue, visited, cMatrix, result)
