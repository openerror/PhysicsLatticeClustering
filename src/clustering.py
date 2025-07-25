# clustering.py
'''
    Apply hierarchical clustering on 1D/2D arrays. Clustering functionality is provided
    by the excellent SciPy package.
'''

import numpy as np

from copy import copy
from scipy.cluster.hierarchy import linkage, fcluster
from numpy.typing import NDArray

from .BFS import connected_components


def detect_clusters(
    data: NDArray, dendrogram_cutoff: float, method: str = "single"
) -> tuple[NDArray, NDArray]:
    """
    Binarize numerical data, and then run hierarchical clustering on it.

    Turn the input data into 0's and 1's by global thresholding; say, that N 1's
    are produced as a result. Then, interpreting the array indexes of the 1's as
    Euclidean coordinates, run hierarchical clustering on them. In machine learning
    terms, the clustering procedure is applied to N observations, each with (at most)
    2 features.

    In the end, return a N-by-3 numpy array. The columns record X,Y coordinates
    and the (integer) id of the cluster, to which this point is assigned.
    Assumption for 2D arrays: origin at top left corner.

    Args:
        data: 1D or 2D numpy array, containing positions
        method: clustering linkage used
        dendrogram_cutoff: cut-off for cluster-merging distance, applied to linkage matrix
    Returns:
        HCStats: a N-by-3 numpy array recording the clustering output

    For additional documentation of SciPy's hierarchical clustering routines,
    see https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    and the official documentation @ https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
    """

    # Assume already binarized; find directly occupied coordinates
    occupied_pts = np.transpose(np.nonzero(data))

    dim = occupied_pts.shape[1]
    if dim == 1:
        X = occupied_pts[:, 0]
        Y = np.zeros(X.shape)
    elif dim == 2:
        X = occupied_pts[:, 1]
        Y = occupied_pts[:, 0]
    else:
        raise NotImplementedError(
            "Function designed to handle only 1D/2D data. Higher\
        dimensions are not yet allowed."
        )

    # Generate linkage matrix, which describes the iterative clustering steps
    # And then, identifiy each point with a cluster
    linkage_matrix = linkage(occupied_pts, method=method, metric="euclidean")
    clusters = fcluster(linkage_matrix, dendrogram_cutoff, criterion="distance")
    clusters = clusters - 1.0  # Broadcasted substraction; change to zero-base index

    # Engineering the feature matrix
    # Columns: (X coordinate, Y coordinate, cluster id)
    # Casting to integer; saves space with no loss of information
    HCStats = np.stack((X, Y, clusters), axis=1)
    return linkage_matrix, HCStats.astype(np.int32)


def extract_cluster_coordinates(HCStats, sim_params_dict):
    '''
        Analyze the output of detect_clusters(), and return
            1) the coordinates of every cluster identified;
            2) whether that cluster straddles the boundary.
        For now, assume data in the shape of rectangular grids.

        Args:
            HCStats: N-by-3 matrix; each row represents a coordinate point, and the columns
                    record the X, Y, and ID of the cluster to which this point belongs
            sim_params_dict: Python dictionary (see paramSpace.py) containing lattice sizes
        Yields:
            coordinates: 4-member tuples. Each tuple corresponds
                         to a cluster: (X coordinates, Y coordinates, straddle True/False, cluster integer id)
    '''

    cluster_ids, _ = np.unique(HCStats[:,-1], return_counts=True)
    cluster_ids = cluster_ids.astype(int)
    X = HCStats[:,0]; Y = HCStats[:,1] #X,Y coordinates of ALL identified clusters

    # Identify ALL clusters, and return its coordinates
    # The coordinates can then be used as a mask to access occupancy NumPy arrays
    for cid in cluster_ids:
        mask = (HCStats[:,2] == cid)
        touch = _detect_touch(X[mask], Y[mask], sim_params_dict) #Boolean var
        yield ( X[mask], Y[mask], touch, cid )


def find_cluster_grouping(cluster_coordinates, sim_params_dict):
    '''
        Determines which clusters touch each other across periodic boundaries,
        and therefore should be merged together.

        Args:
            cluster_coordinates: list output of extractClusterCoordinates()
            from clustering.py. Each list item contains (X coordinates, Y coordinates,
            touch True/False, cluster id)

        Returns:
            Single python list, containing smaller lists of cluster ids. If a smaller
            list has more than 1 item, then it describes a merged cluster.
    '''

    # For each cluster identified, find out whether they touch the boundary
    boundary_touching_clusters = filter(lambda x: x[2], cluster_coordinates)
    points_at_boundary = [_find_boundary_points(pts, sim_params_dict) for pts in boundary_touching_clusters]
    straddle_pairing = np.zeros(shape=(len(cluster_coordinates),
                                       len(cluster_coordinates)), dtype='int16')

    for i,_ in enumerate(points_at_boundary):
        for j in range(i + 1, len(points_at_boundary)):
            if _test_partner_identity( points_at_boundary[i], points_at_boundary[j], 1):
                cluster_id_1 = points_at_boundary[i][4]
                cluster_id_2 = points_at_boundary[j][4]

                straddle_pairing[cluster_id_1, cluster_id_2] = 1
                straddle_pairing[cluster_id_2, cluster_id_1] = 1

    # Use breadth-first search to determine clusters that are touching 'transitively'
    merged_clusters,_ = connected_components(straddle_pairing)
    return merged_clusters


def _detect_touch(X, Y, sim_params_dict, touch_threshold=1):
    '''
        Determine whether a cluster touches the (periodic boundary), up a threshold.

        Specify a sizeY-by-sizeX rectangular grid from sim_params_dict. Then given
        a set of (X, Y) coordinates, determine how many of them are at the grid
        boundary; return T/F according to threshold.
    '''
    
    assert (len(X)==len(Y)), "WARNINGS: lengths of the provided coordinate arrays must match!"
    max_y = sim_params_dict["sizeY"]-1
    max_x = sim_params_dict["sizeX"]-1
    border_pts_x = np.sum((X == 0) | (X == max_x))
    border_pts_y = np.sum((Y == 0) | (Y == max_y))
    return ((border_pts_x >= touch_threshold) or (border_pts_y >= touch_threshold))


def _find_boundary_points(coordinates, paramDict):
    '''
        Given the coordinates of a cluster detected by hierarchical clustering,
        return the coordinates where the cluster touches the boundary

        To reduce redundant information, for instance, for the left boundary
        (x == 0) we return only the y coordinates.

        coordinates: (X numpy, Y numpy, straddle bool)
        paramDict: dict containing KMC simulation parameters
    '''

    #Calculate max possible indices to faciliate later comparison
    maxX, maxY = paramDict['sizeX']-1, paramDict['sizeY']-1
    X, Y, _, cluster_id = coordinates

    left = Y[X == 0]
    right = Y[X == maxX]
    top = X[Y == maxY] #Top and bottom relative to left,bottom origin
    bottom = X[Y == 0]

    return [left, right, top, bottom, cluster_id]


def _test_partner_identity(boundaryA, boundaryB, partner_threshold):
    '''
        Decide whether A and B are actually part of a same cluster that
        straddles the periodic boundary

        boundaryA and boundaryB: Python lists of format [left, right, top, bottom]
        They record the coordinates of the pts touching the boundaries.
        See _findStraddlingPts() above.

        partner_threshold: an integer; if the clusters A and B touch each other for
        more than this number of pixels, then count them as the same cluster
    '''
    Aleft, Aright, Atop, Abottom, _ = boundaryA
    Bleft, Bright, Btop, Bbottom, _ = boundaryB
    overlap = 0

    if (len(Aleft)>0 and len(Bright)>0):
        overlap += len(np.intersect1d(Aleft, Bright))

    if (len(Atop)>0 and len(Bbottom)>0):
        overlap += len(np.intersect1d(Atop, Bbottom))

    if (len(Bleft)>0 and len(Aright)>0):
        overlap += len(np.intersect1d(Bleft, Aright))

    if (len(Btop)>0 and len(Abottom)>0):
        overlap += len(np.intersect1d(Btop, Abottom))

    return (overlap >= partner_threshold)
