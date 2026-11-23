# clustering.py
"""
Apply hierarchical clustering on 1D/2D arrays. Clustering functionality is provided
by the excellent SciPy package.
"""

import numpy as np

from collections.abc import Iterable
from scipy.cluster.hierarchy import linkage, fcluster
from numpy.typing import NDArray

from .bfs import connected_components


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
        linkage_matrix, hc_stats: a N-by-3 numpy array recording the clustering output

    For additional documentation of SciPy's hierarchical clustering routines,
    see https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    and the official documentation @ https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
    """

    # Assume already binarized; find directly occupied coordinates
    occupied_pts = np.transpose(np.nonzero(data))

    dim = occupied_pts.shape[1]
    if dim == 1:
        x = occupied_pts[:, 0]
        y = np.zeros(x.shape)
    elif dim == 2:
        x = occupied_pts[:, 1]
        y = occupied_pts[:, 0]
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
    hc_stats = np.stack((x, y, clusters), axis=1)
    return linkage_matrix, hc_stats.astype(np.int32)


def extract_cluster_coordinates(
    hc_stats: NDArray, sim_params_dict: dict[str, int]
) -> Iterable[tuple[NDArray, NDArray, bool, int]]:
    """
    Analyze the output of detect_clusters(), and return
        1) the coordinates of every cluster identified;
        2) whether that cluster straddles the boundary.
    For now, assume data in the shape of rectangular grids.

    Args:
        hc_stats: N-by-3 matrix; each row represents a coordinate point, and the columns
                record the X, Y, and ID of the cluster to which this point belongs
        sim_params_dict: dict containing lattice sizes, keys 'sizeX'/'sizeY'
    Yields:
        coordinates: 4-member tuples. Each tuple corresponds
                     to a cluster: (X coordinates, Y coordinates, straddle True/False, cluster integer id)
    """

    cluster_ids, _ = np.unique(hc_stats[:, -1], return_counts=True)
    cluster_ids = cluster_ids.astype(int)
    x = hc_stats[:, 0]
    y = hc_stats[:, 1]  # X,Y coordinates of ALL identified clusters

    # Identify ALL clusters, and return its coordinates
    # The coordinates can then be used as a mask to access occupancy NumPy arrays
    for cid in cluster_ids:
        mask = hc_stats[:, 2] == cid
        touch = _detect_touch(x[mask], y[mask], sim_params_dict)  # Boolean var
        yield (x[mask], y[mask], touch, cid)


def find_cluster_grouping(
    cluster_coordinates: list, lattice_params: dict[str, int]
) -> list[list[int]] | None:
    """
    Determines which clusters touch each other across periodic boundaries,
    and therefore should be merged together.

    Args:
        cluster_coordinates: list output of extract_cluster_coordinates()
        from clustering.py. Each list item contains (X coordinates, Y coordinates,
        touch True/False, cluster id)
        lattice_params: dict with lattice sizes, keys 'sizeX'/'sizeY'

    Returns:
        Single python list, containing smaller lists of cluster ids. If a smaller
        list has more than 1 item, then it describes a merged cluster.
        None for degenerate input, matching bfs.connected_components().
    """

    # For each cluster identified, find out whether they touch the boundary
    boundary_touching_clusters = filter(lambda x: x[2], cluster_coordinates)
    points_at_boundary = [
        _find_boundary_points(pts, lattice_params)
        for pts in boundary_touching_clusters
    ]
    straddle_pairing = np.zeros(
        shape=(len(cluster_coordinates), len(cluster_coordinates)), dtype="int16"
    )

    for i, _ in enumerate(points_at_boundary):
        for j in range(i + 1, len(points_at_boundary)):
            if _test_partner_identity(points_at_boundary[i], points_at_boundary[j], 1):
                cluster_id_1 = points_at_boundary[i][4]
                cluster_id_2 = points_at_boundary[j][4]

                straddle_pairing[cluster_id_1, cluster_id_2] = 1
                straddle_pairing[cluster_id_2, cluster_id_1] = 1

    # Use breadth-first search to determine clusters that are touching 'transitively'
    merged_clusters = connected_components(straddle_pairing)
    return merged_clusters


def _detect_touch(
    x: NDArray, y: NDArray, sim_params_dict, touch_threshold=1
):
    """
    Determine whether a cluster touches the (periodic boundary), up a threshold.

    Specify a sizeY-by-sizeX rectangular grid from sim_params_dict. Then given
    a set of (X, Y) coordinates, determine how many of them are at the grid
    boundary; return T/F according to threshold.
    """

    assert len(x) == len(
        y
    ), "WARNINGS: lengths of the provided coordinate arrays must match!"
    max_y = sim_params_dict["sizeY"] - 1
    max_x = sim_params_dict["sizeX"] - 1
    border_points_x = np.sum((x == 0) | (x == max_x))
    border_points_y = np.sum((y == 0) | (y == max_y))
    return (border_points_x >= touch_threshold) or (
        border_points_y >= touch_threshold
    )


def _find_boundary_points(coordinates: tuple, lattice_params: dict[str, int]):
    """
    Given the coordinates of a cluster detected by hierarchical clustering,
    return the coordinates where the cluster touches the boundary

    To reduce redundant information, for instance, for the left boundary
    (x == 0) we return only the y coordinates.

    coordinates: (X numpy, Y numpy, straddle bool)
    lattice_params: dict with lattice sizes, keys 'sizeX'/'sizeY'
    """

    # Calculate max possible indices to faciliate later comparison
    max_x, max_y = lattice_params["sizeX"] - 1, lattice_params["sizeY"] - 1
    x, y, _, cluster_id = coordinates

    left = y[x == 0]
    right = y[x == max_x]
    top = x[y == max_y]  # Top and bottom relative to left,bottom origin
    bottom = x[y == 0]

    return [left, right, top, bottom, cluster_id]


def _test_partner_identity(boundary_a, boundary_b, partner_threshold):
    """
    Decide whether A and B are actually part of a same cluster that
    straddles the periodic boundary

    boundary_a and boundary_b: Python lists of format [left, right, top, bottom]
    They record the coordinates of the pts touching the boundaries.
    See _find_boundary_points() above.

    partner_threshold: an integer; if the clusters A and B touch each other for
    more than this number of pixels, then count them as the same cluster
    """
    a_left, a_right, a_top, a_bottom, _ = boundary_a
    b_left, b_right, b_top, b_bottom, _ = boundary_b
    overlap = 0

    if len(a_left) > 0 and len(b_right) > 0:
        overlap += len(np.intersect1d(a_left, b_right))

    if len(a_top) > 0 and len(b_bottom) > 0:
        overlap += len(np.intersect1d(a_top, b_bottom))

    if len(b_left) > 0 and len(a_right) > 0:
        overlap += len(np.intersect1d(b_left, a_right))

    if len(b_top) > 0 and len(a_bottom) > 0:
        overlap += len(np.intersect1d(b_top, a_bottom))

    return overlap >= partner_threshold
