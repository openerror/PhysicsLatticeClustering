#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .bfs import connected_components


def merge_boundary_clusters(
    cluster_coordinates: list, lattice_params: dict[str, int]
) -> list[list[int]] | None:
    """
    Determines which clusters touch each other across periodic boundaries,
    and therefore should be merged together.

    Args:
        cluster_coordinates: list output of extract_cluster_coordinates() from clustering.py. Each list item contains (X coordinates, Y coordinates, touch True/False, cluster id)

    Returns:
        Single python list, containing smaller lists of cluster ids. If a smaller
        list has more than 1 item, then it describes a merged cluster.
    """

    # For each cluster identified, find out whether they touch the boundary
    boundary_touching_clusters = list(
        filter(lambda cluster: cluster[2], cluster_coordinates)
    )
    points_at_boundary = [
        _find_boundary_points(pts, lattice_params) for pts in boundary_touching_clusters
    ]

    straddle_pairing = np.zeros(
        shape=(len(cluster_coordinates), len(cluster_coordinates)), dtype="int16"
    )

    for i, _ in enumerate(points_at_boundary):
        for j in range(i + 1, len(points_at_boundary)):
            if _test_partner(points_at_boundary[i], points_at_boundary[j], 1):
                cluster_id_1 = points_at_boundary[i][4]
                cluster_id_2 = points_at_boundary[j][4]

                straddle_pairing[cluster_id_1, cluster_id_2] = 1
                straddle_pairing[cluster_id_2, cluster_id_1] = 1

    # Use breadth-first search to determine clusters that are touching 'transitively'
    merged_clusters = connected_components(straddle_pairing)
    return merged_clusters


def _find_boundary_points(coordinates: tuple, lattice_params: dict[str, int]) -> list:
    """
    Given the coordinates of a cluster detected by hierarchical clustering,
    return the coordinates where the cluster touches the boundary

    To reduce redundant information, for instance, for the left boundary
    (x == 0) we return only the y coordinates.

    coordinates: (X numpy, Y numpy, straddle bool, cluster id)
    lattice_params: dict containing KMC simulation parameters
    """

    # Calculate max possible indices to faciliate later comparison
    max_x, max_y = lattice_params["sizeX"] - 1, lattice_params["sizeY"] - 1
    x, y, _, cluster_id = coordinates

    left = y[x == 0]
    right = y[x == max_x]
    top = x[y == max_y]  # Top and bottom relative to left,bottom origin
    bottom = x[y == 0]

    return [left, right, top, bottom, cluster_id]


def _test_partner(boundary_a: list, boundary_b: list, partner_threshold: int) -> bool:
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
