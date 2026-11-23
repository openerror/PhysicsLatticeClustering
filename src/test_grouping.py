"""Unit tests for src/grouping.py."""
"""
Tests for src/grouping.py: merge_boundary_clusters(), _find_boundary_points()
and _test_partner(). Also covers an end-to-end pipeline through
detect_clusters()/extract_cluster_coordinates() in clustering.py.

Behavior verified against the CURRENT implementation:
- merge_boundary_clusters() returns None for degenerate inputs (0 or 1 nodes):
  the straddle pairing matrix has size 0/1 and bfs.connected_components()
  propagates None.
- The adjacency matrix is indexed by the cluster id carried in each tuple, so
  ids need not equal positional indices (see test_merge_straddling_pair_left_right_seam).
- bfs.connected_components() labels come from range() (python int) or from
  np.nonzero() neighbor indices (numpy.int64); every element is thus castable
  via int() — asserted explicitly below.
"""

import numpy as np
import pytest

from src.grouping import (
    _find_boundary_points,
    _test_partner,
    merge_boundary_clusters,
)

PARAMS_66 = {"sizeX": 6, "sizeY": 6}


def _as_groups(result) -> list[set[int]]:
    """Normalize a merge result into a list of sets of plain ints."""
    return [set(int(node) for node in group) for group in result]


# ---------------------------------------------------------------------------
# merge_boundary_clusters
# ---------------------------------------------------------------------------


def test_merge_straddling_pair_left_right_seam():
    # A touches the left edge (x==0); B touches the right edge (x==5) at the
    # same y; both share y=3 so their boundary points overlap across the seam.
    # The interior cluster 1 (x in [1,4]) remains untouched.
    # NOTE: cid 2 is used for the second tuple — the pairing matrix is indexed
    # by the cid in slot 4, not by list position, hence groups {0, 2} and {1}.
    cluster_coordinates = [
        (np.array([0]), np.array([3]), True, 0),
        (np.array([5]), np.array([3]), True, 2),
        (np.array([2]), np.array([1]), False, 1),
    ]
    result = merge_boundary_clusters(cluster_coordinates, PARAMS_66)

    assert result is not None
    assert sorted(_as_groups(result), key=len) == [{1}, {0, 2}]


def test_merge_straddling_pair_top_bottom_seam():
    # A at (x=2, y=0) touches the bottom edge; B at (x=2, y=5) touches the top
    # edge; their top/bottom boundary points overlap on x=2.
    cluster_coordinates = [
        (np.array([2]), np.array([0]), True, 0),
        (np.array([2]), np.array([5]), True, 1),
        (np.array([4]), np.array([4]), False, 2),
    ]
    result = merge_boundary_clusters(cluster_coordinates, PARAMS_66)

    assert result is not None
    assert sorted(_as_groups(result), key=len) == [{2}, {0, 1}]


def test_transitive_merge():
    # A touches left at y=2, B touches BOTH right (y=2) and left (y=4), C
    # touches right at y=4. A-B merge via the y=2 seam, B-C via the y=4 seam,
    # but A and C never touch — the BFS must chain A-B-C transitively
    # (the paper's Fig. 2 story).
    cluster_coordinates = [
        (np.array([0]), np.array([2]), True, 0),
        (np.array([5, 0]), np.array([2, 4]), True, 1),
        (np.array([5]), np.array([4]), True, 2),
    ]
    result = merge_boundary_clusters(cluster_coordinates, PARAMS_66)

    assert result is not None
    assert _as_groups(result) == [{0, 1, 2}]


def test_no_straddle_singletons():
    # No cluster touches the boundary: no edges, one singleton group per cid.
    cluster_coordinates = [
        (np.array([1]), np.array([1]), False, 0),
        (np.array([3]), np.array([3]), False, 1),
        (np.array([4]), np.array([2]), False, 2),
    ]
    result = merge_boundary_clusters(cluster_coordinates, PARAMS_66)

    assert result is not None
    assert _as_groups(result) == [{0}, {1}, {2}]


def test_boundary_cluster_with_no_partner():
    # Exactly one boundary-touching cluster: the pairing matrix stays all
    # zeros, so all clusters remain singletons.
    cluster_coordinates = [
        (np.array([0]), np.array([3]), True, 0),
        (np.array([2]), np.array([1]), False, 1),
        (np.array([4]), np.array([4]), False, 2),
    ]
    result = merge_boundary_clusters(cluster_coordinates, PARAMS_66)

    assert result is not None
    assert _as_groups(result) == [{0}, {1}, {2}]


def test_empty_input_returns_none():
    # Empty input yields a (0, 0) pairing matrix; bfs.connected_components()
    # treats size <= 1 as degenerate and returns None, which propagates here.
    # Verified empirically on the current code.
    result = merge_boundary_clusters([], PARAMS_66)
    assert result is None


def test_single_nonstraddling_cluster_returns_none():
    # A single cluster yields a (1, 1) pairing matrix of size 1; BFS treats
    # size <= 1 as degenerate and returns None. So a lone (even non-straddling)
    # cluster cannot be recovered from merge_boundary_clusters — callers must
    # handle this degenerate case themselves. Verified empirically.
    cluster_coordinates = [(np.array([2]), np.array([1]), False, 0)]
    result = merge_boundary_clusters(cluster_coordinates, PARAMS_66)
    assert result is None


def test_results_are_lists_of_ints():
    # Cluster ids come from tuple slot 4 (python int) and BFS labels are
    # either python ints (from range()) or numpy.int64 (from np.nonzero());
    # every element must therefore be castable to a plain int.
    cluster_coordinates = [
        (np.array([0]), np.array([3]), True, 0),
        (np.array([5]), np.array([3]), True, 2),
        (np.array([2]), np.array([1]), False, 1),
    ]
    result = merge_boundary_clusters(cluster_coordinates, PARAMS_66)

    assert result is not None
    flat = [node for group in result for node in group]
    expected = {0, 1, 2}
    for node in flat:
        assert int(node) in expected  # int(...) never truncates/distorts
    assert set(map(int, flat)) == expected


# ---------------------------------------------------------------------------
# _find_boundary_points
# ---------------------------------------------------------------------------


def test_find_boundary_points_left_right_top_bottom():
    # Cluster spanning x=0..4 with sites (0,2), (4,2), (2,3) on a 5x5 lattice:
    # left  = y at x==0        -> [2]
    # right = y at x==4        -> [2]
    # top   = x at y==4        -> empty (no site at the top edge)
    # bottom= x at y==0        -> empty
    coordinates = (np.array([0, 4, 2]), np.array([2, 2, 3]), True, 7)
    boundary = _find_boundary_points(coordinates, {"sizeX": 5, "sizeY": 5})

    np.testing.assert_array_equal(boundary[0], [2])  # left
    np.testing.assert_array_equal(boundary[1], [2])  # right
    np.testing.assert_array_equal(boundary[2], [])   # top (empty)
    np.testing.assert_array_equal(boundary[3], [])   # bottom (empty)
    assert boundary[4] == 7  # cluster id preserved in slot 4


def test_find_boundary_points_corner():
    # A corner site at (x=0, y=0) touches both the left and bottom edges.
    coordinates = (np.array([0]), np.array([0]), True, 3)
    boundary = _find_boundary_points(coordinates, {"sizeX": 5, "sizeY": 5})

    np.testing.assert_array_equal(boundary[0], [0])  # left
    np.testing.assert_array_equal(boundary[1], [])   # right
    np.testing.assert_array_equal(boundary[2], [])   # top
    np.testing.assert_array_equal(boundary[3], [0])  # bottom
    assert boundary[4] == 3


# ---------------------------------------------------------------------------
# _test_partner
# ---------------------------------------------------------------------------


def test_test_partner_left_right_overlap():
    # A's left boundary overlaps B's right boundary on exactly one value.
    boundary_a = [np.array([2, 3]), np.array([]), np.array([]), np.array([]), 0]
    boundary_b = [np.array([]), np.array([3, 9]), np.array([]), np.array([]), 1]

    assert _test_partner(boundary_a, boundary_b, 1) is True
    assert _test_partner(boundary_a, boundary_b, 2) is False  # overlap == 1 < 2


def test_test_partner_top_bottom_overlap():
    # A's top boundary overlaps B's bottom boundary.
    boundary_a = [np.array([]), np.array([]), np.array([1]), np.array([]), 0]
    boundary_b = [np.array([]), np.array([]), np.array([]), np.array([1]), 1]

    assert _test_partner(boundary_a, boundary_b, 1) is True


def test_test_partner_disjoint_top_bottom_false():
    # Top/bottom boundaries exist but share no coordinates.
    boundary_a = [np.array([]), np.array([]), np.array([2]), np.array([]), 0]
    boundary_b = [np.array([]), np.array([]), np.array([]), np.array([5]), 1]

    assert _test_partner(boundary_a, boundary_b, 1) is False


def test_test_partner_no_overlap_false():
    # Fully empty boundaries (0-length arrays, never None) -> no overlap.
    boundary_a = [np.array([]), np.array([]), np.array([]), np.array([]), 0]
    boundary_b = [np.array([]), np.array([]), np.array([]), np.array([]), 1]

    for arr in boundary_a[:4] + boundary_b[:4]:
        assert arr.size == 0

    assert _test_partner(boundary_a, boundary_b, 1) is False


# ---------------------------------------------------------------------------
# Integration with clustering.py
# ---------------------------------------------------------------------------


def test_end_to_end_with_extract_cluster_coordinates():
    # Full pipeline: binarized 6x6 lattice -> detect_clusters() ->
    # extract_cluster_coordinates() -> merge_boundary_clusters().
    # Two 2-site clusters sit in columns 0 and 5 at rows 2, 3; they share both
    # y values, so they straddle the left-right seam and merge. The interior
    # site at (x=4, y=4) is >= 4 lattice units from any other site and stays a
    # singleton. lattice[row, col]: rows are y, columns are x.
    lattice = np.zeros((6, 6), dtype=int)
    lattice[2, 0] = lattice[3, 0] = 1  # left edge cluster
    lattice[2, 5] = lattice[3, 5] = 1  # right edge cluster (same rows)
    lattice[4, 4] = 1  # isolated interior site

    _, hc_stats = _detect_clusters_helper(lattice)
    coordinates_list = list(
        _extract_cluster_coordinates_helper(hc_stats, PARAMS_66)
    )
    result = merge_boundary_clusters(coordinates_list, PARAMS_66)

    assert result is not None
    touched = {int(coords[3]) for coords in coordinates_list if coords[2]}
    straddler_ids = {int(coords[3]) for coords in coordinates_list if not coords[2]}

    # The two boundary clusters merge together; they must be the touched ones.
    merged_groups = _as_groups(result)
    multi_groups = [group for group in merged_groups if len(group) > 1]
    assert len(multi_groups) == 1
    assert multi_groups[0] == touched

    # The interior cluster appears as its own singleton group.
    singletons = [group for group in merged_groups if len(group) == 1]
    assert straddler_ids in [set(group) for group in singletons]


# Keep clustering imports inside helpers so that grouping tests remain
# independent if clustering.py changes shape; ids/touch flags drive assertions.
def _detect_clusters_helper(lattice):
    from src.clustering import detect_clusters

    return detect_clusters(lattice, 1.1, method="single")


def _extract_cluster_coordinates_helper(hc_stats, lattice_params):
    from src.clustering import extract_cluster_coordinates

    return extract_cluster_coordinates(hc_stats, lattice_params)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
