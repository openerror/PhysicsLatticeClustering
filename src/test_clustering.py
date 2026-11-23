"""Unit tests for src/clustering.py."""

import numpy as np
import pytest

from src.clustering import (
    _detect_touch,
    _find_boundary_points,
    _test_partner_identity,
    detect_clusters,
    extract_cluster_coordinates,
    find_cluster_grouping,
)

LATTICE_5X5 = {"sizeX": 5, "sizeY": 5}

# Paper toy lattice: (row, col) of occupied sites on a 5x5 grid
TOY_LATTICE = np.zeros((5, 5))
for row, col in [(0, 4), (1, 1), (2, 3), (2, 1), (3, 1), (4, 4)]:
    TOY_LATTICE[row, col] = 1


def point_sets_by_id(hc_stats):
    """Map cluster id -> set of (x, y) points, for id-order-independent checks."""
    mapping = {}
    for x, y, cid in hc_stats:
        mapping.setdefault(int(cid), set()).add((int(x), int(y)))
    return mapping


def coords_by_id(cluster_coordinates):
    """Same mapping, built from extract_cluster_coordinates() output."""
    return {
        int(touch_id): {(int(x), int(y)) for x, y in zip(xs, ys)}
        for xs, ys, _, touch_id in cluster_coordinates
    }


def test_detect_clusters_paper_toy_lattice():
    _, hc_stats = detect_clusters(TOY_LATTICE, 1.1, "single")
    clusters = point_sets_by_id(hc_stats)
    assert len(clusters) == 4
    # Columns are (x, y); the 3 occupied sites at col 1, rows 1-3 share one id
    assert set(map(frozenset, clusters.values())) == {
        frozenset({(1, 1), (1, 2), (1, 3)}),
        frozenset({(4, 0)}),
        frozenset({(3, 2)}),
        frozenset({(4, 4)}),
    }


def test_detect_clusters_1d():
    _, hc_stats = detect_clusters(np.array([0, 1, 1, 0, 1]), 1.1)
    assert set(hc_stats[:, 1]) == {0}
    clusters = point_sets_by_id(hc_stats)
    assert len(clusters) == 2
    assert set(map(frozenset, clusters.values())) == {
        frozenset({(1, 0), (2, 0)}), frozenset({(4, 0)})
    }


def test_detect_clusters_output_shape_dtype():
    n_occupied = np.count_nonzero(TOY_LATTICE)
    linkage_matrix, hc_stats = detect_clusters(TOY_LATTICE, 1.1)
    assert hc_stats.shape == (n_occupied, 3)
    assert hc_stats.dtype == np.int32
    assert linkage_matrix.shape == (n_occupied - 1, 4)


def test_detect_clusters_all_zeros():
    # scipy linkage raises on the empty coordinate set produced by np.nonzero
    with pytest.raises(ValueError):
        detect_clusters(np.zeros((5, 5)), 1.1)


def test_extract_cluster_coordinates_straddle_flags():
    _, hc_stats = detect_clusters(TOY_LATTICE, 1.1)
    touches = {
        (int(x), int(y)): bool(touch)
        for xs, ys, touch, _ in extract_cluster_coordinates(hc_stats, LATTICE_5X5)
        for x, y in zip(xs, ys)
    }
    assert touches[(4, 0)] is True  # row 0 = top/bottom-seam boundary
    assert touches[(4, 4)] is True
    assert touches[(1, 1)] is False  # interior strip at col 1
    assert touches[(1, 3)] is False
    assert touches[(3, 2)] is False  # col 3, not at either edge


def test_extract_cluster_coordinates_yields_all_points():
    _, hc_stats = detect_clusters(TOY_LATTICE, 1.1)
    total = sum(
        len(xs) for xs, _, _, _ in extract_cluster_coordinates(hc_stats, LATTICE_5X5)
    )
    assert total == np.count_nonzero(TOY_LATTICE)


def test_extract_cluster_coordinates_corner_cluster():
    lattice = np.zeros((5, 5))
    lattice[0, 0] = lattice[0, 4] = 1
    _, hc_stats = detect_clusters(lattice, 0.5)
    clusters = point_sets_by_id(hc_stats)
    assert len(clusters) == 2
    assert clusters[0] != clusters[1]
    # Both single-site clusters sit on the boundary -> both straddle
    assert all(bool(touch) for _, _, touch, _ in
               extract_cluster_coordinates(hc_stats, LATTICE_5X5))


def test_find_cluster_grouping_paper_example():
    _, hc_stats = detect_clusters(TOY_LATTICE, 1.1)
    coords = list(extract_cluster_coordinates(hc_stats, LATTICE_5X5))
    points = coords_by_id(coords)

    # (0, 4) and (4, 4) sit on the periodic seam and must merge
    id_at_top = next(cid for cid, pts in points.items() if pts == {(4, 0)})
    id_at_bottom = next(cid for cid, pts in points.items() if pts == {(4, 4)})
    merged = {id_at_top, id_at_bottom}
    singletons = set(points) - merged

    grouping = find_cluster_grouping(coords, LATTICE_5X5)
    assert grouping is not None
    groups = {frozenset(int(m) for m in group) for group in grouping}
    assert groups == {frozenset(merged)} | {frozenset({cid}) for cid in singletons}


def test_find_cluster_grouping_transitive():
    # A touches the left edge at y=2; B touches the right edge (y=2) AND the
    # left edge (y=4); C touches the right edge at y=4. A and C never overlap
    # directly, so BFS must chain them transitively across the two seams.
    coords = [
        (np.array([0]), np.array([2]), True, 0),
        (np.array([5, 0]), np.array([2, 4]), True, 1),
        (np.array([5]), np.array([4]), True, 2),
    ]
    grouping = find_cluster_grouping(coords, {"sizeX": 6, "sizeY": 6})
    assert grouping is not None
    assert {frozenset(int(m) for m in group) for group in grouping} == {
        frozenset({0, 1, 2})
    }


def test_find_cluster_grouping_ids_need_not_be_positions():
    # The pairing matrix is indexed by the cluster id (slot 3 of each tuple),
    # not by the tuple's position in the list.
    coords = [
        (np.array([0]), np.array([3]), True, 0),
        (np.array([5]), np.array([3]), True, 2),
        (np.array([2]), np.array([1]), False, 1),
    ]
    grouping = find_cluster_grouping(coords, {"sizeX": 6, "sizeY": 6})
    assert grouping is not None
    assert {frozenset(int(m) for m in group) for group in grouping} == {
        frozenset({0, 2}),
        frozenset({1}),
    }


def test_find_cluster_grouping_no_straddle():
    lattice = np.zeros((5, 5))
    for row, col in [(1, 1), (1, 2), (2, 1)]:
        lattice[row, col] = 1
    _, hc_stats = detect_clusters(lattice, 1.1)
    coords = list(extract_cluster_coordinates(hc_stats, LATTICE_5X5))
    grouping = find_cluster_grouping(coords, LATTICE_5X5)
    # Interior-only lattice: nothing touches, pairing matrix stays all zeros.
    # An all-zero pairing matrix is degenerate (see bfs.connected_components),
    # so no grouping is produced at all rather than singleton lists.
    assert grouping is None
    assert len(coords) == 1


def test_find_cluster_grouping_empty():
    # (0, 0) pairing matrix is degenerate; bfs.connected_components returns None
    assert find_cluster_grouping([], LATTICE_5X5) is None


def test_detect_touch_private():
    interior = (np.array([1, 2]), np.array([1, 2]))
    assert not _detect_touch(*interior, {"sizeX": 5, "sizeY": 5})
    # On a 3x3 lattice index 2 IS the boundary, so the same coords touch
    assert _detect_touch(*interior, {"sizeX": 3, "sizeY": 3})
    assert _detect_touch(np.array([0, 1]), np.array([1, 1]), {"sizeX": 5, "sizeY": 5})
    assert _detect_touch(np.array([1, 2]), np.array([2, 2]), {"sizeX": 3, "sizeY": 3})
    # touch_threshold=2: one boundary point is not enough, two are
    args = ({"sizeX": 5, "sizeY": 5},)
    assert not _detect_touch(np.array([0]), np.array([1]), touch_threshold=2, *args)
    assert _detect_touch(
        np.array([0, 0]), np.array([1, 2]), touch_threshold=2, *args
    )


def test_find_boundary_points_private():
    # Points (x=0, y=2) and (x=4, y=2): left and right edges only
    left, right, top, bottom, cid = _find_boundary_points(
        (np.array([0, 4]), np.array([2, 2]), False, 7), LATTICE_5X5
    )
    assert cid == 7
    assert left.tolist() == [2]
    assert right.tolist() == [2]
    assert top.size == 0
    assert bottom.size == 0

    # Cluster spread over all four edges; corners count on two edges
    left, right, top, bottom, _ = _find_boundary_points(
        (np.array([0, 4, 2, 2]), np.array([0, 4, 4, 0]), False, 9), LATTICE_5X5
    )
    assert left.tolist() == [0]
    assert right.tolist() == [4]
    assert sorted(top.tolist()) == [2, 4]
    assert sorted(bottom.tolist()) == [0, 2]


def test_test_partner_identity_private():
    boundary_a = [np.array([2]), np.array([]), np.array([]), np.array([]), 0]
    partner = [np.array([]), np.array([2]), np.array([]), np.array([]), 1]
    stranger = [np.array([]), np.array([3]), np.array([]), np.array([]), 2]
    assert _test_partner_identity(boundary_a, partner, 1)
    assert not _test_partner_identity(boundary_a, stranger, 1)
    # Same overlap of one pixel, but threshold demands two
    assert not _test_partner_identity(boundary_a, partner, 2)
