import numpy as np

def _intersection_over_union(ptA, ptB, lattice_params):
    '''
        Convert two sets of Cartesian coordinates to scalar index, and then compute their
        intersection-over-union.

        Args:
            pt1, pt2:: (list[int], list[int]) Two-member tuples containing X, Y coordinates
            lattice_params::dict Contains lattice dimensions
        Returns:
            float
    '''

    def _get_scalar_index(x:int, y:int, max_y:int) -> int:
        return int(y + max_y * x)

    max_y = lattice_params['sizeY']
    indices_A = set([ _get_scalar_index(x, y, max_y) for x,y in zip(ptA[0], ptA[1]) ])
    indices_B = set([ _get_scalar_index(x, y, max_y) for x,y in zip(ptB[0], ptB[1]) ])

    intersection = len(indices_A.intersection(indices_B))
    union = len(indices_A.union(indices_B))
    return intersection/1./union


def compute_iou_matrix(cluster_dict_A, cluster_dict_B, lattice_params):
    '''
        Compute pairwise intersection-over-union (IoU) values for clusters identified
        in dictionaries A and B.

        Args:
            cluster_dict_A, cluster_dict_B: outputs of generate_coordinate_dict()
            lattice_params: dict[str]->int containing keys 'sizeY' and 'sizeX'
        Returns:
            iou_matrix: numpy array recording IoU values
    '''

    iou_matrix = np.zeros(shape=(len(cluster_dict_A), len(cluster_dict_B)))

    for i,_ in enumerate(cluster_dict_A):
        for j,_ in enumerate(cluster_dict_B):
            iou_matrix[i, j] = _intersection_over_union(cluster_dict_A[i],
                                                       cluster_dict_B[j],
                                                       lattice_params)
    return iou_matrix


def find_cluster_partners(iou_matrix, threshold=0.0):
    '''
        Given a matrix of intersection-over-union (IoU) values between clusters, find matching partners.
        For each row (cluster), find highest IoU counterpart and return the corresponding column index.
        Applies high-pass thresholding to cut out weak overlaps; default to 0.0 which means no thresholding.

        Args:
            iou_matrix:: 2D numpy array computed by compute_iou_matrix()
            threshold::float
        Returns:
            partners:: 1D numpy integer array containing indices of cluster partners
    '''

    # Apply "high-pass" IoU filter
    assert (0.0 < threshold < 1.0), "IoU threshold must lie between (0, 1)"
    matrix[matrix <= threshold] = 0.0

    # Find potential partners
    partners = matrix.argmax(axis=1)

    # Mark clusters with no counterparts
    partners[np.all(matrix == 0.0, axis=1)] = -1
    return partners

# pt1 = [(1,2,3), (4,5,6)]
# pt2 = [(1,2,4), (4,5,6)]
# intersection_over_union(pt1, pt2, {'sizeY': 225})
# Answer 0.5

# pt1 = [(1,2,3), (4,5,6)]
# pt2 = [(1,2,3), (4,5,6)]
# intersection_over_union(pt1, pt2, {'sizeY': 225})
# Answer 1.0
