

def intersection_over_union(pt1, pt2, lattice_params):
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
    indices_1 = set([ _get_scalar_index(x, y, max_y) for x,y in zip(pt1[0], pt1[1]) ])
    indices_2 = set([ _get_scalar_index(x, y, max_y) for x,y in zip(pt2[0], pt2[1]) ])

    intersection = len(indices_1.intersection(indices_2))
    union = len(indices_1.union(indices_2))

    return intersection/1./union

# pt1 = [(1,2,3), (4,5,6)]
# pt2 = [(1,2,4), (4,5,6)]
# intersection_over_union(pt1, pt2, {'sizeY': 225})
# Answer 0.5

# pt1 = [(1,2,3), (4,5,6)]
# pt2 = [(1,2,3), (4,5,6)]
# intersection_over_union(pt1, pt2, {'sizeY': 225})
# Answer 1.0
