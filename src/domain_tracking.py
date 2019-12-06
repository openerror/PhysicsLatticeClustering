import numpy as np
from collections import deque
from .clustering import detect_clusters, extract_cluster_coordinates, find_cluster_grouping


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


def find_partners(iou_matrix, threshold=0.0):
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
    iou_matrix[iou_matrix <= threshold] = 0.0

    # Find potential partners
    partners = iou_matrix.argmax(axis=1)

    # Mark clusters with no counterparts
    partners[np.all(iou_matrix == 0.0, axis=1)] = -1
    return partners


def generate_coordinate_dict(array, lattice_params, dendrogram_cutoff=1.1):
    '''
        Perform hierarchical clustering on a binarized array, taking into account 
        periodic bounary conditions.
        
        Args:
            array: 2D numpy array 
        Returns:
             dict[int] -> (List[int], List[int]), mapping a "cluster id" to the (X, Y) coordinates
    '''
    
    # Perform hierarchical clustering
    hcs = detect_clusters(data=array, method='single', dendrogram_cutoff=dendrogram_cutoff)
    
    # Collate from clustering output the information about *each* cluster 
    cluster_coordinates = extract_cluster_coordinates(hcs, lattice_params)
    
    # Finally, output lists of cluster ids. Each tuple contains clusters
    # that are merged together across periodic boundaries '''
    cluster_groups = find_cluster_grouping(cluster_coordinates, lattice_params)
    
    assert (cluster_groups is not None), 'WARNING: cluster_groups is None; likely no clusters detected'
    
    cdict = {}
    for final_cid, group in enumerate(cluster_groups):
        mask_collection = [hcs[:,-1]==cid for cid in group]
        final_mask = mask_collection[0]
        for mask in mask_collection[1:]:
            final_mask = final_mask | mask
        
        cdict[final_cid] = (hcs[final_mask,0], hcs[final_mask,1])
        
    return cdict


def track_cluster_lifetime(times, coordinate_dicts, center_index, iou_threshold, lattice_params):
    '''        
        Given the clusters located at array index "center_index", locate the earliest
        and latest time when any of the domains exist in the system. Existence continuity is defined by
        intersection-over-union (IoU) overlap between domains identified at different times.
        
        Args:
            times:: 1D array containing simulation timestamps (hr)
            coordinate_dicts::List[ dict[int] -> (List[int], List[int])]
            center_index::int
            iou_threshold::float [0...1) high-pass threshold for identifying clusters as "same"
        Returns:
            dict[int] -> collections.deque, a dictionary mapping cluster IDs at center_index
            to their time trajectory. Deque contains integer tuples of (time index, cluster id at that time)
    '''
    
    assert len(times)==len(coordinate_dicts), "ERROR: arrays containing simulation times \
                                                and coordinate dictionaries don't have the same length"
    existence_timestamps = {cid:deque([(center_index, cid)]) 
                            for cid in coordinate_dicts[center_index].keys()}
    
    
    def update_single_step(existence_timestamps, current_loc, current_cid, forward=True):
        ''' Refactored logic for partner search '''
        nonlocal lattice_params
        update_loc = (current_loc+1) if (forward==True) else (current_loc-1)
        
        # Compute IoU values of our cluster with those that exist at times[later]
        current_coordinates = {0:coordinate_dicts[current_loc][current_cid]}
        iou_matrix = compute_iou_matrix(current_coordinates, coordinate_dicts[update_loc], lattice_params)
        
        # DEBUG: len(partner)==1, because we are following one cluster only
        # Once partner is identified, update current_cid to reflect which cluster is tracked
        partner = find_partners(iou_matrix, iou_threshold)[0]
        current_cid = partner if (partner != -1) else -1
        if current_cid == -1:
            return current_loc, update_loc, current_cid
        
        # Update existence_timestamps
        if forward:
            existence_timestamps[center_cid].append((update_loc, current_cid))
            current_loc += 1; update_loc += 1
        else:
            existence_timestamps[center_cid].appendleft((update_loc, current_cid))                                        
            current_loc -= 1; update_loc -= 1

        return current_loc, update_loc, current_cid
    
    
    ## For each cluster identified at center_index, find all evolved versions of it in time
    for center_cid, Q in existence_timestamps.items():
        ## Search forward in time
        current, later = center_index, center_index+1
        current_cid = center_cid
        while later < len(times):
            current, later, current_cid = update_single_step(existence_timestamps, 
                                                             current, current_cid, forward=True)
            if current_cid == -1: break
        
        ## Search backward in time
        current, before = center_index, center_index-1
        current_cid = center_cid
        while before >= 0:
            current, before, current_cid = update_single_step(existence_timestamps, 
                                                              current, current_cid, forward=False)
            if current_cid == -1: break
       
    return existence_timestamps

# pt1 = [(1,2,3), (4,5,6)]
# pt2 = [(1,2,4), (4,5,6)]
# intersection_over_union(pt1, pt2, {'sizeY': 225})
# Answer 0.5

# pt1 = [(1,2,3), (4,5,6)]
# pt2 = [(1,2,3), (4,5,6)]
# intersection_over_union(pt1, pt2, {'sizeY': 225})
# Answer 1.0
