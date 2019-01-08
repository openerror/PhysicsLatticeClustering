# clustering.py
'''Apply hierarchical clustering on 1D/2D arrays. Clustering functionality is provided
by the excellent SciPy package.
'''

from scipy.cluster.hierarchy import linkage, fcluster
from copy import copy
import numpy as np

def extractClusterCoordinates(HCStats, sim_params_dict):
    '''
        Analyze the output of detectClusters(), and return
            1) the coordinates of every cluster identified;
            2) whether that cluster straddles the boundary.
        For now, assume data in the shape of rectangular grids.

        Args:
            HCStats: N-by-3 matrix; each row represents a coordinate point, and the columns
                    record the X, Y, and ID of the cluster to which this point belongs
            sim_params_dict: Python dictionary (see paramSpace.py) containing lattice sizes
        Returns:
            coordinates: a Python list containing 3-member tuples. Each tuple corresponds
                        to a cluster: (X coordinates, Y coordinates, straddle True/False)
    '''

    clusterIDs, clusterArea = np.unique(HCStats[:,-1], return_counts=True)
    X = HCStats[:,0]; Y = HCStats[:,1] #X,Y coordinates of ALL identified clusters

    # Identify ALL clusters, and return its coordinates
    # The coordinates can then be used as a mask to access occupancy NumPy arrays
    coordinates = []
    for cid in clusterIDs:
        mask = (HCStats[:,2] == cid)
        touch = detectTouch(X[mask], Y[mask], sim_params_dict) #Boolean var
        coordinates.append(( X[mask], Y[mask], touch, cid )) #Zero-base cids for future use as indices

    return coordinates

def detectClusters(data, dendrogram_cutoff, binarize_threshold=0.0, binarize=False, method='single'):
    '''
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
            binarize: boolean value indicating whether to call binarize() on input 'data'
            threshold: float-valued threshold for binarize()
        Returns:
            bw_array: the binarized data
            L: linkage matrix generated by clustering; can be visualized as dendrogram
            HCStats: a N-by-3 numpy array recording the clustering output

        For additional documentation of SciPy's hierarchical clustering routines,
        see https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
        and the official documentation @ https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
    '''

    if binarize:
        if (binarize_threshold > 0.0):
            bw_array = binarize(data, binarize_threshold)
            occupied_pts = np.transpose(np.nonzero(bw_array))
        else:
            raise ValueError("Must supply *POSITIVE* threshold value, if desired to binarize data.")
    else:
        # Assume already binarized; extract directly the nonzero coordinates
        bw_array = data
        occupied_pts = np.transpose(np.nonzero(bw_array))

    dim = occupied_pts.shape[1]
    if dim == 1:
        X = occupied_pts[:,0]
        Y = np.zeros(X.shape)
    elif dim == 2:
        X = occupied_pts[:,1]
        Y = occupied_pts[:,0]
    else:
        raise NotImplementedError("Function designed to handle only 1D/2D data. Higher\
        dimensions are not yet allowed.")

    # Generate linkage matrix, which describes the iterative clustering steps
    # And then, identifiy each point with a cluster
    linkage_matrix = linkage(occupied_pts, method=method, metric="euclidean")
    clusters = fcluster(linkage_matrix, dendrogram_cutoff, criterion="distance")
    clusters = clusters - 1 # Change to zero-base index

    # Engineering the feature matrix
    # Columns: (X coordinate, Y coordinate, cluster id)
    HCStats = np.stack((X,Y,clusters), axis = 1)

    # Return linkage matrix also, for dendrogram plotting
    return bw_array, linkage_matrix, HCStats


def detectTouch(X, Y, sim_params_dict, touch_threshold = 1):
    '''
        Determine whether a cluster touches the (periodic boundary), up a threshold.
        
        Specify a sizeY-by-sizeX rectangular grid from sim_params_dict. Then given
        a set of (X, Y) coordinates, determine how many of them are at the grid 
        boundary; return T/F according to threshold.
    '''
    
    maxY = sim_params_dict["sizeY"]-1
    maxX = sim_params_dict["sizeX"]-1

    borderPts_X = np.sum(X == 0) + np.sum(X == maxX)
    borderPts_Y = np.sum(Y == 0) + np.sum(Y == maxY)

    if (borderPts_X >= touch_threshold or borderPts_Y >= touch_threshold):
        return True
    else:
        return False

def binarize(data, threshold):
    '''
        Replace all array values above the threshold with 1, and any other values with 0.
        Returns result performed on a copy of the original array.

        Args:
            data: numpy array, or another data type that supports vectorized notation
            threshold: float value
        Returns:
            A tuple containing 1) the binarized data; 2) array indexes where
            the binarized data equals 1
    '''
    #assert(np.max(data) <= 1.0), "Must first normalize data range to [0,1]"
    binarized_data = copy(data)
    binarized_data[data < threshold] = 0
    binarized_data[data >= threshold] = 1
    return binarized_data
