#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from BFS import connectedComponents

def grouping(cluster_coordinates, sim_params_dict):
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
    boundary_touching_clusters = list(filter(lambda x: x[2], cluster_coordinates))
    points_at_boundary = [_findBoundaryPts(pts, sim_params_dict) for pts in boundary_touching_clusters]
    
    straddle_pairing = np.zeros(shape=(len(cluster_coordinates), 
                                       len(cluster_coordinates)),dtype='int16')
    
    for i,_ in enumerate(points_at_boundary):
        for j in range(i + 1, len(points_at_boundary)):
            if _testPartner( points_at_boundary[i], points_at_boundary[j], 1):
                cluster_id_1 = points_at_boundary[i][4]
                cluster_id_2 = points_at_boundary[j][4]
                                  
                straddle_pairing[cluster_id_1, cluster_id_2] = 1
                straddle_pairing[cluster_id_2, cluster_id_1] = 1
        
    # Use breadth-first search to determine clusters that are touching 'transitively'
    merged_clusters = connectedComponents(straddle_pairing)
    return merged_clusters


def _findBoundaryPts(coordinates, paramDict):
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

def _testPartner(boundaryA, boundaryB, partnerThreshold):
    '''
        Decide whether A and B are actually part of a same cluster that
        straddles the periodic boundary

        boundaryA and boundaryB: Python lists of format [left, right, top, bottom]
        They record the coordinates of the pts touching the boundaries.
        See _findStraddlingPts() above.

        partnerThreshold: an integer; if the clusters A and B touch each other for
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

    if overlap >= partnerThreshold:
        return True
    else:
        return False