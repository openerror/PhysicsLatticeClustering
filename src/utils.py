#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from numpy import sum as npSum
from os.path import join as pathJoin

def IoU(prev, current):
    '''
    Calculate the intersection-over-union of two clusters, prev and current. The cluster
    coordinates are in principle, integers, because they are coordinates on a lattice.
    In practice, even when they are casted to floats, this function still works.

    prev: Python list [X, Y], where X and Y are NumPy arrays
    current: ditto
    '''
    prevX, prevY = prev
    currentX, currentY = current

    prevArea, currentArea = len(prevX), len(currentX)
    intersect = 0.0

    for px,py in zip(prevX, prevY):
        for cx,cy in zip(currentX, currentY):
            if (px == cx) and (py == cy):
                intersect += 1.0

    union = prevArea + currentArea - intersect
    iou = intersect/union
    return iou

def detectStraddle(X, Y, paramsDict):
    '''
        paramsDict["sizeY/sizeX"] specifies the grid size
        Then, how many points within a set of coordinates (X, Y) are
        touching the grid bounary? Return True/False if above/below a
        certain threshold
    '''
    maxY = paramsDict["sizeY"]-1
    maxX = paramsDict["sizeX"]-1

    borderPts_X = npSum(X == 0) + npSum(X == maxX)
    borderPts_Y = npSum(Y == 0) + npSum(Y == maxY)

    if (borderPts_X > 5 or borderPts_Y > 5):
        return True
    else:
        return False

        
# def IoU_test():
#     ''' Generate two overlapping LINES and check
#         Each line is 10 pts long, and 5 pts lie in the intersection
#         Expected IoU = 5 / (20 - 5) = 1/3 = 0.333333...
#     '''
#
#     testPrev = (range(10), range(10))
#     testCurrent = (range(5,15), range(5,15))
#
#     return IoU(testPrev, testCurrent)
