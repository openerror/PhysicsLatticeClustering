#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
#from clustering import detectClusters, detectTouch
from BFS import connectedComponents

import numpy as np

#class TestDetectClusters(unittest.TestCase):
#    def test_something():
#        pass 

class TestBFS(unittest.TestCase):
    def setUp(self):
        self.invalid_inputs = [None, np.zeros(1)]
        self.sample_graphs = []

    def test_invalid_inputs(self):
        # Should refuse to process and return None
        for data in self.invalid_inputs:
            self.assertIsNone(connectedComponents(data))
    
#    def test_simple_graphs(self):
#        pass
#        # Return the connected components, given some connectivity matrices
#        # self.assertTrue(np.all())

if __name__ == '__main__':
    unittest.main(verbosity=2)