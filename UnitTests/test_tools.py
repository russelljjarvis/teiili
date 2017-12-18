# -*- coding: utf-8 -*-
# @Author: Moritz Milde
# @Date:   2017-12-17 13:06:18
# @Last Modified by:   Moritz Milde
# @Last Modified time: 2017-12-18 18:01:28
"""
This file contains unittest for tools.py
"""

import unittest
import numpy as np
from NCSBrian2Lib.Tools import tools


class TestTools(unittest.TestCase):

    def test_xy2ind_single(self):
        x = 127
        y = 5
        n2dNeurons = 240
        ind = tools.xy2ind(x, y, n2dNeurons)
        self.assertEqual(ind, int(x) + int(y) * n2dNeurons)

    def test_xy2ind_array(self):
        x = np.arange(0, 128)
        y = np.arange(0, 128)
        n2dNeurons = 128
        ind = tools.xy2ind(x, y, n2dNeurons)
        self.assertEqual(ind.tolist(), (x + (y * n2dNeurons)).tolist())

    def test_ind2x(self):
        ind = 1327
        n2dNeurons = 240
        x = tools.ind2x(ind, n2dNeurons)
        self.assertEqual(x, np.floor_divide(np.round(ind), n2dNeurons))

    def test_ind2y(self):
        ind = 1327
        n2dNeurons = 240
        y = tools.ind2y(ind, n2dNeurons)
        self.assertEqual(y, np.mod(np.round(ind), n2dNeurons))

    def test_ind2xy_square(self):
        ind = 1327
        n2dNeurons = 240
        coordinates = tools.ind2xy(ind, n2dNeurons)
        self.assertEqual(coordinates, np.unravel_index(ind, (n2dNeurons, n2dNeurons)))

    def test_ind2xy_rectangular(self):
        ind = 1327
        n2dNeurons = (240, 180)
        self.assertTupleEqual(n2dNeurons, (240, 180))
        coordinates = tools.ind2xy(ind, n2dNeurons)
        self.assertEqual(coordinates, np.unravel_index(ind, (n2dNeurons[0], n2dNeurons[1])))

    def test_fdist2d(self):
        pass

    def test_makeGaussian(self):
        pass



if __name__ == '__main__':
    unittest.main()
