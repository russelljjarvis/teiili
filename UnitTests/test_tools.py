# -*- coding: utf-8 -*-
# @Author: Moritz Milde
# @Date:   2017-12-17 13:06:18
# @Last Modified by:   mmilde
# @Last Modified time: 2017-12-19 13:13:42
"""
This file contains unittest for tools.py
"""

import unittest
import numpy as np
import os
from NCSBrian2Lib.Tools import tools


class TestTools(unittest.TestCase):

    def test_printStates(self):
        self.assertRaises(UserWarning, tools.printStates, 5)

    def test_returnValueIf(self):
        testVal = 3.7
        greaterThanVal = 2.5
        smallerThanVal = 10.
        returnValTrue = 1337
        returnValFalse = 42
        returnVal = tools.returnValueIf(testVal, greaterThanVal, smallerThanVal,
                                        returnValTrue, returnValFalse)
        self.assertEqual(returnVal, 1337)
        testVal = 2.4
        returnVal = tools.returnValueIf(testVal, greaterThanVal, smallerThanVal,
                                        returnValTrue, returnValFalse)
        self.assertEqual(returnVal, 42)

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

    def test_dist2d(self):
        pass

    def test_fkernel1d(self):
        w = tools.fkernel1d(3, 5, 0.7)
        x = 3 - 5
        exponent = -(x**2) / (2 * 0.7**2)
        self.assertEqual(w, (1 + 2 * exponent) * np.exp(exponent))

    def test_fkernelgauss1d(self):
        i = 3
        j = 5
        gsigma = 0.7
        w = tools.fkernelgauss1d(i, j, gsigma)
        self.assertEqual(w, np.exp(-((i - j)**2) / (2 * gsigma**2)))

    def test_makeGaussian(self):
        pass

    def test_aedat2numpy(self):
        self.assertRaises(ValueError, tools.aedat2numpy, camera='DAVIS128')
        # Create a small/simple aedat file to test all functions which rely on edat2numpy

    def test_dvs2ind(self):
        self.assertRaises(AssertionError, tools.dvs2ind, eventDirectory=1337)
        self.assertRaises(AssertionError, tools.dvs2ind,
                          eventDirectory='/These/Are/Not/Events.txt')
        # os.command('touch /tmp/Events.npy')
        # self.assertRaises(AssertionError, tools.dvs2ind, Events=np.zeros((4, 100)),
        #                   eventDirectory='/tmp/Events.npy')
        # os.command('rm /tmp/Events.npy')


if __name__ == '__main__':
    unittest.main()
