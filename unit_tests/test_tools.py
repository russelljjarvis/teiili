# -*- coding: utf-8 -*-
# @Author: Moritz Milde
# @Date:   2017-12-17 13:06:18
# @Last Modified by:   Moritz Milde
# @Last Modified time: 2018-06-01 16:53:07
"""
This file contains unittest for tools.py
"""

import unittest
import numpy as np
import os
from teili.tools import indexing, converter, misc, synaptic_kernel


class TestTools(unittest.TestCase):

    # def test_printStates(self):
    #     self.assertRaises(UserWarning, misc.printStates, 5)

    def test_return_value_if(self):
        testVal = 3.7
        greater_than_val = 2.5
        smaller_than_val = 10.
        return_val_true = 1337
        return_val_false = 42
        return_val = misc.return_value_if(testVal, greater_than_val, smaller_than_val,
                                          return_val_true, return_val_false)
        self.assertEqual(return_val, 1337)
        testVal = 2.4
        return_val = misc.return_value_if(testVal, greater_than_val, smaller_than_val,
                                          return_val_true, return_val_false)
        self.assertEqual(return_val, 42)

    def test_xy2ind_single(self):
        x = 127
        y = 5
        ncols = 240
        nrows = 240
        ind = indexing.xy2ind(x, y, nrows, ncols)
        self.assertEqual(ind, int(x) * ncols + int(y))

    def test_xy2ind_array(self):
        x = np.arange(0, 128)
        y = np.arange(0, 128)
        n2d_neurons = 128
        ind = indexing.xy2ind(x, y, n2d_neurons, n2d_neurons)
        self.assertEqual(list(ind), list(x + (y * n2d_neurons)))

    def test_ind2x(self):
        ind = 1327
        n2d_neurons = 240
        x = indexing.ind2x(ind, n2d_neurons, n2d_neurons)
        self.assertEqual(x, np.floor_divide(np.round(ind), n2d_neurons))

    def test_ind2y(self):
        ind = 1327
        n2d_neurons = 240
        y = indexing.ind2y(ind, n2d_neurons, n2d_neurons)
        self.assertEqual(y, np.mod(np.round(ind), n2d_neurons))

    def test_ind2xy_square(self):
        ind = 1327
        n2d_neurons = 240
        coordinates = indexing.ind2xy(ind, n2d_neurons, n2d_neurons)
        self.assertEqual(coordinates, np.unravel_index(
            ind, (n2d_neurons, n2d_neurons)))

    def test_ind2xy_rectangular(self):
        ind = 1327
        nrows, ncols = (240, 180)
        #self.assertTupleEqual(n2d_neurons, (240, 180))
        coordinates = indexing.ind2xy(ind, nrows, ncols)
        self.assertEqual(coordinates, np.unravel_index(ind, (nrows, ncols)))

    def test_fdist2d(self):
        pass

    def test_dist2d(self):
        pass

    def test_kernel_mexican_1d(self):
        w = synaptic_kernel.kernel_mexican_1d(3, 5, 0.7)
        x = 3 - 5
        exponent = -(x**2) / (2 * 0.7**2)
        self.assertEqual(w, (1 + 2 * exponent) * np.exp(exponent))

    def test_kernel_gauss_1d(self):
        i = 3
        j = 5
        gsigma = 0.7
        w = synaptic_kernel.kernel_gauss_1d(i, j, gsigma)
        self.assertEqual(w, np.exp(-((i - j)**2) / (2 * gsigma**2)))

    def test_makeGaussian(self):
        pass

    def test_aedat2numpy(self):
        self.assertRaises(FileNotFoundError,
                          converter.aedat2numpy, datafile='/tmp/aerout.aedat')
        # self.assertRaises(ValueError, converter.aedat2numpy, datafile='tmp/aerout.aedat', camera='DAVIS128')
        # Create a small/simple aedat file to test all functions which rely on
        # edat2numpy

    def test_dvs2ind(self):
        self.assertRaises(AssertionError, converter.dvs2ind,
                          event_directory=1337)
        self.assertRaises(AssertionError, converter.dvs2ind,
                          event_directory='/These/Are/Not/Events.txt')
        # os.command('touch /tmp/Events.npy')
        # self.assertRaises(AssertionError, tools.dvs2ind, Events=np.zeros((4, 100)),
        #                   event_directory='/tmp/Events.npy')
        # os.command('rm /tmp/Events.npy')


if __name__ == '__main__':
    unittest.main()
