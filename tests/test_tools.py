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
import copy
from teili.tools import indexing, converter, misc, synaptic_kernel, sorting


class TestToolsMatrix(unittest.TestCase):
    def setUp(self):
        # Definitions
        n_rows = 50
        n_cols = 50
        diag_width = 3
        max_weight = 15
        noise = True
        noise_probability = 0.05
        conn_probability = 0.9
        np.random.seed(26)

        test_matrix = np.zeros((n_rows, n_cols))
        if max_weight > 1:
            test_matrix = max_weight*np.random.rand(n_rows, n_cols)
            test_matrix = test_matrix.astype(int)
            test_matrix = np.clip(test_matrix, 0, int(max_weight/2))
        else:
            test_matrix = (max_weight*np.random.rand(n_rows, n_cols))
            test_matrix = np.clip(test_matrix, 0, max_weight)

        # Construct matrix
        ids = [x for x in range(n_rows)]
        for central_id in ids:
            # Add values on a diagonal with a given width
            add_to_ids = np.arange(central_id-diag_width,
                                   central_id+diag_width+1)
            values_to_add = np.array(
                    [max_weight for _ in range(2*diag_width+1)])
            # Remove values outside dimensions of matrix
            values_to_add = np.delete(values_to_add, np.append(
                np.where(add_to_ids < 0), np.where(add_to_ids > n_rows-1)))
            add_to_ids = np.delete(add_to_ids, np.append(
                np.where(add_to_ids < 0), np.where(add_to_ids > n_rows-1)))
            try:
                test_matrix[central_id][add_to_ids] = values_to_add
            except IndexError:
                continue

        # Add noise
        if noise:
            noise_ind = np.where(
                    np.random.rand(n_rows, n_cols) < noise_probability)
            test_matrix[noise_ind] = 0.3

        # Shuffle matrix
        # test matrices (shuffled along rows and/or columns)
        shuffled_matrix_tmp = np.zeros((n_rows, n_cols))
        self.shuffled_matrix = np.zeros((n_rows, n_cols))
        self.shuffled_input_matrix = np.zeros((n_rows, n_cols))
        # 32 bits is enough for numbers up to about 4 billion
        inds = np.arange(n_cols, dtype='uint32')
        np.random.shuffle(inds)
        shuffled_matrix_tmp[:, inds] = copy.deepcopy(test_matrix)
        self.shuffled_matrix[inds, :] = shuffled_matrix_tmp
        self.shuffled_input_matrix[:, inds] = copy.deepcopy(test_matrix)

        # Add connectivity of recurrent matrix
        conn_ind = np.where(np.random.rand(n_rows, n_cols) < conn_probability)
        source, target = conn_ind[0], conn_ind[1]
        conn_matrix = [[] for x in range(n_cols)]
        rec_matrix = [[] for x in range(n_cols)]
        for ind, val in enumerate(source):
            conn_matrix[val].append(target[ind])
        self.conn_matrix = np.array(conn_matrix, dtype=object)
        for i_source, i_target in enumerate(self.conn_matrix):
            rec_matrix[i_source] = self.shuffled_matrix[i_source, i_target]
        self.rec_matrix = np.array(rec_matrix, dtype=object)

    def test_matrix_permutation(self):
        n_rows = np.size(self.shuffled_matrix, 0)
        n_cols = np.size(self.shuffled_matrix, 1)
        tmp_matrix = copy.deepcopy(self.shuffled_input_matrix)
        sorted_matrix = sorting.SortMatrix(ncols=n_cols, nrows=n_rows, axis=1,
                                           matrix=tmp_matrix)
        permutation_expected = [12, 19, 25, 41, 42, 45, 9, 35, 2, 4, 46,
                                40, 14, 22, 38, 47, 31, 27, 3, 10, 49, 21,
                                34, 36, 43, 28, 44, 6, 33, 24, 39, 17, 18,
                                32, 1, 0, 5, 23, 26, 13, 48, 15, 11, 37, 16,
                                8, 7, 20, 30, 29]
        self.assertEqual(sorted_matrix.permutation, permutation_expected)

    def test_rec_matrix_permutation(self):
        n_rows = np.size(self.shuffled_matrix, 0)
        n_cols = np.size(self.shuffled_matrix, 1)
        tmp_matrix = copy.deepcopy(self.shuffled_matrix)
        sorted_matrix = sorting.SortMatrix(ncols=n_cols, nrows=n_rows, axis=1,
                                           matrix=tmp_matrix, rec_matrix=True)
        permutation_expected = [12, 19, 25, 41, 42, 45, 9, 35, 2, 4, 46,
                                40, 14, 22, 38, 47, 31, 27, 3, 10, 49, 21,
                                34, 36, 43, 28, 44, 6, 33, 24, 39, 17, 18,
                                32, 1, 0, 5, 23, 26, 13, 48, 15, 11, 37, 16,
                                8, 7, 20, 30, 29]
        self.assertEqual(sorted_matrix.permutation, permutation_expected)

    def test_sparse_matrix_permutation(self):
        n_rows = np.size(self.shuffled_matrix, 0)
        n_cols = np.size(self.shuffled_matrix, 1)
        tmp_matrix = copy.deepcopy(self.rec_matrix)
        sorted_matrix = sorting.SortMatrix(ncols=n_cols, nrows=n_rows, axis=1,
                                           matrix=tmp_matrix, rec_matrix=True,
                                           target_indices=self.conn_matrix)
        permutation_expected = [28, 34, 21, 36, 43, 44, 33, 6, 39, 24,
                                17, 18, 32, 26, 5, 1, 23, 0, 31, 49, 10,
                                3, 27, 38, 47, 22, 14, 29, 12, 19, 41, 25,
                                42, 45, 35, 9, 2, 40, 46, 4, 7, 8, 16, 37,
                                30, 20, 11, 15, 13, 48]
        self.assertEqual(sorted_matrix.permutation, permutation_expected)


class TestTools(unittest.TestCase):

    # def test_printStates(self):
    #     self.assertRaises(UserWarning, misc.printStates, 5)

    def test_return_value_if(self):
        testVal = 3.7
        greater_than_val = 2.5
        smaller_than_val = 10.
        return_val_true = 1337
        return_val_false = 42
        return_val = misc.return_value_if(testVal, greater_than_val,
                                          smaller_than_val, return_val_true,
                                          return_val_false)
        self.assertEqual(return_val, 1337)
        testVal = 2.4
        return_val = misc.return_value_if(testVal, greater_than_val,
                                          smaller_than_val, return_val_true,
                                          return_val_false)
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
        # self.assertTupleEqual(n2d_neurons, (240, 180))
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
        # self.assertRaises(ValueError, converter.aedat2numpy,
        #                   datafile='tmp/aerout.aedat', camera='DAVIS128')
        # Create a small/simple aedat file to test all functions which rely on
        # edat2numpy

    def test_dvs2ind(self):
        self.assertRaises(AssertionError, converter.dvs2ind,
                          event_directory=1337)
        self.assertRaises(AssertionError, converter.dvs2ind,
                          event_directory='/These/Are/Not/Events.txt')
        # os.command('touch /tmp/Events.npy')
        # self.assertRaises(AssertionError, tools.dvs2ind,
        #                   Events=np.zeros((4, 100)),
        #                   event_directory='/tmp/Events.npy')
        # os.command('rm /tmp/Events.npy')


if __name__ == '__main__':
    unittest.main()
