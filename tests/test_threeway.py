#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:34:57 2019

@author: dzenn
"""

import unittest
import numpy as np
from teili.building_blocks.threeway import Threeway
from teili.tools.three_way_kernels import A_plus_B_equals_C
from brian2 import prefs

prefs.codegen.target = "numpy"
#TODO: test plotting;


class TestThreeway(unittest.TestCase):

    def setUp(self):
        self.TW = Threeway('TestTW1', hidden_layer_gen_func = A_plus_B_equals_C,
                           cutoff = 2, monitor=True)

    def test_attributes(self):
        self.assertEqual(self.TW.num_neurons, 364)
        self.assertEqual(self.TW.A.num_neurons + self.TW.B.num_neurons +
                         self.TW.C.num_neurons + self.TW.H._groups['n_exc'].N, 48 + 256)

    def test_sub_blocks(self):
        self.assertEqual(len(self.TW.groups), 38)
        self.assertEqual(len(self.TW.sub_blocks), 4)


if __name__ == '__main__':
    unittest.main()
