# -*- coding: utf-8 -*-
# @Author: Moritz Milde
# @Date:   2017-12-18 18:02:30
# @Last Modified by:   mmilde
# @Last Modified time: 2017-12-27 11:40:55
"""
This file contains unittest for testbench.py
"""

import unittest
import numpy as np
from NCSBrian2Lib.stimuli.testbench import octa_testbench, stdp_testbench

octa_testbench = octa_testbench()
stdp_testbench = stdp_testbench()


class TestTestbench(unittest.TestCase):
    def test_aedat2events(self):
        self.assertRaises(AssertionError, octa_testbench.aedat2events, rec=2)
        self.assertRaises(AssertionError, octa_testbench.aedat2events, rec='/tmp/test.aedat')

    def test_infinity(self):
        cAngle = 1.5
        position = octa_testbench.infinity(cAngle)
        self.assertTupleEqual(position, (np.cos(cAngle), np.sin(cAngle) * np.cos(cAngle)))

    def test_dda_round(self):
        x = 2.7
        x_rounded = octa_testbench.dda_round(x)
        self.assertEqual(x_rounded, int(x + 0.5))
        self.assertIs(type(x_rounded), int)

    def test_rotating_bar(self):
        self.assertRaises(UserWarning, octa_testbench.rotating_bar, artifical_stimulus=False)
        self.assertRaises(AssertionError, octa_testbench.rotating_bar, artifical_stimulus=False, rec_path='/tmp/')

    def test_translating_bar_infinity(self):
        self.assertRaises(UserWarning, octa_testbench.translating_bar_infinity, artifical_stimulus=False)
        self.assertRaises(AssertionError, octa_testbench.translating_bar_infinity, artifical_stimulus=False, rec_path='/tmp/')

    def test_rotating_bar_infinity(self):
        self.assertRaises(UserWarning, octa_testbench.rotating_bar_infinity, artifical_stimulus=False)
        self.assertRaises(AssertionError, octa_testbench.rotating_bar_infinity, artifical_stimulus=False, rec_path='/tmp/')

    def test_ball(self):
        self.assertRaises(AssertionError, octa_testbench.ball, rec_path='/tmp/')


if __name__ == '__main__':
    unittest.main()
