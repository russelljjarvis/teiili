# -*- coding: utf-8 -*-
# @Author: Moritz Milde
# @Date:   2017-12-17 13:06:18
# @Last Modified by:   Moritz Milde
# @Last Modified time: 2017-12-17 13:32:40
"""
This file contains unittest for tools.py
"""

import unittest
from NCSBrian2Lib.Tools import tools


class TestTools(unittest.TestCase):

    def test_xy2ind(self):
        x = 127
        y = 5
        n2dNeurons = 240
        ind = tools.xy2ind(x, y, n2dNeurons)
        self.assertEqual(ind, x + (y * n2dNeurons))


if __name__ == '__main__':
    unittest.main()
