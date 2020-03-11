#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:47:46 2019

@author: matteo
"""
import unittest
import numpy as np
from teili.models.parameters.octa_params import wta_params, octa_params

from teili.building_blocks.octa import Octa
from brian2 import prefs

prefs.codegen.target = "numpy"


class TestOcta(unittest.TestCase):

    def test_attributes(self):
        test_octa = Octa(name='test_octa')

        self.assertEqual(test_octa._groups['n_proj'].N,
                         test_octa.sub_blocks['prediction'].groups['n_exc'].N)

    def test_sub_blocks(self):
        test_octa = Octa(name='test_octa')
        self.assertEqual(len(test_octa.groups), 26)
        self.assertEqual(len(test_octa.sub_blocks), 2)


if __name__ == '__main__':
    unittest.main()
