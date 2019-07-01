#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:47:46 2019

@author: matteo
"""
import unittest
import numpy as np
from teili.tools.octa_tools.octa_param import wtaParameters, octaParameters, octa_neuron,\
DPIadp ,SynSTDGM, mismatch_synap_param
from teili.building_blocks.octa_hierarchicalBB import Octa
from brian2 import prefs

prefs.codegen.target = "numpy"


class TestOcta(unittest.TestCase):
    
    def setUp(self):
        self.OCTA =Octa(name='test_OCTA', 
                wtaParams = wtaParameters,
                 octaParams = octaParameters,     
                 neuron_eq_builder=octa_neuron,
                 num_input_neurons= octaParameters['num_input_neurons'],
                 num_neurons = octaParameters['num_neurons'],
                 stacked_inp = True,
                 noise= True,
                 monitor=True,
                 debug=False)


    def test_attributes(self):
        self.assertGreater(self.OCTA.num_input_neurons, self.OCTA.num_neurons)
        self.assertEqual(self.OCTA.sub_blocks['compressionWTA'].groups['spike_gen'].N,
                         self.OCTA.sub_blocks['predictionWTA'].groups['n_exc'].N )

    def test_sub_blocks(self):
        self.assertEqual(len(self.OCTA.groups), 24)
        self.assertEqual(len(self.OCTA.sub_blocks), 2)


if __name__ == '__main__':
    unittest.main()
