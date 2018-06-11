# -*- coding: utf-8 -*-
# @Author: Moritz Milde
# @Date:   2017-12-17 13:06:18
# @Last Modified by:   Moritz Milde
# @Last Modified time: 2018-06-01 16:53:07
"""
This file contains unittest for equations
"""

import unittest
import numpy as np
#import os
from teili import NeuronEquationBuilder, SynapseEquationBuilder, Neurons, teiliNetwork

from brian2 import prefs, ms
prefs.codegen.target = "numpy"

class TestEquations(unittest.TestCase):

    def test_ExpAdaptIF(self):
        ExpAdaptIF = NeuronEquationBuilder.import_eq('ExpAdaptIF', num_inputs=1)
        testNeurons = Neurons(1, equation_builder=ExpAdaptIF(num_inputs=1), name="testNeuron", verbose = False)
        Net = teiliNetwork()
        Net.add(testNeurons)
        Net.run(5*ms)

    def test_DPI(self):
        DPI = NeuronEquationBuilder.import_eq('DPI', num_inputs=1)
        testNeurons = Neurons(1, equation_builder=DPI(num_inputs=1), name="testNeuron", verbose = False)
        Net = teiliNetwork()
        Net.add(testNeurons)
        Net.run(5*ms)

    def test_Izhikevich(self):
        Izhikevich = NeuronEquationBuilder.import_eq('Izhikevich', num_inputs=1)
        testNeurons = Neurons(1, equation_builder=Izhikevich(num_inputs=1), name="testNeuron", verbose = False)
        Net = teiliNetwork()
        Net.add(testNeurons)
        Net.run(5*ms)

if __name__ == '__main__':
    unittest.main()
