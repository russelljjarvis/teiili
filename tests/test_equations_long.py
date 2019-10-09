# -*- coding: utf-8 -*-
# @Author: Moritz Milde
# @Date:   2017-12-17 13:06:18
# @Last Modified by:   mmilde
# @Last Modified time: 2018-08-08 15:57:30
"""
This file contains unittest for equations
"""

import unittest
import matplotlib.pyplot as plt
import numpy as np
#import os
from teili import NeuronEquationBuilder, SynapseEquationBuilder, Neurons, TeiliNetwork
from teili.models.neuron_models import Izhikevich, ExpLIF, DPI, ExpAdaptIF

from brian2 import prefs, ms
prefs.codegen.target = "numpy"
from brian2 import prefs, ms, pA, nA, StateMonitor, device, set_device,\
    second, msecond, defaultclock, TimedArray, mV, pfarad
pF = pfarad


class TestEquations(unittest.TestCase):

    def test_ExpLIF(self):
        testNeurons = Neurons(1,
                              equation_builder=ExpLIF(num_inputs=1),
                              name="testNeurons", verbose=False)
        Net = TeiliNetwork()
        Net.add(testNeurons)
        Net.run(1e-3 * ms)

    def test_ExpAdaptIF(self):
        testNeurons = Neurons(1,
                              equation_builder=ExpAdaptIF(num_inputs=1),
                              name="testNeurons", verbose=False)
        Net = TeiliNetwork()
        Net.add(testNeurons)
        Net.run(1e-3 * ms)

    def test_DPI(self):
        testNeurons = Neurons(1,
                              equation_builder=DPI(num_inputs=1),
                              name="testNeurons", verbose=False)
        Net = TeiliNetwork()
        Net.add(testNeurons)
        Net.run(1e-3 * ms)

    def test_Izhikevich(self):
        testNeurons = Neurons(1,
                              equation_builder=Izhikevich(num_inputs=1),
                              name="testNeurons", verbose=False)
        Net = TeiliNetwork()
        Net.add(testNeurons)
        Net.run(1e-3 * ms)

if __name__ == '__main__':
    unittest.main()
