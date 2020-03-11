# -*- coding: utf-8 -*-
# @Author: Moritz Milde
# @Date:   2017-12-17 13:06:18
# @Last Modified by:   Moritz Milde
# @Last Modified time: 2018-06-11 18:19:14
"""
This file contains unit tests for equations.
"""

import unittest
import numpy as np
import os
from teili import NeuronEquationBuilder, SynapseEquationBuilder, Neurons, TeiliNetwork

from brian2 import prefs, ms
prefs.codegen.target = "numpy"


class TestEquations(unittest.TestCase):

    def test_ExpAdaptIF(self):
        filename = "ExpAdaptIF"

        ExpAdaptIF = NeuronEquationBuilder.import_eq(
            filename, num_inputs=1)
        testNeurons = Neurons(1, equation_builder=ExpAdaptIF(
            num_inputs=1), name="testNeuron", verbose=False)
        Net = TeiliNetwork()
        Net.add(testNeurons)
        Net.run(5 * ms)

    def test_DPI(self):
        filename = "DPI.py"

        DPI = NeuronEquationBuilder.import_eq(
            filename, num_inputs=1)
        testNeurons = Neurons(1, equation_builder=DPI(
            num_inputs=1), name="testNeuron", verbose=False)
        Net = TeiliNetwork()
        Net.add(testNeurons)
        Net.run(5 * ms)

    def test_Izhikevich(self):
        filename = os.path.join(os.path.expanduser('~'),
                                "teiliApps",
                                "equations",
                                "Izhikevich.py")

        Izhikevich = NeuronEquationBuilder.import_eq(
            filename, num_inputs=1)
        testNeurons = Neurons(1, equation_builder=Izhikevich(
            num_inputs=1), name="testNeuron", verbose=False)
        Net = TeiliNetwork()
        Net.add(testNeurons)
        Net.run(5 * ms)

    def test_ExpAdaptIF(self):
        filename = os.path.join(os.path.expanduser('~'),
                                "teiliApps",
                                "equations",
                                "ExpAdaptIF.py")

        ExpAdaptIF = NeuronEquationBuilder.import_eq(
            filename, num_inputs=1)
        testNeurons = Neurons(1, equation_builder=ExpAdaptIF(
            num_inputs=1), name="testNeuron", verbose=False)
        Net = TeiliNetwork()
        Net.add(testNeurons)
        Net.run(5 * ms)

    def test_ExpLIF(self):
        filename = os.path.join(os.path.expanduser('~'),
                                "teiliApps",
                                "equations",
                                "ExpAdaptIF.py")

        ExpLIF = NeuronEquationBuilder.import_eq(
            filename, num_inputs=1)
        testNeurons = Neurons(1, equation_builder=ExpLIF(
            num_inputs=1), name="testNeuron", verbose=False)
        Net = TeiliNetwork()
        Net.add(testNeurons)
        Net.run(5 * ms)
if __name__ == '__main__':
    unittest.main()
