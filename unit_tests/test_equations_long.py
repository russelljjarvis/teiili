# -*- coding: utf-8 -*-
# @Author: Moritz Milde
# @Date:   2017-12-17 13:06:18
# @Last Modified by:   Moritz Milde
# @Last Modified time: 2018-06-11 18:19:14
"""
This file contains unittest for equations
"""

import unittest
import matplotlib.pyplot as plt
import numpy as np
#import os
from teili import NeuronEquationBuilder, SynapseEquationBuilder, Neurons, teiliNetwork
from teili.models.neuron_models import Izhikevich as neuron_model

from brian2 import prefs, ms
prefs.codegen.target = "numpy"
from brian2 import prefs, ms, pA, nA, StateMonitor, device, set_device,\
    second, msecond, defaultclock, TimedArray, mV, pfarad
pF = pfarad

class TestEquations(unittest.TestCase):

    # def test_ExpAdaptIF(self):
    #     ExpAdaptIF = NeuronEquationBuilder.import_eq(
    #         'ExpAdaptIF', num_inputs=1)
    #     testNeurons = Neurons(1, equation_builder=ExpAdaptIF(
    #         num_inputs=1), name="testNeuron", verbose=False)
    #     Net = teiliNetwork()
    #     Net.add(testNeurons)
    #     Net.run(5 * ms)

    # def test_DPI(self):
    #     DPI = NeuronEquationBuilder.import_eq('DPI', num_inputs=1)
    #     testNeurons = Neurons(1, equation_builder=DPI(
    #         num_inputs=1), name="testNeuron", verbose=False)
    #     Net = teiliNetwork()
    #     Net.add(testNeurons)
    #     Net.run(5 * ms)

    def test_Izhikevich(self):
        # Izhikevich = NeuronEquationBuilder.import_eq(
        #     'Izhikevich', num_inputs=1)
        Izhikevich = neuron_model(
            num_inputs=1)
        testNeurons = Neurons(1, equation_builder=Izhikevich(
            num_inputs=1), name="testNeuron", verbose=False)
        Net = teiliNetwork()
        I_bias = 1000 * pA + 500 * pA
        testNeurons.namespace.update({'I_bias':I_bias})
        testNeurons.run_regularly("Iconst = I_bias",dt=1*ms)

        statemon_izh = StateMonitor(testNeurons,
                           ('Ie0', 'Ii0','Iconst','Vm','Iadapt'),
                           record=True,
                           name='statemon_izh')
        Net.add(testNeurons, statemon_izh)
        testNeurons.Vm = -60 * mV
        testNeurons.Cm = 250 * pF
        testNeurons.print_equations()
        Net.run(1e3 * ms)

        for var in [statemon_izh.Vm.T, statemon_izh.Iadapt.T]:
            plt.figure()
            plt.plot(statemon_izh.t/ms, var)
        plt.show(0)
        # import ipdb; ipdb.set_trace()
        
if __name__ == '__main__':
    unittest.main()
