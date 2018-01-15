# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-01-11 14:48:17
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-15 18:30:19
from brian2 import prefs, ms, StateMonitor
import matplotlib.pyplot as plt

from NCSBrian2Lib.building_blocks.wta import WTA, plotWTA
from NCSBrian2Lib.core.groups import Neurons, Connections
from NCSBrian2Lib.stimuli.testbench import wta_testbench
from NCSBrian2Lib import NCSNetwork
from NCSBrian2Lib.models.synapse_models import DPISyn

prefs.codegen.target = 'numpy'
Net = NCSNetwork()
duration = 500
testbench = wta_testbench()

wtaParams = {'weInpWTA': 100,
             'weWTAInh': 150,
             'wiInhWTA': -100,
             'weWTAWTA': 50,
             'sigm': 5,
             'rpWTA': 3 * ms,
             'rpInh': 1 * ms
             }

gtestWTA = WTA(name='testWTA', dimensions=2, num_neurons=16, num_inh_neurons=50, num_inputs=2, block_params=wtaParams)

testbench.stimuli(num_neurons=16, dimensions=2, start_time=100, end_time=duration)
testbench.background_noise(num_neurons=16**2, rate=10)

gtestWTA.inputGroup.set_spikes(indices=testbench.indices, times=testbench.times * ms)
noise_syn = Connections(testbench.noise_input, gtestWTA,
                        equation_builder=DPISyn(), name="noise_syn",)
noise_syn.connect("i==j")
noise_syn.weight = 150

statemon_inhSyn = StateMonitor(gtestWTA.Groups['synInhWTA1i'], variables="Ii_syn", record=True)
Net.add(gtestWTA, testbench.noise_input, noise_syn, statemon_inhSyn)

Net.run(duration * ms)

plotWTA(name='testWTA', start_time=0 * ms, end_time=duration * ms, num_neurons=16, WTAMonitors=gtestWTA.Monitors)
