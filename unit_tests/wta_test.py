# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-01-11 14:48:17
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-25 16:29:42
from brian2 import prefs, ms, StateMonitor
import matplotlib.pyplot as plt
import numpy as np
from NCSBrian2Lib.building_blocks.wta import WTA, plotWTA
from NCSBrian2Lib.core.groups import Neurons, Connections
from NCSBrian2Lib.stimuli.testbench import wta_testbench
from NCSBrian2Lib import NCSNetwork
from NCSBrian2Lib.models.synapse_models import DPISyn, DPIstdp

prefs.codegen.target = 'numpy'
Net = NCSNetwork()
duration = 500
testbench = wta_testbench()

wtaParams = {'weInpWTA': 700,
             'weWTAInh': 175,
             'wiInhWTA': -250,
             'weWTAWTA': 75,
             'sigm': 3,
             'rpWTA': 3 * ms,
             'rpInh': 1 * ms
             }

gtestWTA = WTA(name='testWTA', dimensions=2, num_neurons=10, num_inh_neurons=50,
               num_input_neurons=10, num_inputs=2, block_params=wtaParams)
gtestWTA.Groups['synInpWTA1e'] = Connections(gtestWTA.Groups['gWTAInpGroup'], gtestWTA.Groups['gWTAGroup'],
                                             equation_builder=DPIstdp(),
                                             method='euler', name=gtestWTA.Groups['synInpWTA1e'].name)
# this needs to be changed to some random init
gtestWTA.Groups['synInpWTA1e'].connect('i==j')
gtestWTA.Groups['synInpWTA1e'].weight = wtaParams['weInpWTA']
gtestWTA.Groups['synInpWTA1e'].w_plast = 0.5

testbench.stimuli(num_neurons=10, dimensions=2, start_time=100, end_time=duration)
testbench.background_noise(num_neurons=10**2, rate=10)

gtestWTA.inputGroup.set_spikes(indices=testbench.indices, times=testbench.times * ms)
noise_syn = Connections(testbench.noise_input, gtestWTA,
                        equation_builder=DPISyn(), name="noise_syn",)
noise_syn.connect("i==j")
noise_syn.weight = 150

statemon_w_plast = StateMonitor(gtestWTA.Groups['synInpWTA1e'], variables="w_plast", record=True)
Net.add(gtestWTA, testbench.noise_input, noise_syn, statemon_w_plast)

Net.run(duration * ms)

num_synapses = np.floor(np.sqrt(np.shape(statemon_w_plast.w_plast)[0]))
print(num_synapses)
print(np.shape(statemon_w_plast.w_plast))
cm = plt.cm.get_cmap('jet')
x = np.arange(0, num_synapses, 1)
y = np.arange(0, num_synapses, 1)
X, Y = np.meshgrid(x, y)
data = np.reshape(statemon_w_plast.w_plast[0:num_synapses**2, -2:-1],
                  (num_synapses, num_synapses))
fig = plt.figure()
plt.pcolor(X, Y, data, cmap=cm, vmin=0, vmax=1)
plt.ylim((0, num_synapses))
plt.xlim((0, num_synapses))
plt.colorbar()
plt.show()
plotWTA(name='testWTA', start_time=0 * ms, end_time=duration * ms, WTAMonitors=gtestWTA.Monitors)
