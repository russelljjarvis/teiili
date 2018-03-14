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
duration = 10000
testbench = wta_testbench()

wtaParams = {'weInpWTA': 1000,
             'weWTAInh': 175,
             'wiInhWTA': -250,
             'weWTAWTA': 75,
             'sigm': 3,
             'rpWTA': 3 * ms,
             'rpInh': 1 * ms
             }
num_neurons = 10
num_input_neurons = 10

gtestWTA = WTA(name='testWTA', dimensions=2, num_neurons=num_neurons, num_inh_neurons=50,
               num_input_neurons=num_input_neurons, num_inputs=2, block_params=wtaParams)
gtestWTA.Groups['synInpWTA1e'] = Connections(gtestWTA.Groups['gWTAInpGroup'], gtestWTA.Groups['gWTAGroup'],
                                             equation_builder=DPIstdp(),
                                             method='euler', name=gtestWTA.Groups['synInpWTA1e'].name)
# this needs to be changed to some random init
gtestWTA.Groups['synInpWTA1e'].connect('True', p=0.3)
gtestWTA.Groups['synInpWTA1e'].weight = wtaParams['weInpWTA']
gtestWTA.Groups['synInpWTA1e'].w_plast = 0.5

gtestWTA.Groups['synInpWTA1e'].run_regularly('''w_plast *= 0.98 ''',
                                             dt=500 * ms)

testbench.stimuli(num_neurons=num_neurons, dimensions=2, start_time=100, end_time=duration)
testbench.background_noise(num_neurons=num_neurons, rate=10)

gtestWTA.inputGroup.set_spikes(indices=testbench.indices, times=testbench.times * ms)
noise_syn = Connections(testbench.noise_input, gtestWTA,
                        equation_builder=DPISyn(), name="noise_syn",)
noise_syn.connect("i==j")
noise_syn.weight = 150

statemon_w_plast = StateMonitor(gtestWTA.Groups['synInpWTA1e'], variables=["w_plast"], record=True)
statemon_apre = StateMonitor(gtestWTA.Groups['synInpWTA1e'], variables=["Apre", "Apost"], record=True)
Net.add(gtestWTA, testbench.noise_input, noise_syn, statemon_w_plast, statemon_apre)

Net.run(duration * ms, report='text')

num_source_neurons = gtestWTA.Groups['gWTAInpGroup'].N
num_target_neurons = gtestWTA.Groups['gWTAGroup'].N
cm = plt.cm.get_cmap('jet')
x = np.arange(0, num_target_neurons, 1)
y = np.arange(0, num_source_neurons, 1)
X, Y = np.meshgrid(x, y)
data = np.zeros((num_target_neurons, num_source_neurons)) * np.nan
# Getting sparse weights
w_plast = statemon_w_plast.w_plast[:, -2:-1]
data[gtestWTA.Groups['synInpWTA1e'].i, gtestWTA.Groups['synInpWTA1e'].j] = w_plast[:, 0]
data[np.isnan(data)] = 0
fig = plt.figure()
plt.pcolor(X, Y, data, cmap=cm, vmin=0, vmax=1)
plt.colorbar()
plt.xlim((0, np.max(x)))
plt.ylim((0, np.max(y)))
plt.ylabel('Source neuron index')
plt.xlabel('Target neuron index')
plt.draw()
plt.show()
plotWTA(name='testWTA', start_time=0 * ms, end_time=duration * ms, WTAMonitors=gtestWTA.Monitors)
