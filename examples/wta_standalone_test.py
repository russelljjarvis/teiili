# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-01-11 14:48:17
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-25 16:29:42
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import scipy
from scipy.optimize import minimize
from scipy import ndimage

from brian2 import prefs, ms, pA, nA, StateMonitor, device, set_device,\
 second, msecond, defaultclock


from teili.building_blocks.wta import WTA, plotWTA
from teili.core.groups import Neurons, Connections
from teili.stimuli.testbench import WTA_Testbench
from teili import teiliNetwork
from teili.models.synapse_models import DPISyn, DPIstdp
from teili.tools.synaptic_kernel import kernel_gauss_1d

prefs.codegen.target = 'numpy'

run_as_standalone = True

if run_as_standalone:
    standaloneDir = os.path.expanduser('~/WTA_standalone')
    set_device('cpp_standalone', directory=standaloneDir, build_on_run=False)
    device.reinit()
    device.activate(directory=standaloneDir, build_on_run=False)
    prefs.devices.cpp_standalone.openmp_threads = 4

num_neurons = 50
num_input_neurons = num_neurons

x = np.arange(0,num_neurons-1,1)

Net = teiliNetwork()
duration = 500#10000
testbench = WTA_Testbench()

wtaParams = {'weInpWTA': 500,
             'weWTAInh': 175,
             'wiInhWTA': -100,#-250,
             'weWTAWTA': 200,#75,
             'sigm': 1,
             'rpWTA': 3 * ms,
             'rpInh': 1 * ms,
             'EI_connection_probability' : 0.7,
             }

gtestWTA = WTA(name='testWTA', dimensions=1, num_neurons=num_neurons, num_inh_neurons=40,
               num_input_neurons=num_input_neurons, num_inputs=2, block_params=wtaParams,
               spatial_kernel = "kernel_gauss_1d")

syn_in_ex = gtestWTA.Groups["synInpWTA1e"]
syn_ex_ex = gtestWTA.Groups['synWTAWTA1e']
syn_ex_ih = gtestWTA.Groups['synWTAInh1e']
syn_ih_ex = gtestWTA.Groups['synInhWTA1i']

testbench.stimuli(num_neurons=num_neurons, dimensions=1, start_time=100, end_time=duration)
testbench.background_noise(num_neurons=num_neurons, rate=10)

#gtestWTA.inputGroup.set_spikes(indices=testbench.indices, times=testbench.times * ms)
noise_syn = Connections(testbench.noise_input, gtestWTA,
                        equation_builder=DPISyn(), name="noise_syn",)
noise_syn.connect("i==j")
noise_syn.weight = 500

statemonWTAin = StateMonitor(gtestWTA.Groups['gWTAGroup'], ('Ie0', 'Ii0','Ie1', 'Ii1','Ie2', 'Ii2','Ie3', 'Ii3'), record=True,
                                       name='statemonWTAin')

Net.add(gtestWTA, testbench.noise_input, noise_syn, statemonWTAin)

Net.standaloneParams.update({'gtestWTA_Iconst' : 1*pA})

if run_as_standalone:
    Net.build()

#%%
#parameters are nonsense, please find good ones!
standaloneParams=OrderedDict([('duration', 0.5 * second),
             ('stestWTA_e_latWeight', 400),#280),
             ('stestWTA_e_latSigma', 2),
             ('stestWTA_Inpe_weight', 300),
             ('stestWTA_Inhe_weight', 200),#300),
             ('stestWTA_Inhi_weight', -20),

             ('gtestWTA_refP', 5. * msecond),
             ('gtestWTA_Inh_refP', 5. * msecond),
             ('gtestWTA_Iconst', 5000 * pA)])

duration=standaloneParams['duration']/ms
Net.run(duration=duration*ms, standaloneParams=standaloneParams, report='text')

wta_plot = plotWTA(wta_monitors=gtestWTA.Monitors, name='testWTA',
                     start_time=0 * ms, end_time=duration * ms, plot_states=False)
wta_plot.show()

spikemonWTA = gtestWTA.Groups['spikemonWTA']
spiketimes = spikemonWTA.t
dt = defaultclock.dt
spikeinds = spiketimes/dt

data_sparse = scipy.sparse.coo_matrix((np.ones(len(spikeinds)),(spikeinds,[i for i in spikemonWTA.i])))
data_dense = data_sparse.todense()

#data_dense.shape
filtersize = 500*ms
data_filtered = ndimage.uniform_filter1d(data_dense, size=int(filtersize / dt), axis=0, mode='constant') * second / dt
plt.plot(data_filtered[-10]) #[400,:])