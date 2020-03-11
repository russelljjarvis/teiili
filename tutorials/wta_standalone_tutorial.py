# -*- coding: utf-8 -*-
# @Author: alpren
# @Date:   2018-01-11 14:48:17

# On a fresh Ubuntu install, you need to install the compilers via:
# sudo apt-get install build-essential


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

import scipy
from scipy import ndimage

from brian2 import prefs, ms, pA, StateMonitor, SpikeMonitor,\
    device, set_device,\
    second, msecond, pamp, defaultclock

from teili.building_blocks.wta import WTA
from teili.core.groups import Neurons, Connections
from teili.stimuli.testbench import WTA_Testbench
from teili import TeiliNetwork
from teili.models.synapse_models import DPISyn

from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataModels import StateVariablesModel
from teili.tools.visualizer.DataControllers import Rasterplot

prefs.codegen.target = 'numpy'

run_as_standalone = True

if run_as_standalone:
    standaloneDir = os.path.expanduser('~/WTA_standalone')
    set_device('cpp_standalone', directory=standaloneDir, build_on_run=False)
    device.reinit()
    device.activate(directory=standaloneDir, build_on_run=False)
    #prefs.devices.cpp_standalone.openmp_threads = 2
    # on systems with limited memory, make sure to limit the number of compiler threads:
    prefs.devices.cpp_standalone.extra_make_args_unix = ["-j$(nproc)"]

num_neurons = 50
num_input_neurons = num_neurons

x = np.arange(0, num_neurons - 1, 1)

Net = TeiliNetwork()
duration = 500  # 10000
duration_s = duration * 1e-3
testbench = WTA_Testbench()


wta_params = {'we_inp_exc': 900,
             'we_exc_inh': 500,
             'wi_inh_exc': -550,  # -250,
             'we_exc_exc': 650,  # 75,
             'sigm': 2,
             'rp_exc': 3 * ms,
             'rp_inh': 1 * ms,
             'ei_connection_probability': 0.7,
             }

test_WTA = WTA(name='test_WTA', dimensions=1, num_neurons=num_neurons, num_inh_neurons=40,
               num_input_neurons=num_input_neurons, num_inputs=2, block_params=wta_params,
               spatial_kernel="kernel_gauss_1d")

testbench.stimuli(num_neurons=num_neurons, dimensions=1,
                  start_time=100, end_time=duration)
testbench.background_noise(num_neurons=num_neurons, rate=10)

test_WTA.spike_gen.set_spikes(
    indices=testbench.indices, times=testbench.times * ms)
noise_syn = Connections(testbench.noise_input, test_WTA,
                        equation_builder=DPISyn(), name="noise_syn")
noise_syn.connect("i==j")
noise_syn.weight = 3000

statemonWTAin = StateMonitor(test_WTA.output_groups['n_exc'],
                             ('Iin0', 'Iin1', 'Iin2','Iin3'),
                             record=True,
                             name='statemonWTAin')

spikemonitor_input = SpikeMonitor(
    test_WTA._groups['spike_gen'], name="spikemonitor_input")
spikemonitor_noise = SpikeMonitor(
    testbench.noise_input, name="spikemonitor_noise")

Net.add(test_WTA, testbench.noise_input, noise_syn,
        statemonWTAin, spikemonitor_noise, spikemonitor_input)

Net.standalone_params.update({test_WTA.input_groups['n_exc'].name+'_Iconst': 1 * pA})

#Net.standalone_params = {}

if run_as_standalone:
    Net.build()

standalone_params = OrderedDict([('duration', 1. * second),
             ('test_WTA__s_exc_exc_lateral_weight', 650),
             ('test_WTA__s_exc_exc_lateral_sigma', 2),
             ('test_WTA__s_inp_exc_weight', 900),
             ('test_WTA__s_inh_exc_weight', -550),
             ('test_WTA__n_inh_refP', 1. * msecond),
             ('test_WTA__s_exc_inh_weight', 500),
             ('test_WTA__n_exc_refP', 3. * msecond),
             ('test_WTA__n_exc_Iconst', 1. * pamp)])

duration = standalone_params['duration'] / ms
Net.run(duration=duration * ms, standalone_params=standalone_params, report='text')

# Visualization
win_wta = pg.GraphicsWindow(title="WTA")
win_wta.resize(2500, 1500)
win_wta.setWindowTitle("WTA")
p1 = win_wta.addPlot()
win_wta.nextRow()
p2 = win_wta.addPlot()
win_wta.nextRow()
p3 = win_wta.addPlot()

spikemonWTA = test_WTA.monitors['spikemon_exc']
spiketimes = spikemonWTA.t

Rasterplot(MyEventsModels = [spikemonitor_noise],
            time_range=(0, duration_s),
            title="Noise input",
            xlabel='Time (s)',
            ylabel=None,
            backend='pyqtgraph',
            mainfig=win_wta,
            subfig_rasterplot=p1)

Rasterplot(MyEventsModels=[spikemonWTA],
            time_range=(0, duration_s),
            title="WTA activity",
            xlabel='Time (s)',
            ylabel=None,
            backend='pyqtgraph',
            mainfig=win_wta,
            subfig_rasterplot=p2)

Rasterplot(MyEventsModels=[spikemonitor_input],
            time_range=(0, duration_s),
            title="Actual signal",
            xlabel='Time (s)',
            ylabel=None,
            backend='pyqtgraph',
            mainfig=win_wta,
            subfig_rasterplot=p3,
            show_immediately=True)

