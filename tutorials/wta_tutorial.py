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
        second, msecond, defaultclock

from teili.building_blocks.wta import WTA
from teili.core.groups import Neurons, Connections
from teili.stimuli.testbench import WTA_Testbench
from teili import TeiliNetwork
from teili.models.synapse_models import DPISyn

prefs.codegen.target = 'numpy'
run_as_standalone = False

if run_as_standalone:
        standaloneDir = os.path.expanduser('~/WTA_standalone')
        set_device('cpp_standalone', directory=standaloneDir, build_on_run=False)
        device.reinit()
        device.activate(directory=standaloneDir, build_on_run=False)
        prefs.devices.cpp_standalone.openmp_threads = 2
num_neurons = 50
num_input_neurons = num_neurons
num_inh_neurons = 40

Net = TeiliNetwork()
duration = 500
testbench = WTA_Testbench()

wta_params = {'we_inp_exc': 900,
             'we_exc_inh': 500,
             'wi_inh_exc': -550,
             'we_exc_exc': 650,
             'sigm': 2,
             'rp_exc': 3 * ms,
             'rp_inh': 1 * ms,
             'ei_connection_probability': 0.7,
                }

test_WTA = WTA(name='test_WTA', dimensions=1,
               num_neurons=num_neurons, 
               num_inh_neurons=num_inh_neurons,
               num_input_neurons=num_input_neurons,
               num_inputs=2, block_params=wta_params,
               spatial_kernel="kernel_gauss_1d")

testbench.stimuli(num_neurons=num_input_neurons, 
                  dimensions=1,
                  start_time=100, end_time=duration)

testbench.background_noise(num_input_neurons=num_neurons, rate=10)

test_WTA.spike_gen.set_spikes(
    indices=testbench.indices, times=testbench.times * ms)

noise_syn = Connections(testbench.noise_input, 
                        test_WTA._groups['n_exc'],
                        equation_builder=DPISyn(), 
                        name="noise_syn")
noise_syn.connect("i==j")

noise_syn.weight = 3000

statemonWTAin = StateMonitor(test_WTA._groups['n_exc'],
                             ('Iin0', 'Iin1', 'Iin2', 'Iin3'),
                             record=True,
                             name='statemonWTAin')

spikemonitor_input = SpikeMonitor(
    test_WTA.spike_gen, name="spikemonitor_input")
spikemonitor_noise = SpikeMonitor(
    testbench.noise_input, name="spikemonitor_noise")


Net.add(test_WTA, testbench.noise_input, noise_syn,
        statemonWTAin, spikemonitor_noise, spikemonitor_input)

Net.standalone_params.update({'test_WTA_Iconst': 1 * pA})

if run_as_standalone:
    Net.build()

standalone_params = OrderedDict([('duration', 0.7 * second),
                                ('stestWTA_e_latWeight', 650),
                                ('stestWTA_e_latSigma', 2),
                                ('stestWTA_Inpe_weight', 900),
                                ('stestWTA_Inhe_weight', 500),
                                ('stestWTA_Inhi_weight', -550),
                                ('test_WTA_refP', 1. * msecond),
                                ('testWTA_Inh_refP', 1. * msecond)])

duration = standalone_params['duration'] / ms
Net.run(duration=duration * ms, standalone_params=standalone_params, report='text')


app = QtGui.QApplication.instance()
if app is None:
        app = QtGui.QApplication(sys.argv)
else:
        print('QApplication instance already exists: %s' % str(app))

pg.setConfigOptions(antialias=True)

win_wta = pg.GraphicsWindow(title="STDP Unit Test")
win_wta.resize(2500, 1500)
win_wta.setWindowTitle("Spike Time Dependet Plasticity")
colors = [(255, 0, 0), (89, 198, 118), (0, 0, 255), (247, 0, 255),
                    (0, 0, 0), (255, 128, 0), (120, 120, 120), (0, 171, 255)]
labelStyle = {'color': '#FFF', 'font-size': '12pt'}

p1 = win_wta.addPlot(title="Noise input")
win_wta.nextRow()
p2 = win_wta.addPlot(title="WTA activity")
win_wta.nextRow()
p3 = win_wta.addPlot(title="Actual signal")

p1.setXRange(0, duration, padding=0)
p2.setXRange(0, duration, padding=0)
p3.setXRange(0, duration, padding=0)


spikemonWTA = test_WTA.monitors['spikemon_exc']
spiketimes = spikemonWTA.t

p1.plot(x=np.asarray(spikemonitor_noise.t / ms), y=np.asarray(spikemonitor_noise.i),
                pen=None, symbol='s', symbolPen=None,
                symbolSize=7, symbolBrush=(255, 0, 0),
                name='Noise input')

p2.plot(x=np.asarray(spikemonWTA.t / ms), y=np.asarray(spikemonWTA.i),
                pen=None, symbol='s', symbolPen=None,
                symbolSize=7, symbolBrush=(255, 0, 0),
                name='WTA Rasterplot')

p3.plot(x=np.asarray(spikemonitor_input.t / ms), y=np.asarray(spikemonitor_input.i),
                pen=None, symbol='s', symbolPen=None,
                symbolSize=7, symbolBrush=(255, 0, 0),
                name='Desired signal')

app.exec()
