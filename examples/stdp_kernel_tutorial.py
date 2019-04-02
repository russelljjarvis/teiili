"""
Created on 30.11.2017

@author: Moritz Milde
Email: mmilde@ini.uzh.ch

This script is adapted from https://code.ini.uzh.ch/alpren/gridcells/blob/master/STDP_IE_HaasKernel.py

This script contains a simple event based way to simulate complex STDP kernels
"""

from brian2 import ms, prefs, SpikeMonitor, run
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import matplotlib.pyplot as plt
import numpy as np

from teili.core.groups import Neurons, Connections
from teili.models.synapse_models import DPIstdp

prefs.codegen.target = "numpy"
visualization_backend = 'pyqt'  # Or set it to 'pyplot' to use matplotlib.pyplot to plot

font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
        }


tmax = 30 * ms
N = 100

# Presynaptic neurons G spike at times from 0 to tmax
# Postsynaptic neurons G spike at times from tmax to 0
# So difference in spike times will vary from -tmax to +tmax
pre_neurons = Neurons(N, model='''tspike:second''', threshold='t>tspike', refractory=100 * ms)

pre_neurons.namespace.update({'tmax': tmax})
post_neurons = Neurons(N, model='''
                Ii0 : amp
                Ie0 : amp
                tspike:second''', threshold='t>tspike', refractory=100 * ms)
post_neurons.namespace.update({'tmax': tmax})

pre_neurons.tspike = 'i*tmax/(N-1)'
post_neurons.tspike = '(N-1-i)*tmax/(N-1)'


stdp_synapse = Connections(pre_neurons, post_neurons,
                equation_builder=DPIstdp(), name='stdp_synapse')

stdp_synapse.connect('i==j')

# Setting parameters
stdp_synapse.w_plast = 0.5
stdp_synapse.dApre = 0.01
stdp_synapse.taupre = 10 * ms
stdp_synapse.taupost = 10 * ms


spikemon_pre_neurons = SpikeMonitor(pre_neurons, record=True)
spikemon_post_neurons = SpikeMonitor(post_neurons, record=True)

run(tmax + 1 * ms)


if visualization_backend == 'pyqt':
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication(sys.argv)
    else:
        print('QApplication instance already exists: %s' % str(app))

    labelStyle = {'color': '#FFF', 'font-size': '12pt'}
    pg.GraphicsView(useOpenGL=True)
    win = pg.GraphicsWindow(title="STDP Kernel")
    win.resize(1024, 768)
    toPlot = win.addPlot(title="Spike-time dependent plasticity")

    toPlot.plot(x=np.asarray((post_neurons.tspike - pre_neurons.tspike) / ms), y=np.asarray(stdp_synapse.w_plast),
                pen=pg.mkPen((255, 128, 0), width=2))

    toPlot.setLabel('left', '<font>w</font>', **labelStyle)
    toPlot.setLabel('bottom', '<font>&Delta; t</font>', **labelStyle)
    b = QtGui.QFont("Sans Serif", 10)
    toPlot.getAxis('bottom').tickFont = b
    toPlot.getAxis('left').tickFont = b
    app.exec_()

elif visualization_backend == 'pyplot':
    plt.plot((post_neurons.tspike - pre_neurons.tspike) / ms, stdp_synapse.w_plast, color="black", linewidth=2.5, linestyle="-")
    plt.title("STDP", fontdict=font)
    plt.xlabel(r'$\Delta t$ (ms)')
    plt.ylabel(r'$w$')

    fig = plt.figure()
    plt.plot(spikemon_pre_neurons.t / ms, spikemon_pre_neurons.i, '.k')
    plt.plot(spikemon_post_neurons.t / ms, spikemon_post_neurons.i, '.k')
    plt.xlabel('Time [ms]')
    plt.ylabel('Neuron ID')
    plt.show()
