"""
Created on 30.11.2017

@author: Moritz Milde
Email: mmilde@ini.uzh.ch

This script is adapted from https://code.ini.uzh.ch/alpren/gridcells/blob/master/STDP_IE_HaasKernel.py

This script contains a simple event based way to simulate complex STDP kernels
"""

from brian2 import *
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.exporters
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

start_scope()

tmax = 30 * ms
N = 100

# Presynaptic neurons G spike at times from 0 to tmax
# Postsynaptic neurons G spike at times from tmax to 0
# So difference in spike times will vary from -tmax to +tmax
G = Neurons(N, model='''tspike:second''', threshold='t>tspike', refractory=100 * ms)

G.namespace.update({'tmax': tmax})
H = Neurons(N, model='''
                Ii0 : amp
                Ie0 : amp
                tspike:second''', threshold='t>tspike', refractory=100 * ms)
H.namespace.update({'tmax': tmax})
G.tspike = 'i*tmax/(N-1)'
H.tspike = '(N-1-i)*tmax/(N-1)'

# syn = DPIstdp()
# S = Connections(G, H, model=syn.model, on_pre=syn.on_pre, on_post=syn.on_post, name='STDP_syn')
S = Connections(G, H,
                equation_builder=DPIstdp(), name='STDP_syn')
S.connect('i==j')
S.w_plast = 0.5
# S.taupre = 5 * ms
# S.taupost = 5 * ms

spikemonG = SpikeMonitor(G, record=True)
spikemonH = SpikeMonitor(H, record=True)

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

    toPlot.plot(x=np.asarray((H.tspike - G.tspike) / ms), y=np.asarray(S.w_plast),
                pen=pg.mkPen((255, 128, 0), width=2))

    toPlot.setLabel('left', '<font>&Delta; w</font>', **labelStyle)
    toPlot.setLabel('bottom', '<font>&Delta; t</font>', **labelStyle)
    b = QtGui.QFont("Sans Serif", 10)
    toPlot.getAxis('bottom').tickFont = b
    toPlot.getAxis('left').tickFont = b
    app.exec_()

elif visualization_backend == 'pyplot':
    plt.plot((H.tspike - G.tspike) / ms, S.w_plast, color="black", linewidth=2.5, linestyle="-")
    plt.title("STDP", fontdict=font)
    plt.xlabel(r'$\Delta t$ (ms)')
    plt.ylabel(r'$\Delta w$')
    plt.savefig('STDP_IE.png')

    # figure()
    # for ii in range(100):
    #    plot(statemon.t/ms,statemon.Apre[ii,:])
    fig = plt.figure()
    plt.plot(spikemonG.t / ms, spikemonG.i, '.k')
    plt.plot(spikemonH.t / ms, spikemonH.i, '.k')
    plt.xlabel('Time [ms]')
    plt.ylabel('Neuron ID')
    plt.show()
