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

from NCSBrian2Lib.Groups.Groups import Connections
from NCSBrian2Lib.Models.SynapseModels import DPI_stdp

prefs.codegen.target = "numpy"

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
G = NeuronGroup(N, '''tspike:second''', threshold='t>tspike', refractory=100 * ms)
H = NeuronGroup(N, '''
                Ii0 : amp
                Ie0 : amp
                tspike:second''', threshold='t>tspike', refractory=100 * ms)
G.tspike = 'i*tmax/(N-1)'
H.tspike = '(N-1-i)*tmax/(N-1)'

DPISynEq, DPISynparam = DPI_stdp(1)
S = Connections(G, H,
                **DPISynEq, params=DPISynparam)
S.connect('i==j')
S.w = 0.5
S.taupre = 5 * ms
S.taupost = 5 * ms

spikemonG = SpikeMonitor(G, record=True)
spikemonH = SpikeMonitor(H, record=True)

run(tmax + 1 * ms)

plt.plot((H.tspike - G.tspike) / ms, S.w, color="black", linewidth=2.5, linestyle="-")
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
