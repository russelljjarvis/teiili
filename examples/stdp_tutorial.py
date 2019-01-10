# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-01-16 17:57:35

"""
This file provides an example of how to use neuron and synapse models which are present
on neurmorphic chips in the context of synaptic plasticity based on precise timing of spikes.
We use a standard STDP protocal with a exponentioally decaying window.

"""
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np
import os


from brian2 import ms, us, pA, prefs,\
    SpikeMonitor, StateMonitor, defaultclock

from teili.core.groups import Neurons, Connections
from teili import teiliNetwork
from teili.models.neuron_models import DPI
from teili.models.synapse_models import DPISyn, DPIstdp
from teili.stimuli.testbench import STDP_Testbench


prefs.codegen.target = "numpy"
defaultclock.dt = 50 * us
Net = teiliNetwork()

stdp = STDP_Testbench()
pre_spikegenerator, post_spikegenerator = stdp.stimuli(isi=30)

pre_neurons = Neurons(2, equation_builder=DPI(num_inputs=1),
                      name='pre_neurons')

post_neurons = Neurons(2, equation_builder=DPI(num_inputs=2),
                       name='post_neurons')


pre_synapse = Connections(pre_spikegenerator, pre_neurons,
                          equation_builder=DPISyn(), name='pre_synapse')

post_synapse = Connections(post_spikegenerator, post_neurons,
                           equation_builder=DPISyn(), name='post_synapse')

stdp_synapse = Connections(pre_neurons, post_neurons,
                           equation_builder=DPIstdp(), name='stdp_synapse')

pre_synapse.connect(True)
post_synapse.connect(True)
# Set parameters:
pre_neurons.refP = 3 * ms
pre_neurons.Itau = 6 * pA

post_neurons.Itau = 6 * pA

pre_synapse.weight = 4000.

post_synapse.weight = 4000.

stdp_synapse.connect("i==j")
stdp_synapse.weight = 300.
stdp_synapse.Ie_tau = 10 * pA
stdp_synapse.dApre = 0.01
stdp_synapse.taupre = 3 * ms
stdp_synapse.taupost = 3 * ms

# Setting up monitors
spikemon_pre_neurons = SpikeMonitor(pre_neurons, name='spikemon_pre_neurons')
statemon_pre_neurons = StateMonitor(pre_neurons, variables='Imem',
                                    record=0, name='statemon_pre_neurons')

spikemon_post_neurons = SpikeMonitor(
    post_neurons, name='spikemon_post_neurons')
statemon_post_neurons = StateMonitor(
    post_neurons, variables='Imem', record=0, name='statemon_post_neurons')


statemon_pre_synapse = StateMonitor(
    pre_synapse, variables=['Ie_syn'], record=0, name='statemon_pre_synapse')

statemon_post_synapse = StateMonitor(stdp_synapse, variables=[
    'Ie_syn', 'w_plast', 'weight'],
    record=True, name='statemon_post_synapse')

Net.add(pre_spikegenerator, post_spikegenerator,
        pre_neurons, post_neurons,
        pre_synapse, post_synapse, stdp_synapse,
        spikemon_pre_neurons, spikemon_post_neurons,
        statemon_pre_neurons, statemon_post_neurons,
        statemon_pre_synapse, statemon_post_synapse)

duration = 2000
Net.run(duration * ms)

# Visualize
# Visualize simulation results
app = QtGui.QApplication.instance()
if app is None:
    app = QtGui.QApplication(sys.argv)
else:
    print('QApplication instance already exists: %s' % str(app))

pg.setConfigOptions(antialias=True)

win_stdp = pg.GraphicsWindow(title="STDP Unit Test")
win_stdp.resize(2500, 1500)
win_stdp.setWindowTitle("Spike Time Dependet Plasticity")
colors = [(255, 0, 0), (89, 198, 118), (0, 0, 255), (247, 0, 255),
          (0, 0, 0), (255, 128, 0), (120, 120, 120), (0, 171, 255)]
labelStyle = {'color': '#FFF', 'font-size': '12pt'}

p1 = win_stdp.addPlot(title="STDP protocol")
win_stdp.nextRow()
p2 = win_stdp.addPlot(title="Plastic synaptic weight")
win_stdp.nextRow()
p3 = win_stdp.addPlot(title="Post synaptic current")

p1.setXRange(0, duration, padding=0)
p1.setYRange(-0.1, 1.1, padding=0)
p2.setXRange(0, duration, padding=0)
p3.setXRange(0, duration, padding=0)

p1.plot(x=np.asarray(spikemon_pre_neurons.t / ms), y=np.asarray(spikemon_pre_neurons.i),
        pen=None, symbol='o', symbolPen=None,
        symbolSize=7, symbolBrush=(255, 255, 255),
        name='Pre synaptic neuron')

text1 = pg.TextItem(text='Homoeostasis', anchor=(-0.3, 0.5))
text2 = pg.TextItem(text='Weak Pot.', anchor=(-0.3, 0.5))
text3 = pg.TextItem(text='Weak Dep.', anchor=(-0.3, 0.5))
text4 = pg.TextItem(text='Strong Pot.', anchor=(-0.3, 0.5))
text5 = pg.TextItem(text='Strong Dep.', anchor=(-0.3, 0.5))
text6 = pg.TextItem(text='Homoeostasis', anchor=(-0.3, 0.5))
p1.addItem(text1)
p1.addItem(text2)
p1.addItem(text3)
p1.addItem(text4)
p1.addItem(text5)
p1.addItem(text6)

text1.setPos(0, 0.5)
text2.setPos(250, 0.5)
text3.setPos(550, 0.5)
text4.setPos(850, 0.5)
text5.setPos(1150, 0.5)
text6.setPos(1450, 0.5)


p1.plot(x=np.asarray(spikemon_post_neurons.t / ms), y=np.asarray(spikemon_post_neurons.i),
        pen=None, symbol='s', symbolPen=None,
        symbolSize=7, symbolBrush=(255, 0, 0),
        name='Post synaptic neuron')

for i, data in enumerate(np.asarray(statemon_post_synapse.w_plast)):
    if i == 1:
        p2.plot(x=np.asarray(statemon_post_synapse.t / ms), y=data,
                pen=pg.mkPen(colors[i], width=3))

p3.plot(x=np.asarray(statemon_post_synapse.t / ms), y=np.asarray(statemon_post_synapse.Ie_syn[1]),
        pen=pg.mkPen(colors[3], width=2))


p1.setLabel('left', "Neuron ID", **labelStyle)
p1.setLabel('bottom', "Time (ms)", **labelStyle)
p2.setLabel('bottom', "Time (ms)", **labelStyle)
p2.setLabel('left', "Synpatic weight w_plast", **labelStyle)
p3.setLabel('left', "Synapic current Ie", units='A', **labelStyle)
p3.setLabel('bottom', "Time (ms)", **labelStyle)

b = QtGui.QFont("Sans Serif", 10)
p1.getAxis('bottom').tickFont = b
p1.getAxis('left').tickFont = b
p2.getAxis('bottom').tickFont = b
p2.getAxis('left').tickFont = b
p3.getAxis('bottom').tickFont = b
p3.getAxis('left').tickFont = b

app.exec()
