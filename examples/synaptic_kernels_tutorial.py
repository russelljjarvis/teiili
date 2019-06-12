
# -*- coding: utf-8 -*-
# @Author: kburel
# @Date:   2018-26-11 12:34:16
# @Last Modified by:   Karla Burelo
# -*- coding: utf-8 -*-

"""
This is a tutorial example used to learn how the different synaptic kernels behave and 
how to set them in your network
"""

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import matplotlib.pyplot as plt
import numpy as np
import os

from brian2 import ms, pA, nA, prefs,\
        SpikeMonitor, StateMonitor,\
        SpikeGeneratorGroup

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import DPI as neuron_model
from teili.models.synapse_models import Alpha, Resonant
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.models.parameters.dpi_neuron_param import parameters as neuron_model_param


prefs.codegen.target = "numpy"

input_timestamps = np.asarray([1, 1.5, 1.8, 2.0, 2.0, 2.3, 2.5, 3]) * ms
input_indices = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])

input_spikegenerator = SpikeGeneratorGroup(2, indices=input_indices,
                                times=input_timestamps, name='gtestInp')

Net = TeiliNetwork()

testNeurons = Neurons(1, equation_builder=neuron_model(num_inputs=2), name="testNeuron")
testNeurons.set_params(neuron_model_param)
testNeurons.refP = 1 * ms

testNeurons2 = Neurons(1, equation_builder=neuron_model(num_inputs=2), name="testNeuron2")
testNeurons2.set_params(neuron_model_param)
testNeurons2.refP = 1 * ms

#Set synapses using different kernels
InpSynAlpha = Connections(input_spikegenerator, testNeurons,
                     equation_builder=Alpha(), name="testSynAlpha", verbose=False)
InpSynAlpha.connect(True)
InpSynAlpha.weight = np.asarray([10,-10])

InpSynResonant = Connections(input_spikegenerator, testNeurons2,
                     equation_builder=Resonant(), name="testSynResonant", verbose=False)
InpSynResonant.connect(True)
InpSynResonant.weight = np.asarray([10,-10])


#Set monitors
spikemonInp = SpikeMonitor(input_spikegenerator, name='spikemonInp')
statemonInpSynAlpha = StateMonitor(
    InpSynAlpha, variables='I_syn', record=True, name='statemonInpSynAlpha')
statemonInpSynResonant = StateMonitor(
    InpSynResonant, variables='I_syn', record=True, name='statemonInpSynResonant')
statemonNeuOut = StateMonitor(testNeurons, variables=[
                              'Iin'], record=0, name='statemonNeuOut')
statemonNeuOut2 = StateMonitor(testNeurons2, variables=[
                              'Iin'], record=0, name='statemonNeuOut2')

Net.add(input_spikegenerator, testNeurons, testNeurons2,
        InpSynAlpha, InpSynResonant, spikemonInp,
        statemonInpSynAlpha, statemonInpSynResonant,
        statemonNeuOut, statemonNeuOut2)

duration = 10
Net.run(duration * ms)


#Visualize simulation results
pg.setConfigOptions(antialias=True)

labelStyle = {'color': '#FFF', 'font-size': '12pt'}
win = pg.GraphicsWindow(title='Kernels Simulation')
win.resize(900, 600)
win.setWindowTitle('Simple SNN')

p1 = win.addPlot(title="Alpha Kernel Synapse")
p2 = win.addPlot(title="Neuron response")
win.nextRow()
p3 = win.addPlot(title='Resonant Kernel Synapse')
p4 = win.addPlot(title="Neuron response")

colors = [(255, 0, 0), (89, 198, 118), (0, 0, 255), (247, 0, 255),
          (0, 0, 0), (255, 128, 0), (120, 120, 120), (0, 171, 255)]

p1.setXRange(0, duration, padding=0)
p2.setXRange(0, duration, padding=0)
p3.setXRange(0, duration, padding=0)
p4.setXRange(0, duration, padding=0)


# Kernel synapses
for i, data in enumerate(np.asarray(statemonInpSynAlpha.I_syn)):
    if i == 1:
        data = data * -1
    name = 'Syn_{}'.format(i)
    p1.plot(x=np.asarray(statemonInpSynAlpha.t / ms), y=data,
            pen=pg.mkPen(colors[3], width=2), name=name)

for i, data in enumerate(np.asarray(spikemonInp.t / ms)): 
   vLine = pg.InfiniteLine(pen=pg.mkPen(color=(200, 200, 255), style=QtCore.Qt.DotLine),pos=data, angle=90, movable=False,)
   p1.addItem(vLine, ignoreBounds=True)

for i, data in enumerate(np.asarray(statemonInpSynResonant.I_syn)):
    if i == 1:
        data = data * -1
    name = 'Syn_{}'.format(i)
    p3.plot(x=np.asarray(statemonInpSynResonant.t / ms), y=data,
            pen=pg.mkPen(colors[3], width=2), name=name)

for i, data in enumerate(np.asarray(spikemonInp.t / ms)):
   vLine = pg.InfiniteLine(pen=pg.mkPen(color=(200, 200, 255), style=QtCore.Qt.DotLine),pos=data, angle=90, movable=False,)
   p3.addItem(vLine, ignoreBounds=True)


for data in np.asarray(statemonNeuOut.Iin):
    p2.plot(x=np.asarray(statemonNeuOut.t / ms), y=data,
            pen=pg.mkPen(colors[5], width=3))

for data in np.asarray(statemonNeuOut2.Iin):
    p4.plot(x=np.asarray(statemonNeuOut2.t / ms), y=data,
            pen=pg.mkPen(colors[5], width=3))


#Set labels
p1.setLabel('left', "Synaptic current I", units='A', **labelStyle)
p1.setLabel('bottom', "Time (ms)", **labelStyle)
p2.setLabel('left', "Membrane current I_mem", units='A', **labelStyle)
p2.setLabel('bottom', "Time (ms)", **labelStyle)
p3.setLabel('left', "Synaptic current I", units="A", **labelStyle)
p3.setLabel('bottom', "Time (ms)", **labelStyle)
p4.setLabel('left', "Membrane current I_mem", units="A", **labelStyle)
p4.setLabel('bottom', "Time (ms)", **labelStyle)

b = QtGui.QFont("Sans Serif", 10)
p1.getAxis('bottom').tickFont = b
p1.getAxis('left').tickFont = b
p2.getAxis('bottom').tickFont = b
p2.getAxis('left').tickFont = b
p3.getAxis('bottom').tickFont = b
p3.getAxis('left').tickFont = b
p4.getAxis('bottom').tickFont = b
p4.getAxis('left').tickFont = b


QtGui.QApplication.instance().exec_()

