
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

from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second, Network, prefs,\
    SpikeMonitor, StateMonitor, figure, plot, show, xlabel, ylabel,\
    seed, xlim, ylim, subplot, network_operation, TimedArray,\
    defaultclock, SpikeGeneratorGroup, asarray, pamp, set_device, device

from teili.core.groups import Neurons, Connections
from teili import teiliNetwork
from teili.models.parameters.dpi_neuron_param import parameters as DPIparam
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder

# For this example you must first run models/neuron_models.py and synapse_models.py,
# which will create the equation template. This will be stored in models/equations
# Building neuron objects
builder_object1 = NeuronEquationBuilder.import_eq(
    'teili/models/equations/DPI', num_inputs=2)
builder_object2 = NeuronEquationBuilder.import_eq(
    'teili/models/equations/DPI', num_inputs=2)
builder_object6 = NeuronEquationBuilder.import_eq(
    'teili/models/equations/DPI', num_inputs=2)
# Building synapses objects
builder_object3 = SynapseEquationBuilder.import_eq(
    'teili/models/equations/Alpha')
builder_object4 = SynapseEquationBuilder.import_eq(
    'teili/models/equations/Resonant')
builder_object5 = SynapseEquationBuilder.import_eq(
    'teili/models/equations/Gaussian')

prefs.codegen.target = "numpy"
# defaultclock.dt = 10 * us

tsInp = asarray([1, 1.1, 1.25, 1.38, 1.5, 1.67, 1.8, 2.5, 4, 4.2, 4.37, 9]) * ms
indInp = np.zeros(tsInp.size)

gInpGroup = SpikeGeneratorGroup(1, indices=indInp,
                                times=tsInp, name='gtestInp')

Net = teiliNetwork()

testNeurons = Neurons(1, equation_builder=builder_object1, name="testNeuron")
# Example of how to set parameters, saved as a dictionary
testNeurons.set_params(DPIparam)
testNeurons.refP = 1 * ms

testNeurons2 = Neurons(1, equation_builder=builder_object2, name="testNeuron2")
# Example of how to set parameters, saved as a dictionary
testNeurons2.set_params(DPIparam)
testNeurons2.refP = 1 * ms

testNeurons3 = Neurons(1, equation_builder=builder_object6, name="testNeuron3")
# Example of how to set parameters, saved as a dictionary
testNeurons3.set_params(DPIparam)
testNeurons3.refP = 1 * ms

#Set synapses using different kernels
InpSynAlpha = Connections(gInpGroup, testNeurons,
                     equation_builder=builder_object3, name="testSynAlpha", verbose=False)
InpSynAlpha.connect(True)
InpSynAlpha.t_spike = 0 * ms
InpSynAlpha.weight = 10

InpSynResonant = Connections(gInpGroup, testNeurons2,
                     equation_builder=builder_object4, name="testSynResonant", verbose=False)
InpSynResonant.connect(True)
InpSynResonant.t_spike = 0 * ms
InpSynResonant.weight = 30

InpSynGaussian = Connections(gInpGroup, testNeurons3,
                     equation_builder=builder_object5, name="testSynGaussian", verbose=False)
InpSynGaussian.connect(True)
InpSynGaussian.t_spike = 40 * ms
InpSynGaussian.weight = 5
InpSynGaussian.delta_t = 5 * ms

#Set monitors
spikemonInp = SpikeMonitor(gInpGroup, name='spikemonInp')
statemonInpSynAlpha = StateMonitor(
    InpSynAlpha, variables='Ie_syn', record=True, name='statemonInpSynAlpha')
statemonInpSynResonant = StateMonitor(
    InpSynResonant, variables='Ie_syn', record=True, name='statemonInpSynResonant')
statemonInpSynGaussian = StateMonitor(
    InpSynGaussian, variables='Ie_syn', record=True, name='statemonInpSynGaussian')
statemonNeuOut = StateMonitor(testNeurons, variables=[
                              'Imem'], record=0, name='statemonNeuOut')
statemonNeuOut2 = StateMonitor(testNeurons2, variables=[
                              'Imem'], record=0, name='statemonNeuOut2')
statemonNeuOut3 = StateMonitor(testNeurons3, variables=[
                              'Imem'], record=0, name='statemonNeuOut3')

Net.add(gInpGroup, testNeurons, testNeurons2, testNeurons3,
        InpSynAlpha, InpSynResonant, InpSynGaussian, spikemonInp,
        statemonInpSynAlpha, statemonInpSynResonant, statemonInpSynGaussian, 
        statemonNeuOut, statemonNeuOut2, statemonNeuOut3)

duration = 40 #500
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
win.nextRow()
p5 = win.addPlot(title="Gaussian Kernel Synapse")
p6 = win.addPlot(title="Neuron response")

colors = [(255, 0, 0), (89, 198, 118), (0, 0, 255), (247, 0, 255),
          (0, 0, 0), (255, 128, 0), (120, 120, 120), (0, 171, 255)]

p1.setXRange(0, duration, padding=0)
p2.setXRange(0, duration, padding=0)
p3.setXRange(0, duration, padding=0)
p4.setXRange(0, duration, padding=0)
p5.setXRange(0, duration, padding=0)
p6.setXRange(0, duration, padding=0)


# Spike generator
#p1.plot(x=np.asarray(spikemonInp.t / ms), y=np.asarray(spikemonInp.i),
 #       pen=None, symbol='o', symbolPen=None,
  #      symbolSize=7, symbolBrush=(255, 255, 255))

# Kernel synapses


for i, data in enumerate(np.asarray(statemonInpSynAlpha.Ie_syn)):
    name = 'Syn_{}'.format(i)
    p1.plot(x=np.asarray(statemonInpSynAlpha.t / ms), y=data,
            pen=pg.mkPen(colors[3], width=2), name=name)

for i, data in enumerate(np.asarray(spikemonInp.t / ms)): 
   vLine = pg.InfiniteLine(pen=pg.mkPen(color=(200, 200, 255), style=QtCore.Qt.DotLine),pos=data, angle=90, movable=False,)
   p1.addItem(vLine, ignoreBounds=True)

for i, data in enumerate(np.asarray(statemonInpSynResonant.Ie_syn)):
    name = 'Syn_{}'.format(i)
    p3.plot(x=np.asarray(statemonInpSynResonant.t / ms), y=data,
            pen=pg.mkPen(colors[3], width=2), name=name)

for i, data in enumerate(np.asarray(spikemonInp.t / ms)): 
   vLine = pg.InfiniteLine(pen=pg.mkPen(color=(200, 200, 255), style=QtCore.Qt.DotLine),pos=data, angle=90, movable=False,)
   p3.addItem(vLine, ignoreBounds=True)

for i, data in enumerate(np.asarray(statemonInpSynGaussian.Ie_syn)):
    name = 'Syn_{}'.format(i)
    p5.plot(x=np.asarray(statemonInpSynGaussian.t / ms), y=data,
            pen=pg.mkPen(colors[3], width=2), name=name)

for i, data in enumerate(np.asarray(spikemonInp.t / ms)): 
   vLine = pg.InfiniteLine(pen=pg.mkPen(color=(200, 200, 255), style=QtCore.Qt.DotLine),pos=data, angle=90, movable=False,)
   p5.addItem(vLine, ignoreBounds=True)

for data in np.asarray(statemonNeuOut.Imem):
    p2.plot(x=np.asarray(statemonNeuOut.t / ms), y=data,
            pen=pg.mkPen(colors[5], width=3))

for data in np.asarray(statemonNeuOut2.Imem):
    p4.plot(x=np.asarray(statemonNeuOut2.t / ms), y=data,
            pen=pg.mkPen(colors[5], width=3))

for data in np.asarray(statemonNeuOut3.Imem):
    p6.plot(x=np.asarray(statemonNeuOut3.t / ms), y=data,
            pen=pg.mkPen(colors[5], width=3))

#Set labels
p1.setLabel('left', "Synaptic current I_e", units='A', **labelStyle)
p1.setLabel('bottom', "Time (ms)", **labelStyle)
p2.setLabel('left', "Membrane current I_mem", units='A', **labelStyle)
p2.setLabel('bottom', "Time (ms)", **labelStyle)
p3.setLabel('left', "Synaptic current I_e", units="A", **labelStyle)
p3.setLabel('bottom', "Time (ms)", **labelStyle)
p4.setLabel('left', "Membrane current I_mem", units="A", **labelStyle)
p4.setLabel('bottom', "Time (ms)", **labelStyle)
p5.setLabel('left', "Synaptic current I_e", units="A", **labelStyle)
p5.setLabel('bottom', "Time (ms)", **labelStyle)
p6.setLabel('left', "Membrane current I_mem", units='A', **labelStyle)
p6.setLabel('bottom', "Time (ms)", **labelStyle)

b = QtGui.QFont("Sans Serif", 10)
p1.getAxis('bottom').tickFont = b
p1.getAxis('left').tickFont = b
p2.getAxis('bottom').tickFont = b
p2.getAxis('left').tickFont = b
p3.getAxis('bottom').tickFont = b
p3.getAxis('left').tickFont = b
p4.getAxis('bottom').tickFont = b
p4.getAxis('left').tickFont = b
p5.getAxis('bottom').tickFont = b
p5.getAxis('left').tickFont = b
p6.getAxis('bottom').tickFont = b
p6.getAxis('left').tickFont = b


QtGui.QApplication.instance().exec_()


