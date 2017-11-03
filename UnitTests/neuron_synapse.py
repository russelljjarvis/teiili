# -*- coding: utf-8 -*-
"""
This is a tutorial example used to learn the basics of the Brian2 INI library.

Created on 25.8.2017
"""

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np


from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second, Network, prefs,\
    SpikeMonitor, StateMonitor, figure, plot, show, xlabel, ylabel,\
    seed, xlim, ylim, subplot, network_operation, TimedArray,\
    defaultclock, SpikeGeneratorGroup, asarray, pamp, set_device, device

from brian2 import *

#from NCSBrian2Lib.BuildingBlocks.BuildingBlock import BuildingBlock
from NCSBrian2Lib.Groups.Groups import Neurons, Connections
from NCSBrian2Lib.Equations.NeuronEquation import NeuronEquation
from NCSBrian2Lib import StandaloneNetwork, activate_standalone, deactivate_standalone, NeuronEquation
from NCSBrian2Lib.Models.ExpAdapIF_chip import ExpAdapIF_chip
from NCSBrian2Lib.Parameters.ExpAdapIF_chip_param import parameters
from NCSBrian2Lib.Models.ExpAdapIF_chip import ExpAdapIF_chip
from NCSBrian2Lib.Models.Exp_chip_stdp_syn import Exp_chip_stdp_syn
from NCSBrian2Lib.Models.Exp_chip_syn import Exp_chip_syn
from NCSBrian2Lib.Models.Exp_syn import Exp_syn


prefs.codegen.target = "numpy"
defaultclock.dt = 10 * us
#set_device('cpp_standalone', directory='Tutorial_standalone', build_on_run=True)

# activate_standalone(directory='Tutorial_standalone', build_on_run=True) #this is the Lib function to activate standalone mode (c++ code generation)
# deactivate_standalone()

tsInp = asarray([1, 3, 4, 5, 6, 7, 8, 9]) * ms
indInp = asarray([0, 0, 0, 0, 0, 0, 0, 0])
gInpGroup = SpikeGeneratorGroup(1, indices=indInp,
                                times=tsInp, name='gtestInp')


Net = StandaloneNetwork()


testNeurons = Neurons(2, model=ExpAdapIF_chip, name="testNeuron")

testNeurons2 = Neurons(2, model=ExpAdapIF_chip, numInputs=2, name="testNeuron2")
testNeurons2.setParams(parameters)


InpSyn = Connections(gInpGroup, testNeurons,
                     name="sInpTest_e", baseUnit='current',
                     kernel='alpha', plasticity='nonplastic')
InpSyn.connect(True)

#testInpSyn.Iw_exc =100*pamp
InpSyn.weight = 3
# You can also give different weigths to different synapses of the group:
#testInpSyn.Iw_exc = [100*pamp,50*pamp]



#Syn = Connections(testNeurons, testNeurons2,
#                  name="testSyn", baseUnit='current',
#                  kernel='exponential', plasticity='nonplastic')



Syn = Connections(testNeurons, testNeurons2,
                  model=Exp_chip_syn)




Syn.connect(True)
# you can change all the parameters like this after creation of the neurongroup:
Syn.weight = 0.5

testNeurons2.Iconst = 1.5 * nA
# testNeurons2.Itau = 13 * pA
# testNeurons2.Iath = 80 * pA
# testNeurons2.Iagain = 20 * pA
# testNeurons2.Ianorm = 8 * pA

spikemonInp = SpikeMonitor(gInpGroup, name='spikemonInp')
spikemon = SpikeMonitor(testNeurons, name='spikemon')
spikemonOut = SpikeMonitor(testNeurons2, name='spikemonOut')
statemonInpSyn = StateMonitor(InpSyn, variables='Ie_syn', record=True, name='statemonInpSyn')
statemonNeuOut = StateMonitor(testNeurons2, variables=['Imem'], record=0, name='statemonNeuOut')
statemonNeuIn = StateMonitor(testNeurons, variables=["Iin", "Imem"], record=[0, 1], name='statemonNeu')
statemonSynOut = StateMonitor(Syn, variables='Ie_syn', record=True, name='statemonSynOut')
# statemonSynTest=StateMonitor(testInpSyn,variables=["Isyn_exc"],record=[0],name='statemonSyn')

Net.add(gInpGroup, testNeurons, testNeurons2, InpSyn, Syn, spikemonInp, spikemon,
        spikemonOut, statemonNeuIn, statemonNeuOut, statemonSynOut, statemonInpSyn)

Net.run(100 * ms)

# Visualize simulation results
#app = QtGui.QApplication([])
pg.setConfigOptions(antialias=True)

labelStyle = {'color': '#FFF', 'font-size': '12pt'}
win = pg.GraphicsWindow(title='NCSBrian2Lib Test Simulation')
win.resize(1900, 600)
win.setWindowTitle('Simple SNN')

p1 = win.addPlot(title="Spike generator")
p2 = win.addPlot(title="Input synapses")
win.nextRow()
p3 = win.addPlot(title='Intermediate neuron')
p4 = win.addPlot(title="Output synapses")
win.nextRow()
p5 = win.addPlot(title="Output spikes")
p6 = win.addPlot(title="Output membrane current")

# p1.addLegend()
p2.addLegend()
# p3.addLegend()
p4.addLegend()
colors = [(255, 0, 0), (89, 198, 118), (0, 0, 255), (247, 0, 255),
          (0, 0, 0), (255, 128, 0), (120, 120, 120), (0, 171, 255)]

# Spike generator
p1.plot(x=np.asarray(spikemonInp.t / ms), y=np.asarray(spikemonInp.i),
        pen=None, symbol='o', symbolPen=None,
        symbolSize=7, symbolBrush=(255, 255, 255))

# p2.plot(x=np.asarray(spikemon.t / ms), y=np.asarray(spikemon.i),
#         pen=None, symbol='o', symbolPen=None,
#         symbolSize=7, symbolBrush=(255, 255, 255))

# Input synapses
for i, data in enumerate(np.asarray(statemonInpSyn.Ie_syn)):
  name = 'Syn_{}'.format(i)
  print (name, i, colors[i])
  p2.plot(x=np.asarray(statemonInpSyn.t / ms), y=data,
          pen=pg.mkPen(colors[3], width=2), name=name)

# Intermediate neurons
for i, data in enumerate(np.asarray(statemonNeuIn.Imem)):
  p3.plot(x=np.asarray(statemonNeuIn.t / ms), y=data,
          pen=pg.mkPen(colors[6], width=2))

# Output synapses
for i, data in enumerate(np.asarray(statemonSynOut.Ie_syn)):
  name = 'Syn_{}'.format(i)
  print (name, i, colors[i])
  p4.plot(x=np.asarray(statemonSynOut.t / ms), y=data,
          pen=pg.mkPen(colors[1], width=2), name=name)

for data in np.asarray(statemonNeuOut.Imem):
  p6.plot(x=np.asarray(statemonNeuOut.t / ms), y=data,
          pen=pg.mkPen(colors[5], width=3))

p5.plot(x=np.asarray(spikemonOut.t / ms), y=np.asarray(spikemonOut.i),
        pen=None, symbol='o', symbolPen=None,
        symbolSize=7, symbolBrush=(255, 0, 0))

p1.setLabel('left', "Neuron ID", **labelStyle)
p1.setLabel('bottom', "Time (ms)", **labelStyle)
p2.setLabel('left', "Synaptic current", units='A', **labelStyle)
p2.setLabel('bottom', "Time (ms)", **labelStyle)
p3.setLabel('left', "Membrane current Imem", units="A", **labelStyle)
p3.setLabel('bottom', "Time (ms)", **labelStyle)
p4.setLabel('left', "Synaptic current I_e", units="A", **labelStyle)
p4.setLabel('bottom', "Time (ms)", **labelStyle)
p6.setLabel('left', "Membrane current I_mem", units="A", **labelStyle)
p6.setLabel('bottom', "Time (ms)", **labelStyle)
p5.setLabel('left', "Neuron ID", **labelStyle)
p5.setLabel('bottom', "Time (ms)", **labelStyle)

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


#QtGui.QApplication.instance().exec_()


# fig = figure()
# plot(spikemon.t / ms, spikemon.i, ".k")
# fig2 = figure()
# plot(spikemonInp.t / ms, spikemonInp.i, ".k")

# fig3 = figure()
# plot(statemonTest.t, statemonTest.Iin[0] / pamp)
# plot(statemonTest.t, statemonTest.Imem[0] / pamp)
# # plot(statemonSynTest.t,statemonSynTest.Isyn_exc[0]/pamp)
