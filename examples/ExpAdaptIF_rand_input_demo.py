#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:27:55 2018

This is just showing the behavior of the ExpAdaptIF model for some random input
(Should be documented properly and improved)

@author: alpha
"""
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

from brian2 import ms, mV, pA, nS, nA, uA, pF, us, volt, second, Network, prefs,\
    SpikeMonitor, StateMonitor, figure, plot, show, xlabel, ylabel,\
    seed, xlim, ylim, subplot, network_operation, TimedArray, start_scope,\
    defaultclock, SpikeGeneratorGroup, asarray, pamp, set_device, device

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili import NeuronEquationBuilder, SynapseEquationBuilder
from teili.tools.random_walk import pink

ExpAdaptIF = NeuronEquationBuilder.import_eq('ExpAdaptIF', num_inputs=1)

eq_builder = NeuronEquationBuilder(base_unit='voltage', adaptation='calcium_feedback',
                               integration_mode='exponential', leak='non_leaky',
                               position='spatial', noise='none')
eq_builder.add_input_currents(2)



prefs.codegen.target = "numpy"
defaultclock.dt = 10 * us

start_scope()
Net = TeiliNetwork()

parameters = {
'Cm' : '281. * pfarad',
'refP' : '2. * msecond',
'Ileak' : '0. * amp',
'Inoise' : '1. * nA',
'Iconst' : '0. * amp',
'Vres' : '-70.6 * mvolt',
'gAdapt' : '4. * nsiemens',
'wIadapt' : '80.5 * pamp',
'tauIadapt' : '30. * msecond',
'EL' : '-70.6 * mvolt',
'gL' : '4.3 * nsiemens',
'DeltaT' : '2. * mvolt',
'VT' : '-50.4 * mvolt',
}

# Here we manually add an activity variable to the neuron equation.
# The activity trace basically is the spike trace convolved with an exponential kernel (here tau=20*ms)
equation_builder=ExpAdaptIF(num_inputs=1)
equation_builder.keywords['reset']=equation_builder.keywords['reset']+'Activity+=1'
equation_builder.keywords['model']=equation_builder.keywords['model']+'\n dActivity/dt=-Activity/(20*ms) : 1'

testNeurons = Neurons(1, equation_builder=equation_builder, name="testNeuron", verbose = True)
testNeurons.refP = 1 * ms

duration = 200 *ms
sg_dt = 10*ms
n_pink = int(duration/sg_dt)+1
pink_x = abs(pink(n_pink))* 0.005*uA
pink_x_array = TimedArray(pink_x, dt = sg_dt)
testNeurons.namespace.update({'pink_x_array':pink_x_array})
testNeurons.run_regularly("Iin0 = pink_x_array(t)",dt = defaultclock.dt) #0.005*uA#

testNeurons.set_params(parameters)

spikemon = SpikeMonitor(testNeurons, name='spikemon')
statemonNeuIn = StateMonitor(testNeurons, variables=[
                              "Iin0", "Vm", "Activity"], record=[0], name='statemonNeu')

Net.add(testNeurons, spikemon, statemonNeuIn)
Net.run(duration)

# Visualize simulation results
pg.setConfigOptions(antialias=True)

labelStyle = {'color': '#FFF', 'font-size': '12pt'}
win = pg.GraphicsWindow(title='teili Test Simulation')
win.resize(1900, 600)
win.setWindowTitle('random input to ExpAdaptIF')

p1 = win.addPlot(title='adaptive exponential integrate and fire neuron')
win.nextRow()
p2 = win.addPlot(title="I_In")
win.nextRow()
p3 = win.addPlot(title="Activity")
win.nextRow()
p4 = win.addPlot(title="1/IFR")

colors = [(255, 0, 0), (89, 198, 118), (0, 0, 255), (247, 0, 255),
          (0, 0, 0), (255, 128, 0), (120, 120, 120), (0, 171, 255)]
p1.setXRange(0, duration/ms, padding=0)
p2.setXRange(0, duration/ms, padding=0)
p3.setXRange(0, duration/ms, padding=0)
p4.setXRange(0, duration/ms, padding=0)

for i,data  in enumerate(np.asarray(statemonNeuIn.Vm/mV)):
    p1.plot(x=np.asarray(statemonNeuIn.t / ms), y=np.asarray(data),
            pen=pg.mkPen(colors[0], width=2))

for i,data  in enumerate(np.asarray(statemonNeuIn.Iin0/nA)):
    p2.plot(x=np.asarray(statemonNeuIn.t / ms), y=np.asarray(data),
            pen=pg.mkPen(colors[1], width=2))

for i,data  in enumerate(np.asarray(statemonNeuIn.Activity)):
    p3.plot(x=np.asarray(statemonNeuIn.t / ms), y=np.asarray(data),
            pen=pg.mkPen(colors[2], width=2))

for i,data  in enumerate(np.asarray(spikemon.t)):
    p4.plot(x=np.asarray(spikemon.t[:-1]/ms), y=np.asarray(1/np.diff(spikemon.t)),
            pen=pg.mkPen(colors[3], width=2))

p1.setLabel('left', "voltage", units="mV", **labelStyle)
p2.setLabel('left', "input current", units="nA", **labelStyle)
p3.setLabel('left', "activity", units="", **labelStyle)
p4.setLabel('left', "instantaneous frequency", units="Hz", **labelStyle)

p4.setLabel('bottom', "Time (ms)", **labelStyle)

b = QtGui.QFont("Sans Serif", 10)
p1.getAxis('bottom').tickFont = b
p1.getAxis('left').tickFont = b
p2.getAxis('left').tickFont = b
p3.getAxis('left').tickFont = b
p4.getAxis('left').tickFont = b


QtGui.QApplication.instance().exec_()