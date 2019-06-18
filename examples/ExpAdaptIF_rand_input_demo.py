#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:27:55 2018

This is just showing the behavior of the ExpAdaptIF model for some random input
(Should be documented properly and improved)

@author: alpha
"""
import pyqtgraph as pg
import numpy as np

from brian2 import ms, uA, us, second, prefs, SpikeMonitor, StateMonitor, TimedArray, start_scope, defaultclock

from teili.core.groups import Neurons
from teili import TeiliNetwork
from teili import NeuronEquationBuilder
from teili.tools.random_walk import pink


from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataModels import StateVariablesModel
from teili.tools.visualizer.DataControllers import Lineplot

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

duration = 0.2 *second
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
win = pg.GraphicsWindow(title='teili Test Simulation')
win.resize(1900, 600)
win.setWindowTitle('random input to ExpAdaptIF')
p1 = win.addPlot()
win.nextRow()
p2 = win.addPlot()
win.nextRow()
p3 = win.addPlot()
win.nextRow()
p4 = win.addPlot()

Lineplot(DataModel_to_x_and_y_attr=[(statemonNeuIn, ('t', 'Vm'))],
            MyPlotSettings=PlotSettings(colors=['r']),
            x_range=(0, float(duration)),
            title='adaptive exponential integrate and fire neuron',
            ylabel='voltage (V)',
            backend='pyqtgraph',
            mainfig=win,
            subfig=p1)

Lineplot(DataModel_to_x_and_y_attr=[(statemonNeuIn, ('t', 'Iin0'))],
            MyPlotSettings=PlotSettings(colors=['g']),
            x_range=(0, float(duration)),
            title="I_In",
            ylabel='input current (A)',
            backend='pyqtgraph',
            mainfig=win,
            subfig=p2)

Lineplot(DataModel_to_x_and_y_attr=[(statemonNeuIn, ('t', 'Activity'))],
            MyPlotSettings=PlotSettings(colors=['b']),
            x_range=(0, float(duration)),
            title='Activity',
            ylabel='activity',
            backend='pyqtgraph',
            mainfig=win,
            subfig=p3)

datamodel = StateVariablesModel(state_variable_names=['i'],
                                state_variables=[np.asarray(1./np.diff(spikemon.t))],
                                state_variables_times=[np.asarray(spikemon.t[:-1])])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_i', 'i'))],
            MyPlotSettings=PlotSettings(colors=['m']),
            x_range=(0, float(duration)),
            title='1/IFR',
            xlabel='Time (ms)',
            ylabel='instantaneous frequency (Hz)',
            backend='pyqtgraph',
            mainfig=win,
            subfig=p4,
            show_immediately=True)