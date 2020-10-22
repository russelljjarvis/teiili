
# -*- coding: utf-8 -*-
# @Author: kburel
# @Date:   2018-26-11 12:34:16
# @Last Modified by:   Karla Burelo
# -*- coding: utf-8 -*-

"""
This is a tutorial example used to learn how the different synaptic kernels behave and 
how to set them in your network
"""
import sys
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

from brian2 import ExplicitStateUpdater, mV, second, ms, prefs,\
    SpikeMonitor, StateMonitor,\
    SpikeGeneratorGroup, defaultclock

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import StochasticLIF as neuron_model
from teili.models.synapse_models import StochasticSyn_decay as synapse_model
from teili.models.parameters.dpi_neuron_param import parameters as neuron_model_param
from teili.tools.add_run_reg import add_lfsr

from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataModels import StateVariablesModel
from teili.tools.visualizer.DataControllers import Lineplot


prefs.codegen.target = "numpy"
defaultclock.dt = 1*ms

input_timestamps = np.asarray([1, 2, 3, 3.0, 4, 4, 5, 6]) * ms
input_indices = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])

input_spikegenerator = SpikeGeneratorGroup(2, indices=input_indices,
                                           times=input_timestamps, name='gtestInp')

Net = TeiliNetwork()

stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')
test_neurons1 = Neurons(1, equation_builder=neuron_model(
    num_inputs=2), method=stochastic_decay,name="test_neurons1")

# Set synapses using different kernels
syn_stoch = Connections(input_spikegenerator, test_neurons1,method=stochastic_decay,
                        equation_builder=synapse_model(), name="test_syn_alpha", verbose=False)
syn_stoch.connect(True)
syn_stoch.weight = np.asarray([10, -10])

num_bits = 6
seed = 12
test_neurons1.lfsr_num_bits = num_bits
test_neurons1.Vm = 3*mV
syn_stoch.lfsr_num_bits_syn = num_bits

add_lfsr(test_neurons1, seed, defaultclock.dt)
add_lfsr(syn_stoch, seed, defaultclock.dt)

# Set monitors
spikemon_inp = SpikeMonitor(input_spikegenerator, name='spikemon_inp')
statemon_syn = StateMonitor(
    syn_stoch, variables='I_syn', record=True, name='statemon_syn')

statemon_test_neuron1 = StateMonitor(test_neurons1, variables=[
    'Iin'], record=0, name='statemon_test_neuron1')

Net.add(input_spikegenerator, test_neurons1,
        syn_stoch, spikemon_inp,
        statemon_syn,
        statemon_test_neuron1)

duration = 0.010
Net.run(duration * second)


# Visualize simulation results
app = QtGui.QApplication.instance()
if app is None:
    app = QtGui.QApplication(sys.argv)
else:
    print('QApplication instance already exists: %s' % str(app))

pg.setConfigOptions(antialias=True)
labelStyle = {'color': '#FFF', 'font-size': 12}
MyPlotSettings = PlotSettings(fontsize_title=labelStyle['font-size'],
                              fontsize_legend=labelStyle['font-size'],
                              fontsize_axis_labels=10,
                              marker_size=7)

win = pg.GraphicsWindow(title='Kernels Simulation')
win.resize(900, 600)
win.setWindowTitle('Simple SNN')

p1 = win.addPlot()
p2 = win.addPlot()

# Alpha kernel synapse
data = statemon_syn.I_syn.T
data[:, 1] *= -1.
datamodel_syn_alpha = StateVariablesModel(state_variable_names=['I_syn'],
                                          state_variables=[data],
                                          state_variables_times=[statemon_syn.t])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel_syn_alpha, ('t_I_syn', 'I_syn'))],
         MyPlotSettings=MyPlotSettings,
         x_range=(0, duration),
         y_range=None,
         title='Stochastic Synapse',
         xlabel='Time (s)',
         ylabel='Synaptic current I (A)',
         backend='pyqtgraph',
         mainfig=win,
         subfig=p1,
         QtApp=app)
for i, data in enumerate(np.asarray(spikemon_inp.t)):
    vLine = pg.InfiniteLine(pen=pg.mkPen(color=(200, 200, 255),
                                         style=QtCore.Qt.DotLine), pos=data, angle=90, movable=False,)
    p1.addItem(vLine, ignoreBounds=True)

# Neuron response
Lineplot(DataModel_to_x_and_y_attr=[(statemon_test_neuron1, ('t', 'Iin'))],
         MyPlotSettings=MyPlotSettings,
         x_range=(0, duration),
         y_range=None,
         title="Neuron's input response",
         xlabel='Time (s)',
         ylabel='Iin (A)',
         backend='pyqtgraph',
         mainfig=win,
         subfig=p2,
         QtApp=app)
