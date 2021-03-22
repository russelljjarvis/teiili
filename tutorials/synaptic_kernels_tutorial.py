
# -*- coding: utf-8 -*-
# @Author: kburel
# @Date:   2018-26-11 12:34:16
# @Last Modified by: Pablo Urbizagastegui
# -*- coding: utf-8 -*-

"""
This is a tutorial example used to learn how the different synaptic kernels behave and 
how to set them in your network
"""
import sys
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

from brian2 import second, ms, prefs,\
    SpikeMonitor, StateMonitor,\
    SpikeGeneratorGroup, ExplicitStateUpdater

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import DPI as neuron_model
from teili.models.neuron_models import QuantStochLIF as q_neuron_model
from teili.models.synapse_models import Alpha, Resonant, DPISyn, QuantStochSyn
from teili.models.parameters.dpi_neuron_param import parameters as neuron_model_param

from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataModels import StateVariablesModel
from teili.tools.visualizer.DataControllers import Lineplot


prefs.codegen.target = "numpy"

input_timestamps = np.asarray([1, 1.5, 1.8, 2.0, 2.0, 2.3, 2.5, 3]) * ms
input_indices = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])

input_spikegenerator1 = SpikeGeneratorGroup(2, indices=input_indices,
                                           times=input_timestamps, name='gtestInp1')
input_spikegenerator2 = SpikeGeneratorGroup(2, indices=input_indices,
                                           times=input_timestamps, name='gtestInp2')

Net = TeiliNetwork()

test_neurons1 = Neurons(1, equation_builder=neuron_model(
    num_inputs=2), name="test_neurons1")
test_neurons1.set_params(neuron_model_param)
test_neurons1.refP = 1 * ms

test_neurons2 = Neurons(1, equation_builder=neuron_model(
    num_inputs=2), name="test_neurons2")
test_neurons2.set_params(neuron_model_param)
test_neurons2.refP = 1 * ms

test_neurons3 = Neurons(1, equation_builder=neuron_model(
    num_inputs=2), name="test_neurons3")
test_neurons3.set_params(neuron_model_param)
test_neurons3.refP = 1 * ms

test_neurons4 = Neurons(1, equation_builder=q_neuron_model(
    num_inputs=2), method=ExplicitStateUpdater('''x_new = f(x,t)'''),
    name="test_neurons4")

# Set synapses using different kernels
syn_alpha = Connections(input_spikegenerator1, test_neurons1,
                        equation_builder=Alpha(), name="test_syn_alpha", verbose=False)
syn_alpha.connect(True)
syn_alpha.weight = np.asarray([10, -10])

syn_resonant = Connections(input_spikegenerator1, test_neurons2,
                           equation_builder=Resonant(), name="test_syn_resonant", verbose=False)
syn_resonant.connect(True)
syn_resonant.weight = np.asarray([10, -10])

syn_dpi = Connections(input_spikegenerator1, test_neurons3,
                      equation_builder=DPISyn(), name="test_syn_dpi", verbose=False)
syn_dpi.connect(True)
syn_dpi.weight = np.asarray([10, -10])

syn_q_stoch = Connections(input_spikegenerator2, test_neurons4,
                          equation_builder=QuantStochSyn(),
                          method=ExplicitStateUpdater('''x_new = f(x,t)'''),
                          name="test_syn_q_stoch", verbose=False)
syn_q_stoch.connect(True)
syn_q_stoch.weight = np.asarray([10, -10])

# Set monitors
spikemon_inp = SpikeMonitor(input_spikegenerator1, name='spikemon_inp')
statemon_syn_alpha = StateMonitor(
    syn_alpha, variables='I_syn', record=True, name='statemon_syn_alpha')
statemon_syn_resonant = StateMonitor(
    syn_resonant, variables='I_syn', record=True, name='statemon_syn_resonant')
statemon_syn_dpi = StateMonitor(
    syn_dpi, variables='I_syn', record=True, name='statemon_syn_dpi')
statemon_syn_q_stoch = StateMonitor(
    syn_q_stoch, variables='I_syn', record=True, name='statemon_syn_q_stoch')

statemon_test_neuron1 = StateMonitor(test_neurons1, variables=[
    'Iin'], record=0, name='statemon_test_neuron1')
statemon_test_neuron2 = StateMonitor(test_neurons2, variables=[
    'Iin'], record=0, name='statemon_test_neuron2')
statemon_test_neuron3 = StateMonitor(test_neurons3, variables=[
    'Iin'], record=0, name='statemon_test_neuron3')
statemon_test_neuron4 = StateMonitor(test_neurons4, variables=[
    'Iin'], record=0, name='statemon_test_neuron4')

Net.add(input_spikegenerator1, test_neurons1, test_neurons2, test_neurons3,
        syn_alpha, syn_resonant, syn_dpi, spikemon_inp,
        statemon_syn_alpha, statemon_syn_resonant, statemon_syn_dpi,
        statemon_test_neuron1, statemon_test_neuron2, statemon_test_neuron3)
duration = 0.010
Net.run(duration * second)

# Run models that use a different integration method
Net = TeiliNetwork()
Net.add(input_spikegenerator2, test_neurons4, syn_q_stoch, statemon_syn_q_stoch,
        statemon_test_neuron4)
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
win.nextRow()
p3 = win.addPlot()
p4 = win.addPlot()
win.nextRow()
p5 = win.addPlot()
p6 = win.addPlot()
win.nextRow()
p7 = win.addPlot()
p8 = win.addPlot()

# Alpha kernel synapse
data = statemon_syn_alpha.I_syn.T
data[:, 1] *= -1.
datamodel_syn_alpha = StateVariablesModel(state_variable_names=['I_syn'],
                                          state_variables=[data],
                                          state_variables_times=[statemon_syn_alpha.t])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel_syn_alpha, ('t_I_syn', 'I_syn'))],
         MyPlotSettings=MyPlotSettings,
         x_range=(0, duration),
         y_range=None,
         title='Alpha Kernel Synapse',
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

# Resonant kernel synapse
data = statemon_syn_resonant.I_syn.T
data[:, 1] *= -1.
datamodel_syn_resonant = StateVariablesModel(state_variable_names=['I_syn'],
                                             state_variables=[data],
                                             state_variables_times=[statemon_syn_resonant.t])

Lineplot(DataModel_to_x_and_y_attr=[(datamodel_syn_resonant, ('t_I_syn', 'I_syn'))],
         MyPlotSettings=MyPlotSettings,
         x_range=(0, duration),
         y_range=None,
         title='Resonant Kernel Synapse',
         xlabel='Time (s)',
         ylabel='Synaptic current I (A)',
         backend='pyqtgraph',
         mainfig=win,
         subfig=p3,
         QtApp=app)
for i, data in enumerate(np.asarray(spikemon_inp.t)):
    vLine = pg.InfiniteLine(pen=pg.mkPen(color=(200, 200, 255),
                                         style=QtCore.Qt.DotLine), pos=data, angle=90, movable=False,)
    p3.addItem(vLine, ignoreBounds=True)

# Neuron response
Lineplot(DataModel_to_x_and_y_attr=[(statemon_test_neuron2, ('t', 'Iin'))],
         MyPlotSettings=MyPlotSettings,
         x_range=(0, duration),
         y_range=None,
         title="Neuron's input response",
         xlabel='Time (s)',
         ylabel='Iin (A)',
         backend='pyqtgraph',
         mainfig=win,
         subfig=p4,
         QtApp=app)


# DPI synapse
data = statemon_syn_dpi.I_syn.T
data[:, 1] *= -1.
datamodel_syn_dpi = StateVariablesModel(state_variable_names=['I_syn'],
                                        state_variables=[data],
                                        state_variables_times=[statemon_syn_dpi.t])

Lineplot(DataModel_to_x_and_y_attr=[(datamodel_syn_dpi, ('t_I_syn', 'I_syn'))],
         MyPlotSettings=MyPlotSettings,
         x_range=(0, duration),
         y_range=None,
         title='DPI Synapse',
         xlabel='Time (s)',
         ylabel='Synaptic current I (A)',
         backend='pyqtgraph',
         mainfig=win,
         subfig=p5,
         QtApp=app)
for i, data in enumerate(np.asarray(spikemon_inp.t)):
    vLine = pg.InfiniteLine(pen=pg.mkPen(color=(200, 200, 255),
                                         style=QtCore.Qt.DotLine), pos=data, angle=90, movable=False,)
    p5.addItem(vLine, ignoreBounds=True)

# Neuron response
Lineplot(DataModel_to_x_and_y_attr=[(statemon_test_neuron3, ('t', 'Iin'))],
         MyPlotSettings=MyPlotSettings,
         x_range=(0, duration),
         y_range=None,
         title="Neuron's input response",
         xlabel='Time (s)',
         ylabel='Iin (A)',
         backend='pyqtgraph',
         mainfig=win,
         subfig=p6,
         QtApp=app)

# Quantized stochastic synapse
data = statemon_syn_q_stoch.I_syn.T
data[:, 1] *= -1.
datamodel_syn_q_stoch = StateVariablesModel(state_variable_names=['I_syn'],
                                          state_variables=[data],
                                          state_variables_times=[statemon_syn_q_stoch.t])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel_syn_q_stoch, ('t_I_syn', 'I_syn'))],
         MyPlotSettings=MyPlotSettings,
         x_range=(0, duration),
         y_range=None,
         title='Quantized Stochastic Synapse',
         xlabel='Time (s)',
         ylabel='Synaptic current I (A)',
         backend='pyqtgraph',
         mainfig=win,
         subfig=p7,
         QtApp=app)
for i, data in enumerate(np.asarray(spikemon_inp.t)):
    vLine = pg.InfiniteLine(pen=pg.mkPen(color=(200, 200, 255),
                                         style=QtCore.Qt.DotLine), pos=data, angle=90, movable=False,)
    p7.addItem(vLine, ignoreBounds=True)

# Neuron response
Lineplot(DataModel_to_x_and_y_attr=[(statemon_test_neuron4, ('t', 'Iin'))],
         MyPlotSettings=MyPlotSettings,
         x_range=(0, duration),
         y_range=None,
         title="Neuron's input response",
         xlabel='Time (s)',
         ylabel='Iin (A)',
         backend='pyqtgraph',
         mainfig=win,
         subfig=p8,
         QtApp=app,
         show_immediately=True)
