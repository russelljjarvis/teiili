
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

from brian2 import second, ms, prefs,\
    SpikeMonitor, StateMonitor,\
    SpikeGeneratorGroup

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import DPI as neuron_model
from teili.models.synapse_models import Alpha, Resonant, DPISyn
from teili.models.parameters.dpi_neuron_param import parameters as neuron_model_param

from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataModels import StateVariablesModel
from teili.tools.visualizer.DataControllers import Lineplot


prefs.codegen.target = "numpy"

input_timestamps = np.asarray([1, 1.5, 1.8, 2.0, 2.0, 2.3, 2.5, 3]) * ms
input_indices = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])

input_spikegenerator = SpikeGeneratorGroup(2, indices=input_indices,
                                           times=input_timestamps, name='gtestInp')

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

# Set synapses using different kernels
syn_alpha = Connections(input_spikegenerator, test_neurons1,
                        equation_builder=Alpha(), name="test_syn_alpha", verbose=False)
syn_alpha.connect(True)
syn_alpha.weight = np.asarray([10, -10])

syn_resonant = Connections(input_spikegenerator, test_neurons2,
                           equation_builder=Resonant(), name="test_syn_resonant", verbose=False)
syn_resonant.connect(True)
syn_resonant.weight = np.asarray([10, -10])

syn_dpi = Connections(input_spikegenerator, test_neurons3,
                      equation_builder=DPISyn(), name="test_syn_dpi", verbose=False)
syn_dpi.connect(True)
syn_dpi.weight = np.asarray([10, -10])

# Set monitors
spikemon_inp = SpikeMonitor(input_spikegenerator, name='spikemon_inp')
statemon_syn_alpha = StateMonitor(
    syn_alpha, variables='I_syn', record=True, name='statemon_syn_alpha')
statemon_syn_resonant = StateMonitor(
    syn_resonant, variables='I_syn', record=True, name='statemon_syn_resonant')
statemon_syn_dpi = StateMonitor(
    syn_dpi, variables='I_syn', record=True, name='statemon_syn_dpi')

statemon_test_neuron1 = StateMonitor(test_neurons1, variables=[
    'Iin'], record=0, name='statemon_test_neuron1')
statemon_test_neuron2 = StateMonitor(test_neurons2, variables=[
    'Iin'], record=0, name='statemon_test_neuron2')
statemon_test_neuron3 = StateMonitor(test_neurons3, variables=[
    'Iin'], record=0, name='statemon_test_neuron3')

Net.add(input_spikegenerator, test_neurons1, test_neurons2, test_neurons3,
        syn_alpha, syn_resonant, syn_dpi, spikemon_inp,
        statemon_syn_alpha, statemon_syn_resonant, statemon_syn_dpi,
        statemon_test_neuron1, statemon_test_neuron2, statemon_test_neuron3)

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
win.nextRow()
p3 = win.addPlot()
p4 = win.addPlot()
win.nextRow()
p5 = win.addPlot()
p6 = win.addPlot()

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
         QtApp=app,
         show_immediately=True)
