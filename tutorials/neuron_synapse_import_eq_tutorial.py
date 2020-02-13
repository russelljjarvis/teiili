# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2017-25-08 13:43:10
# @Last Modified by:   mmilde
# @Last Modified time: 2019-01-10 14:51:45
# -*- coding: utf-8 -*-

"""
This is a tutorial example used to learn the basics of the Brian2 INI library.
The emphasise is on neuron groups and non-plastic synapses.
"""
import os
import sys
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import matplotlib.pyplot as plt
import numpy as np

from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second, Network, prefs,\
    SpikeMonitor, StateMonitor, figure, plot, show, xlabel, ylabel,\
    seed, xlim, ylim, subplot, network_operation, TimedArray,\
    defaultclock, SpikeGeneratorGroup, asarray, pamp, set_device, device

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.parameters.dpi_neuron_param import parameters as neuron_model_param
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder

from teili.tools.visualizer.DataControllers.Rasterplot import Rasterplot
from teili.tools.visualizer.DataControllers.Lineplot import Lineplot
from teili.tools.visualizer.DataViewers import PlotSettings

# For this example you must first run models/neuron_models.py and synapse_models.py,
# which will create the equation template. This will be stored in models/equations
# Building neuron objects
path = os.path.expanduser("~")
model_path = os.path.join(path, "teiliApps", "equations", "")

builder_object1 = NeuronEquationBuilder.import_eq(
    filename=model_path + 'DPI.py', num_inputs=2)
builder_object2 = NeuronEquationBuilder.import_eq(
    model_path + 'DPI.py', num_inputs=2)

builder_object3 = SynapseEquationBuilder.import_eq(
    model_path + 'DPISyn.py')
builder_object4 = SynapseEquationBuilder.import_eq(
    model_path + 'DPISyn.py')

prefs.codegen.target = "numpy"

input_timestamps = np.asarray([1, 3, 4, 5, 6, 7, 8, 9]) * ms
input_indices = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])
input_spikegenerator = SpikeGeneratorGroup(1, indices=input_indices,
                                           times=input_timestamps, name='gtestInp')

Net = TeiliNetwork()

test_neurons1 = Neurons(2, equation_builder=builder_object1, name="test_neurons1")

test_neurons2 = Neurons(2, equation_builder=builder_object2, name="test_neurons2")

input_synapse = Connections(input_spikegenerator, test_neurons1,
                     equation_builder=builder_object3, name="input_synapse")
input_synapse.connect(True)


test_synapse = Connections(test_neurons1, test_neurons2,
                  equation_builder=builder_object4, name="test_synapse")
test_synapse.connect(True)

'''
You can change all the parameters like this after creation
of the neurongroup or synapsegroup.
Note that the if condition is inly there for
convinience to switch between voltage- or current-based models.
Normally, you have one or the other in yur simulation, thus
you will not need the if condition.
'''
# Example of how to set parameters, saved as a dictionary
test_neurons1.set_params(neuron_model_param)
# Example of how to set a single parameter
test_neurons1.refP = 1 * ms
test_neurons2.set_params(neuron_model_param)
test_neurons2.refP = 1 * ms
if 'Imem' in builder_object1.keywords['model']:
    input_synapse.weight = 5000
    test_synapse.weight = 800
    test_neurons1.Iconst = 10 * nA
elif 'Vm' in builder_object1.keywords['model']:
    input_synapse.weight = 1.5
    test_synapse.weight = 8.0
    test_neurons1.Iconst = 3 * nA

spikemon_input = SpikeMonitor(input_spikegenerator, name='spikemon_input')
spikemon_test_neurons1 = SpikeMonitor(
    test_neurons1, name='spikemon_test_neurons1')
spikemon_test_neurons2 = SpikeMonitor(
    test_neurons2, name='spikemon_test_neurons2')

statemon_input_synapse = StateMonitor(
    input_synapse, variables='I_syn', record=True, name='statemon_input_synapse')

statemon_test_synapse = StateMonitor(
    test_synapse, variables='I_syn', record=True, name='statemon_test_synapse')

if 'Imem' in builder_object2.keywords['model']:
    statemon_test_neurons2 = StateMonitor(test_neurons2,
                                          variables=['Imem'],
                                          record=0, name='statemon_test_neurons2')
    statemon_test_neurons1 = StateMonitor(test_neurons1, variables=[
        "Iin", "Imem", "Iahp"], record=[0, 1], name='statemon_test_neurons1')
elif 'Vm' in builder_object2.keywords['model']:
    statemon_test_neurons2 = StateMonitor(test_neurons2,
                                          variables=['Vm'],
                                          record=0, name='statemon_test_neurons2')
    statemon_test_neurons1 = StateMonitor(test_neurons1, variables=[
        "Iin", "Vm", "Iadapt"], record=[0, 1], name='statemon_test_neurons1')


Net.add(input_spikegenerator, test_neurons1, test_neurons2,
        input_synapse, test_synapse,
        spikemon_input, spikemon_test_neurons1, spikemon_test_neurons2,
        statemon_test_neurons1, statemon_test_neurons2, statemon_test_synapse, statemon_input_synapse)

duration = 0.500
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

win = pg.GraphicsWindow()
win.resize(2100, 1200)
win.setWindowTitle('Simple Spiking Neural Network')
p1 = win.addPlot()
p2 = win.addPlot()
win.nextRow()
p3 = win.addPlot()
p4 = win.addPlot()
win.nextRow()
p5 = win.addPlot()
p6 = win.addPlot()


# Spike generator
Rasterplot(MyEventsModels=[spikemon_input],
                     MyPlotSettings=MyPlotSettings,
                     time_range=[0, duration],
                     neuron_id_range=None,
                     title="Input spike generator",
                     xlabel='Time (ms)',
                     ylabel="Neuron ID",
                     backend='pyqtgraph',
                     mainfig=win,
                     subfig_rasterplot=p1,
                     QtApp=app,
                     show_immediately=False)

Lineplot(DataModel_to_x_and_y_attr=[(statemon_input_synapse, ('t', 'I_syn'))],
                   MyPlotSettings=MyPlotSettings,
                   x_range=[0, duration],
                   title="Input synapses",
                   xlabel="Time (ms)",
                   ylabel="EPSC (A)",
                   backend='pyqtgraph',
                   mainfig=win,
                   subfig=p2,
                   QtApp=app,
                   show_immediately=False)

# Intermediate neurons
if hasattr(statemon_test_neurons1, 'Imem'):
    MyData_intermed_neurons = [(statemon_test_neurons1, ('t', 'Imem'))]
if hasattr(statemon_test_neurons1, 'Vm'):
    MyData_intermed_neurons = [(statemon_test_neurons1, ('t', 'Vm'))]

i_current_name = 'Imem' if 'Imem' in builder_object1.keywords['model'] else 'Vm'
Lineplot(DataModel_to_x_and_y_attr=MyData_intermed_neurons,
                   MyPlotSettings=MyPlotSettings,
                   x_range=[0, duration],
                   title='Intermediate test neurons 1',
                   xlabel="Time (ms)",
                   ylabel=i_current_name,
                   backend='pyqtgraph',
                   mainfig=win,
                   subfig=p3,
                   QtApp=app,
                   show_immediately=False)


Lineplot(DataModel_to_x_and_y_attr=[(statemon_test_synapse, ('t', 'I_syn'))],
                   MyPlotSettings=MyPlotSettings,
                   x_range=[0, duration],
                   title="Test synapses",
                   xlabel="Time (ms)",
                   ylabel="EPSC (A)",
                   backend='pyqtgraph',
                   mainfig=win,
                   subfig=p4,
                   QtApp=app,
                   show_immediately=False)


Rasterplot(MyEventsModels=[spikemon_test_neurons2],
                     MyPlotSettings=MyPlotSettings,
                     time_range=[0, duration],
                     neuron_id_range=None,
                     title="Rasterplot of output test neurons 2",
                     xlabel='Time (ms)',
                     ylabel="Neuron ID",
                     backend='pyqtgraph',
                     mainfig=win,
                     subfig_rasterplot=p5,
                     QtApp=app,
                     show_immediately=False)


if hasattr(statemon_test_neurons2, 'Imem'):
    MyData_output = [(statemon_test_neurons2, ('t','Imem'))]
if hasattr(statemon_test_neurons2, 'Vm'):
    MyData_output = [(statemon_test_neurons2, ('t','Vm'))]

Lineplot(DataModel_to_x_and_y_attr=MyData_output,
                   MyPlotSettings=MyPlotSettings,
                   x_range=[0, duration],
                   title="Output test neurons 2",
                   xlabel="Time (ms)",
                   ylabel="%s" %i_current_name,
                   backend='pyqtgraph',
                   mainfig=win,
                   subfig=p6,
                   QtApp=app,
                   show_immediately=False)

app.exec()
