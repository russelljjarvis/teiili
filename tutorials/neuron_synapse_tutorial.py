# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2017-25-08 13:43:10
"""
This is a tutorial to construct a simple network of neurons
using the teili framework.
The emphasise is on neuron groups and non-plastic synapses.
"""

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import numpy as np
import sys

from brian2 import ms, second, pA, nA, prefs,\
    SpikeMonitor, StateMonitor, \
    SpikeGeneratorGroup

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import DPI as neuron_model
from teili.models.synapse_models import DPISyn as synapse_model
from teili.models.parameters.dpi_neuron_param import parameters as neuron_model_param

from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataControllers import Rasterplot, Lineplot

prefs.codegen.target = "numpy"

input_timestamps = np.asarray([1, 3, 4, 5, 6, 7, 8, 9]) * ms
input_indices = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])
input_spikegenerator = SpikeGeneratorGroup(1, indices=input_indices,
                                           times=input_timestamps, name='input_spikegenerator')


Net = TeiliNetwork()

test_neurons1 = Neurons(2, equation_builder=neuron_model(
    num_inputs=2), name="test_neurons1")

test_neurons2 = Neurons(2, equation_builder=neuron_model(
    num_inputs=2), name="test_neurons2")

input_synapse = Connections(input_spikegenerator, test_neurons1,
                            equation_builder=synapse_model(), name="input_synapse", verbose=False)
input_synapse.connect(True)

test_synapse = Connections(test_neurons1, test_neurons2,
                           equation_builder=synapse_model(), name="test_synapse")
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
if 'Imem' in neuron_model().keywords['model']:
    input_synapse.weight = 5000
    test_synapse.weight = 800
    test_neurons1.Iconst = 10 * nA
elif 'Vm' in neuron_model().keywords['model']:
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

if 'Imem' in neuron_model().keywords['model']:
    statemon_test_neurons2 = StateMonitor(test_neurons2,
                                          variables=['Imem'],
                                          record=0, name='statemon_test_neurons2')
    statemon_test_neurons1 = StateMonitor(test_neurons1, variables=[
        "Iin", "Imem", "Iahp"], record=[0, 1], name='statemon_test_neurons1')
elif 'Vm' in neuron_model().keywords['model']:
    statemon_test_neurons2 = StateMonitor(test_neurons2,
                                          variables=['Vm'],
                                          record=0, name='statemon_test_neurons2')
    statemon_test_neurons1 = StateMonitor(test_neurons1, variables=[
        "Iin", "Vm", "Iadapt"], record=[0, 1], name='statemon_test_neurons1')


Net.add(input_spikegenerator, test_neurons1, test_neurons2,
        input_synapse, test_synapse,
        spikemon_input, spikemon_test_neurons1, spikemon_test_neurons2,
        statemon_test_neurons1, statemon_test_neurons2,
        statemon_test_synapse, statemon_input_synapse)

duration = 0.5
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

p1 = win.addPlot(title="Input spike generator")
p2 = win.addPlot(title="Input synapses")
win.nextRow()
p3 = win.addPlot(title='Intermediate test neurons 1')
p4 = win.addPlot(title="Test synapses")
win.nextRow()
p5 = win.addPlot(title="Rasterplot of output test neurons 2")
p6 = win.addPlot(title="Output test neurons 2")


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

# Input synapses
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

i_current_name = 'Imem' if 'Imem' in neuron_model().keywords['model'] else 'Vm'
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

# Output synapses
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
    MyData_output = [(statemon_test_neurons2, ('t', 'Imem'))]
if hasattr(statemon_test_neurons2, 'Vm'):
    MyData_output = [(statemon_test_neurons2, ('t', 'Vm'))]

Lineplot(DataModel_to_x_and_y_attr=MyData_output,
         MyPlotSettings=MyPlotSettings,
         x_range=[0, duration],
         title="Output test neurons 2",
         xlabel="Time (ms)",
         ylabel="%s" % i_current_name,
         backend='pyqtgraph',
         mainfig=win,
         subfig=p6,
         QtApp=app,
         show_immediately=False)

app.exec()
