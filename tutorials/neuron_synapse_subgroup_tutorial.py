# -*- coding: utf-8 -*-
"""
This is the same as the tutorial file but uses subgroups.
It is used to test issues with subroups in brian2 and the library.
Later, a subroup unit test should be created

Created on 25.8.2017
"""

from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataControllers.Lineplot import Lineplot
from teili.tools.visualizer.DataControllers.Rasterplot import Rasterplot
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import numpy as np


from brian2 import ms, nA, second, prefs, SpikeMonitor, StateMonitor, SpikeGeneratorGroup, asarray

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import DPI as neuron_model
from teili.models.synapse_models import DPISyn as synapse_model
from teili.models.parameters.dpi_neuron_param import parameters as neuron_model_param

prefs.codegen.target = "numpy"
# defaultclock.dt = 10 * us

input_timestamps = np.asarray([1, 3, 4, 5, 6, 7, 8, 9]) * ms
input_indices = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])
input_spikegenerator = SpikeGeneratorGroup(1, indices=input_indices,
                                           times=input_timestamps, name='input_spikegenerator')


Net = TeiliNetwork()

test_neurons1 = Neurons(2, equation_builder=neuron_model(
    num_inputs=2), name="test_neurons1")

test_neurons2 = Neurons(2, equation_builder=neuron_model(
    num_inputs=2), name="test_neurons2")

test_neurons1_sub = test_neurons1[0:1]
test_neurons2_sub = test_neurons1[1:2]


input_synapse1 = Connections(input_spikegenerator,
                             test_neurons1_sub,
                             equation_builder=synapse_model(),
                             name="testSyn1a",
                             verbose=False)
input_synapse1.connect(True)

input_synapse2 = Connections(input_spikegenerator,
                             test_neurons2_sub,
                             equation_builder=synapse_model(),
                             name="testSyn1b",
                             verbose=False)
input_synapse2.connect(True)

input_synapse1.weight = 10
input_synapse2.weight = 10

test_synapse = Connections(test_neurons1,
                           test_neurons2,
                           equation_builder=synapse_model(),
                           name="testSyn2")
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
    input_synapse1.weight = 5000
    input_synapse2.weight = 5000
    test_synapse.weight = 800
    test_neurons1.Iconst = 10 * nA
elif 'Vm' in neuron_model().keywords['model']:
    input_synapse1.weight = 1.5
    input_synapse2.weight = 5000
    test_synapse.weight = 8.0
    test_neurons1.Iconst = 3 * nA


spikemonInp = SpikeMonitor(input_spikegenerator, name='spikemonInp')
spikemon = SpikeMonitor(test_neurons1, name='spikemon')
spikemonOut = SpikeMonitor(test_neurons2, name='spikemonOut')
statemonInpSyn = StateMonitor(
    input_synapse1, variables='I_syn', record=True, name='statemonInpSyn')
statemonNeuOut = StateMonitor(test_neurons2, variables=[
                              'Imem'], record=0, name='statemonNeuOut')
statemonNeuIn = StateMonitor(test_neurons1, variables=[
                             "Iin", "Imem", "Iahp"], record=[0, 1], name='statemonNeu')
statemonSynOut = StateMonitor(
    test_synapse, variables='I_syn', record=True, name='statemonSynOut')

Net.add(input_spikegenerator, test_neurons1, test_neurons2, input_synapse1, input_synapse2,
        test_synapse, spikemonInp, spikemon,
        spikemonOut, statemonNeuIn, statemonNeuOut, statemonSynOut, statemonInpSyn)

duration = 0.500
Net.run(duration * second)

# Visualize simulation results

app = QtGui.QApplication.instance()
pg.setConfigOptions(antialias=True)
MyPlotSettings = PlotSettings(fontsize_title=12,
                              fontsize_legend=12,
                              fontsize_axis_labels=10,
                              marker_size=7)

win = pg.GraphicsWindow(title='teili Test Simulation')
win.resize(1900, 600)
win.setWindowTitle('Simple SNN')

p1 = win.addPlot()
p2 = win.addPlot()
win.nextRow()
p3 = win.addPlot()
p4 = win.addPlot()
win.nextRow()
p5 = win.addPlot()
p6 = win.addPlot()


# Spike generator
Rasterplot(MyEventsModels=[spikemonInp],
           MyPlotSettings=MyPlotSettings,
           time_range=[0, duration],
           neuron_id_range=None,
           title="Spike generator",
           xlabel='Time (s)',
           ylabel="Neuron ID",
           backend='pyqtgraph',
           mainfig=win,
           subfig_rasterplot=p1,
           QtApp=app,
           show_immediately=False)

Lineplot(DataModel_to_x_and_y_attr=[(statemonInpSyn, ('t', 'I_syn'))],
         MyPlotSettings=MyPlotSettings,
         x_range=[0, duration],
         title="Input synapses",
         xlabel="Time (s)",
         ylabel="Synaptic current (A)",
         backend='pyqtgraph',
         mainfig=win,
         subfig=p2,
         QtApp=app,
         show_immediately=False)

Lineplot(DataModel_to_x_and_y_attr=[(statemonNeuIn, ('t', 'Imem'))],
         MyPlotSettings=MyPlotSettings,
         x_range=[0, duration],
         title='Intermediate neuron',
         xlabel="Time (s)",
         ylabel="Membrane current Imem (A)",
         backend='pyqtgraph',
         mainfig=win,
         subfig=p3,
         QtApp=app,
         show_immediately=False)

Lineplot(DataModel_to_x_and_y_attr=[(statemonSynOut, ('t', 'I_syn'))],
         MyPlotSettings=MyPlotSettings,
         x_range=[0, duration],
         title="Output synapses",
         xlabel="Time (s)",
         ylabel="Synaptic current I_e",
         backend='pyqtgraph',
         mainfig=win,
         subfig=p4,
         QtApp=app,
         show_immediately=False)

Rasterplot(MyEventsModels=[spikemonOut],
           MyPlotSettings=MyPlotSettings,
           time_range=[0, duration],
           neuron_id_range=None,
           title="Output spikes",
           xlabel='Time (s)',
           ylabel="Neuron ID",
           backend='pyqtgraph',
           mainfig=win,
           subfig_rasterplot=p5,
           QtApp=app,
           show_immediately=False)

Lineplot(DataModel_to_x_and_y_attr=[(statemonNeuOut, ('t', 'Imem'))],
         MyPlotSettings=MyPlotSettings,
         x_range=[0, duration],
         title="Output membrane current",
         xlabel="Time (s)",
         ylabel="Membrane current I_mem",
         backend='pyqtgraph',
         mainfig=win,
         subfig=p6,
         QtApp=app,
         show_immediately=False)

app.exec()
