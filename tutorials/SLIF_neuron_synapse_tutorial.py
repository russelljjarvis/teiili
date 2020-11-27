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

from brian2 import ExplicitStateUpdater, ms, mV, ohm, second, mA, prefs,\
    SpikeMonitor, StateMonitor, \
    SpikeGeneratorGroup, defaultclock, TimedArray

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import StochasticLIF as neuron_model
from teili.models.synapse_models import StochasticSyn_decay as synapse_model
from teili.tools.add_run_reg import add_lfsr
from lfsr import create_lfsr

from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataControllers import Rasterplot, Lineplot

defaultclock.dt = 1*ms

prefs.codegen.target = "numpy"

input_timestamps = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) * ms
input_indices = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
input_spikegenerator = SpikeGeneratorGroup(1, indices=input_indices,
                                           times=input_timestamps, 
                                           name='input_spikegenerator')


Net = TeiliNetwork()

stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')
test_neurons1 = Neurons(N=2, 
                        equation_builder=neuron_model(num_inputs=2), 
                        name="test_neurons1",
                        method=stochastic_decay,
                        verbose=True)

test_neurons2 = Neurons(N=2, 
                        equation_builder=neuron_model(num_inputs=2), 
                        name="test_neurons2",
                        method=stochastic_decay,
                        verbose=True)

input_synapse = Connections(input_spikegenerator, test_neurons1,
                            equation_builder=synapse_model(), 
                            name="input_synapse",
                            method=stochastic_decay,
                            verbose=True)
input_synapse.connect(True)

test_synapse = Connections(test_neurons1, test_neurons2,
                           equation_builder=synapse_model(),
                           name="test_synapse",
                           method=stochastic_decay,
                           verbose=True)
test_synapse.connect(True)

'''
You can change all the parameters like this after creation
of the neurongroup or synapsegroup.
Note that the if condition is inly there for
convinience to switch between voltage- or current-based models.
Normally, you have one or the other in yur simulation, thus
you will not need the if condition.
'''
num_bits = 6
seed = 12
test_neurons1.lfsr_num_bits = num_bits
test_neurons2.lfsr_num_bits = num_bits
lfsr1 = create_lfsr([test_neurons1, test_neurons2], [input_synapse, test_synapse])
test_neurons1.lfsr_max_value = lfsr1['max_value']*ms
test_neurons1.lfsr_init = lfsr1['init']*ms
test_neurons1.seed = lfsr1['seed']*ms
neuron_timedarray = TimedArray(lfsr1['array'], dt=defaultclock.dt)
test_neurons1.namespace['neuron_timedarray'] = neuron_timedarray 
test_neurons2.namespace['neuron_timedarray'] = neuron_timedarray 
#add_lfsr(test_neurons1, seed, defaultclock.dt)
test_neurons1.Vm = 3*mV
test_neurons2.lfsr_max_value = lfsr1['max_value']*ms
test_neurons2.lfsr_init = lfsr1['init']*ms
test_neurons2.seed = lfsr1['seed']*ms
#add_lfsr(test_neurons2, seed, defaultclock.dt)
test_neurons2.Vm = 3*mV

input_synapse.lfsr_num_bits_syn = num_bits
test_synapse.lfsr_num_bits_syn = num_bits
#add_lfsr(input_synapse, seed, defaultclock.dt)
#add_lfsr(test_synapse, seed, defaultclock.dt)
input_synapse.lfsr_max_value_syn = lfsr1['max_value']*ms
input_synapse.lfsr_init_syn = lfsr1['init']*ms
input_synapse.seed_syn = lfsr1['seed']*ms
lfsr2 = create_lfsr(test_synapse.lfsr_num_bits_syn)
test_synapse.lfsr_max_value_syn = lfsr2['max_value']*ms
test_synapse.lfsr_init_syn = lfsr2['init']*ms
test_synapse.seed_syn = lfsr2['seed']*ms
syn_timedarray = TimedArray(lfsr1['array'], dt=defaultclock.dt)
input_synapse.namespace['syn_timedarray'] = syn_timedarray 
test_synapse.namespace['syn_timedarray'] = syn_timedarray 

# Example of how to set a single parameter
# Fast neuron to allow more spikes
test_neurons1.refrac_tau = 1 * ms
test_neurons2.refrac_tau = 1 * ms
test_neurons1.tau = 10 * ms
test_neurons2.tau = 10 * ms

# long EPSC or big weight to allow summations
input_synapse.tau_syn = 10*ms
test_synapse.tau_syn = 10*ms
input_synapse.weight = 2
test_synapse.weight = 15
test_neurons1.Iconst = 13.0 * mA


spikemon_input = SpikeMonitor(input_spikegenerator, name='spikemon_input')
spikemon_test_neurons1 = SpikeMonitor(
    test_neurons1, name='spikemon_test_neurons1')
spikemon_test_neurons2 = SpikeMonitor(
    test_neurons2, name='spikemon_test_neurons2')

statemon_input_synapse = StateMonitor(
    input_synapse, variables=['I_syn', 'decay_probability_syn'], record=True,
    name='statemon_input_synapse')

statemon_test_synapse = StateMonitor(
    test_synapse, variables=['I_syn', 'decay_probability_syn'], record=True,
    name='statemon_test_synapse')

if 'Imem' in neuron_model().keywords['model']:
    statemon_test_neurons2 = StateMonitor(test_neurons2,
                                          variables=['Imem'],
                                          record=0, name='statemon_test_neurons2')
    statemon_test_neurons1 = StateMonitor(test_neurons1, variables=[
        "Iin", "Imem", "Iahp"], record=[0, 1], name='statemon_test_neurons1')
elif 'Vm' in neuron_model().keywords['model']:
    statemon_test_neurons2 = StateMonitor(test_neurons2,
                                          variables=['Vm'],
                                          record=True, name='statemon_test_neurons2')
    statemon_test_neurons1 = StateMonitor(test_neurons1, variables=[
        "Iin", "Vm"], record=True, name='statemon_test_neurons1')


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

p2.setXLink(p1)
p3.setXLink(p2)
p4.setXLink(p3)
p4.setXLink(p3)
p5.setXLink(p4)
p6.setXLink(p5)

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
#from bokeh.plotting import figure, show
#from bokeh.layouts import gridplot
#p1 = figure(y_range=[-.5, .5], x_axis_label='Time (ms)',
#        y_axis_label='Neuron ID', width=650, height=300, x_range=[-5,100])
#p1.circle(np.array(spikemon_input.t/ms), np.array(spikemon_input.i), line_color='black')
#p2 = figure(x_axis_label='Time (ms)',
#        y_axis_label='EPCS (mA)', width=650, height=300, x_range=p1.x_range)
#p2.line(np.array(statemon_input_synapse.t/ms), np.array(statemon_input_synapse[0].I_syn/mA), line_color='black', line_width=2)
#p3 = figure(x_axis_label='Time (ms)',
#        y_axis_label='Vm (mV)', width=650, height=300, x_range=p1.x_range)
#p3.line(np.array(statemon_test_neurons1.t/ms), np.array(statemon_test_neurons1[0].Vm/mV), line_color='black', line_width=2)
#p4 = figure(x_axis_label='Time (ms)',
#        y_axis_label='Vm (mV)', width=650, height=300, x_range=p1.x_range)
#p4.line(np.array(statemon_test_neurons2.t/ms), np.array(statemon_test_neurons2[0].Vm/mV), line_color='black', line_width=2)
#pf = gridplot([[p1, p2], [p3, p4]])
#show(pf)
