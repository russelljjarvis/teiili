# -*- coding: utf-8 -*-
# @Author: pabloabur
# @Date:   2020-07-12 11:50:45

"""
This file provides an example of how to use neuron and synapse models which are present
on neurmorphic chips in the context of synaptic plasticity based on precise timing of spikes.
We use a standard STDP protocal with a exponentioally decaying window.

"""
import pyqtgraph as pg
import numpy as np
import os

from brian2 import ms, ohm, mV, us, second, pA, prefs,\
    SpikeMonitor, StateMonitor, defaultclock,\
    implementation, check_units, ExplicitStateUpdater

from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import StochasticLIF as neuron_model
from teili.models.synapse_models import StochasticSyn_decay_stdp as stdp_synapse_model
from teili.models.synapse_models import StochasticSyn_decay as synapse_model
from teili.models.synapse_models import DPIstdp
from teili.stimuli.testbench import STDP_Testbench
from teili.tools.add_run_reg import add_lfsr

from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataModels import StateVariablesModel
from teili.tools.visualizer.DataControllers import Lineplot, Rasterplot

prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms
Net = TeiliNetwork()

stdp = STDP_Testbench()
pre_spikegenerator, post_spikegenerator = stdp.stimuli(isi=30)

stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')
pre_neurons = Neurons(2,
                      equation_builder=neuron_model(num_inputs=2),
                      method=stochastic_decay,
                      name='pre_neurons',
                      verbose=True)

post_neurons = Neurons(2,
                       equation_builder=neuron_model(num_inputs=2),
                       method=stochastic_decay,
                       name='post_neurons',
                       verbose=True)

pre_synapse = Connections(pre_spikegenerator, pre_neurons,
                          method=stochastic_decay,
                          equation_builder=synapse_model(),
                          name='pre_synapse')

post_synapse = Connections(post_spikegenerator, post_neurons,
                           method=stochastic_decay,
                           equation_builder=synapse_model(),
                           name='post_synapse')

stdp_synapse = Connections(pre_neurons, post_neurons,
                           method=stochastic_decay,
                           equation_builder=stdp_synapse_model(),
                           name='stdp_synapse')

pre_synapse.connect(True)
post_synapse.connect(True)
stdp_synapse.connect("i==j")

# Set parameters:
seed = 12
add_lfsr(pre_neurons, seed, defaultclock.dt)
pre_neurons.Vm = 3*mV
add_lfsr(post_neurons, seed, defaultclock.dt)
post_neurons.Vm = 3*mV

pre_synapse.weight = 91
add_lfsr(pre_synapse, seed, defaultclock.dt)
post_synapse.weight = 91
add_lfsr(post_synapse, seed, defaultclock.dt)

stdp_synapse.tau_syn = 5*ms
post_synapse.tau_syn = 5*ms
pre_synapse.tau_syn = 5*ms

add_lfsr(stdp_synapse, seed, defaultclock.dt)

# Setting up monitors
spikemon_pre_neurons = SpikeMonitor(pre_neurons, name='spikemon_pre_neurons')
statemon_pre_neurons = StateMonitor(pre_neurons, variables=['Vm', 'Iin'],
                                    record=0, name='statemon_pre_neurons')

spikemon_post_neurons = SpikeMonitor(
    post_neurons, name='spikemon_post_neurons')
statemon_post_neurons = StateMonitor(
    post_neurons, variables=['Vm', 'Iin', 'decay_probability'], record=0,
    name='statemon_post_neurons')

statemon_pre_synapse = StateMonitor(
    pre_synapse, variables=['I_syn'], record=0, name='statemon_pre_synapse')

statemon_post_synapse = StateMonitor(stdp_synapse, variables=[
    'I_syn', 'w_plast', 'weight'],
    record=True, name='statemon_post_synapse')

Net.add(pre_spikegenerator, post_spikegenerator,
        pre_neurons, post_neurons,
        pre_synapse, post_synapse, stdp_synapse,
        spikemon_pre_neurons, spikemon_post_neurons,
        statemon_pre_neurons, statemon_post_neurons,
        statemon_pre_synapse, statemon_post_synapse)

duration = 2.
Net.run(duration * second)


# Visualize
win_stdp = pg.GraphicsWindow(title="STDP Unit Test")
win_stdp.resize(2500, 1500)
win_stdp.setWindowTitle("Spike Time Dependent Plasticity")

p1 = win_stdp.addPlot()
win_stdp.nextRow()
p2 = win_stdp.addPlot()
win_stdp.nextRow()
p3 = win_stdp.addPlot()

p2.setXLink(p1)
p3.setXLink(p2)

text1 = pg.TextItem(text='Homoeostasis', anchor=(-0.3, 0.5))
text2 = pg.TextItem(text='Weak Pot.', anchor=(-0.3, 0.5))
text3 = pg.TextItem(text='Weak Dep.', anchor=(-0.3, 0.5))
text4 = pg.TextItem(text='Strong Pot.', anchor=(-0.3, 0.5))
text5 = pg.TextItem(text='Strong Dep.', anchor=(-0.3, 0.5))
text6 = pg.TextItem(text='Homoeostasis', anchor=(-0.3, 0.5))
p1.addItem(text1)
p1.addItem(text2)
p1.addItem(text3)
p1.addItem(text4)
p1.addItem(text5)
p1.addItem(text6)
text1.setPos(0, 0.5)
text2.setPos(0.300, 0.5)
text3.setPos(0.600, 0.5)
text4.setPos(0.900, 0.5)
text5.setPos(1.200, 0.5)
text6.setPos(1.500, 0.5)

Rasterplot(MyEventsModels=[spikemon_pre_neurons, spikemon_post_neurons],
            MyPlotSettings=PlotSettings(colors=['w', 'r']),
            time_range=(0, duration),
            neuron_id_range=(-1, 2),
            title="STDP protocol",
            xlabel="Time (s)",
            ylabel="Neuron ID",
            backend='pyqtgraph',
            mainfig=win_stdp,
            subfig_rasterplot=p1)

Lineplot(DataModel_to_x_and_y_attr=[(statemon_post_synapse, ('t', 'w_plast'))],
            MyPlotSettings=PlotSettings(colors=['g']),
            x_range=(0, duration),
            title="Plastic synaptic weight",
            xlabel="Time (s)",
            ylabel="w_plast",
            backend='pyqtgraph',
            mainfig=win_stdp,
            subfig=p2)

datamodel = StateVariablesModel(state_variable_names=['I_syn'],
                                state_variables=[np.asarray(statemon_post_synapse.I_syn[0])],
                                state_variables_times=[np.asarray(statemon_post_synapse.t)])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_I_syn', 'I_syn'))],
            MyPlotSettings=PlotSettings(colors=['m']),
            x_range=(0, duration),
            title="Post synaptic current",
            xlabel="Time (s)",
            ylabel="I_syn (A)",
            backend='pyqtgraph',
            mainfig=win_stdp,
            subfig=p3,
            show_immediately=True)

win_traces = pg.GraphicsWindow(title="STDP Unit Test")
win_traces.resize(2500, 1500)
win_traces.setWindowTitle("States during STDP")

p1 = win_traces.addPlot()
win_traces.nextRow()
p2 = win_traces.addPlot()
win_traces.nextRow()
p3 = win_traces.addPlot()
p2.setXLink(p1)
p3.setXLink(p2)

datamodel = StateVariablesModel(state_variable_names=['Vm'],
                                state_variables=[np.asarray(statemon_post_neurons.Vm[0])],
                                state_variables_times=[np.asarray(statemon_post_neurons.t)])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_Vm', 'Vm'))],
            MyPlotSettings=PlotSettings(colors=['m']),
            x_range=(0, duration),
            title="Post neuron 0 Vm",
            xlabel="Time (s)",
            ylabel="voltage",
            backend='pyqtgraph',
            mainfig=win_stdp,
            subfig=p1)

datamodel = StateVariablesModel(state_variable_names=['decay_probability'],
                                state_variables=[np.asarray(statemon_post_neurons.decay_probability[0])],
                                state_variables_times=[np.asarray(statemon_post_neurons.t)])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_decay_probability', 'decay_probability'))],
            MyPlotSettings=PlotSettings(colors=['m']),
            x_range=(0, duration),
            xlabel="Time (s)",
            ylabel="decay_probability",
            backend='pyqtgraph',
            mainfig=win_stdp,
            subfig=p2)

datamodel = StateVariablesModel(state_variable_names=['Iin'],
                                state_variables=[np.asarray(statemon_post_neurons.Iin[0])],
                                state_variables_times=[np.asarray(statemon_post_neurons.t)])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_Iin', 'Iin'))],
            MyPlotSettings=PlotSettings(colors=['m']),
            x_range=(0, duration),
            title="Post neuron 0 Iin",
            xlabel="Time (s)",
            ylabel="Iin",
            backend='pyqtgraph',
            mainfig=win_stdp,
            subfig=p3,
            show_immediately=True)
