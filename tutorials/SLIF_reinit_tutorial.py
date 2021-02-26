import numpy as np
from scipy.stats import norm

from brian2 import Hz, ms, mV, prefs, SpikeMonitor, StateMonitor, defaultclock,\
    ExplicitStateUpdater, SpikeGeneratorGroup, TimedArray, PoissonGroup

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import StochasticLIF as neuron_model
from teili.stimuli.testbench import SequenceTestbench
from teili.tools.group_tools import add_group_activity_proxy,\
        add_group_params_re_init
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.tools.converter import delete_doublets

from lfsr import create_lfsr
from reinit_functions import get_prune_indices, get_spawn_indices,\
        wplast_re_init, weight_re_init, tau_re_init, delay_re_init
        #reset_re_init_counter#, get_re_init_indices

from SLIF_utils import neuron_group_from_spikes

import sys
import pickle
import os
from datetime import datetime

#############
# Load models
path = os.path.expanduser("/home/pablo/git/teili")
model_path = os.path.join(path, "teili", "models", "equations", "")
neuron_model_Adapt = NeuronEquationBuilder.import_eq(
        model_path + 'StochLIFAdapt.py')
stdp_synapse_model = SynapseEquationBuilder.import_eq(
        model_path + 'StochStdpNew.py')

# Initialize simulation preferences
prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms
stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')

# Initialize input sequence: Poisson pattern presented many times
num_inputs = 50
num_items = 1
input_rate = 30
pattern_duration = 50
pattern_repetitions = 200

repeated_input = SequenceTestbench(num_inputs, num_items, pattern_duration,
                                   rate=input_rate,
                                   cycle_repetitions=pattern_repetitions)
spike_indices, spike_times = repeated_input.stimuli()

sim_time = pattern_duration * pattern_repetitions

# Reproduce activity in a neuron group (necessary for STDP compatibility)
seq_cells = neuron_group_from_spikes(spike_indices, spike_times/ms, num_inputs,
                                     defaultclock.dt,
                                     sim_time)

#################
# Building network
# cells
num_exc = 5
exc_cells = Neurons(num_exc,
                    equation_builder=neuron_model_Adapt(num_inputs=3),
                    method=stochastic_decay,
                    name='exc_cells',
                    verbose=True)
# Connections
feedforward_exc = Connections(seq_cells, exc_cells,
                              equation_builder=stdp_synapse_model(),
                              method=stochastic_decay,
                              name='feedforward_exc')
feedforward_exc.connect()
# Set sparsity
for i in range(num_exc):
    ffe_zero_w = np.random.choice(num_inputs, int(num_inputs*.7), replace=False)
    feedforward_exc.weight[ffe_zero_w,i] = 0
    feedforward_exc.w_plast[ffe_zero_w,i] = 0

# Time constants
feedforward_exc.tau_syn = 5*ms
exc_cells.tau = 19*ms

# LFSR lengths
feedforward_exc.lfsr_num_bits_syn = 5
exc_cells.lfsr_num_bits = 5

seed = 12
exc_cells.Vm = 3*mV

# Set LFSRs for each group
ta = create_lfsr([exc_cells],
                 [feedforward_exc],
                 defaultclock.dt)

##################
# Synaptic homeostasis
#feedforward_exc.variables.add_array('prune_indices', size=len(feedforward_exc.weight))
#feedforward_exc.variables.add_array('spawn_indices', size=len(feedforward_exc.weight))
#feedforward_exc.namespace.update({'get_prune_indices': get_prune_indices})
#feedforward_exc.namespace.update({'get_spawn_indices': get_spawn_indices})
#feedforward_exc.namespace.update({'wplast_re_init': wplast_re_init})
#feedforward_exc.namespace.update({'tau_re_init': tau_re_init})
#feedforward_exc.namespace.update({'delay_re_init': delay_re_init})
#feedforward_exc.namespace.update({'weight_re_init': weight_re_init})
#feedforward_exc.namespace.update({'reset_re_init_counter': reset_re_init_counter})

################################
add_group_params_re_init(groups=[feedforward_exc],
                         variable='w_plast',
                         re_init_variable='re_init_counter',
                         re_init_threshold=1,
                         re_init_dt=1000*ms,
                         dist_param=3,#TODO
                         scale=1,#TODO
                         distribution='gamma',
                         sparsity=.7,#TODO
                         clip_min=0,
                         clip_max=15,
                         reference='synapse_counter')

################################
#feedforward_exc.namespace['reference'] = 2
#feedforward_exc.namespace['re_init_threshold'] = 1
#feedforward_exc.namespace.update({'get_re_init_indices': get_re_init_indices})
#feedforward_exc.variables.add_array('re_init_indices', size=np.int(len(feedforward_exc)))
#feedforward_exc.run_regularly('''re_init_indices = get_re_init_indices(w_plast,\
#                                                                       re_init_counter,\
#                                                                       1,\
#                                                                       1,\
#                                                                       reference,\
#                                                                       re_init_threshold,\
#                                                                       1*ms,\
#                                                                       t)''',
#                              dt=1000*ms)

#feedforward_exc.run_regularly('''prune_indices = get_prune_indices(\
#                                                    weight,\
#                                                    re_init_counter,\
#                                                    t)''',
#                                                    dt=re_init_dt,
#                                                    order=0)
#feedforward_exc.run_regularly('''spawn_indices = get_spawn_indices(\
#                                                    prune_indices,\
#                                                    weight,\
#                                                    t)''',
#                                                    dt=re_init_dt,
#                                                    order=1)
#
#feedforward_exc.run_regularly('''w_plast = wplast_re_init(w_plast,\
#                                                          spawn_indices,\
#                                                          t)''',
#                                                          dt=re_init_dt,
#                                                          order=2)
#feedforward_exc.run_regularly('''tau_syn = tau_re_init(tau_syn,\
#                                                       spawn_indices,\
#                                                       t)''',
#                                                       dt=re_init_dt,
#                                                       order=3)
##feedforward_exc.run_regularly('''delay = delay_re_init(delay,\
##                                                       spawn_indices,\
##                                                       t)''',
##                                                       dt=re_init_dt,
##                                                       order=4)
#feedforward_exc.run_regularly('''weight = weight_re_init(weight,\
#                                                         spawn_indices,\
#                                                         prune_indices,\
#                                                         t)''',
#                                                         dt=re_init_dt,
#                                                         order=5)
#feedforward_exc.run_regularly('''re_init_counter = reset_re_init_counter(re_init_counter)''',
#                                                                         dt=re_init_dt,
#                                                                         order=6)

##################
# Setting up monitors
spikemon_exc_neurons = SpikeMonitor(exc_cells, name='spikemon_exc_neurons')
spikemon_seq_neurons = SpikeMonitor(seq_cells, name='spikemon_seq_neurons')
statemon_wplast = StateMonitor(feedforward_exc, variables=['w_plast'],
                               record=True, name='statemon_wplast')
statemon_weight = StateMonitor(feedforward_exc, variables=['weight'],
                               record=True, name='statemon_weight')
statemon_counter = StateMonitor(feedforward_exc, variables=['re_init_counter'],
                               record=True, name='statemon_counter')
#statemon_pruned = StateMonitor(feedforward_exc, variables=['prune_indices'],
#                               record=True, name='statemon_pruned')

net = TeiliNetwork()
net.add(exc_cells, seq_cells, feedforward_exc, spikemon_exc_neurons,
        spikemon_seq_neurons, statemon_wplast, statemon_weight,
        statemon_counter)#, statemon_pruned)
net.run(sim_time*ms, report='stdout', report_period=100*ms)

# Plots
from teili.tools.visualizer.DataModels import EventsModel, StateVariablesModel
from teili.tools.visualizer.DataControllers import Rasterplot, Lineplot
from teili.tools.visualizer.DataViewers import PlotSettings
import pyqtgraph as pg
from PyQt5 import QtGui

QtApp = QtGui.QApplication([])

colors = [
    (0, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 255, 255)
]
cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 4), color=colors)

win1 = QtGui.QMainWindow()
image_axis = pg.PlotItem()
image_axis.setLabel(axis='bottom', text='Neuron index')
image_axis.setLabel(axis='left', text='Input channels')
m1 = pg.ImageView(view=image_axis)
win1.setCentralWidget(m1)
win1.show()
m1.setImage(np.reshape(statemon_wplast.w_plast, (num_inputs, num_exc, -1)), axes={'t':2, 'y':0, 'x':1})
m1.setColorMap(cmap)
image_axis = pg.PlotItem()
image_axis.setLabel(axis='bottom', text='postsynaptic neuron')
image_axis.setLabel(axis='left', text='presynaptic neuron')

win2 = QtGui.QMainWindow()
image_axis = pg.PlotItem()
image_axis.setLabel(axis='bottom', text='Neuron index')
image_axis.setLabel(axis='left', text='Input channels')
m2 = pg.ImageView(view=image_axis)
win2.setCentralWidget(m2)
win2.show()
m2.setImage(np.reshape(statemon_weight.weight, (num_inputs, num_exc, -1)), axes={'t':2, 'y':0, 'x':1})
m2.setColorMap(cmap)
image_axis = pg.PlotItem()
image_axis.setLabel(axis='bottom', text='postsynaptic neuron')
image_axis.setLabel(axis='left', text='presynaptic neuron')

exc_raster = EventsModel.from_brian_spike_monitor(spikemon_exc_neurons)
seq_raster = EventsModel.from_brian_spike_monitor(spikemon_seq_neurons)

state_variable_names = ['re_init_counter']
counter_copy = np.array(statemon_counter.re_init_counter)
counter_copy[np.isnan(counter_copy)] = 0
state_variables = [counter_copy]
state_variables_times = [statemon_counter.t/ms]
counter_line = StateVariablesModel(state_variable_names, state_variables, state_variables_times)
#reinit_ratio = []
#for i in range(int(sim_time)):
#    reinit_ratio.append(len(np.where(statemon_pruned.prune_indices[:,i]==1)[0]))
#state_variable_names = ['reinit_ratio']
#state_variables = [reinit_ratio]
#state_variables_times = [statemon_pruned.t/ms]
#ratio_line = StateVariablesModel(state_variable_names, state_variables, state_variables_times)

#line_plot1 = Lineplot(DataModel_to_x_and_y_attr=[(counter_line, ('t_re_init_counter', 're_init_counter'))],
#                      title='reinit counters with time', xlabel='time (s)',
#                      ylabel='counter value', backend='pyqtgraph', QtApp=QtApp)
#line_plot2 = Lineplot(DataModel_to_x_and_y_attr=[(ratio_line, ('t_reinit_ratio', 'reinit_ratio'))],
#                      MyPlotSettings = PlotSettings(marker_size=30),
#                      title='Number of pruned/spawned synapses with time',
#                      xlabel='time (s)', ylabel='# pruned/spawned',
#                      backend='pyqtgraph', QtApp=QtApp)
raster_plot1 = Rasterplot(MyEventsModels=[exc_raster], backend='pyqtgraph', QtApp=QtApp)
raster_plot2 = Rasterplot(MyEventsModels=[seq_raster], backend='pyqtgraph', QtApp=QtApp,
                show_immediately=True)
#np.savez('reinit.npz', ratio=reinit_ratio, time=statemon_pruned.t/ms)
from brian2 import *
plot(counter_line.re_init_counter.T)
show()
