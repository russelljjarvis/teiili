import numpy as np
from scipy.stats import norm

from brian2 import Hz, ms, mV, prefs, SpikeMonitor, StateMonitor, defaultclock,\
    ExplicitStateUpdater, SpikeGeneratorGroup, TimedArray, PoissonGroup

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import StochasticLIF as neuron_model
from teili.stimuli.testbench import SequenceTestbench
from teili.tools.add_run_reg import add_lfsr
from teili.tools.group_tools import add_group_activity_proxy
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.tools.converter import delete_doublets

from lfsr import create_lfsr
from reinit_functions import get_prune_indices, get_spawn_indices,\
        wplast_re_init, weight_re_init, tau_re_init, delay_re_init,\
        reset_re_init_counter

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
net = TeiliNetwork()

# Initialize input sequence
num_items = 1
num_channels = 50
sequence_duration = 50
noise_prob = None
item_rate = 30
spike_times, spike_indices = [], []
sequence_repetitions = 10
sim_time = sequence_duration * sequence_repetitions * ms
sequence = SequenceTestbench(num_channels, num_items, sequence_duration,
                                     noise_prob, item_rate)
tmp_i, tmp_t = sequence.stimuli()
poisson_spikes = SpikeGeneratorGroup(num_channels, tmp_i, tmp_t,
                period=sequence_duration*ms)
# Reproduce activity in a neuron group (necessary for STDP compatibility)
input_monitor = SpikeMonitor(poisson_spikes)
net.add(poisson_spikes, input_monitor)
print('Generating input...')
net.run(sim_time, report='stdout', report_period=100*ms)
spike_indices = np.array(input_monitor.i)
spike_times = np.array(input_monitor.t/ms)
spike_times = [spike_times[np.where(spike_indices==i)[0]]*ms for i in range(num_channels)]
# Create matrix where each row (neuron id) is associated with time when there
# is a spike or -1 when there is not
converted_input = (np.zeros((num_channels, int(sim_time/defaultclock.dt))) - 1)*ms
for ind, val in enumerate(spike_times):
    converted_input[ind, np.around(val/defaultclock.dt).astype(int)] = val
converted_input = np.transpose(converted_input)
converted_input = TimedArray(converted_input, dt=defaultclock.dt)
# t is simulation time, and will be equal to tspike when there is a spike
# Cell remains refractory when there is no spike, i.e. tspike=-1
seq_cells = Neurons(num_channels, model='tspike=converted_input(t, i): second',
        threshold='t==tspike', refractory='tspike < 0*ms')
seq_cells.namespace.update({'converted_input':converted_input})

#################
# Building network
# cells
num_exc = 20
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
    ffe_zero_w = np.random.choice(num_channels, int(num_channels*.3), replace=False)
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
feedforward_exc.variables.add_array('prune_indices', size=len(feedforward_exc.weight))
feedforward_exc.variables.add_array('spawn_indices', size=len(feedforward_exc.weight))
feedforward_exc.namespace.update({'get_prune_indices': get_prune_indices})
feedforward_exc.namespace.update({'get_spawn_indices': get_spawn_indices})
feedforward_exc.namespace.update({'wplast_re_init': wplast_re_init})
feedforward_exc.namespace.update({'tau_re_init': tau_re_init})
feedforward_exc.namespace.update({'delay_re_init': delay_re_init})
feedforward_exc.namespace.update({'weight_re_init': weight_re_init})
feedforward_exc.namespace.update({'reset_re_init_counter': reset_re_init_counter})

reinit_period = 20*ms

feedforward_exc.run_regularly('''prune_indices = get_prune_indices(\
                                                    prune_indices,\
                                                    weight,\
                                                    re_init_counter,\
                                                    t)''',
                                                    dt=reinit_period,
                                                    when='start')
feedforward_exc.run_regularly('''spawn_indices = get_spawn_indices(\
                                                    spawn_indices,\
                                                    prune_indices,\
                                                    weight,\
                                                    t)''',
                                                    dt=reinit_period,
                                                    when='start')

feedforward_exc.run_regularly('''w_plast = wplast_re_init(w_plast,\
                                                          spawn_indices,\
                                                          t)''',
                                                          dt=reinit_period,
                                                          when='end')
feedforward_exc.run_regularly('''tau_syn = tau_re_init(tau_syn,\
                                                       spawn_indices,\
                                                       t)''',
                                                       dt=reinit_period,
                                                       when='end')
#feedforward_exc.run_regularly('''delay = delay_re_init(delay,\
#                                                       spawn_indices,\
#                                                       t)''',
#                                                       dt=reinit_period,
#                                                       when='end')
feedforward_exc.run_regularly('''weight = weight_re_init(weight,\
                                                         spawn_indices,\
                                                         prune_indices,\
                                                         t)''',
                                                         dt=reinit_period,
                                                         when='end')
feedforward_exc.run_regularly('''re_init_counter = reset_re_init_counter(re_init_counter)''',
                                                                         dt=reinit_period,
                                                                         when='end')

##################
# Setting up monitors
spikemon_exc_neurons = SpikeMonitor(exc_cells, name='spikemon_exc_neurons')
spikemon_seq_neurons = SpikeMonitor(seq_cells, name='spikemon_seq_neurons')
statemon_wplast = StateMonitor(feedforward_exc, variables=['w_plast'],
                               record=True, name='statemon_wplast')
statemon_weight = StateMonitor(feedforward_exc, variables=['weight'],
                               record=True, name='statemon_weight')

net = TeiliNetwork()
net.add(exc_cells, seq_cells, feedforward_exc, spikemon_exc_neurons,
        spikemon_seq_neurons, statemon_wplast, statemon_weight)
net.run(sim_time, report='stdout', report_period=100*ms)

# Plots
from teili.tools.visualizer.DataModels import EventsModel, StateVariablesModel
from teili.tools.visualizer.DataControllers import Rasterplot, Lineplot
from teili.tools.visualizer.DataViewers import PlotSettings
import pyqtgraph as pg
from PyQt5 import QtGui

QtApp = QtGui.QApplication([])

exc_raster = EventsModel.from_brian_spike_monitor(spikemon_exc_neurons)
seq_raster = EventsModel.from_brian_spike_monitor(spikemon_seq_neurons)
skip_not_rec_neuron_ids = True

RC = Rasterplot(MyEventsModels=[exc_raster], backend='pyqtgraph', QtApp=QtApp)
RC = Rasterplot(MyEventsModels=[seq_raster], backend='pyqtgraph', QtApp=QtApp,
                show_immediately=True)

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
m1.setImage(np.reshape(statemon_wplast.w_plast, (num_channels, num_exc, -1)), axes={'t':2, 'y':0, 'x':1})
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
m2.setImage(np.reshape(statemon_weight.weight, (num_channels, num_exc, -1)), axes={'t':2, 'y':0, 'x':1})
m2.setColorMap(cmap)
image_axis = pg.PlotItem()
image_axis.setLabel(axis='bottom', text='postsynaptic neuron')
image_axis.setLabel(axis='left', text='presynaptic neuron')
