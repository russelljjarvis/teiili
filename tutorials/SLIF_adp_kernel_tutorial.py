import numpy as np
from scipy.stats import norm

from brian2 import ms, mV, prefs, SpikeMonitor, StateMonitor, defaultclock,\
    ExplicitStateUpdater, SpikeGeneratorGroup, TimedArray

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import StochasticLIF as neuron_model
from teili.models.synapse_models import StochasticSyn_decay as static_synapse_model
from teili.stimuli.testbench import SequenceTestbench
from teili.tools.add_run_reg import add_lfsr
from teili.tools.group_tools import add_group_activity_proxy
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.tools.converter import delete_doublets

from lfsr import create_lfsr

import sys
import pickle
import os
from datetime import datetime

#############
# Load models
path = os.path.expanduser("/home/pablo/git/teili")
model_path = os.path.join(path, "teili", "models", "equations", "")
adp_synapse_model = SynapseEquationBuilder.import_eq(
        model_path + 'StochSynAdp.py')

# Initialize simulation preferences
prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms
stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')
net = TeiliNetwork()

# Initialize input sequence
num_items = 3
num_channels = 102
sequence_duration = 300
noise_prob = None
item_rate = 20
spike_times, spike_indices = [], []
sequence_repetitions = 25
training_duration = sequence_repetitions*sequence_duration*ms
sequence = SequenceTestbench(num_channels, num_items, sequence_duration,
                                     noise_prob, item_rate)
tmp_i, tmp_t = sequence.stimuli()
input_spikes = SpikeGeneratorGroup(num_channels, tmp_i, tmp_t,
                period=sequence_duration*ms)
input_monitor = SpikeMonitor(input_spikes)
net.add(input_spikes, input_monitor)
print('Generating input...')
net.run(training_duration, report='stdout', report_period=100*ms)
spike_indices = np.array(input_monitor.i)
spike_times = np.array(input_monitor.t/ms)

# Reproduce activity in a neuron group (necessary for STDP compatibility)
spike_times = [spike_times[np.where(spike_indices==i)[0]]*ms for i in range(num_channels)]
# Create matrix where each row (neuron id) is associated with time when there
# is a spike or -1 when there is not
converted_input = (np.zeros((num_channels, int(training_duration/defaultclock.dt))) - 1)*ms
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
num_exc = 1
num_inh = 27
exc_cells = Neurons(num_exc,
                    equation_builder=neuron_model(num_inputs=2),
                    method=stochastic_decay,
                    name='exc_cells',
                    verbose=True)
# Register proxy arrays
dummy_unit = 1*mV
exc_cells.variables.add_array('activity_proxy', 
                               size=exc_cells.N,
                               dimensions=dummy_unit.dim)
exc_cells.variables.add_array('normalized_activity_proxy', 
                               size=exc_cells.N)

inh_cells = Neurons(num_inh,
                    equation_builder=neuron_model(num_inputs=1),
                    method=stochastic_decay,
                    name='inh_cells',
                    verbose=True)

inh_exc_conn = Connections(inh_cells, exc_cells,
                           equation_builder=adp_synapse_model,
                           method=stochastic_decay,
                           name='inh_exc_conn')
feedforward_exc = Connections(seq_cells, exc_cells,
                              equation_builder=static_synapse_model(),
                              method=stochastic_decay,
                              name='feedforward_exc')
feedforward_inh = Connections(seq_cells, inh_cells,
                              equation_builder=static_synapse_model(),
                              method=stochastic_decay,
                              name='feedforward_inh')

feedforward_exc.connect()
feedforward_inh.connect()
inh_exc_conn.connect()

# Time constants
inh_exc_conn.tau_syn = 10*ms
feedforward_exc.tau_syn = 10*ms
feedforward_inh.tau_syn = 10*ms
exc_cells.tau = 19*ms
inh_cells.tau = 10*ms

# LFSR lengths
exc_cells.lfsr_num_bits = 5
inh_cells.lfsr_num_bits = 5
inh_exc_conn.lfsr_num_bits_syn = 5
feedforward_exc.lfsr_num_bits_syn = 5
feedforward_inh.lfsr_num_bits_syn = 4

seed = 12
exc_cells.Vm = 3*mV
inh_cells.Vm = 3*mV

inh_exc_conn.weight = -1
# 1 = no inhibition, 0 = maximum inhibition
inh_exc_conn.variance_th = .99
inh_exc_conn.w_plast = 3
# Sets preference for inputs
feedforward_exc.weight = norm.pdf([x for x in range(num_channels)], scale=5, loc=int(num_channels/2))*50+6
for i in range(num_items):
    symbol_weights = int(num_channels*(num_inh/num_items))
    if i != 1:
        weights = 6
    else:
        weights = norm.pdf([x for x in range(symbol_weights)], scale=5, loc=int(symbol_weights/2))*50+6
    init_id = int(symbol_weights*i)
    final_id = int(symbol_weights*(i+1))
    feedforward_inh.weight[init_id:final_id] = weights

# Set LFSRs for each group
ta = create_lfsr([exc_cells, inh_cells],
                 [inh_exc_conn, feedforward_exc,
                     feedforward_inh],
                 defaultclock.dt)

# Add proxy activity group
activity_proxy_group = [exc_cells]
add_group_activity_proxy(activity_proxy_group,
                         buffer_size=200,
                         decay=150)
inh_exc_conn.variance_th = np.random.uniform(
        low=inh_exc_conn.variance_th - 0.1,
        high=inh_exc_conn.variance_th + 0.1,
        size=len(inh_exc_conn))

##################
# Setting up monitors
spikemon_exc_neurons = SpikeMonitor(exc_cells, name='spikemon_exc_neurons')
spikemon_inh_neurons = SpikeMonitor(inh_cells, name='spikemon_inh_neurons')
spikemon_seq_neurons = SpikeMonitor(seq_cells, name='spikemon_seq_neurons')
statemon_ffe_conns = StateMonitor(feedforward_exc, variables=['I_syn'], record=True,
                                  name='statemon_ffe_conns')
statemon_ie_conns = StateMonitor(inh_exc_conn, variables=['I_syn'], record=True,
                                  name='statemon_ie_conns')
statemon_inh_conns = StateMonitor(inh_exc_conn, variables=['w_plast', 'normalized_activity_proxy'], record=True,
                                  name='statemon_inh_conns')
statemon_exc_cells = StateMonitor(exc_cells, variables=['Vm'], record=True,
                                  name='statemon_exc_cells')
statemon_inh_cells = StateMonitor(inh_cells, variables=['Vm'], record=True,
                                  name='statemon_inh_cells')

net = TeiliNetwork()
net.add(seq_cells, exc_cells, inh_cells, inh_exc_conn, feedforward_exc,
        feedforward_inh, spikemon_exc_neurons, spikemon_inh_neurons,
        spikemon_seq_neurons, statemon_ffe_conns, statemon_inh_conns,
        statemon_ie_conns, statemon_exc_cells, statemon_inh_cells)
net.run(training_duration, report='stdout', report_period=100*ms)

# Plots
from brian2 import *
figure()
y = np.mean(statemon_inh_conns.normalized_activity_proxy, axis=0)
stdd=np.std(statemon_inh_conns.normalized_activity_proxy, axis=0)
plot(statemon_inh_conns.t/ms, y)
fill_between(statemon_inh_conns.t/ms, y-stdd, y+stdd, facecolor='lightblue')

figure()
plot(statemon_ffe_conns.t/ms, np.sum(statemon_ffe_conns.I_syn, axis=0)/amp, color='r', label='summed exc. current')
plot(statemon_inh_conns.t/ms, np.sum(statemon_ie_conns.I_syn, axis=0)/amp, color='b', label='summed inh. current')
ylabel('Current [amp]')
xlabel('time [ms]')
legend()

figure()
plot(statemon_ffe_conns.t/ms, np.sum(statemon_ffe_conns.I_syn[:34,:], axis=0)/amp, color='r', label='summed exc. current')
plot(statemon_inh_conns.t/ms, np.sum(statemon_ie_conns.I_syn[:9,:], axis=0)/amp, color='b', label='summed inh. current')
ylabel('Current [amp]')
xlabel('time [ms]')
title('1st symbol')
legend()

figure()
plot(statemon_ffe_conns.t/ms, np.sum(statemon_ffe_conns.I_syn[34:68,:], axis=0)/amp, color='r', label='summed exc. current')
plot(statemon_inh_conns.t/ms, np.sum(statemon_ie_conns.I_syn[9:18,:], axis=0)/amp, color='b', label='summed inh. current')
ylabel('Current [amp]')
xlabel('time [ms]')
title('2nd symbol')
legend()

figure()
plot(statemon_ffe_conns.t/ms, np.sum(statemon_ffe_conns.I_syn[68:,:], axis=0)/amp, color='r', label='summed exc. current')
plot(statemon_inh_conns.t/ms, np.sum(statemon_ie_conns.I_syn[18:,:], axis=0)/amp, color='b', label='summed inh. current')
ylabel('Current [amp]')
xlabel('time [ms]')
title('3rd symbol')
legend()

figure()
plot(spikemon_exc_neurons.t/ms, spikemon_exc_neurons.i, '.')
title('Exc. neurons')
figure()
plot(spikemon_inh_neurons.t/ms, spikemon_inh_neurons.i, '.')
title('Inh. neurons')
figure()
plot(spikemon_seq_neurons.t/ms, spikemon_seq_neurons.i, '.')
title('Input spikes')
figure()
plot(statemon_exc_cells.t/ms, statemon_exc_cells.Vm[0])
show()

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')

app = QtGui.QApplication([])

## Create window with ImageView widget
win = QtGui.QMainWindow()
win.resize(800,800)
imv = pg.ImageView()
win.setCentralWidget(imv)
win.show()
win.setWindowTitle('pyqtgraph example: ImageView')

imv.setImage(np.reshape(statemon_inh_conns.w_plast, (5, 5, -1)), axes={'t':2, 'y':0, 'x':1})

## Set a custom color map
colors = [
    (0, 0, 0),
    (45, 5, 61),
    (84, 42, 55),
    (150, 87, 60),
    (208, 171, 141),
    (255, 255, 255)
]
cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
imv.setColorMap(cmap)
