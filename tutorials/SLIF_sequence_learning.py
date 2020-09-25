"""
This code implements a sequence learning using and Excitatory-Inhibitory 
network with STDP.
"""
import numpy as np

from brian2 import ms, mV, prefs, SpikeMonitor, StateMonitor, defaultclock,\
    ExplicitStateUpdater, SpikeGeneratorGroup, TimedArray

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import StochasticLIF as neuron_model
from teili.models.synapse_models import StochasticSyn_decay_stoch_stdp as stdp_synapse_model
from teili.models.synapse_models import StochasticSyn_decay as static_synapse_model
from teili.stimuli.testbench import SequenceTestbench
from teili.tools.add_run_reg import add_lfsr
from teili.tools.sorting import SortMatrix

# Initialize simulation preferences
prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms
stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')

# Initialize input sequence
num_items = 2
num_channels = 100
sequence_duration = 2000*ms
sub_sequence_duration = 200
noise_prob = .007
item_rate = 80
spike_times, spike_indices = [], []
sequence_repetitions = 10
for i in range(sequence_repetitions):
    sequence = SequenceTestbench(num_channels, num_items, sub_sequence_duration,
                                 noise_prob, item_rate)
    tmp_i, tmp_t = sequence.stimuli()
    spike_indices.extend(tmp_i)
    tmp_t = [(x/ms+i*sub_sequence_duration) for x in tmp_t]
    spike_times.extend(tmp_t)
spike_indices = np.array(spike_indices)
spike_times = np.array(spike_times) * ms
# Save them for comparison
spk_i, spk_t = spike_indices, spike_times

# Reproduce activity in a neuron group (necessary for STDP compatibility)
spike_times = [spike_times[np.where(spike_indices==i)[0]] for i in range(num_channels)]
converted_input = (np.zeros((num_channels, int(sequence_duration/defaultclock.dt))) - 1)*ms
for ind, val in enumerate(spike_times):
    converted_input[ind, (val/defaultclock.dt).astype(int)] = val
converted_input = np.transpose(converted_input)
converted_input = TimedArray(converted_input, dt=defaultclock.dt)
#seq_cells = SpikeGeneratorGroup(num_channels, spike_indices, spike_times)
seq_cells = Neurons(num_channels, model='tspike=converted_input(t, i): second',
        threshold='t==tspike', refractory='tspike < 0*ms')
seq_cells.namespace.update({'converted_input':converted_input})

# Create neuron groups
num_exc = 85
num_inh = 15
exc_cells = Neurons(num_exc,
                    equation_builder=neuron_model(num_inputs=3),
                    method=stochastic_decay,
                    name='exc_cells',
                    verbose=True)
inh_cells = Neurons(num_inh,
                    equation_builder=neuron_model(num_inputs=2),
                    method=stochastic_decay,
                    name='inh_cells',
                    verbose=True)

# Create synapses
exc_exc_conn = Connections(exc_cells, exc_cells,
                           equation_builder=stdp_synapse_model(),
                           method=stochastic_decay,
                           name='exc_exc_conn')
exc_inh_conn = Connections(exc_cells, inh_cells,
                           equation_builder=static_synapse_model(),
                           method=stochastic_decay,
                           name='exc_inh_conn')
inh_exc_conn = Connections(inh_cells, exc_cells,
                           equation_builder=static_synapse_model(),
                           method=stochastic_decay,
                           name='inh_exc_conn')
feedforward_exc = Connections(seq_cells, exc_cells,
                              equation_builder=stdp_synapse_model(),
                              method=stochastic_decay,
                              name='feedforward_exc')
feedforward_inh = Connections(seq_cells, inh_cells,
                              equation_builder=static_synapse_model(),
                              method=stochastic_decay,
                              name='feedforward_inh')

# Connect synapses
feedforward_exc.connect(True)
feedforward_inh.connect(True)
exc_exc_conn.connect('True', p=.85)
exc_inh_conn.connect('True', p=.15)
inh_exc_conn.connect('True', p=.85)

# Setting parameters
seed = 12
exc_cells.Vm = 3*mV
add_lfsr(exc_cells, seed, defaultclock.dt)
inh_cells.Vm = 3*mV
add_lfsr(inh_cells, seed, defaultclock.dt)
exc_exc_conn.weight = 1
add_lfsr(exc_exc_conn, seed, defaultclock.dt)
exc_inh_conn.weight = 1
add_lfsr(exc_inh_conn, seed, defaultclock.dt)
inh_exc_conn.weight = -1
add_lfsr(inh_exc_conn, seed, defaultclock.dt)
feedforward_exc.weight = 2
add_lfsr(feedforward_exc, seed, defaultclock.dt)
feedforward_inh.weight = -1
add_lfsr(feedforward_inh, seed, defaultclock.dt)

# Setting up monitors
spikemon_exc_neurons = SpikeMonitor(exc_cells, name='spikemon_exc_neurons')
spikemon_seq_neurons = SpikeMonitor(seq_cells, name='spikemon_seq_neurons')
statemon_exc_cells = StateMonitor(exc_cells, variables=['Vm'], record=True,
                                  name='statemon_exc_cells')
statemon_rec_conns = StateMonitor(exc_exc_conn, variables=['w_plast'], record=True,
                                  name='statemon_rec_conns')

net = TeiliNetwork()
net.add(seq_cells, exc_cells, inh_cells, exc_exc_conn, exc_inh_conn, inh_exc_conn,
        feedforward_exc, feedforward_inh, statemon_exc_cells, spikemon_exc_neurons,
        spikemon_seq_neurons)
net.run(sequence_duration, report='stdout', report_period=100*ms)

n_rows = num_exc
n_cols = n_rows
w_plast = []
recurrent_ids = []
for i in range(n_rows):
    w_plast.append(list(exc_exc_conn.w_plast[i, :]))
    recurrent_ids.append(list(exc_exc_conn.j[i, :]))
sorted_w = SortMatrix(ncols=n_cols, nrows=n_rows, matrix=np.array(w_plast, dtype=object), 
        fill_ids=np.array(recurrent_ids, dtype=object))
sorted_i = np.asarray([np.where(
                np.asarray(sorted_w.permutation) == int(i))[0][0] for i in spikemon_exc_neurons.i])

import matplotlib.pyplot as plt

plt.figure()
plt.plot(spikemon_seq_neurons.t/ms, spikemon_seq_neurons.i, '.')
plt.plot(spk_t/ms, spk_i, '+')
plt.title('Input')
plt.figure()
plt.plot(spikemon_exc_neurons.t/ms, spikemon_exc_neurons.i, '.')
plt.title('Raster')
plt.figure()
plt.plot(spikemon_exc_neurons.t/ms, sorted_i, '.')
plt.title('Sorted raster')
plt.figure()
plt.plot(statemon_rec_conns.w_plast[0])
plt.title('receptive field 0')
plt.figure()
plt.plot(statemon_exc_cells.Vm[0])
plt.title('Membrane potential 0')
plt.show()
