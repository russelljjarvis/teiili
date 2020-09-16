"""
This code implements a sequence learning using and Excitatory-Inhibitory 
network with STDP.
"""
import numpy as np

from brian2 import ms, mV, prefs, SpikeMonitor, StateMonitor, defaultclock,\
    ExplicitStateUpdater, SpikeGeneratorGroup

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import StochasticLIF as neuron_model
from teili.models.synapse_models import StochasticSyn_decay_stoch_stdp as stdp_synapse_model
from teili.models.synapse_models import StochasticSyn_decay as static_synapse_model
from teili.stimuli.testbench import SequenceTestbench
from teili.tools.add_run_reg import add_lfsr

# Initialize simulation preferences
prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms
stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')

# Initialize input sequence
num_items = 7
num_channels = 100
sequence_duration = 500
noise_prob = .007
item_rate = 80
sequence = SequenceTestbench(num_channels, num_items, sequence_duration,
                             noise_prob, item_rate)
spike_indices, spike_times = sequence.stimuli()
seq_cells = SpikeGeneratorGroup(num_channels, spike_indices, spike_times)

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
                              equation_builder=static_synapse_model(),
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
feedforward_exc.weight = 1
add_lfsr(feedforward_exc, seed, defaultclock.dt)
feedforward_inh.weight = -1
add_lfsr(feedforward_inh, seed, defaultclock.dt)

# Setting up monitors
spikemon_exc_neurons = SpikeMonitor(exc_cells, name='spikemon_exc_neurons')
statemon_exc_cells = StateMonitor(exc_cells, variables=['Vm'], record=True,
                                  name='statemon_exc_cells')
statemon_rec_conns = StateMonitor(exc_exc_conn, variables=['w_plast'], record=True,
                                  name='statemon_rec_conns')

net = TeiliNetwork()
net.add(seq_cells, exc_cells, inh_cells, exc_exc_conn, exc_inh_conn, inh_exc_conn,
        feedforward_exc, feedforward_inh, spikemon_exc_neurons, statemon_exc_cells)
duration = 10*ms
net.run(duration, report='stdout', report_period=100*ms)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(spike_times/ms, spike_indices, '.')
plt.figure()
plt.plot(spikemon_exc_neurons.t/ms, spikemon_exc_neurons.i, '.')
plt.figure()
plt.plot(statemon_rec_conns.w_plast[0])
plt.figure()
plt.plot(statemon_exc_cells.Vm[0])
plt.show()
import pdb;pdb.set_trace()
