# -*- coding: utf-8 -*-
# @Author: pabloabur
# @Date:   2020-27-05 18:05:05
"""
This is a simple tutorial to simulate 4 stochastic leaky integrate and fire
neurons charging and firing action potentials.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from teili.tools.add_run_reg import add_lfsr

from brian2 import ExplicitStateUpdater
from brian2 import StateMonitor, SpikeMonitor, ms, mV, defaultclock,\
        implementation, check_units, SpikeGeneratorGroup

from teili import TeiliNetwork
from teili.core.groups import Neurons, Connections
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder

defaultclock.dt = 1*ms

path = os.path.expanduser("~")
model_path = os.path.join(path, "teiliApps", "equations", "")

neuron_model = NeuronEquationBuilder.import_eq(
    filename=model_path + 'StochasticLIF.py', num_inputs=2)
synapse_model = SynapseEquationBuilder.import_eq(
    filename=model_path + 'StochasticLIFSyn.py')

input_timestamps = np.array(range(1, 400, 100))*ms
input_indices = np.zeros(len(input_timestamps))
input_spike_generator = SpikeGeneratorGroup(1, indices=input_indices,
                                            times=input_timestamps)

num_neurons = 4
# With this state updater, the abstract state update code generated by Brian2
# will be the same as the equation provided
stochastic_decay = ExplicitStateUpdater('''x_new = dt*f(x,t)''')
neuron = Neurons(num_neurons, equation_builder=neuron_model,
                 method=stochastic_decay, name="test_neurons", verbose=True)

synapse = Connections(input_spike_generator, neuron, method=stochastic_decay,
                      equation_builder=synapse_model, name="test_synapse", verbose=True)
synapse.connect(True)

#synapse.weight = 3 # No spikes
synapse.weight = 4 # Spikes
neuron.Vm = 3*mV
add_lfsr(neuron, 12345, defaultclock.dt)
add_lfsr(synapse, 12345, defaultclock.dt)

spikemon = SpikeMonitor(neuron, name='spike_monitor')
neuron_monitor = StateMonitor(neuron, variables=['Vm', 'Iin', 'decay_probability'], record=True, name='state_monitor_neu')
synapse_monitor = StateMonitor(synapse, variables=['I_syn'], record=True, name='state_monitor_syn')

duration = 400*ms
Net = TeiliNetwork()
Net.add(input_spike_generator, neuron, synapse, spikemon, neuron_monitor, synapse_monitor)
Net.run(duration)

# Visualize compare random numbers generated with rand, for 250s
#plt.figure()
#Z = np.random.rand(250000)   # Test data
#Z = np.reshape(Z, (500,500))
#plt.imshow(Z, cmap='gray', interpolation='nearest')
#plt.show()
#for i in range(num_neurons):
#    plt.figure()
#    Z = neuron_monitor.decay_probability[i,:]
#    Z = np.reshape(Z, (500,500))
#    plt.imshow(Z, cmap='gray', interpolation='nearest')
#    plt.show()

#plt.figure()
#spike_times = spikemon[0].t/ms
#ISI = spike_times[1:-1] - spike_times[0:-2]
#_ = plt.hist(ISI, bins=20, range=(0, 100))

plt.figure()
plt.plot(spikemon.t/ms, spikemon.i, 'ko')
plt.ylabel('Neuron index')
plt.xlabel('Time (samples)')

plt.figure()
colors = ['k', 'r', 'g', 'b']
for i in range(num_neurons):
    plt.plot(neuron_monitor.t/ms, neuron_monitor.Vm[i], colors[i], label=str(i))
plt.xlabel('Time (samples)')
plt.ylabel('membrane potential (a.u.)')
plt.legend()

plt.figure()
for i in range(num_neurons):
    plt.plot(neuron_monitor.t/ms, neuron_monitor.Iin[i], 'k')
plt.xlabel('Time (samples)')
plt.ylabel('Iin (a.u.)');

plt.figure()
for i in range(num_neurons):
    plt.plot(synapse_monitor.t/ms, synapse_monitor.I_syn[i], 'k')
plt.xlabel('Time (samples)')
plt.ylabel('I_syn (a.u.)');
plt.show()

#with open('teili.npy', 'wb') as f:
#    np.save(f, neuron_monitor.Vm)
