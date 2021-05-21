from teili import TeiliNetwork
from teili.core.groups import Neurons, Connections
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.tools.misc import neuron_group_from_spikes
from teili.models.neuron_models import QuantStochLIF as teili_neu
from teili.models.synapse_models import QuantStochSynStdp as teili_syn

from brian2 import Hz, mV, second, ms, prefs, SpikeMonitor, StateMonitor,\
        defaultclock, PoissonGroup, ExplicitStateUpdater

import os
import matplotlib.pyplot as plt
import numpy as np

sim_duration = 100*second
#defaultclock.dt = 1 * ms

# Preparing input
N = 1000
F = 15*Hz
input_group = PoissonGroup(N, rates=F)

net = TeiliNetwork()
temp_monitor = SpikeMonitor(input_group, name='temp_monitor')
net.add(input_group, temp_monitor)
print('Converting Poisson input into neuro group...')
net.run(sim_duration, report='text')
input_group = neuron_group_from_spikes(
    N, defaultclock.dt, sim_duration,
    spike_indices=np.array(temp_monitor.i),
    spike_times=np.array(temp_monitor.t)*second)

# Loading models
path = os.path.expanduser("~")
model_path = os.path.join(path, 'git', 'teili', 'tutorials')
song_neu = NeuronEquationBuilder.import_eq(model_path+'/song_neu.py')
song_syn = SynapseEquationBuilder.import_eq(model_path+'/song_syn.py')
quant_stdp = SynapseEquationBuilder.import_eq(model_path+'/quant_stdp.py')

#neurons = Neurons(N=1, method = ExplicitStateUpdater('''x_new = f(x,t)'''), equation_builder=teili_neu)
neurons = Neurons(N=1, equation_builder=song_neu)
#S = Connections(input_group, neurons, method=ExplicitStateUpdater('''x_new = f(x,t)'''), equation_builder=teili_syn)
S = Connections(input_group, neurons, equation_builder=song_syn)
#S = Connections(input_group, neurons, method=ExplicitStateUpdater('''x_new = f(x,t)'''), equation_builder=quant_stdp)

# Initializations
S.connect()
S.w_plast = np.random.rand(len(S.w_plast)) * S.w_max
mon = StateMonitor(S, ['w_plast', 'Apre', 'Apost'], record=[0, 1])
in_mon = SpikeMonitor(input_group)
neu_mon = StateMonitor(neurons, ['Vm'], record=True)
# For local quant syn file
#S.taupre = 20*ms
#S.taupost = 20*ms
#S.w_max = .01
#S.lr = .0001
#S.deltaApre = 15
#S.deltaApost = 15
#S.rand_num_bits_pre1 = 4
#S.rand_num_bits_post1 = 4
#S.stdp_thres = 1
# For teili syn
#S.taupre = 20*ms
#S.taupost = 20*ms
#S.w_max = 15
#S.dApre = 15
#S.rand_num_bits_Apre = 4
#S.rand_num_bits_Apost = 4
#S.stdp_thres = 1

net = TeiliNetwork()
net.add(neurons, input_group, S, mon, in_mon, neu_mon)
net.run(sim_duration, report='text')

plt.subplot(311)
plt.plot(S.w_plast / S.w_max, '.k')
plt.ylabel('Weight / w_max')
plt.xlabel('Synapse index')
plt.subplot(312)
plt.hist(S.w_plast / S.w_max, 20)
plt.xlabel('Weight / w_max')
plt.subplot(313)
plt.plot(mon.t/second, mon.w_plast.T/S.w_max[0])
plt.xlabel('Time (s)')
plt.ylabel('Weight / w_max')
plt.tight_layout()

plt.figure()
plt.plot(mon.Apre[0], 'r')
plt.plot(mon.Apost[0], 'b')

plt.show()
