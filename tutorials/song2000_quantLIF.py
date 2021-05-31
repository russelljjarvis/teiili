from brian2 import *
from teili.models.neuron_models import QuantStochLIF as teili_neu
from teili.models.synapse_models import QuantStochSynStdp as teili_syn
from teili.core.groups import Neurons, Connections
from teili.tools.misc import neuron_group_from_spikes

defaultclock.dt = 1 * ms

sim_duration = 100*second
N = 1000
F = 15*Hz
input = PoissonGroup(N, rates=F)
temp_monitor = SpikeMonitor(input, name='temp_monitor')
run(sim_duration, report='text')
input = neuron_group_from_spikes(
    N, defaultclock.dt, sim_duration,
    spike_indices=np.array(temp_monitor.i),
    spike_times=np.array(temp_monitor.t)*second)
del temp_monitor

neurons = Neurons(1, method=ExplicitStateUpdater('''x_new = f(x,t)'''),
    equation_builder=teili_neu(num_inputs=1))
neurons.tau = 10*ms
neurons.Vrest = -12*mV
neurons.Vreset = 0*mV
neurons.Vthr = 15*mV
neurons.refP = 0*ms
neurons.Vm_min = -12
neurons.Vm_max = 15
neurons.refrac_tau = 0*ms
#neurons.Iconst = 8.0*mA # 8.1 already changes mean to 1
#Ee = 0*mV N.A. I_syn already does it.

S = Connections(input, neurons, method=ExplicitStateUpdater('''x_new = f(x,t)'''),
    equation_builder=teili_syn)
S.connect()
S.tausyn = 5*ms
S.taupre = 20*ms
S.taupost = 30*ms # Change rand nums seems to cause too much inhibition
S.w_max = 15
S.dApre = 15
S.rand_num_bits_Apre = 4
S.rand_num_bits_Apost = 4
S.stdp_thres = 1
S.weight = 1
S.gain_syn = (1/32)*mA #Needed for high N and rate
S.w_plast = 'rand() * w_max'

mon = StateMonitor(S, 'w_plast', record=[0, 1])
s_mon = SpikeMonitor(input)
monmon = StateMonitor(neurons, 'Vm', record=True)

run(sim_duration, report='text')

subplot(311)
plot(S.w_plast / S.w_max, '.k')
ylabel('Weight / w_max')
xlabel('Synapse index')
subplot(312)
hist(S.w_plast / S.w_max, 20)
xlabel('Weight / w_max')
subplot(313)
plot(mon.t/second, mon.w_plast[0].T/S.w_max[0])
plot(mon.t/second, mon.w_plast[1].T/S.w_max[0])
xlabel('Time (s)')
ylabel('Weight / w_max')
tight_layout()

figure()
plot(monmon.Vm[0])
show()
