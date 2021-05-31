from brian2 import *
from teili.models.neuron_models import ExpLIF as teili_neu
from teili.models.synapse_models import ExponentialStdp as teili_syn
from teili.core.groups import Neurons, Connections

N = 1000
F = 15*Hz
input = PoissonGroup(N, rates=F)

neurons = Neurons(1, equation_builder=teili_neu(num_inputs=1))
#taum = 10*ms
neurons.Cm = 140*pF # 281
neurons.gL = 14*nS # 4.3
neurons.EL = -74*mV # -55
neurons.Vres = -60*mV # -70.6
neurons.VT = -54*mV # -50.4
neurons.DeltaT = 0.001*mV # -2
neurons.refP = 0*ms # 2
#Ee = 0*mV N.A. I_syn already does it.

S = Connections(input, neurons, equation_builder=teili_syn)
S.connect()
S.tausyn = 5*ms
S.taupre = 20*ms # 10
S.taupost = S.taupre
S.w_max = .01 # 1
S.dApre = .01 # 0.1
S.Q_diffAPrePost = 1.05
S.w_plast = 'rand() * w_max'
S.weight = 1

mon = StateMonitor(S, 'w_plast', record=[0, 1])
s_mon = SpikeMonitor(input)
monmon = StateMonitor(neurons, 'Vm', record=True)

run(100*second, report='text')

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
