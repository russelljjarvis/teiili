#from brian2 import *
#
#defaultclock.dt = 1*ms
#
## Generate lfsr numbers
#lfsr_numbers = np.array([[1,2,3,0,0,0,0],[1,2,3,4,5,6,7]])
#inp = TimedArray(np.transpose(lfsr_numbers)*mV, dt=1*ms)
##time_index = (t+ts)%lfsr_len : second
#
## groups
#neurons = NeuronGroup(3, '''dv/dt = (-v + inp(t, i % 2)) / tau : volt
#        ts : second
#        target_neuron : 1
#        seed : 1
#        lfsr_len : second
#        ''')
#neurons.lfsr_len = [4, 4, 3]*ms
#tau=1*ms
#
## gets who needs which lfsr and set seeds (as position of array)
## TODO generates seeds randomly, exclusing repeated and measuring final len
## TODO but keep original array to map to linked neurons
## TODO seed not neessary in group, set counter like this instead
## TODO map seeds into linking index e.g. {0:0, 15:1, 3:2, 0:0, 3:2, 1:3}
#neurons.seed[0], neurons.seed[1] = 0, 4
#neurons.seed[2] = 0
#lfsr_info = [{'L': 4, 'seed_len': 2}, {'L': 3, 'seed_len': 1}]
#
## Create one group for each lfsr, size depends on how many diff seeds
##lfsr = NeuronGroup(neurons.N, '''v = I((t+ts)%lfsr_len, target_neuron): volt (shared)
##        counter:second
##        ts:second
##        lfsr_len:1
##        target_neuron:1''')
#
## TODO add the if condition to reset
##lfsr.run_regularly('''counter=counter+1*ms if True''', dt=1*ms)
#
## link variables
## TODO get ranges from indexes of lfsr_len, for simplicity, they should be already next to one another
## TODO looks like I cannot set I for some neurons inside a group only,
##      so I need LFSR group to contain all the LFSRs
##neurons.I = linked_var(lfsr, 'v', index=[0,1,0])
#
#statemon = StateMonitor(neurons, ['v'], record=True)
#seeds = [0,0,1]
##lfsr.counter=seeds*ms
#neurons.ts=seeds*ms
##lfsr.lfsr_len=[3,3,4]
#neurons.target_neuron=[0,0,1]
#net = Network(neurons, statemon)
#net.run(4*ms, report='stdout', report_period=1*ms)
from brian2 import *
ta = TimedArray([1, 2, 3, 4] * mV, dt=0.1*ms)
G = NeuronGroup(1, 'v = ta(t) : volt')
mon = StateMonitor(G, 'v', record=True)
net = Network(G, mon)
net.run(1*ms)
print(mon[0].v)
ta2d = TimedArray([[1, 2], [3, 4], [5, 6]]*mV, dt=0.1*ms)
G = NeuronGroup(4, '''v = ta2d(t, i%2) : volt''')
mon = StateMonitor(G, 'v', record=True)
net = Network(G, mon)
net.run(0.2*ms)
print(mon.v[:])
