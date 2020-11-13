from brian2 import *

defaultclock.dt = 1*ms

# groups
neurons = NeuronGroup(3, '''inp : volt (linked)
        seed : 1
        lfsr_len : 1
        dv/dt = (-v + inp) / tau : volt''')
neurons.lfsr_len = [4, 4, 3]

# gets who needs which lfsr and set seeds (as position of array)
# TODO generates randomly, exclusing repeated and measuring final len
# TODO but keep original array to map to linked neurons
# TODO seed not neessary in group, set counter like this instead
# TODO map seeds into linking index e.g. {0:0, 15:1, 3:2, 0:0, 3:2, 1:3}
neurons.seed[0], neurons.seed[1] = 0, 4
neurons.seed[2] = 0
lfsr_info = [{'L': 4, 'seed_len': 2}, {'L': 3, 'seed_len': 1}]

# Create one group for each lfsr, size depends on how many diff seeds
lfsr4 = NeuronGroup(lfsr_info[0]['seed_len'], '''v = inp1(counter): volt
        counter:second''')
lfsr3 = NeuronGroup(lfsr_info[1]['seed_len'], '''v = inp2(counter): volt
        counter:second''')

# TODO add the if condition to reset
lfsr4.run_regularly('''counter=counter+1*ms''', dt=1*ms)
lfsr3.run_regularly('''counter=counter+1*ms''', dt=1*ms)

# Generate lfsr numbers
inp1 = TimedArray([1,2,3,4]*mV, dt=1*ms)
inp2 = TimedArray([5,6,7,8]*mV, dt=1*ms)

# link variables so that 
# TODO get ranges from indexes of lfsr_len, for simplicity, they should be already next to one another
neurons.inp[0:2] = linked_var(lfsr4, 'v', index=linking1)
neurons.inp[2] = linked_var(lfsr3, 'v', index=linking2)

statemon = StateMonitor(lfsr4, ['counter', 'v'], record=True)
lfsr4.counter=[0,0,1,2]*ms
net = Network(lfsr4, statemon)
net.run(4*ms, report='stdout', report_period=1*ms)
