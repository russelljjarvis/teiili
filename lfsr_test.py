from brian2 import *

defaultclock.dt = 1*ms
prefs.codegen.target = "numpy"

# Generate lfsr numbers
lfsr_numbers = np.array([1,2,3,1,2,3,4,5,6,7])*mV
ta = TimedArray(lfsr_numbers, dt=1*ms)
G = NeuronGroup(4, '''v = ta( ((seed+t) % lfsr_len) + lfsr_init ) : volt
                      lfsr_len : second
                      seed : second
                      lfsr_init : second''')
G.lfsr_len = [3, 7, 3, 7]*ms # get it considering whole array
G.lfsr_init = [0, 3, 0, 3]*ms # get it considering whole array
G.seed = [1, 1, 2, 3]*ms # get it thinking like each has an array
mon = StateMonitor(G, 'v', record=True)
net = Network(G, mon)
net.run(7*ms)
print(mon.v[0])
print(mon.v[1])
print(mon.v[2])
print(mon.v[3])
