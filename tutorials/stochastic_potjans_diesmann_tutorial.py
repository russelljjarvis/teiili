import numpy as np

from brian2 import defaultclock, prefs, ms, ExplicitStateUpdater, pA

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import QuantStochLIF as neuron_model
from teili.models.synapse_models import QuantStochSyn as synapse_model

defaultclock.dt = 1*ms
prefs.codegen.target = "numpy"
method = ExplicitStateUpdater('''x_new = f(x,t)''')

# Population size per layer
#          2/3e   2/3i   4e    4i    5e    5i    6e     6i    Th
num_layer = [20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948, 902]
# Total cortical Population
N = sum(num_layer[:-1])
# Number of neurons accumulated
neurons_accum = [0]
neurons_accum.extend(np.cumsum(num_layer))
# External populations related to background activity
#                            2/3e  2/3i  4e    4i    5e    5i    6e    6i
background_layer = np.array([1600, 1500 ,2100, 1900, 2000, 1900, 2900, 2100])
# Prob. connection table
p_conn = np.array([[0.101,  0.169, 0.044, 0.082, 0.032, 0.,     0.008, 0.,     0.    ],
                  [0.135,  0.137, 0.032, 0.052, 0.075, 0.,     0.004, 0.,     0.    ],
                  [0.008,  0.006, 0.050, 0.135, 0.007, 0.0003, 0.045, 0.,     0.0983],
                  [0.069,  0.003, 0.079, 0.160, 0.003, 0.,     0.106, 0.,     0.0619],
                  [0.100,  0.062, 0.051, 0.006, 0.083, 0.373,  0.020, 0.,     0.    ],
                  [0.055,  0.027, 0.026, 0.002, 0.060, 0.316,  0.009, 0.,     0.    ],
                  [0.016,  0.007, 0.021, 0.017, 0.057, 0.020,  0.040, 0.225,  0.0512],
                  [0.036,  0.001, 0.003, 0.001, 0.028, 0.008,  0.066, 0.144,  0.0196]])

# Initializations
# TODO membrane potential from gaussian?
# TODO equivalent of below
d_ex = 1.5*ms      	# Excitatory delay
std_d_ex = 0.75*ms 	# Std. Excitatory delay
d_in = 0.80*ms      # Inhibitory delay
std_d_in = 0.4*ms  	# Std. Inhibitory delay
tau_syn = 0.5*ms    # Post-synaptic current time constant
w_ex = 87.8*pA		   	# excitatory synaptic weight
std_w_ex = 0.1*w_ex     # standard deviation weigth

neurons = Neurons(N, equation_builder=neuron_model(num_inputs=2),
    name="neurons", method=method)
neuron_pops = []
num_pops = 8
for pop in range(0, num_pops):
        neuron_pops.append(neurons[neurons_accum[pop]:neurons_accum[pop+1]])
