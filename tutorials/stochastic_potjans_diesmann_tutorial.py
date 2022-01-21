from brian2 import defaultclock, prefs, ms

from teili import TeiliNetwork
from orca_wta import ORCA_WTA
from orca_column import orcaColumn

import numpy as np

defaultclock.dt = 1*ms
prefs.codegen.target = "numpy"

cmc = orcaColumn()

# Number of thalamic projections
num_thal = 902
# Number of neurons accumulated
neurons_accum = [0]
neurons_accum.extend(np.cumsum(num_layer))
# External populations related to background activity
#                            2/3e  2/3i  4e    4i    5e    5i    6e    6i
background_layer = np.array([1600, 1500 ,2100, 1900, 2000, 1900, 2900, 2100])
# Prob. connection table: from colum to row
#                 L2/3e 	L2/3i 	L4e 	L4i   L5e   L5i 	L6e    L6i 	   Th
p_conn = np.array([[     ,  0.169,      , 0.082,      , 0.,          , 0.,     0.    ], #L2/3e
                   [     ,  0.137,      , 0.052,      , 0.,          , 0.,     0.    ], #L2/3i
                   [     ,  0.006,      , 0.135,      , 0.0003,      , 0.,     0.0983], #L4e
                   [     ,  0.003,      , 0.160,      , 0.,          , 0.,     0.0619], #L4i
                   [     ,  0.062,      , 0.006,      , 0.373,       , 0.,     0.    ], #L5e
                   [     ,  0.027,      , 0.002,      , 0.316,       , 0.,     0.    ], #L5i
                   [     ,  0.007,      , 0.017,      , 0.020,       , 0.225,  0.0512], #L6e
                   [     ,  0.001,      , 0.001,      , 0.008,       , 0.144,  0.0196]]) #L6i
# TODO set
# Use defaults of runParamsParellel(). Protocol=0; Fig.6; g=4,
# bg_type=0 (layer specific), bg_freq=8.0, stim=0 (bg noise)
#  N.B. nsyn_type=0, but I am not using nsyn, just probabilities

# Initializations
# TODO exc weight, exc/inh delay, membrane potential from gaussian?
#   vm='-58.0*mV + 10.0*mV*randn()'
#   thal_con[r].w = 'clip((w_thal + std_w_thal*randn()),w_thal*0.0, w_thal*inf)'
#       std_w_thal = w_thal*0.1
#       w_thal = w_ex*pA
#   mem_tau   = 10.0*ms
#   refrac_tau = 2*ms
#   std_w_ex = 0.1*w_ex
# TODO equivalent of below
d_ex = 1.5*ms      	# Excitatory delay
std_d_ex = 0.75*ms 	# Std. Excitatory delay
d_in = 0.80*ms      # Inhibitory delay
std_d_in = 0.4*ms  	# Std. Inhibitory delay
tau_syn = 0.5*ms    # Post-synaptic current time constant
w_ex = 87.8*pA		   	# excitatory synaptic weight
std_w_ex = 0.1*w_ex     # standard deviation weigth
