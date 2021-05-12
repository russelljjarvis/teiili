"""
This code implements a sequence learning using and Excitatory-Inhibitory
network with STDP.
"""
import numpy as np
from scipy.stats import gamma, truncnorm
from scipy.signal import find_peaks

from brian2 import second, mA, ms, mV, Hz, prefs, SpikeMonitor, StateMonitor,\
        defaultclock, ExplicitStateUpdater, SpikeGeneratorGroup,\
        PopulationRateMonitor, run, PoissonGroup, collect

from teili import TeiliNetwork
from teili.core.groups import Neurons, Connections
from teili.models.neuron_models import QuantStochLIF as static_neuron_model
from teili.models.synapse_models import QuantStochSyn as static_synapse_model
from teili.models.synapse_models import QuantStochSynStdp as stdp_synapse_model
from teili.stimuli.testbench import SequenceTestbench, OCTA_Testbench
from teili.tools.add_run_reg import add_lfsr
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder

from teili.tools.lfsr import create_lfsr
from teili.tools.misc import neuron_group_from_spikes
from SLIF_utils import neuron_rate, rate_correlations, ensemble_convergence,\
        permutation_from_rate, load_merge_multiple

from orca_wta import ORCA_WTA

import sys
import pickle
import os
from datetime import datetime

#############
# Utils functions
def save_data():
    np.savez(path + f'rasters_{block}.npz',
             input_t=np.array(monitors['spikemon_seq_neurons']['monitor'].t/ms),
             input_i=np.array(monitors['spikemon_seq_neurons']['monitor'].i),
             exc_spikes_t=np.array(monitors['spikemon_exc_neurons']['monitor'].t/ms),
             exc_spikes_i=np.array(monitors['spikemon_exc_neurons']['monitor'].i),
             inh_spikes_t=np.array(monitors['spikemon_inh_neurons']['monitor'].t/ms),
             inh_spikes_i=np.array(monitors['spikemon_inh_neurons']['monitor'].i),
            )

    # If there are only a few samples, smoothing operation can create an array
    # with is incompatible with array with spike times. This is then addressed
    # before saving to disk
    exc_rate_t = np.array(monitors['rate_exc_neurons']['monitor'].t/ms)
    exc_rate = np.array(monitors['rate_exc_neurons']['monitor'].smooth_rate(width=10*ms)/Hz)
    inh_rate_t = np.array(monitors['rate_inh_neurons']['monitor'].t/ms)
    inh_rate = np.array(monitors['rate_inh_neurons']['monitor'].smooth_rate(width=10*ms)/Hz)
    if len(exc_rate_t) != len(exc_rate):
        exc_rate = np.array(monitors['rate_exc_neurons']['monitor'].rate/Hz)
        inh_rate = np.array(monitors['rate_inh_neurons']['monitor'].rate/Hz)
    np.savez(path + f'traces_{block}.npz',
             Vm_e=monitors['statemon_exc_cells']['monitor'].Vm,
             Vm_i=monitors['statemon_inh_cells']['monitor'].Vm,
             exc_rate_t=exc_rate_t, exc_rate=exc_rate,
             inh_rate_t=inh_rate_t, inh_rate=inh_rate,
            )

    # Save targets of recurrent connections as python object
    recurrent_ids = []
    recurrent_weights = []
    for row in range(num_exc):
        recurrent_weights.append(list(orca._groups['pyr_pyr'].w_plast[row, :]))
        recurrent_ids.append(list(orca._groups['pyr_pyr'].j[row, :]))
    np.savez_compressed(path + f'matrices_{block}.npz',
             rf=monitors['statemon_ffe']['monitor'].w_plast.astype(np.uint8),
             rec_ids=recurrent_ids, rec_w=recurrent_weights
            )
    pickled_monitor = monitors['spikemon_exc_neurons']['monitor'].get_states()
    with open(path + f'pickled_{block}', 'wb') as f:
        pickle.dump(pickled_monitor, f)

def create_monitors():
    monitors = {'spikemon_exc_neurons': {'group': 'pyr_cells',
                                         'monitor': None},
                'spikemon_inh_neurons': {'group': 'pv_cells',
                                         'monitor': None},
                'spikemon_seq_neurons': {'group': 'seq_cells',
                                         'monitor': None},
                'statemon_exc_cells': {'group': 'pyr_cells',
                                       'variable': ['Vm'],
                                       'monitor': None},
                'statemon_inh_cells': {'group': 'pv_cells',
                                       'variable': ['Vm'],
                                       'monitor': None},
                'statemon_ffe': {'group': 'input_pyr',
                                 'variable': ['w_plast'],
                                 'monitor': None},
                'rate_exc_neurons': {'group': 'pyr_cells',
                                     'monitor': None},
                'rate_inh_neurons': {'group': 'pv_cells',
                                     'monitor': None}}

    for key, val in monitors.items():
        if 'spike' in key:
            monitors[key]['monitor'] = SpikeMonitor(orca._groups[val['group']],
                                                    name=key)
        elif 'state' in key:
            monitors[key]['monitor'] = StateMonitor(orca._groups[val['group']],
                                                    variables=val['variable'],
                                                    record=True,
                                                    name=key)
        elif 'rate' in key:
            monitors[key]['monitor'] = PopulationRateMonitor(orca._groups[val['group']],
                                                             name=key)

    return monitors

# Prepare parameters of the simulation
i_plast = sys.argv[1]

# Initialize simulation preferences
prefs.codegen.target = "numpy"

# Initialize rotating bar
testbench_stim = OCTA_Testbench()
num_channels = 100
num_items = None
noise_prob = None
item_rate = None
sequence_repetitions = 600
testbench_stim.rotating_bar(length=10, nrows=10,
                            direction='cw',
                            ts_offset=3, angle_step=10,
                            #noise_probability=0.2,
                            repetitions=sequence_repetitions,
                            debug=False)
training_duration = np.max(testbench_stim.times)*ms
test_duration = 1000*ms
input_indices = testbench_stim.indices
input_times = testbench_stim.times * ms
sequence_duration = 105*ms

num_exc = 200
Net = TeiliNetwork()
orca = ORCA_WTA(num_channels, input_indices, input_times, num_exc_neurons=num_exc,
    ratio_pv=1, ratio_sst=0.02, ratio_vip=0.02)

##################
# Setting up monitors
monitors = create_monitors()

# Temporary monitors
if i_plast == 'plastic_inh':
    statemon_proxy = StateMonitor(orca._groups['pyr_cells'],
        variables=['normalized_activity_proxy'], record=True,
        name='statemon_proxy')
statemon_net_current = StateMonitor(orca._groups['pyr_cells'],
    variables=['Iin', 'Iin0', 'Iin1', 'Iin2', 'Iin3'], record=True,
    name='statemon_net_current')

# Training
date_time = datetime.now()
path = f"""{date_time.strftime('%Y.%m.%d')}_{date_time.hour}.{date_time.minute}/"""
os.mkdir(path)

Net.add(orca, [x['monitor'] for x in monitors.values()])
if i_plast == 'plastic_inh':
    Net.add(statemon_proxy)
Net.add(statemon_net_current)

training_blocks = 10
remainder_time = int(np.around(training_duration/defaultclock.dt)
                     % training_blocks) * defaultclock.dt
for block in range(training_blocks):
    block_duration = int(np.around(training_duration/defaultclock.dt)
                         / training_blocks) * defaultclock.dt
    Net.run(block_duration, report='stdout', report_period=100*ms)
    # Free up memory
    save_data()

    # Reinitialization. Remove first so obj reference is not lost
    Net.remove([x['monitor'] for x in monitors.values()])
    del monitors
    monitors = create_monitors()
    Net.add([x['monitor'] for x in monitors.values()])

if remainder_time != 0:
    block += 1
    Net.run(remainder_time, report='stdout', report_period=100*ms)
    save_data()
    Net.remove([x['monitor'] for x in monitors.values()])
    del monitors
    monitors = create_monitors()
    Net.add([x['monitor'] for x in monitors.values()])

# Testing
Net.run(test_duration, report='stdout', report_period=100*ms)
block += 1
save_data()

# Recover data pickled from monitor
spikemon_exc_neurons = load_merge_multiple(path, 'pickled*')

last_sequence_t = (training_duration-sequence_duration)/ms
neu_rates = neuron_rate(spikemon_exc_neurons, kernel_len=200,
    kernel_var=10, kernel_min=0.001,
    interval=[int(last_sequence_t), int(training_duration/ms)])
#foo = ensemble_convergence(seq_rates, neu_rates, [[0, 48], [48, 96], [96, 144]],
#                           sequence_duration, sequence_repetitions)
#
#corrs = rate_correlations(neu_rates, sequence_duration, sequence_repetitions)

############
# Saving results
# Calculating permutation indices from firing rates
permutation_ids = permutation_from_rate(neu_rates, sequence_duration,
                                        defaultclock.dt)

# Save data
np.savez(path+f'permutation.npz',
         ids = permutation_ids
        )
  
Metadata = {'time_step': defaultclock.dt,
            'num_symbols': num_items,
            'num_exc': num_exc,
            'num_channels': num_channels,
            'sequence_duration': sequence_duration,
            'input_noise': noise_prob,
            'input_rate': item_rate,
            'sequence_repetitions': sequence_repetitions,
        }
with open(path+'general.data', 'wb') as f:
    pickle.dump(Metadata, f)

Metadata = {'exc': orca._groups['pyr_cells'].get_params(),
            'inh': orca._groups['pv_cells'].get_params()}
with open(path+'population.data', 'wb') as f:
    pickle.dump(Metadata, f)

Metadata = {'e->i': orca._groups['pyr_pv'].get_params(),
            'i->e': orca._groups['pv_pyr'].get_params(),
            'ffe': orca._groups['input_pyr'].get_params(),
            'ffi': orca._groups['input_pv'].get_params(),
            'e->e': orca._groups['pyr_pyr'].get_params()
            }
with open(path+'connections.data', 'wb') as f:
    pickle.dump(Metadata, f)

#from brian2 import *
#import pandas as pd
#from scipy.signal import savgol_filter
#_ = hist(corrs, bins=20)
#xlabel('Correlation values')
#ylabel('count')
#title('Correlations of average response to every sequence presentation (all neurons)')
#
#figure()
#neu=1
#y1 = pd.Series(foo[0,neu,:])
#y1=savgol_filter(y1.interpolate(), 31, 4)
#y2 = pd.Series(foo[1,neu,:])
#y2=savgol_filter(y2.interpolate(), 31, 4)
#y3 = pd.Series(foo[2,neu,:])
#y3=savgol_filter(y3.interpolate(), 31, 4)
#
#plot(y1, label='symbol 1')
#plot(y2, label='symbol 2')
#plot(y3, label='symbol 3')
#xlabel('# sequence presentation')
#ylabel('correlation value')
#legend()

def plot_norm_act():
    import matplotlib.pyplot as plt 
    plt.figure()
    plt.plot(statemon_proxy.normalized_activity_proxy.T)
    plt.xlabel('time (ms)')
    plt.ylabel('normalized activity value')
    plt.title('Normalized activity of all neurons')
    plt.show()

def plot_inh_w():
    import matplotlib.pyplot as plt 
    _ = plt.hist(orca._groups['pv_pyr'].w_plast)
    plt.show()

def plot_EI_balance(idx):
    import matplotlib.pyplot as plt
    win_len = 100
    rec_Iin = np.convolve(statemon_net_current.Iin0[idx], np.ones(win_len)/win_len, mode='valid')
    inh_Iin = np.convolve(statemon_net_current.Iin1[idx], np.ones(win_len)/win_len, mode='valid')
    ffe_Iin = np.convolve(statemon_net_current.Iin2[idx], np.ones(win_len)/win_len, mode='valid')
    noise_Iin = np.convolve(statemon_net_current.Iin3[idx], np.ones(win_len)/win_len, mode='valid')
    total_Iin = np.convolve(statemon_net_current.Iin[idx], np.ones(win_len)/win_len, mode='valid')
    plt.plot(rec_Iin, 'r', label='rec. current')
    plt.plot(inh_Iin, 'g', label='inh. current')
    plt.plot(ffe_Iin, 'b', label='input current')
    plt.plot(total_Iin, 'k', label='net current')
    plt.plot(noise_Iin, 'k--', label='spont. activity')
    plt.legend()
    plt.ylabel('Current [amp]')
    plt.xlabel('time [ms]')
    plt.title('EI balance')
    plt.show()
