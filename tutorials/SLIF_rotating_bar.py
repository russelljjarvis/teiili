"""
This code implements a sequence learning using and Excitatory-Inhibitory
network with STDP.
"""
import numpy as np

from brian2 import ms, Hz, prefs, SpikeMonitor, StateMonitor,\
        defaultclock, ExplicitStateUpdater,\
        PopulationRateMonitor, seed

from teili import TeiliNetwork
from teili.stimuli.testbench import OCTA_Testbench
from teili.tools.misc import neuron_group_from_spikes

from SLIF_utils import neuron_rate, rate_correlations, ensemble_convergence,\
        permutation_from_rate, load_merge_multiple

from orca_wta import ORCA_WTA
from orca_params import excitatory_synapse_dend, excitatory_synapse_soma

import sys
import pickle
import os
from datetime import datetime


#############
# Utils functions
def save_data():
    # Concatenate data from inhibitory population
    pv_times = np.array(monitors['spikemon_pv_neurons']['monitor'].t/ms)
    pv_indices = np.array(monitors['spikemon_pv_neurons']['monitor'].i)
    sst_times = np.array(monitors['spikemon_sst_neurons']['monitor'].t/ms)
    sst_indices = np.array(monitors['spikemon_sst_neurons']['monitor'].i)
    sst_indices += orca._groups['pv_cells'].N
    vip_times = np.array(monitors['spikemon_vip_neurons']['monitor'].t/ms)
    vip_indices = np.array(monitors['spikemon_vip_neurons']['monitor'].i)
    vip_indices += (orca._groups['pv_cells'].N + orca._groups['sst_cells'].N)

    inh_spikes_t = np.concatenate((pv_times, sst_times, vip_times))
    inh_spikes_i = np.concatenate((pv_indices, sst_indices, vip_indices))
    sorting_index = np.argsort(inh_spikes_t)
    inh_spikes_t = inh_spikes_t[sorting_index]
    inh_spikes_i = inh_spikes_i[sorting_index]

    np.savez(path + f'rasters_{block}.npz',
             input_t=np.array(monitors['spikemon_seq_neurons']['monitor'].t/ms),
             input_i=np.array(monitors['spikemon_seq_neurons']['monitor'].i),
             exc_spikes_t=np.array(monitors['spikemon_exc_neurons']['monitor'].t/ms),
             exc_spikes_i=np.array(monitors['spikemon_exc_neurons']['monitor'].i),
             inh_spikes_t=inh_spikes_t,
             inh_spikes_i=inh_spikes_i,
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
                'spikemon_pv_neurons': {'group': 'pv_cells',
                                        'monitor': None},
                'spikemon_sst_neurons': {'group': 'sst_cells',
                                         'monitor': None},
                'spikemon_vip_neurons': {'group': 'vip_cells',
                                         'monitor': None},
                'spikemon_seq_neurons': {'group': 'relay_cells',
                                         'monitor': None},
                'statemon_exc_cells': {'group': 'pyr_cells',
                                       'variable': ['Vm'],
                                       'monitor': None},
                'statemon_inh_cells': {'group': 'pv_cells',
                                       'variable': ['Vm'],
                                       'monitor': None},
                'statemon_ffe': {'group': 'ff_pyr',
                                 'variable': ['w_plast'],
                                 'monitor': None},
                'rate_exc_neurons': {'group': 'pyr_cells',
                                     'monitor': None},
                'rate_inh_neurons': {'group': 'pv_cells',
                                     'monitor': None}}

    for key, val in monitors.items():
        if 'spike' in key:
            try:
                monitors[key]['monitor'] = SpikeMonitor(orca._groups[val['group']],
                                                        name=key)
            except KeyError:
                print(val['group'] + 'added after exception')
                monitors[key]['monitor'] = SpikeMonitor(eval(val['group']),
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


# Initialize simulation preferences
seed(13)
prefs.codegen.target = "numpy"

# Initialize rotating bar
testbench_stim = OCTA_Testbench()
num_channels = 100
sequence_repetitions = 600#200
testbench_stim.rotating_bar(length=10, nrows=10,
                            direction='cw',
                            ts_offset=3, angle_step=10,#3
                            #noise_probability=0.2,
                            repetitions=sequence_repetitions,
                            debug=False)
input_indices = testbench_stim.indices
input_times = testbench_stim.times * ms
sequence_duration = 105*ms#357*ms
testing_duration = 0*ms
training_duration = np.max(testbench_stim.times)*ms - testing_duration

# Convert input into neuron group (necessary for STDP compatibility)
sim_duration = input_times[-1]
relay_cells = neuron_group_from_spikes(num_channels,
                                       defaultclock.dt,
                                       sim_duration,
                                       spike_indices=input_indices,
                                       spike_times=input_times)

num_exc = 200
Net = TeiliNetwork()
orca = ORCA_WTA(num_exc_neurons=num_exc)#,
    #ratio_pv=1, ratio_sst=0.02, ratio_vip=0.02)
orca.add_input(relay_cells, 'ff', ['pyr_cells'], 'reinit', 'excitatory',
    sparsity=.3)
orca.add_input(relay_cells, 'ff', ['pv_cells', 'sst_cells'],
    'static', 'inhibitory')

##################
# Setting up monitors
monitors = create_monitors()

# Temporary monitors
statemon_net_current = StateMonitor(orca._groups['pyr_cells'],
    variables=['Iin', 'Iin0', 'Iin1', 'Iin2', 'Iin3', 'I', 'Vm'], record=True,
    name='statemon_net_current')

# Training
date_time = datetime.now()
path = f"""{date_time.strftime('%Y.%m.%d')}_{date_time.hour}.{date_time.minute}/"""
os.mkdir(path)

Net.add(orca, relay_cells, [x['monitor'] for x in monitors.values()])
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

# Testing network
if testing_duration:
    block += 1
    orca._groups['ff_pyr'].stdp_thres = 0
    orca._groups['pyr_pyr'].stdp_thres = 0
    orca._groups['sst_pyr'].inh_learning_rate = 0
    orca._groups['pv_pyr'].inh_learning_rate = 0

    # deactivate bottom-up
    orca._groups['ff_pyr'].weight = 0
    orca._groups['ff_pv'].weight = 0
    orca._groups['ff_sst'].weight = 0
    Net.run(testing_duration, report='stdout', report_period=100*ms)
    save_data()

# Recover data pickled from monitor
spikemon_exc_neurons = load_merge_multiple(path, 'pickled*')

last_sequence_t = training_duration - sequence_duration
neu_rates = neuron_rate(spikemon_exc_neurons, kernel_len=10*ms,
    kernel_var=0.5, simulation_dt=defaultclock.dt,
    interval=[last_sequence_t, training_duration], smooth=True)
    #interval=[0*ms, training_duration], smooth=True,
    #trials=sequence_repetitions)
#foo = ensemble_convergence(seq_rates, neu_rates, [[0, 48], [48, 96], [96, 144]],
#                           sequence_duration, sequence_repetitions)
#
#corrs = rate_correlations(neu_rates, sequence_duration, sequence_repetitions)

############
# Saving results
# Calculating permutation indices from firing rates
permutation_ids = permutation_from_rate(neu_rates)

# Save data
np.savez(path+f'permutation.npz',
         ids = permutation_ids
        )
  
Metadata = {'time_step': defaultclock.dt,
            'num_exc': num_exc,
            'num_channels': num_channels,
            'sequence_duration': sequence_duration,
            'sequence_repetitions': sequence_repetitions
        }

with open(path+'metadata', 'wb') as f:
    pickle.dump(Metadata, f)

Net.store(filename=f'{path}network')

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
    plt.plot(statemon_net_current.normalized_activity_proxy.T)
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
    iin0 = np.convolve(statemon_net_current.Iin0[idx], np.ones(win_len)/win_len, mode='valid')
    iin1 = np.convolve(statemon_net_current.Iin1[idx], np.ones(win_len)/win_len, mode='valid')
    iin2 = np.convolve(statemon_net_current.Iin2[idx], np.ones(win_len)/win_len, mode='valid')
    iin3 = np.convolve(statemon_net_current.Iin3[idx], np.ones(win_len)/win_len, mode='valid')
    total_Iin = np.convolve(statemon_net_current.Iin[idx], np.ones(win_len)/win_len, mode='valid')
    plt.plot(iin0, 'r', label='Iin0')
    plt.plot(iin1, 'g', label='Iin1')
    plt.plot(iin2, 'b', label='Iin2')
    plt.plot(iin3, 'k--', label='Iin3')
    plt.plot(total_Iin, 'k', label='net current')
    plt.legend()
    plt.ylabel('Current [amp]')
    plt.xlabel('time [ms]')
    plt.title('EI balance')
    plt.show()
