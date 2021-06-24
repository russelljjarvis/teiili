"""
This code implements a sequence learning using and Excitatory-Inhibitory
network with STDP.
"""
import numpy as np

from brian2 import ms, Hz, prefs, SpikeMonitor, StateMonitor,\
        defaultclock, ExplicitStateUpdater,\
        PopulationRateMonitor

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


# Prepare parameters of the simulation
i_plast = sys.argv[1]

# Initialize simulation preferences
prefs.codegen.target = "numpy"

# Initialize rotating bar
testbench_stim = OCTA_Testbench()
num_channels = 100
sequence_repetitions = 100
testbench_stim.rotating_bar(length=10, nrows=10,
                            direction='cw',
                            ts_offset=3, angle_step=10,
                            #noise_probability=0.2,
                            repetitions=sequence_repetitions,
                            debug=False)
input_indices = testbench_stim.indices
input_times = testbench_stim.times * ms
sequence_duration = 105*ms
testing_duration = 300*ms
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
orca = ORCA_WTA(num_exc_neurons=num_exc, noise=True)#,
    #ratio_pv=1, ratio_sst=0.02, ratio_vip=0.02)
orca.add_input(relay_cells, 'ff', ['pyr_cells'], 'reinit', 'excitatory',
    sparsity=.3)
orca.add_input(relay_cells, 'ff', ['pv_cells', 'sst_cells'],
    'static', 'inhibitory')

orca2 = ORCA_WTA(num_exc_neurons=num_exc, name='top_down_', noise=True)#,
    #ratio_pv=1, ratio_sst=0.02, ratio_vip=0.02)
orca2.add_input(relay_cells, 'ff', ['pyr_cells'], 'reinit', 'excitatory',
    sparsity=.3)
orca2.add_input(relay_cells, 'ff', ['pv_cells', 'sst_cells'],
    'static', 'inhibitory')
orca.add_input(orca2._groups['pyr_cells'], 'fb', ['pyr_cells'], 'reinit',
    'excitatory', exc_params=excitatory_synapse_dend, sparsity=.3)
orca.add_input(orca2._groups['pyr_cells'], 'fb', ['vip_cells'], 'static',
    'inhibitory', exc_params=excitatory_synapse_soma)

##################
# Setting up monitors
monitors = create_monitors()

# Temporary monitors
if i_plast == 'plastic_inh':
    statemon_proxy = StateMonitor(orca._groups['pyr_cells'],
        variables=['normalized_activity_proxy'], record=True,
        name='statemon_proxy')
statemon_net_current = StateMonitor(orca._groups['pv_cells'],
    variables=['Iin', 'Iin0', 'Iin1', 'Iin2', 'Iin3', 'I', 'Vm', 'normalized_activity_proxy'], record=True,
    name='statemon_net_current')
orca2_mon = SpikeMonitor(orca2._groups['pyr_cells'], name='orca2_mon')

# Training
date_time = datetime.now()
path = f"""{date_time.strftime('%Y.%m.%d')}_{date_time.hour}.{date_time.minute}/"""
os.mkdir(path)

Net.add(orca, relay_cells, [x['monitor'] for x in monitors.values()], orca2)
if i_plast == 'plastic_inh':
    Net.add(statemon_proxy)
Net.add(statemon_net_current, orca2_mon)

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

# Use line below to try testing script
#Net.remove(statemon_net_current, relay_cells)

# Testing network
test_trial = 3
block += 1
orca._groups['ff_pyr'].stdp_thres = 0
orca._groups['fb_pyr'].stdp_thres = 0
orca._groups['pyr_pyr'].stdp_thres = 0
orca._groups['sst_pyr'].inh_learning_rate = 0
orca._groups['pv_pyr'].inh_learning_rate = 0
orca2._groups['ff_pyr'].stdp_thres = 0
orca2._groups['pyr_pyr'].stdp_thres = 0
orca2._groups['sst_pyr'].inh_learning_rate = 0
orca2._groups['pv_pyr'].inh_learning_rate = 0

# deactivate top-down only
orca._groups['fb_pyr'].weight = 0
orca._groups['fb_vip'].weight = 0
Net.run(testing_duration/test_trial, report='stdout', report_period=100*ms)
save_data()

# deactivate bottom-up only
orca._groups['fb_pyr'].weight = 1
orca._groups['ff_pyr'].weight = 0
orca._groups['ff_pv'].weight = 0
orca._groups['ff_sst'].weight = 0
Net.run(testing_duration/test_trial, report='stdout', report_period=100*ms)
save_data()

# No input
orca._groups['fb_pyr'].weight = 0
orca._groups['pv_pyr'].weight = 0
orca._groups['pv_pv'].weight = 0
orca._groups['sst_pyr'].weight = 0
orca._groups['sst_pv'].weight = 0
orca._groups['sst_vip'].weight = 0
orca._groups['vip_sst'].weight = 0
orca._groups['pyr_pyr'].weight = 0
orca._groups['pyr_pv'].weight = 0
orca._groups['pyr_sst'].weight = 0
orca._groups['pyr_vip'].weight = 0
Net.run(1000*ms, report='stdout', report_period=100*ms)
save_data()

# Recover data pickled from monitor
spikemon_exc_neurons = load_merge_multiple(path, 'pickled*')

last_sequence_t = training_duration - sequence_duration
neu_rates = neuron_rate(spikemon_exc_neurons, kernel_len=200,
    kernel_var=10, kernel_min=0.001, simulation_dt=defaultclock.dt,
    interval=[last_sequence_t, training_duration])
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
            'num_exc': num_exc,
            'num_channels': num_channels,
            'sequence_duration': sequence_duration,
            'sequence_repetitions': sequence_repetitions
        }

with open(path+'metadata', 'wb') as f:
    pickle.dump(Metadata, f)

Net.store(filename=f'{path}network')

############ Extra analysis of orca2
last_sequence_t = training_duration - sequence_duration
neu_rates = neuron_rate(orca2_mon, kernel_len=200,
    kernel_var=10, kernel_min=0.001, simulation_dt=defaultclock.dt,
    interval=[last_sequence_t, training_duration])
permutation_ids = permutation_from_rate(neu_rates, sequence_duration,
                                        defaultclock.dt)

#G = orca2._groups['pyr_pyr']
G = orca._groups['fb_pyr']
fb_ids = []
for row in range(num_exc):
    fb_ids.append(list(G.j[row, :]))
matrix = [None for _ in range(num_exc)]
interval1 = 0
interval2 = 0
for source_neu, values in enumerate(fb_ids):
    interval2 += len(values)
    matrix[source_neu] = G.w_plast[interval1:interval2]
    interval1 += len(values)
matrix = np.array(matrix, dtype=object)

#sorted_rf.matrix[:, permutation_ids][permutation_ids, :]

from SLIF_utils import plot_weight_matrix
from teili.tools.sorting import SortMatrix
sorted_rf = SortMatrix(ncols=num_exc, nrows=num_exc, matrix=matrix, axis=1,similarity_metric='euclidean', target_indices=fb_ids)#, rec_matrix=True)
#plot_weight_matrix(sorted_rf.sorted_matrix, title='asd', xlabel='x', ylabel='y')
# raster
#import matplotlib.pyplot as plt
#sorted_i = np.asarray([np.where(
#                np.asarray(permutation_ids) == int(i))[0][0] for i in orca2_mon.i])
#plt.plot(orca2_mon.t/ms, sorted_i, '.k')
#plt.show()

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
