import numpy as np
import matplotlib.pyplot as plt

from brian2 import ms, mA, second, Hz, prefs, SpikeMonitor, StateMonitor,\
        SpikeGeneratorGroup, defaultclock, ExplicitStateUpdater,\
        PopulationRateMonitor, seed

from teili import TeiliNetwork
from teili.stimuli.testbench import SequenceTestbench
from teili.tools.misc import neuron_group_from_spikes

from SLIF_utils import neuron_rate, rate_correlations, ensemble_convergence,\
        permutation_from_rate, load_merge_multiple, recorded_bar_testbench,\
        run_batches

from orca_column import orcaColumn
from monitor_params import monitor_params, selected_cells

import sys
import pickle
import os
from datetime import datetime


def change_params_conn1(desc):
    # Changes some intralaminar connections
    desc.plasticities['sst_pv'] = 'static'
    desc.plasticities['pyr_pyr'] = 'stdp'
    desc.filter_params()
    desc.base_vals['pyr_pyr']['w_max'] = 15

def change_params_conn2(desc):
    # Changes input parameters
    desc.plasticities['ff_pyr'] = 'stdp'
    desc.filter_params()
    desc.base_vals['ff_pyr']['w_max'] = 15

def change_params_conn3(desc):
    # Changes input parameters
    desc.plasticities['ff_pyr'] = 'stdp'
    desc.probabilities['ff_pyr'] = 0.7
    desc.probabilities['ff_pv'] = 1
    desc.probabilities['ff_sst'] = 1
    desc.filter_params()
    desc.base_vals['ff_pyr']['w_max'] = 15

def change_params_pop1(desc):
    # Changes proportions of the EI population
    desc.e_ratio = .00224
    desc.group_prop['ei_ratio'] = 3
    desc.group_prop['inh_ratio']['pv_cells'] = .75
    desc.group_prop['inh_ratio']['sst_cells'] = .125
    desc.group_prop['inh_ratio']['vip_cells'] = .125
    desc.filter_params()
    for pop in desc.base_vals:
        desc.base_vals[pop]['I_min'] = -256*mA
        desc.base_vals[pop]['I_max'] = 256*mA

# Initialize simulation preferences
seed(13)
rng = np.random.default_rng(12345)
prefs.codegen.target = "numpy"
defaultclock.dt = 1*ms

# Initialize input sequence
num_items = 4
item_duration = 60
item_superposition = 0
num_channels = 144
noise_prob = None#0.005
item_rate = 100
#repetitions = 700# 350
repetitions = 200

sequence = SequenceTestbench(num_channels, num_items, item_duration,
                             item_superposition, noise_prob, item_rate,
                             repetitions, surprise_item=True)
input_indices, input_times = sequence.stimuli()
training_duration = np.max(input_times)
sequence_duration = sequence.cycle_length * ms
testing_duration = 0*ms

# Adding alternative sequence at the end of simulation
alternative_sequences = 10
include_symbols = [[0, 1, 2, 3] for _ in range(alternative_sequences)]
test_duration = alternative_sequences * sequence_duration
symbols = sequence.items
for alt_seq in range(alternative_sequences):
    for incl_symb in include_symbols[alt_seq]:
        tmp_symb = [(x*ms + alt_seq*sequence_duration + training_duration)
                        for x in symbols[incl_symb]['t']]
        input_times = np.append(input_times, tmp_symb)
        input_indices = np.append(input_indices, symbols[incl_symb]['i'])
# Get back unit that was remove by append operation
input_times = input_times*second

# Adding noise at the end of simulation
#alternative_sequences = 5
#test_duration = alternative_sequences*sequence_duration*ms
#noise_prob = 0.01
#noise_spikes = np.random.rand(num_channels, int(test_duration/ms))
#noise_indices = np.where(noise_spikes < noise_prob)[0]
#noise_times = np.where(noise_spikes < noise_prob)[1]
#input_indices = np.concatenate((input_indices, noise_indices))
#input_times = np.concatenate((input_times, noise_times+training_duration/ms))

training_duration = np.max(input_times) - testing_duration
sim_duration = input_times[-1]
# Convert input into neuron group (necessary for STDP compatibility)
relay_cells = neuron_group_from_spikes(num_channels,
                                       defaultclock.dt,
                                       sim_duration,
                                       spike_indices=input_indices,
                                       spike_times=input_times)

Net = TeiliNetwork()
column = orcaColumn(['L4', 'L5'])
conn_modifier = {'L4': change_params_conn1, 'L5': change_params_conn1}
pop_modifier = {'L4': change_params_pop1, 'L5': change_params_pop1}
column.create_layers(pop_modifier, conn_modifier)
column.connect_layers()
conn_modifier = {'L4': change_params_conn2, 'L5': change_params_conn3}
column.connect_inputs(relay_cells, 'ff', conn_modifier)

# Prepare for saving data
date_time = datetime.now()
path = f"""{date_time.strftime('%Y.%m.%d')}_{date_time.hour}.{date_time.minute}/"""
os.mkdir(path)
num_exc = column.col_groups['L4'].groups['pyr_cells'].N
Metadata = {'time_step': defaultclock.dt,
            'num_exc': num_exc,
            'num_pv': column.col_groups['L4'].groups['pv_cells'].N,
            'num_channels': num_channels,
            'full_rotation': sequence_duration,
            'repetitions': repetitions,
            'selected_cells': selected_cells,
            're_init_dt': None
        }

with open(path+'metadata', 'wb') as f:
    pickle.dump(Metadata, f)

##################
# Setting up monitors
monitor_params['statemon_static_conn_ff_pyr']['group'] = 'L4_ff_pyr'
monitor_params['statemon_conn_ff_pv']['group'] = 'L4_ff_pv'
monitor_params['statemon_static_conn_ff_pv']['group'] = 'L4_ff_pv'
monitor_params['statemon_conn_ff_pyr']['group'] = 'L4_ff_pyr'
column.col_groups['L4'].create_monitors(monitor_params)

# Temporary monitors
spikemon_input = SpikeMonitor(relay_cells, name='input_spk')
spkmon_l5 = SpikeMonitor(column.col_groups['L5']._groups['pyr_cells'],
                         name='l5_spk')

# Training
Net.add([x for x in column.col_groups.values()])
Net.add([x.input_groups for x in column.col_groups.values()])
Net.add(spikemon_input, spkmon_l5)

training_blocks = 10
run_batches(Net, column.col_groups['L4'], training_blocks, training_duration,
            defaultclock.dt, path, monitor_params)

# Recover data pickled from monitor
spikemon_exc_neurons = load_merge_multiple(path, 'pickled*')

last_sequence_t = training_duration - int(sequence_duration/num_items/defaultclock.dt)*defaultclock.dt*3
neu_rates = neuron_rate(spikemon_exc_neurons, kernel_len=10*ms,
    kernel_var=1*ms, simulation_dt=defaultclock.dt,
    interval=[last_sequence_t, training_duration], smooth=True,#)
    trials=3)

############
# Saving results
# Calculating permutation indices from firing rates
permutation_ids = permutation_from_rate(neu_rates)

# Save data
np.savez(path+f'input_raster.npz',
         input_t=np.array(spikemon_input.t/ms),
         input_i=np.array(spikemon_input.i)
        )

np.savez(path+f'permutation.npz',
         ids = permutation_ids
        )
  
Net.store(filename=f'{path}network')

############ Extra analysis of orca2
#import matplotlib.pyplot as plt
#last_sequence_t = training_duration - sequence_duration
#neu_rates = neuron_rate(orca2_mon, kernel_len=200,
#    kernel_var=10, kernel_min=0.001, simulation_dt=defaultclock.dt,
#    interval=[last_sequence_t, training_duration])
#permutation_ids = permutation_from_rate(neu_rates, sequence_duration,
#                                        defaultclock.dt)
#
#G = orca2._groups['pyr_pyr']
#G = orca._groups['fb_pyr']
#fb_ids = []
#for row in range(num_exc):
#    fb_ids.append(list(G.j[row, :]))
#matrix = [None for _ in range(num_exc)]
#interval1 = 0
#interval2 = 0
#for source_neu, values in enumerate(fb_ids):
#    interval2 += len(values)
#    matrix[source_neu] = G.w_plast[interval1:interval2]
#    interval1 += len(values)
#matrix = np.array(matrix, dtype=object)

#plt.plot(sorted_rf.matrix[:, permutation_ids][permutation_ids, :])

#from SLIF_utils import plot_weight_matrix
#from teili.tools.sorting import SortMatrix
#sorted_rf = SortMatrix(ncols=num_exc, nrows=num_exc, matrix=matrix, axis=1,similarity_metric='euclidean', target_indices=fb_ids)#, rec_matrix=True)
#plot_weight_matrix(sorted_rf.sorted_matrix, title='asd', xlabel='x', ylabel='y')
# raster
#import matplotlib.pyplot as plt
#sorted_i = np.asarray([np.where(
#                np.asarray(permutation_ids) == int(i))[0][0] for i in orca2_mon.i])
#plt.plot(orca2_mon.t/ms, sorted_i, '.k')
#plt.show()

