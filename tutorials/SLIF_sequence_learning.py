"""
This code implements a sequence learning using and Excitatory-Inhibitory
network with STDP.
"""
import numpy as np
import matplotlib.pyplot as plt

from brian2 import ms, second, Hz, prefs, SpikeMonitor, StateMonitor,\
        SpikeGeneratorGroup, defaultclock, ExplicitStateUpdater,\
        PopulationRateMonitor, seed

from teili import TeiliNetwork
from teili.stimuli.testbench import SequenceTestbench
from teili.tools.misc import neuron_group_from_spikes

from SLIF_utils import neuron_rate, rate_correlations, ensemble_convergence,\
        permutation_from_rate, load_merge_multiple, recorded_bar_testbench

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
    if monitors['rate_exc_neurons']['monitor'].rate:
        exc_rate = np.array(monitors['rate_exc_neurons']['monitor'].smooth_rate(width=10*ms)/Hz)
    else:
        exc_rate = np.array(monitors['rate_exc_neurons']['monitor'].rate/Hz)
    inh_rate_t = np.array(monitors['rate_inh_neurons']['monitor'].t/ms)
    if monitors['rate_inh_neurons']['monitor'].rate:
        inh_rate = np.array(monitors['rate_inh_neurons']['monitor'].smooth_rate(width=10*ms)/Hz)
    else:
        inh_rate = np.array(monitors['rate_inh_neurons']['monitor'].rate/Hz)
    if len(exc_rate_t) != len(exc_rate):
        exc_rate = np.array(monitors['rate_exc_neurons']['monitor'].rate/Hz)
        inh_rate = np.array(monitors['rate_inh_neurons']['monitor'].rate/Hz)
    np.savez(path + f'traces_{block}.npz',
             Iin0=monitors['statemon_cells']['monitor'].Iin0,
             Iin1=monitors['statemon_cells']['monitor'].Iin1,
             Iin2=monitors['statemon_cells']['monitor'].Iin2,
             Iin3=monitors['statemon_cells']['monitor'].Iin3,
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
        rfw=monitors['reinit_conn_e']['monitor'].weight.astype(np.uint8),
        rfi=monitors['statemon_ffi']['monitor'].w_plast.astype(np.uint8),
        rfwi=monitors['reinit_conn_i']['monitor'].weight.astype(np.uint8),
        rec_mem=monitors['statemon_rec']['monitor'].w_plast.astype(np.uint8),
        rec_ids=recurrent_ids, rec_w=recurrent_weights
        )
    pickled_monitor = monitors['spikemon_exc_neurons']['monitor'].get_states()
    with open(path + f'pickled_{block}', 'wb') as f:
        pickle.dump(pickled_monitor, f)


def create_monitors():
    mon_dt = 500*ms
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
                'statemon_cells': {'group': 'pyr_cells',
                                    'variable': ['Iin0', 'Iin1', 'Iin2', 'Iin3'],
                                    'monitor': None},
                'statemon_ffe': {'group': 'ff_pyr',
                                 'variable': ['w_plast'],
                                 'monitor': None},
                'reinit_conn_e': {'group': 'ff_pyr',
                                 'variable': ['weight'],
                                 'monitor': None},
                'statemon_ffi': {'group': 'ff_pv',
                                 'variable': ['w_plast'],
                                 'monitor': None},
                'reinit_conn_i': {'group': 'ff_pv',
                                 'variable': ['weight'],
                                 'monitor': None},
                'statemon_rec': {'group': 'pyr_pyr',
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
                print(val['group'] + ' added after exception')
                monitors[key]['monitor'] = SpikeMonitor(eval(val['group']),
                                                        name=key)
        elif 'state' in key:
            if 'cells' in key:
                monitors[key]['monitor'] = StateMonitor(orca._groups[val['group']],
                                                        variables=val['variable'],
                                                        record=selected_cells,
                                                        name=key)
            else:  # Connections
                monitors[key]['monitor'] = StateMonitor(orca._groups[val['group']],
                                                        variables=val['variable'],
                                                        record=True,
                                                        dt=mon_dt,
                                                        name=key)
        elif 'rate' in key:
            monitors[key]['monitor'] = PopulationRateMonitor(orca._groups[val['group']],
                                                             name=key)
        elif 'reinit' in key:
            if re_init_dt is None:
                temp_dt = sim_duration - 1*ms  # Gets only one sample from end of simulation
            else:
                temp_dt = re_init_dt + 1*ms  # Tries to avoid missing update by +1
            monitors[key]['monitor'] = StateMonitor(orca._groups[val['group']],
                                                    variables=val['variable'],
                                                    record=True,
                                                    dt=temp_dt,
                                                    name=key)

    return monitors


# Initialize simulation preferences
seed(13)
rng = np.random.default_rng(12345)
prefs.codegen.target = "numpy"

# Initialize input sequence
num_items = 3
item_duration = 60
item_superposition = 0
num_channels = 144
noise_prob = None#0.005
item_rate = 100
#repetitions = 700# 350
repetitions = 200

sequence = SequenceTestbench(num_channels, num_items, item_duration,
                             item_superposition, noise_prob, item_rate,
                             repetitions)
input_indices, input_times = sequence.stimuli()
training_duration = np.max(input_times)
sequence_duration = sequence.cycle_length * ms
testing_duration = 0*ms

# Adding incomplete sequence at the end of simulation
#incomplete_sequences = 3
#include_symbols = [[2], [1], [0]]
#test_duration = incomplete_sequences * sequence_duration
#symbols = sequence.items
#for incomp_seq in range(incomplete_sequences):
#    for incl_symb in include_symbols[incomp_seq]:
#        tmp_symb = [(x*ms + incomp_seq*sequence_duration + training_duration)
#                        for x in symbols[incl_symb]['t']]
#        input_times = np.append(input_times, tmp_symb)
#        input_indices = np.append(input_indices, symbols[incl_symb]['i'])
## Get back unit that was remove by append operation
#input_times = input_times*second

# Adding noise at the end of simulation
#incomplete_sequences = 5
#test_duration = incomplete_sequences*sequence_duration*ms
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

# Or alternatively send it to an input layer
# TODO organize it somewhere else
#from teili.models.neuron_models import QuantStochLIF as static_neuron_model
#from teili.core.groups import Neurons, Connections
#from teili.models.synapse_models import QuantStochSyn as static_synapse_model
#stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')
#input_cells = SpikeGeneratorGroup(num_channels, input_indices, input_times)
#relay_cells = Neurons(num_channels,
#                      equation_builder=static_neuron_model(num_inputs=1),
#                      method=stochastic_decay,
#                      name='relay_cells',
#                      verbose=True)
#s_inp_exc = Connections(input_cells, relay_cells,
#                        equation_builder=static_synapse_model(),
#                        method=stochastic_decay,
#                        name='s_inp_exc')
#s_inp_exc.connect('i==j')
#s_inp_exc.weight = 4096

num_exc = 49
Net = TeiliNetwork()
orca = ORCA_WTA(num_exc_neurons=num_exc,
    ratio_pv=1, ratio_sst=0.02, ratio_vip=0.02)
re_init_dt = None#60000*ms#
orca.add_input(relay_cells, 'ff', ['pyr_cells'], 'reinit', 'excitatory',
    sparsity=.3, re_init_dt=re_init_dt)
orca.add_input(relay_cells, 'ff', ['pv_cells'],#, 'sst_cells'],
    'static', 'inhibitory', sparsity=.3, re_init_dt=re_init_dt)

# Prepare for saving data
date_time = datetime.now()
path = f"""{date_time.strftime('%Y.%m.%d')}_{date_time.hour}.{date_time.minute}/"""
os.mkdir(path)
selected_cells = rng.choice(range(orca._groups['pv_cells'].N),
                            int(orca._groups['pv_cells'].N*.4), replace=False)
Metadata = {'time_step': defaultclock.dt,
            'num_exc': num_exc,
            'num_pv': orca._groups['pv_cells'].N,
            'num_channels': num_channels,
            'full_rotation': sequence_duration,
            'repetitions': repetitions,
            'selected_cells': selected_cells,
            're_init_dt': re_init_dt
        }

with open(path+'metadata', 'wb') as f:
    pickle.dump(Metadata, f)

##################
# Setting up monitors
monitors = create_monitors()

# Temporary monitors
statemon_net_current = StateMonitor(orca._groups['pyr_cells'],
    variables=['Iin', 'Iin0', 'Iin1', 'Iin2', 'Iin3', 'I', 'Vm'], record=True,
    name='statemon_net_current')
statemon_net_current2 = StateMonitor(orca._groups['pv_cells'],
    variables=['normalized_activity_proxy', 'Iin'], record=True,
    name='statemon_net_current2')
statemon_net_current3 = StateMonitor(orca._groups['sst_cells'],
    variables=['Iin'], record=True,
    name='statemon_net_current3')
statemon_rate4 = SpikeMonitor(relay_cells, name='input_spk')

# Training
# TODO remove things here that should go somewhere else
#statemon_rate2 = SpikeMonitor(input_cells, name='input_cells')
Net.add(orca, relay_cells, [x['monitor'] for x in monitors.values()])#, input_cells, s_inp_exc)
Net.add(statemon_net_current, statemon_net_current2, statemon_net_current3, statemon_rate4)

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

last_sequence_t = training_duration - int(sequence_duration/num_items/defaultclock.dt)*defaultclock.dt*3
neu_rates = neuron_rate(spikemon_exc_neurons, kernel_len=10*ms,
    kernel_var=1*ms, simulation_dt=defaultclock.dt,
    interval=[last_sequence_t, training_duration], smooth=True,#)
    trials=3)
#foo = ensemble_convergence(seq_rates, neu_rates, [[0, 48], [48, 96], [96, 144]],
#                           sequence_duration, repetitions)
#
#corrs = rate_correlations(neu_rates, sequence_duration, repetitions)

############
# Saving results
# Calculating permutation indices from firing rates
permutation_ids = permutation_from_rate(neu_rates)

# Save data
np.savez(path+f'permutation.npz',
         ids = permutation_ids
        )
  
Net.store(filename=f'{path}network')

#inr = neuron_rate(statemon_rate4, 100*ms, 80*ms, defaultclock.dt, interval=[0*ms, 4000*ms], smooth=True)
#plt.plot(inr['t']/ms, np.average(inr['smoothed']/Hz,  axis=0), label='avg. smoothed rate')
#plt.xlabel('second')
#plt.ylabel('Hz')
#plt.legend()
#plt.show()

#plt.figure()
#plt.plot(statemon_rate4.t/second, statemon_rate4.rate/Hz)
#plt.xlabel('s')
#plt.ylabel('Hz')
#plt.title(f'ca. 9Hz, {num_channels} channels')
#plt.xlim([0, .2])
#plt.figure()
#plt.plot(statemon_rate4.t/second, statemon_rate4.smooth_rate(width=10*ms)/Hz)
#plt.xlabel('s')
#plt.ylabel('Hz')
#plt.title(f'ca. 9Hz, {num_channels} channels')
#plt.xlim([0, .2])
#plt.show()
#plt.imshow(np.reshape(orca._groups['ff_pyr'].w_plast[:,162], (np.sqrt(num_channels).astype(int), np.sqrt(num_channels).astype(int))))

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

def plot_norm_act(monitor):
    plt.figure()
    plt.plot(monitor.normalized_activity_proxy.T)
    plt.xlabel('time (ms)')
    plt.ylabel('normalized activity value')
    plt.title('Normalized activity of all neurons')
    plt.show()

def plot_w(conn, plast=True, hist=True, idx=0):
    if hist:
        if plast:
            _ = plt.hist(orca._groups[conn].w_plast)
        else:
            _ = plt.hist(orca._groups[conn].weight)
    else:
        if plast:
            plt.imshow(np.reshape(orca._groups[conn].w_plast[:, idx], (np.sqrt(num_channels).astype(int), np.sqrt(num_channels).astype(int))))
        else:
            plt.imshow(np.reshape(orca._groups[conn].weight[:, idx], (np.sqrt(num_channels).astype(int), np.sqrt(num_channels).astype(int))))
    plt.show()

def plot_EI_balance(idx=None, win_len=None, limits=None):
    if limits is None:
        limits = [max(statemon_net_current.t)/defaultclock.dt - 500, max(statemon_net_current.t)/defaultclock.dt]

    if idx is not None:
        if win_len:
            iin0 = np.convolve(statemon_net_current.Iin0[idx], np.ones(win_len)/win_len, mode='valid')
            iin1 = np.convolve(statemon_net_current.Iin1[idx], np.ones(win_len)/win_len, mode='valid')
            iin2 = np.convolve(statemon_net_current.Iin2[idx], np.ones(win_len)/win_len, mode='valid')
            iin3 = np.convolve(statemon_net_current.Iin3[idx], np.ones(win_len)/win_len, mode='valid')
            total_Iin = np.convolve(statemon_net_current.Iin[idx], np.ones(win_len)/win_len, mode='valid')
        else:
            iin0 = statemon_net_current.Iin0[idx]
            iin1 = statemon_net_current.Iin1[idx]
            iin2 = statemon_net_current.Iin2[idx]
            iin3 = statemon_net_current.Iin3[idx]
            total_Iin = statemon_net_current.Iin[idx]
    else:
        if win_len:
            iin0 = np.convolve(np.mean(statemon_net_current.Iin0, axis=0), np.ones(win_len)/win_len, mode='valid')
            iin1 = np.convolve(np.mean(statemon_net_current.Iin1, axis=0), np.ones(win_len)/win_len, mode='valid')
            iin2 = np.convolve(np.mean(statemon_net_current.Iin2, axis=0), np.ones(win_len)/win_len, mode='valid')
            iin3 = np.convolve(np.mean(statemon_net_current.Iin3, axis=0), np.ones(win_len)/win_len, mode='valid')
            total_Iin = np.convolve(np.mean(statemon_net_current.Iin, axis=0), np.ones(win_len)/win_len, mode='valid')
        else:
            iin0 = np.mean(statemon_net_current.Iin0, axis=0)
            iin1 = np.mean(statemon_net_current.Iin1, axis=0)
            iin2 = np.mean(statemon_net_current.Iin2, axis=0)
            iin3 = np.mean(statemon_net_current.Iin3, axis=0)
            total_Iin = np.mean(statemon_net_current.Iin, axis=0)
    plt.plot(iin0, 'r', label='pyr')
    plt.plot(iin1, 'g', label='pv')
    plt.plot(iin2, 'b', label='sst')
    plt.plot(iin3, 'k--', label='input')
    plt.plot(total_Iin, 'k', label='net current')
    plt.legend()
    if limits is not None:
        plt.xlim(limits)
    plt.ylabel('Current [amp]')
    plt.xlabel('time [ms]')
    plt.title('EI balance')
    plt.show()
