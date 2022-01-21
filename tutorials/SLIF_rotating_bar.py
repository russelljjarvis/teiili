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
from teili.stimuli.testbench import OCTA_Testbench
from teili.tools.misc import neuron_group_from_spikes

from SLIF_utils import neuron_rate, rate_correlations, ensemble_convergence,\
        permutation_from_rate, load_merge_multiple, recorded_bar_testbench

from orca_wta import ORCA_WTA
from monitor_params import monitor_params, selected_cells
from orca_params import ConnectionDescriptor

import sys
import pickle
import os
from datetime import datetime


# Initialize simulation preferences
seed(13)
rng = np.random.default_rng(12345)
prefs.codegen.target = "numpy"
defaultclock.dt = 1*ms

# Initialize rotating bar
sequence_duration = 357*ms#105*ms#
#sequence_duration = 950*ms#1900*ms#
testing_duration = 0*ms
repetitions = 400
#repetitions = 13#8#10#
num_samples = 100
num_channels = num_samples
# Simulated bar
testbench_stim = OCTA_Testbench()
testbench_stim.rotating_bar(length=10, nrows=10,
                            direction='cw',
                            ts_offset=3, angle_step=3,#10,#
                            #noise_probability=0.2,
                            repetitions=repetitions,
                            debug=False)
input_indices = testbench_stim.indices
input_times = testbench_stim.times * ms
# Recorded bar
#input_times, input_indices = recorded_bar_testbench('../raw_7p5V_normallight_fullbar.aedat4_events.npz', num_samples, repetitions)

training_duration = np.max(input_times) - testing_duration
sim_duration = input_times[-1]
# Convert input into neuron group (necessary for STDP compatibility)
ff_cells = neuron_group_from_spikes(num_channels,
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
#ff_cells = Neurons(num_channels,
#                      equation_builder=static_neuron_model(num_inputs=1),
#                      method=stochastic_decay,
#                      name='ff_cells',
#                      verbose=True)
#s_inp_exc = Connections(input_cells, ff_cells,
#                        equation_builder=static_synapse_model(),
#                        method=stochastic_decay,
#                        name='s_inp_exc')
#s_inp_exc.connect('i==j')
#s_inp_exc.weight = 4096

# TODO e_ratio=.00224 to get to 49
# TODO values below were used before. New parameters need testing
# 'pyr_pyr': 0.5, # this has been standard. Commented is H&S
# 'pyr_pv': 0.15, # 0.45
# 'pyr_sst': 0.15, # 0.35
# 'pyr_vip': 0.10, # 0.10
# 'pv_pyr': 1.0, # 0.60
# 'pv_pv': 1.0, # 0.50
# 'sst_pv': 0.9, # 0.60
# 'sst_pyr': 1.0, # 0.55
# 'sst_vip': 0.9, # 0.45
# 'vip_sst': 0.65}, # 0.50
num_exc = 49
Net = TeiliNetwork()
layer='L4'
path = '/Users/Pablo/git/teili/'
conn_desc = ConnectionDescriptor(layer, path)
# TODO not working with altadp
conn_desc.intra_plast['sst_pv'] = 'static'
conn_desc.update_params()
orca = ORCA_WTA(layer=layer,
                conn_params=conn_desc,
                monitor=True)
re_init_dt = None#60000*ms#
orca.add_input(ff_cells, 'ff', ['pyr_cells'])
orca.add_input(ff_cells, 'ff', ['pv_cells'])

# Prepare for saving data
date_time = datetime.now()
path = f"""{date_time.strftime('%Y.%m.%d')}_{date_time.hour}.{date_time.minute}/"""
os.mkdir(path)
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
orca.create_monitors(monitor_params)

# Temporary monitors
statemon_net_current = StateMonitor(orca._groups['pyr_cells'],
    variables=['Iin', 'Iin0', 'Iin1', 'Iin2', 'Iin3', 'I', 'Vm'], record=True,
    name='statemon_net_current')
statemon_net_current2 = StateMonitor(orca._groups['pv_cells'],
    variables=['Iin'], record=True,
    name='statemon_net_current2')
statemon_net_current3 = StateMonitor(orca._groups['sst_cells'],
    variables=['Iin'], record=True,
    name='statemon_net_current3')
spikemon_input = SpikeMonitor(ff_cells, name='input_spk')

# Training
#statemon_rate2 = SpikeMonitor(input_cells, name='input_cells')
Net.add(orca, ff_cells)#, s_inp_exc)
Net.add(statemon_net_current, statemon_net_current2, statemon_net_current3, spikemon_input)

training_blocks = 10
remainder_time = int(np.around(training_duration/defaultclock.dt)
                     % training_blocks) * defaultclock.dt
for block in range(training_blocks):
    block_duration = int(np.around(training_duration/defaultclock.dt)
                         / training_blocks) * defaultclock.dt
    Net.run(block_duration, report='stdout', report_period=100*ms)
    # Free up memory
    orca.save_data(monitor_params, path, block)

    # Reinitialization. Remove first so obj reference is not lost
    Net.remove([x for x in orca.monitors.values()])
    orca.monitors = {}
    orca.create_monitors(monitor_params)
    Net.add([x for x in orca.monitors.values()])

if remainder_time != 0:
    block += 1
    Net.run(remainder_time, report='stdout', report_period=100*ms)
    orca.save_data(monitor_params, path, block)
    Net.remove([x for x in orca.monitors.values()])
    orca.monitors = {}
    orca.create_monitors(monitor_params)
    Net.add([x for x in orca.monitors.values()])

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
    orca.save_data(monitor_params, path, block)

# Recover data pickled from monitor
spikemon_exc_neurons = load_merge_multiple(path, 'pickled*')

last_sequence_t = training_duration - int(sequence_duration/2/defaultclock.dt)*defaultclock.dt*3
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
np.savez(path+f'input_raster.npz',
         input_t=np.array(spikemon_input.t/ms),
         input_i=np.array(spikemon_input.i)
        )

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
