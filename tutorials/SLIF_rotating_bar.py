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

from teili.core.groups import Neurons, Connections
from teili.models.neuron_models import QuantStochLIF as static_neuron_model
from teili.models.synapse_models import QuantStochSyn as static_synapse_model
from teili.models.synapse_models import QuantStochSynStdp as stdp_synapse_model
from teili.stimuli.testbench import SequenceTestbench, OCTA_Testbench
from teili.tools.add_run_reg import add_lfsr
from teili.tools.group_tools import add_group_activity_proxy,\
    add_group_params_re_init
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
             input_t=np.array(spikemon_seq_neurons.t/ms), input_i=np.array(spikemon_seq_neurons.i),
             exc_spikes_t=np.array(spikemon_exc_neurons.t/ms), exc_spikes_i=np.array(spikemon_exc_neurons.i),
             inh_spikes_t=np.array(spikemon_inh_neurons.t/ms), inh_spikes_i=np.array(spikemon_inh_neurons.i),
            )

    # If there are only a few samples, smoothing operation can create an array
    # with is incompatible with array with spike times. This is then addressed
    # before saving to disk
    exc_rate_t = np.array(statemon_pop_rate_e.t/ms)
    exc_rate = np.array(statemon_pop_rate_e.smooth_rate(width=10*ms)/Hz)
    inh_rate_t = np.array(statemon_pop_rate_i.t/ms)
    inh_rate = np.array(statemon_pop_rate_i.smooth_rate(width=10*ms)/Hz)
    if len(exc_rate_t) != len(exc_rate):
        exc_rate = np.array(statemon_pop_rate_e.rate/Hz)
        inh_rate = np.array(statemon_pop_rate_i.rate/Hz)
    np.savez(path + f'traces_{block}.npz',
             Vm_e=statemon_exc_cells.Vm, Vm_i=statemon_inh_cells.Vm,
             exc_rate_t=exc_rate_t, exc_rate=exc_rate,
             inh_rate_t=inh_rate_t, inh_rate=inh_rate,
            )

    # Save targets of recurrent connections as python object
    recurrent_ids = []
    recurrent_weights = []
    for row in range(num_exc):
        recurrent_weights.append(list(exc_exc_conn.w_plast[row, :]))
        recurrent_ids.append(list(exc_exc_conn.j[row, :]))
    np.savez_compressed(path + f'matrices_{block}.npz',
             rf=statemon_ffe_conns.w_plast.astype(np.uint8),
             rec_ids=recurrent_ids, rec_w=recurrent_weights
            )
    pickled_monitor = spikemon_exc_neurons.get_states()
    with open(path + f'pickled_{block}', 'wb') as f:
        pickle.dump(pickled_monitor, f)

def delete_monitors(monitors):
    for mon in monitors:
        del mon

def create_monitors():
    global spikemon_exc_neurons, spikemon_inh_neurons, spikemon_seq_neurons,\
        statemon_exc_cells, statemon_inh_cells, statemon_ffe_conns,\
        statemon_pop_rate_e, statemon_pop_rate_i

    spikemon_exc_neurons = SpikeMonitor(exc_cells, name='spikemon_exc_neurons')
    spikemon_inh_neurons = SpikeMonitor(inh_cells, name='spikemon_inh_neurons')
    spikemon_seq_neurons = SpikeMonitor(seq_cells, name='spikemon_seq_neurons')
    statemon_exc_cells = StateMonitor(exc_cells, variables=['Vm'], record=True,
                                      name='statemon_exc_cells')
    statemon_inh_cells = StateMonitor(inh_cells, variables=['Vm'], record=True,
                                      name='statemon_inh_cells')
    statemon_ffe_conns = StateMonitor(feedforward_exc, variables=['w_plast'],
                                      record=True, name='statemon_ffe_conns')
    statemon_pop_rate_e = PopulationRateMonitor(exc_cells,
                                                name='rate_exc_neurons')
    statemon_pop_rate_i = PopulationRateMonitor(inh_cells,
                                                name='rate_inh_neurons')

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

orca = ORCA_WTA(input_indices, input_times, ratio_pv=1, ratio_sst=0.02, ratio_vip=0.02)

# Weight initializations
ei_w = 3  # 1
mean_ie_w = 4  # 1
mean_ee_w = 1
mean_ffe_w = 3
mean_ffi_w = 1  # 2
mean_ii_w = 1

if i_plast == 'plastic_inh' or i_plast == 'plastic_inh0':
    inh_exc_conn.weight = -1
    # 1 = no inhibition, 0 = maximum inhibition
    #var_th = .1
    var_th = 0.50
for neu in range(num_inh):
    weight_length = np.shape(inh_exc_conn.weight[neu, :])
    sampled_weights = gamma.rvs(a=mean_ie_w, loc=1, size=weight_length).astype(int)
    sampled_weights = np.clip(sampled_weights, 0, 15)
    #sampled_weights = truncnorm.rvs(-3, 4, loc=mean_ie_w, size=weight_length).astype(int)
    if i_plast == 'plastic_inh' or i_plast == 'plastic_inh0':
        inh_exc_conn.w_plast[neu, :] = sampled_weights
    else:
        inh_exc_conn.weight[neu, :] = -sampled_weights
    weight_length = np.shape(inh_inh_conn.weight[neu, :])
    sampled_weights = gamma.rvs(a=mean_ii_w, loc=1, size=weight_length).astype(int)
    inh_inh_conn.weight[neu, :] = -np.clip(sampled_weights, 0, 15)

exc_exc_conn.weight = 1
for neu in range(num_exc):
    weight_length = np.shape(exc_exc_conn.w_plast[neu,:])
    exc_exc_conn.w_plast[neu,:] = gamma.rvs(a=mean_ee_w, size=weight_length).astype(int)
    #exc_exc_conn.w_plast[neu,:] = truncnorm.rvs(-1, 4, loc=mean_ee_w, size=weight_length).astype(int)
    weight_length = np.shape(exc_inh_conn.weight[neu,:])
    sampled_weights = gamma.rvs(a=ei_w, loc=1, size=weight_length).astype(int)
    sampled_weights = np.clip(sampled_weights, 0, 15)
    #sampled_weights = truncnorm.rvs(-2, 4, loc=ei_w, size=weight_length).astype(int)
    exc_inh_conn.weight[neu,:] = sampled_weights
feedforward_exc.weight = 1
num_inh_weight = np.shape(feedforward_inh.weight[0,:])[0]
for ch in range(num_channels):
    wplast_length = np.shape(feedforward_exc.w_plast[ch,:])
    feedforward_exc.w_plast[ch,:] = np.clip(
            gamma.rvs(a=mean_ffe_w, size=wplast_length).astype(int),
            0,
            15)
    #feedforward_exc.w_plast[ch,:] = truncnorm.rvs(-3, 7, loc=mean_ffe_w, size=wplast_length).astype(int)
    feedforward_inh.weight[ch,:] = np.clip(
            gamma.rvs(a=mean_ffi_w, size=num_inh_weight).astype(int),
            0,
            15)
    #feedforward_inh.weight[ch,:] = truncnorm.rvs(-1, 7, loc=mean_ffi_w, size=num_inh_weight).astype(int)
background_activity.weight = 50

# Set LFSRs for each group
#neu_groups = [exc_cells, inh_cells]
# syn_groups = [exc_exc_conn, exc_inh_conn, inh_exc_conn, feedforward_exc,
#                 feedforward_inh, inh_inh_conn]
#ta = create_lfsr(neu_groups, syn_groups, defaultclock.dt)

if i_plast == 'plastic_inh':
    # Add proxy activity group
    activity_proxy_group = [exc_cells]
    add_group_activity_proxy(activity_proxy_group,
                             buffer_size=400,
                             decay=150)
    inh_exc_conn.variance_th = np.random.uniform(
            low=var_th - 0.1,
            high=var_th + 0.1,
            size=len(inh_exc_conn))

# Adding mismatch
mismatch_neuron_param = {
    'tau': 0.1  # 0.2
}

mismatch_synap_param = {
    'tausyn': 0.1  # 0.2
}
mismatch_plastic_param = {
    'taupre': 0.1,  # 0.2
    'taupost': 0.1  # 0.2
}

exc_cells.add_mismatch(std_dict=mismatch_neuron_param)
exc_cells.__setattr__('tau', np.array(exc_cells.tau/ms).astype(int)*ms)
inh_cells.add_mismatch(std_dict=mismatch_neuron_param)
inh_cells.__setattr__('tau', np.array(inh_cells.tau/ms).astype(int)*ms)

exc_exc_conn.add_mismatch(std_dict=mismatch_synap_param)
exc_exc_conn.__setattr__('tausyn', np.array(exc_exc_conn.tausyn/ms).astype(int)*ms)
exc_exc_conn.add_mismatch(std_dict=mismatch_plastic_param)
exc_exc_conn.__setattr__('taupre', np.array(exc_exc_conn.taupre/ms).astype(int)*ms)
exc_exc_conn.__setattr__('taupost', np.array(exc_exc_conn.taupost/ms).astype(int)*ms)
exc_inh_conn.add_mismatch(std_dict=mismatch_synap_param)
exc_inh_conn.__setattr__('tausyn', np.array(exc_inh_conn.tausyn/ms).astype(int)*ms)
inh_exc_conn.add_mismatch(std_dict=mismatch_synap_param)
inh_exc_conn.__setattr__('tausyn', np.array(inh_exc_conn.tausyn/ms).astype(int)*ms)
inh_inh_conn.add_mismatch(std_dict=mismatch_synap_param)
inh_inh_conn.__setattr__('tausyn', np.array(inh_inh_conn.tausyn/ms).astype(int)*ms)
feedforward_exc.add_mismatch(std_dict=mismatch_synap_param)
feedforward_exc.__setattr__('tausyn', np.array(feedforward_exc.tausyn/ms).astype(int)*ms)
feedforward_exc.add_mismatch(std_dict=mismatch_plastic_param)
feedforward_exc.__setattr__('taupre', np.array(feedforward_exc.taupre/ms).astype(int)*ms)
feedforward_exc.__setattr__('taupost', np.array(feedforward_exc.taupost/ms).astype(int)*ms)
feedforward_inh.add_mismatch(std_dict=mismatch_synap_param)
feedforward_inh.__setattr__('tausyn', np.array(feedforward_inh.tausyn/ms).astype(int)*ms)

###################
# Adding homeostatic mechanisms
re_init_dt = 15000*ms
add_group_params_re_init(groups=[feedforward_exc],
                         variable='w_plast',
                         re_init_variable='re_init_counter',
                         re_init_threshold=1,
                         re_init_dt=re_init_dt,
                         dist_param=3,
                         scale=1,
                         distribution='gamma',
                         clip_min=0,
                         clip_max=15,
                         variable_type='int',
                         reference='synapse_counter')
add_group_params_re_init(groups=[feedforward_exc],
                         variable='weight',
                         re_init_variable='re_init_counter',
                         re_init_threshold=1,
                         re_init_dt=re_init_dt,
                         distribution='deterministic',
                         const_value=1,
                         reference='synapse_counter')
add_group_params_re_init(groups=[feedforward_exc],
                         variable='tausyn',
                         re_init_variable='re_init_counter',
                         re_init_threshold=1,
                         re_init_dt=re_init_dt,
                         dist_param=5.5,
                         scale=1,
                         distribution='normal',
                         clip_min=4,
                         clip_max=7,
                         variable_type='int',
                         unit='ms',
                         reference='synapse_counter')

##################
# Setting up monitors
create_monitors()
monitors = [statemon_ffe_conns, statemon_exc_cells, statemon_inh_cells,
            statemon_pop_rate_e, statemon_pop_rate_i, spikemon_seq_neurons,
            spikemon_exc_neurons, spikemon_inh_neurons]

# Temporary monitors
if i_plast == 'plastic_inh':
    statemon_proxy = StateMonitor(exc_cells, variables=['normalized_activity_proxy'], record=True,
                                      name='statemon_proxy')
statemon_net_current = StateMonitor(exc_cells, variables=['Iin', 'Iin0', 'Iin1', 'Iin2', 'Iin3'], record=True,
                                  name='statemon_net_current')

# Training
date_time = datetime.now()
path = f"""{date_time.strftime('%Y.%m.%d')}_{date_time.hour}.{date_time.minute}/"""
os.mkdir(path)

training_blocks = 10
remainder_time = int(np.around(training_duration/defaultclock.dt)
                     % training_blocks) * defaultclock.dt
for block in range(training_blocks):
    block_duration = int(np.around(training_duration/defaultclock.dt)
                         / training_blocks) * defaultclock.dt
    run(block_duration, report='stdout', report_period=100*ms)
    # Free up memory
    save_data()

    # Reinitialization
    delete_monitors(monitors)
    create_monitors()

if remainder_time != 0:
    block += 1
    run(remainder_time, report='stdout', report_period=100*ms)
    save_data()
    delete_monitors(monitors)
    create_monitors()

# Testing
run(test_duration, report='stdout', report_period=100*ms)
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
            'num_channels': num_channels,
            'sequence_duration': sequence_duration,
            'input_noise': noise_prob,
            'input_rate': item_rate,
            'sequence_repetitions': sequence_repetitions,
            'num_exc': num_exc,
            'num_inh': num_inh,
            'e->i p': ei_p,
            'i->e p': ie_p,
            'e->e p': ee_p,
            'mean e->i w': ei_w,
            'mean i->e w': mean_ie_w,
            'mean e->e w': mean_ee_w,
            'mean ffe w': mean_ffe_w,
            'mean ffi w': mean_ffi_w,
            'i_plast': i_plast
        }
with open(path+'general.data', 'wb') as f:
    pickle.dump(Metadata, f)

Metadata = {'exc': exc_cells.get_params(),
            'inh': inh_cells.get_params()}
with open(path+'population.data', 'wb') as f:
    pickle.dump(Metadata, f)

Metadata = {'e->i': exc_inh_conn.get_params(),
            'i->e': inh_exc_conn.get_params(),
            'ffe': feedforward_exc.get_params(),
            'ffi': feedforward_inh.get_params(),
            'e->e': exc_exc_conn.get_params()
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
    _ = plt.hist(inh_exc_conn.w_plast)
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
