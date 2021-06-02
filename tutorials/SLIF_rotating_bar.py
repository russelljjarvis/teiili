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
sequence_repetitions = 600
testbench_stim.rotating_bar(length=10, nrows=10,
                            direction='cw',
                            ts_offset=3, angle_step=10,
                            #noise_probability=0.2,
                            repetitions=sequence_repetitions,
                            debug=False)
training_duration = np.max(testbench_stim.times)*ms
input_indices = testbench_stim.indices
input_times = testbench_stim.times * ms
sequence_duration = 105*ms

num_exc = 200
Net = TeiliNetwork()
orca = ORCA_WTA(num_channels, input_indices, input_times, num_exc_neurons=num_exc)#,
    #ratio_pv=1, ratio_sst=0.02, ratio_vip=0.02)

# TODO this on a separate file
from teili.core.groups import Connections
from teili.models.synapse_models import QuantStochSyn as static_synapse_model
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from orca_params import excitatory_synapse_soma, excitatory_synapse_dend
from teili.tools.group_tools import add_group_param_init
reinit_synapse_model = SynapseEquationBuilder(base_unit='quantized',
    plasticity='quantized_stochastic_stdp',
    structural_plasticity='stochastic_counter')
orca2 = ORCA_WTA(num_channels, input_indices, input_times, num_exc_neurons=num_exc, name='top_down_')#,
    #ratio_pv=1, ratio_sst=0.02, ratio_vip=0.02)
fb_pyr = Connections(orca2._groups['pyr_cells'], orca._groups['pyr_cells'],
    equation_builder=reinit_synapse_model(),
    method=ExplicitStateUpdater('''x_new = f(x,t)'''),
    name='feedback_pyr_conn')
fb_vip = Connections(orca2._groups['pyr_cells'], orca._groups['vip_cells'],
    equation_builder=static_synapse_model(),
    method=ExplicitStateUpdater('''x_new = f(x,t)'''),
    name='feedback_vip_conn')
fb_pyr.connect(p=.8)
fb_vip.connect(p=.8)
fb_pyr.set_params(excitatory_synapse_dend)
fb_pyr.tausyn = 3*ms
fb_vip.set_params(excitatory_synapse_soma)
fb_pyr.weight = 1
add_group_param_init([fb_pyr],
                     variable='w_plast',
                     dist_param=1,
                     scale=1,
                     distribution='gamma',
                     clip_min=0,
                     clip_max=15)
fb_pyr.__setattr__('w_plast', np.array(fb_pyr.w_plast).astype(int))
add_group_param_init([fb_vip],
                     variable='weight',
                     dist_param=1,
                     scale=1,
                     distribution='gamma',
                     clip_min=0,
                     clip_max=15)
fb_vip.__setattr__('weight', np.array(fb_vip.w_plast).astype(int))

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

Net.add(orca, orca2, fb_pyr, fb_vip, [x['monitor'] for x in monitors.values()])
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
Net.remove(statemon_net_current)

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
            'num_exc': num_exc,
            'num_channels': num_channels,
            'sequence_duration': sequence_duration,
            'sequence_repetitions': sequence_repetitions,
            'training_duration': Net.t
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
