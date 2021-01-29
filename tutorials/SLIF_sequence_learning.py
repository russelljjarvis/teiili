"""
This code implements a sequence learning using and Excitatory-Inhibitory
network with STDP.
"""
import numpy as np
from scipy.stats import gamma

from brian2 import ms, mV, Hz, prefs, SpikeMonitor, StateMonitor,\
        defaultclock, ExplicitStateUpdater, SpikeGeneratorGroup,\
        PopulationRateMonitor

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import StochasticLIF as neuron_model
#from teili.models.synapse_models import StochasticSyn_decay_stoch_stdp as stdp_synapse_model
from teili.models.synapse_models import StochasticSyn_decay as static_synapse_model
from teili.stimuli.testbench import SequenceTestbench
from teili.tools.add_run_reg import add_lfsr
from teili.tools.group_tools import add_group_activity_proxy
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.tools.converter import delete_doublets

from lfsr import create_lfsr
from SLIF_utils import neuron_group_from_spikes, neuron_rate,\
        rate_correlations, ensemble_convergence
from reinit_functions import get_prune_indices, get_spawn_indices,\
        wplast_re_init, weight_re_init, tau_re_init, delay_re_init,\
        reset_re_init_counter

import sys
import pickle
import os
from datetime import datetime

#############
# Load models
path = os.path.expanduser("/home/pablo/git/teili")
model_path = os.path.join(path, "teili", "models", "equations", "")
adp_synapse_model = SynapseEquationBuilder.import_eq(
        model_path + 'StochSynAdp.py')
stdp_synapse_model = SynapseEquationBuilder.import_eq(
        model_path + 'StochStdpNew.py')
neuron_model_Adapt = NeuronEquationBuilder.import_eq(
        model_path + 'StochLIFAdapt.py')

#############
# Prepare parameters of the simulation
# Defines if recurrent connections are included
if sys.argv[1] == 'no_rec':
    simple = True
elif sys.argv[1] == 'rec':
    simple = False
else:
    print('Provide correct argument')
    sys.exit(0)
if sys.argv[2] == 'plastic_inh':
    i_plast = True
elif sys.argv[2] == 'static_inh':
    i_plast = False
else:
    print('Provide correct argument')
    sys.exit(0)

# Initialize simulation preferences
prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms
stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')
net = TeiliNetwork()

# Initialize input sequence
num_items = 3
num_channels = 144
sequence_duration = 150  # 300
noise_prob = None
item_rate = 25
spike_times, spike_indices = [], []
sequence_repetitions = 200#700  # 350
training_duration = sequence_repetitions*sequence_duration*ms
sequence = SequenceTestbench(num_channels, num_items, sequence_duration,
                             noise_prob, item_rate)
tmp_i, tmp_t = sequence.stimuli()

# Replicates sequence throughout simulation
input_spikes = SpikeGeneratorGroup(num_channels, tmp_i, tmp_t,
                                   period=sequence_duration*ms)
input_monitor = SpikeMonitor(input_spikes)
net.add(input_spikes, input_monitor)
print('Generating input...')
net.run(training_duration, report='stdout', report_period=100*ms)
spike_indices = np.array(input_monitor.i)
spike_times = np.array(input_monitor.t/ms)

# Adding incomplete sequence
symbols = {}
symbol_duration = int(sequence_duration/num_items)
for item in range(num_items):
    item_interval = (tmp_t>=(symbol_duration*item*ms)) & (tmp_t<(symbol_duration*(item+1)*ms))
    symbols[item] = {'t':tmp_t[item_interval],
                     'i':tmp_i[item_interval]}

incomplete_sequences = 5
test_duration = incomplete_sequences*sequence_duration*ms
include_symbols = [[1], [0], [0,1], [0,2], [1,2]]
for incomp_seq in range(incomplete_sequences):
    for incl_symb in include_symbols[incomp_seq]:
        tmp_symb = [(x/ms + incomp_seq*sequence_duration + training_duration/ms)
                        for x in symbols[incl_symb]['t']]
        spike_times = np.append(spike_times, tmp_symb)
        spike_indices = np.append(spike_indices, symbols[incl_symb]['i'])

# Creating and adding noise
#noise_prob = 0.002
#noise_spikes = np.random.rand(num_channels, int(training_duration/ms + test_duration/ms))
#noise_indices = np.where(noise_spikes < noise_prob)[0]
#noise_times = np.where(noise_spikes < noise_prob)[1]
#spike_indices = np.concatenate((spike_indices, noise_indices))
#spike_times = np.concatenate((spike_times, noise_times))
#sorting_index = np.argsort(spike_times)
#spike_indices = spike_indices[sorting_index]
#spike_times = spike_times[sorting_index]
#spike_times, spike_indices = delete_doublets(spike_times, spike_indices)

# Save them for comparison
spk_i, spk_t = np.array(spike_indices), np.array(spike_times)*ms

# Reproduce activity in a neuron group (necessary for STDP compatibility)
seq_cells = neuron_group_from_spikes(spike_indices, spike_times, num_channels,
                                     defaultclock.dt,
                                     int((training_duration+test_duration)/defaultclock.dt))

#################
# Building network
num_exc = 48
num_inh = 30
exc_cells = Neurons(num_exc,
                    equation_builder=neuron_model_Adapt(num_inputs=3),
                    method=stochastic_decay,
                    name='exc_cells',
                    verbose=True)
# Register proxy arrays
if i_plast:
    dummy_unit = 1*mV
    exc_cells.variables.add_array('activity_proxy',
                                   size=exc_cells.N,
                                   dimensions=dummy_unit.dim)

    exc_cells.variables.add_array('normalized_activity_proxy',
                                   size=exc_cells.N)

inh_cells = Neurons(num_inh,
                    equation_builder=neuron_model(num_inputs=3),
                    method=stochastic_decay,
                    name='inh_cells',
                    verbose=True)

# Connections
ei_p = 0.50
ie_p = 0.70
ee_p = 0.60

if not simple:
    exc_exc_conn = Connections(exc_cells, exc_cells,
                               equation_builder=stdp_synapse_model(),
                               method=stochastic_decay,
                               name='exc_exc_conn')
exc_inh_conn = Connections(exc_cells, inh_cells,
                           equation_builder=static_synapse_model(),
                           method=stochastic_decay,
                           name='exc_inh_conn')
if i_plast:
    inh_exc_conn = Connections(inh_cells, exc_cells,
                               equation_builder=adp_synapse_model,
                               method=stochastic_decay,
                               name='inh_exc_conn')
else:
    inh_exc_conn = Connections(inh_cells, exc_cells,
                               equation_builder=static_synapse_model(),
                               method=stochastic_decay,
                               name='inh_exc_conn')
inh_inh_conn = Connections(inh_cells, inh_cells,
                           equation_builder=static_synapse_model(),
                           method=stochastic_decay,
                           name='inh_inh_conn')
feedforward_exc = Connections(seq_cells, exc_cells,
                              equation_builder=stdp_synapse_model(),
                              method=stochastic_decay,
                              name='feedforward_exc')
feedforward_inh = Connections(seq_cells, inh_cells,
                              equation_builder=static_synapse_model(),
                              method=stochastic_decay,
                              name='feedforward_inh')

feedforward_exc.connect()
feedforward_inh.connect()
inh_inh_conn.connect(p=.1)
#TODO test with delays
if not simple:
    exc_exc_conn.connect('i!=j', p=ee_p)
    exc_exc_conn.delay = np.random.randint(0, 15, size=np.shape(exc_exc_conn.j)[0]) * ms
exc_inh_conn.connect(p=ei_p)
exc_inh_conn.delay = np.random.randint(0, 15, size=np.shape(exc_inh_conn.j)[0]) * ms
feedforward_exc.delay = np.random.randint(0, 15, size=np.shape(feedforward_exc.j)[0]) * ms
feedforward_inh.delay = np.random.randint(0, 15, size=np.shape(feedforward_inh.j)[0]) * ms
inh_exc_conn.connect(p=ie_p)

###################
# Set paramters of the network
# FIXME Worse RFs if values below are used
#feedforward_exc.gain_syn = 0.5*mA
#feedforward_inh.gain_syn = 0.5*mA
#exc_exc_conn.gain_syn = 0.5*mA
#exc_inh_conn.gain_syn = 0.5*mA
#inh_exc_conn.gain_syn = 0.5*mA

# Time constants
# Values similar to those in Klampfl&Maass(2013), Joglekar etal(2018), Vogels&Abbott(2009)
exc_exc_conn.tau_syn = 5*ms
exc_exc_conn.taupre = 20*ms
exc_exc_conn.taupost = 60*ms
exc_exc_conn.stdp_thres = 1
exc_inh_conn.tau_syn = 5*ms
inh_exc_conn.tau_syn = 10*ms
inh_inh_conn.tau_syn = 10*ms
feedforward_exc.tau_syn = 5*ms
feedforward_exc.taupre = 20*ms
feedforward_exc.taupost = 60*ms
feedforward_exc.stdp_thres = 1
feedforward_inh.tau_syn = 5*ms
exc_cells.tau = 19*ms
inh_cells.tau = 10*ms

# LFSR lengths
exc_cells.lfsr_num_bits = 5
inh_cells.lfsr_num_bits = 5
exc_exc_conn.lfsr_num_bits_syn = 5
exc_exc_conn.lfsr_num_bits_Apre = 5
exc_exc_conn.lfsr_num_bits_Apost = 6
exc_inh_conn.lfsr_num_bits_syn = 5
inh_exc_conn.lfsr_num_bits_syn = 5
inh_inh_conn.lfsr_num_bits_syn = 5
feedforward_exc.lfsr_num_bits_syn = 5
feedforward_exc.lfsr_num_bits_Apre = 5
feedforward_exc.lfsr_num_bits_Apost = 6
feedforward_inh.lfsr_num_bits_syn = 4

seed = 12
exc_cells.Vm = 3*mV
inh_cells.Vm = 3*mV
learn_factor = 4
#feedforward_exc.A_gain = learn_factor

# Weight initializations
ei_w = 3
mean_ie_w = 4
mean_ee_w = 1
mean_ffe_w = 3
mean_ffi_w = 1

inh_inh_conn.weight = -1
if i_plast:
    inh_exc_conn.weight = -1
    # 1 = no inhibition, 0 = maximum inhibition
    variance_th = 0.50
for i in range(num_inh):
    weight_length = np.shape(inh_exc_conn.weight[i,:])
    sampled_weights = gamma.rvs(a=mean_ie_w, loc=1, size=weight_length).astype(int)
    sampled_weights = np.clip(sampled_weights, 0, 15)
    if i_plast:
        inh_exc_conn.w_plast[i,:] = sampled_weights
    else:
        inh_exc_conn.weight[i,:] = -sampled_weights
if not simple:
    exc_exc_conn.weight = 1
for i in range(num_exc):
    if not simple:
        weight_length = np.shape(exc_exc_conn.w_plast[i,:])
        exc_exc_conn.w_plast[i,:] = gamma.rvs(a=mean_ee_w, size=weight_length).astype(int)
    weight_length = np.shape(exc_inh_conn.weight[i,:])
    sampled_weights = gamma.rvs(a=ei_w, loc=1, size=weight_length).astype(int)
    sampled_weights = np.clip(sampled_weights, 0, 15)
    exc_inh_conn.weight[i,:] = sampled_weights
feedforward_exc.weight = 1
num_inh_weight = np.shape(feedforward_inh.weight[i,:])[0]
for i in range(num_channels):
    wplast_length = np.shape(feedforward_exc.w_plast[i,:])
    feedforward_exc.w_plast[i,:] = np.clip(
            gamma.rvs(a=mean_ffe_w, size=wplast_length).astype(int),
            0,
            15)
    feedforward_inh.weight[i,:] = np.clip(
            gamma.rvs(a=mean_ffi_w, size=num_inh_weight).astype(int),
            0,
            15)
# Set sparsity for ffe connections
for i in range(num_exc):
    ffe_zero_w = np.random.choice(num_channels, int(num_channels*.3), replace=False)
    feedforward_exc.weight[ffe_zero_w,i] = 0
    feedforward_exc.w_plast[ffe_zero_w,i] = 0

# Set LFSRs for each group
ta = create_lfsr([exc_cells, inh_cells],
                 [exc_exc_conn, exc_inh_conn, inh_exc_conn, feedforward_exc,
                     feedforward_inh, inh_inh_conn],
                 defaultclock.dt)
# Necessary for new stdp equations TODO update when it becomes proper equations
exc_exc_conn.lfsr_max_value_condApost2 = 14*ms
exc_exc_conn.lfsr_max_value_condApre2 = 14*ms
feedforward_exc.lfsr_max_value_condApost2 = 14*ms
feedforward_exc.lfsr_max_value_condApre2 = 14*ms

if i_plast:
    # Add proxy activity group
    activity_proxy_group = [exc_cells]
    add_group_activity_proxy(activity_proxy_group,
                             buffer_size=400,
                             decay=150)
    inh_exc_conn.variance_th = np.random.uniform(
            low=variance_th - 0.1,
            high=variance_th + 0.1,
            size=len(inh_exc_conn))

# Adding mismatch
#mismatch_neuron_param = {
#    'tau': 0.05
#}
#mismatch_synap_param = {
#    'tau_syn': 0.05
#}
#mismatch_plast_param = {
#    'taupre': 0.05,
#    'taupost': 0.05
#}
#
#exc_cells.add_mismatch(std_dict=mismatch_neuron_param, seed=10)
#inh_cells.add_mismatch(std_dict=mismatch_neuron_param, seed=10)
#exc_exc_conn.add_mismatch(std_dict=mismatch_synap_param, seed=11)
#exc_inh_conn.add_mismatch(std_dict=mismatch_synap_param, seed=11)
#inh_exc_conn.add_mismatch(std_dict=mismatch_synap_param, seed=11)
#feedforward_exc.add_mismatch(std_dict=mismatch_synap_param, seed=11)
#feedforward_inh.add_mismatch(std_dict=mismatch_synap_param, seed=11)
#exc_exc_conn.add_mismatch(std_dict=mismatch_plast_param, seed=11)
#feedforward_exc.add_mismatch(std_dict=mismatch_plast_param, seed=11)

###################
# Adding homeostatic mechanisms
feedforward_exc.variables.add_array('prune_indices', size=len(feedforward_exc.weight))
feedforward_exc.variables.add_array('spawn_indices', size=len(feedforward_exc.weight))
feedforward_exc.namespace.update({'get_prune_indices': get_prune_indices})
feedforward_exc.namespace.update({'get_spawn_indices': get_spawn_indices})
feedforward_exc.namespace.update({'wplast_re_init': wplast_re_init})
feedforward_exc.namespace.update({'tau_re_init': tau_re_init})
feedforward_exc.namespace.update({'delay_re_init': delay_re_init})
feedforward_exc.namespace.update({'weight_re_init': weight_re_init})
feedforward_exc.namespace.update({'reset_re_init_counter': reset_re_init_counter})

reinit_period = 50000*ms
feedforward_exc.run_regularly('''prune_indices = get_prune_indices(\
                                                    prune_indices,\
                                                    weight,\
                                                    re_init_counter,\
                                                    t)''',
                                                    dt=reinit_period,
                                                    order=0)
feedforward_exc.run_regularly('''spawn_indices = get_spawn_indices(\
                                                    spawn_indices,\
                                                    prune_indices,\
                                                    weight,\
                                                    t)''',
                                                    dt=reinit_period,
                                                    order=1)

feedforward_exc.run_regularly('''w_plast = wplast_re_init(w_plast,\
                                                          spawn_indices,\
                                                          t)''',
                                                          dt=reinit_period,
                                                          order=2)
feedforward_exc.run_regularly('''tau_syn = tau_re_init(tau_syn,\
                                                       spawn_indices,\
                                                       t)''',
                                                       dt=reinit_period,
                                                       order=3)
feedforward_exc.run_regularly('''weight = weight_re_init(weight,\
                                                         spawn_indices,\
                                                         prune_indices,\
                                                         t)''',
                                                         dt=reinit_period,
                                                         order=5)
feedforward_exc.run_regularly('''re_init_counter = reset_re_init_counter(re_init_counter)''',
                                                                         dt=reinit_period,
                                                                         order=6)

##################
# Setting up monitors
spikemon_exc_neurons = SpikeMonitor(exc_cells, name='spikemon_exc_neurons')
spikemon_inh_neurons = SpikeMonitor(inh_cells, name='spikemon_inh_neurons')
spikemon_seq_neurons = SpikeMonitor(seq_cells, name='spikemon_seq_neurons')
statemon_exc_cells = StateMonitor(exc_cells, variables=['Vm'], record=np.random.randint(0, num_exc),
                                  name='statemon_exc_cells')
statemon_inh_cells = StateMonitor(inh_cells, variables=['Vm'], record=np.random.randint(0, num_inh),
                                  name='statemon_inh_cells')
statemon_ei_conns = StateMonitor(exc_inh_conn, variables=['I_syn'], record=True,
                                  name='statemon_ei_conns')
statemon_ie_conns = StateMonitor(inh_exc_conn, variables=['I_syn'], record=True,
                                  name='statemon_ie_conns')
if not simple:
    statemon_rec_conns = StateMonitor(exc_exc_conn, variables=['w_plast'], record=True,
                                      name='statemon_rec_conns')
if i_plast:
    statemon_inh_conns = StateMonitor(inh_exc_conn, variables=['w_plast'], record=True,
                                      name='statemon_inh_conns')
statemon_ffe_conns = StateMonitor(feedforward_exc, variables=['w_plast'], record=True,
                                  name='statemon_ffe_conns')
statemon_pop_rate_e = PopulationRateMonitor(exc_cells)
statemon_pop_rate_i = PopulationRateMonitor(inh_cells)

net = TeiliNetwork()
if not simple:
    if i_plast:
        net.add(seq_cells, exc_cells, inh_cells, exc_exc_conn, exc_inh_conn, inh_exc_conn,
                feedforward_exc, statemon_exc_cells, statemon_inh_cells, feedforward_inh,
                statemon_rec_conns, spikemon_exc_neurons, spikemon_inh_neurons,
                spikemon_seq_neurons, statemon_ffe_conns, statemon_pop_rate_e,
                statemon_pop_rate_i, statemon_inh_conns, statemon_ei_conns, statemon_ie_conns,
                inh_inh_conn)
    else:
        net.add(seq_cells, exc_cells, inh_cells, exc_exc_conn, exc_inh_conn, inh_exc_conn,
                feedforward_exc, statemon_exc_cells, statemon_inh_cells, feedforward_inh,
                statemon_rec_conns, spikemon_exc_neurons, spikemon_inh_neurons,
                spikemon_seq_neurons, statemon_ffe_conns, statemon_pop_rate_e,
                statemon_pop_rate_i, statemon_ei_conns, statemon_ie_conns, inh_inh_conn)
else:
    net.add(seq_cells, exc_cells, inh_cells, exc_inh_conn, inh_exc_conn,
            feedforward_exc, statemon_exc_cells, statemon_inh_cells, feedforward_inh,
            spikemon_exc_neurons, spikemon_inh_neurons,
            spikemon_seq_neurons, statemon_ffe_conns, statemon_pop_rate_e,
            statemon_pop_rate_i, statemon_ei_conns, statemon_ie_conns, inh_inh_conn)
net.run(training_duration + test_duration, report='stdout', report_period=100*ms)

##########
# Evaluations
if not np.array_equal(spk_t, spikemon_seq_neurons.t):
    print('Proxy activity and generated input do not match.')
    sys.exit()

neu_rates = neuron_rate(spikemon_exc_neurons, 200, 10, 0.001,
                        [0, training_duration/ms])
seq_rates = neuron_rate(spikemon_seq_neurons, 200, 10, 0.001,
                        [0, sequence_duration])
foo = ensemble_convergence(seq_rates, neu_rates, [[0, 48], [48, 96], [96, 144]],
                           sequence_duration, sequence_repetitions)

corrs = rate_correlations(neu_rates, sequence_duration, sequence_repetitions)


############
# Saving results
# Save targets of recurrent connections as python object
n_rows = num_exc
recurrent_ids = []
recurrent_weights = []
if not simple:
    for i in range(n_rows):
        recurrent_weights.append(list(exc_exc_conn.w_plast[i, :]))
        recurrent_ids.append(list(exc_exc_conn.j[i, :]))

# Calculating permutation indices from firing rates
tmp_spike_trains = spikemon_exc_neurons.spike_trains()
neuron_rate = {}
peak_instants = {}
last_sequence_t = training_duration/ms-sequence_duration
interval = range(int(last_sequence_t), int(training_duration/ms)+1)
# Create normalized and truncated gaussian time window
kernel = np.exp(-(np.arange(-100, 100)) ** 2 / (2 * 10 ** 2))
kernel = kernel[np.where(kernel>0.001)]
kernel = kernel / kernel.sum()
for key, val in tmp_spike_trains.items():
    selected_spikes = [x for x in val/ms if x>last_sequence_t]
    # Use histogram to get values that will be convolved
    h, b = np.histogram(selected_spikes, bins=interval, range=(min(interval), max(interval)))
    neuron_rate[key] = {'rate': np.convolve(h, kernel, mode='same'), 't': b[:-1]}
    peak_index = np.where(neuron_rate[key]['rate'] == max(neuron_rate[key]['rate']))[0]
    if neuron_rate[key]['rate'].any():
        peak_instants[key] = neuron_rate[key]['t'][peak_index]
# Remove unspecific cases or choose a single peak TODO no, just average peaks
double_peaks = [key for key, val in peak_instants.items() if len(val)>1]
#triple_peaks = [key for key, val in peak_instants.items() if len(val)>2]
#if triple_peaks:
for i in double_peaks:
    h, b = np.histogram(peak_instants[i], bins=num_items, range=(min(interval), max(interval)))
    if any(h==1):
        peak_instants.pop(i)
    else:
        peak_instants[i] = np.array(peak_instants[i][0])
sorted_peaks = dict(sorted(peak_instants.items(), key=lambda x: x[1]))
permutation_ids = [x[0] for x in sorted_peaks.items()]
[permutation_ids.append(i) for i in range(num_exc) if not i in permutation_ids]

# Save data
date_time = datetime.now()
path = f"""{date_time.strftime('%Y.%m.%d')}_{date_time.hour}.{date_time.minute}/"""
os.mkdir(path)
np.savez(path+f'rasters.npz',
         input_t=np.array(spikemon_seq_neurons.t/ms), input_i=np.array(spikemon_seq_neurons.i),
         exc_spikes_t=np.array(spikemon_exc_neurons.t/ms), exc_spikes_i=np.array(spikemon_exc_neurons.i),
         inh_spikes_t=np.array(spikemon_inh_neurons.t/ms), inh_spikes_i=np.array(spikemon_inh_neurons.i),
        )
del spikemon_seq_neurons, spikemon_inh_neurons#, spikemon_exc_neurons

np.savez(path+f'traces.npz',
         Vm_e=statemon_exc_cells.Vm, Vm_i=statemon_inh_cells.Vm,
         exc_rate_t=np.array(statemon_pop_rate_e.t/ms), exc_rate=np.array(statemon_pop_rate_e.smooth_rate(width=10*ms)/Hz),
         inh_rate_t=np.array(statemon_pop_rate_i.t/ms), inh_rate=np.array(statemon_pop_rate_i.smooth_rate(width=10*ms)/Hz),
        )
del statemon_inh_cells, statemon_pop_rate_e, statemon_pop_rate_i#,statemon_exc_cells

np.savez_compressed(path+f'matrices.npz',
         rf=statemon_ffe_conns.w_plast.astype(np.uint8),
         rec_ids=recurrent_ids, rec_w=recurrent_weights
        )
del recurrent_ids, recurrent_weights#, statemon_ffe_conns

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
            'learn_factor': learn_factor,
            'mean ffe w': mean_ffe_w,
            'mean ffi w': mean_ffi_w,
            'i_plast': i_plast,
            'simple': simple
        }
with open(path+'general.data', 'wb') as f:
    pickle.dump(Metadata, f)

Metadata = {'exc': exc_cells.get_params(),
            'inh': inh_cells.get_params()}
with open(path+'population.data', 'wb') as f:
    pickle.dump(Metadata, f)

Metadata = {'e->i': exc_inh_conn.get_params(),
            'i->e': inh_exc_conn.get_params(),
            'e->e': exc_exc_conn.get_params(),
            'ffe': feedforward_exc.get_params(),
            'ffi': feedforward_inh.get_params()
            }
with open(path+'connections.data', 'wb') as f:
    pickle.dump(Metadata, f)

from brian2 import *
_ = hist(corrs, bins=20)
show()
