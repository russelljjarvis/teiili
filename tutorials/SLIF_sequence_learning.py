"""
This code implements a sequence learning using and Excitatory-Inhibitory 
network with STDP.
"""
import numpy as np
from scipy.stats import gamma
from scipy.signal import savgol_filter

from brian2 import ms, mV, Hz, prefs, SpikeMonitor, StateMonitor, defaultclock,\
    ExplicitStateUpdater, SpikeGeneratorGroup, TimedArray, PopulationRateMonitor

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import StochasticLIF as neuron_model
#from teili.models.synapse_models import StochasticSyn_decay_stoch_stdp as stdp_synapse_model
from teili.models.synapse_models import StochasticSyn_decay as static_synapse_model
from teili.stimuli.testbench import SequenceTestbench
from teili.tools.add_run_reg import add_lfsr
from lfsr import create_lfsr
from teili.tools.group_tools import add_group_activity_proxy
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.tools.converter import delete_doublets

import sys
import pickle
import os
from datetime import datetime

# Load ADP synapse
path = os.path.expanduser("/home/pablo/git/teili")
model_path = os.path.join(path, "teili", "models", "equations", "")
adp_synapse_model = SynapseEquationBuilder.import_eq(
        model_path + 'StochSynAdp.py')
path = os.path.expanduser("/home/pablo/git/teili")
model_path = os.path.join(path, "teili", "models", "equations", "")
stdp_synapse_model = SynapseEquationBuilder.import_eq(
        model_path + 'StochStdpNew.py')

# process inputs
learn_factor = 4
ei_p = 0.50
ie_p = 0.70
ee_p = 0.30
ei_w = 3
mean_ie_w = 4
mean_ee_w = 1
mean_ffe_w = 2
mean_ffi_w = 1

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
sequence_duration = 150
noise_prob = None
item_rate = 7
spike_times, spike_indices = [], []
sequence_repetitions = 350
training_duration = sequence_repetitions*sequence_duration*ms
test_duration = 1000*ms
sequence = SequenceTestbench(num_channels, num_items, sequence_duration,
                                     noise_prob, item_rate)
tmp_i, tmp_t = sequence.stimuli()
input_spikes = SpikeGeneratorGroup(num_channels, tmp_i, tmp_t,
                period=sequence_duration*ms)
input_monitor = SpikeMonitor(input_spikes)
net.add(input_spikes, input_monitor)
print('Generating input...')
net.run(training_duration, report='stdout', report_period=100*ms)
spike_indices = np.array(input_monitor.i)
spike_times = np.array(input_monitor.t/ms)
# Creating and adding noise
noise_prob = 0.001
noise_spikes = np.random.rand(num_channels, int(training_duration/ms + test_duration/ms))
noise_indices = np.where(noise_spikes < noise_prob)[0]
noise_times = np.where(noise_spikes < noise_prob)[1]
spike_indices = np.concatenate((spike_indices, noise_indices))
spike_times = np.concatenate((spike_times, noise_times))
sorting_index = np.argsort(spike_times)
spike_indices = spike_indices[sorting_index]
spike_times = spike_times[sorting_index]
spike_times, spike_indices = delete_doublets(spike_times, spike_indices)
# Save them for comparison
spk_i, spk_t = np.array(spike_indices), np.array(spike_times)*ms

# Reproduce activity in a neuron group (necessary for STDP compatibility)
spike_times = [spike_times[np.where(spike_indices==i)[0]]*ms for i in range(num_channels)]
# Create matrix where each row (neuron id) is associated with time when there
# is a spike or -1 when there is not
converted_input = (np.zeros((num_channels, int((training_duration+test_duration)/defaultclock.dt))) - 1)*ms
for ind, val in enumerate(spike_times):
    converted_input[ind, np.around(val/defaultclock.dt).astype(int)] = val
converted_input = np.transpose(converted_input)
converted_input = TimedArray(converted_input, dt=defaultclock.dt)
# t is simulation time, and will be equal to tspike when there is a spike
# Cell remains refractory when there is no spike, i.e. tspike=-1
seq_cells = Neurons(num_channels, model='tspike=converted_input(t, i): second',
        threshold='t==tspike', refractory='tspike < 0*ms')
seq_cells.namespace.update({'converted_input':converted_input})

# Create neuron groups
num_exc = 36
num_inh = 20
exc_cells = Neurons(num_exc,
                    equation_builder=neuron_model(num_inputs=3),
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
                    equation_builder=neuron_model(num_inputs=2),
                    method=stochastic_decay,
                    name='inh_cells',
                    verbose=True)

# Create synapses
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
feedforward_exc = Connections(seq_cells, exc_cells,
                              equation_builder=stdp_synapse_model(),
                              method=stochastic_decay,
                              name='feedforward_exc')
feedforward_inh = Connections(seq_cells, inh_cells,
                              equation_builder=static_synapse_model(),
                              method=stochastic_decay,
                              name='feedforward_inh')

# Connect synapses
feedforward_exc.connect()
feedforward_inh.connect()
if not simple:
    exc_exc_conn.connect('i!=j', p=ee_p)
    #exc_exc_conn.delay = np.random.randint(0, 3, size=np.shape(exc_exc_conn.j)[0]) * ms
exc_inh_conn.connect(p=ei_p)
#exc_inh_conn.delay = np.random.randint(0, 3, size=np.shape(exc_inh_conn.j)[0]) * ms
inh_exc_conn.connect(p=ie_p)

# Setting parameters
# Time constants
exc_exc_conn.tau_syn = 30*ms
exc_exc_conn.taupre = 20*ms
exc_exc_conn.taupost = 30*ms
exc_exc_conn.stdp_thres = 1
exc_inh_conn.tau_syn = 30*ms
inh_exc_conn.tau_syn = 15*ms
feedforward_exc.tau_syn = 30*ms
feedforward_exc.taupre = 20*ms
feedforward_exc.taupost = 30*ms
feedforward_exc.stdp_thres = 1
feedforward_inh.tau_syn = 10*ms

# LFSR lengths
exc_cells.lfsr_num_bits = 5
inh_cells.lfsr_num_bits = 5
exc_exc_conn.lfsr_num_bits_syn = 5
exc_exc_conn.lfsr_num_bits_Apre = 5
exc_exc_conn.lfsr_num_bits_Apost = 5
exc_inh_conn.lfsr_num_bits_syn = 5
inh_exc_conn.lfsr_num_bits_syn = 5
feedforward_exc.lfsr_num_bits_syn = 5
feedforward_exc.lfsr_num_bits_Apre = 5
feedforward_exc.lfsr_num_bits_Apost = 5
feedforward_inh.lfsr_num_bits_syn = 4

seed = 12
exc_cells.Vm = 3*mV
inh_cells.Vm = 3*mV
#feedforward_exc.A_gain = learn_factor
# Weight initializations
if i_plast:
    inh_exc_conn.weight = 1
    # 1 = no inhibition, 0 = maximum inhibition
    inh_exc_conn.variance_th = 0.80
for i in range(num_inh):
    weight_length = np.shape(inh_exc_conn.weight[i,:])
    sampled_weights = gamma.rvs(a=mean_ie_w, loc=1, size=weight_length).astype(int)
    sampled_weights = -np.clip(sampled_weights, 0, 15)
    if i_plast:
        inh_exc_conn.w_plast[i,:] = sampled_weights
    else:
        inh_exc_conn.weight[i,:] = sampled_weights
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
for i in range(num_channels):
    weight_length = np.shape(feedforward_exc.w_plast[i,:])
    feedforward_exc.w_plast[i,:] = gamma.rvs(a=mean_ffe_w, size=weight_length).astype(int)
    weight_length = np.shape(feedforward_inh.weight[i,:])
    feedforward_inh.weight[i,:] = gamma.rvs(a=mean_ffi_w, size=weight_length).astype(int)
#a=1.3
#x = np.linspace(gamma.ppf(0.01, a, loc=1),gamma.ppf(0.99, a, loc=1), 100)
#plt.plot(x, gamma.pdf(x, a,loc=1),'r-', lw=5, alpha=0.6, label='gamma pdf')
#plt.show()

ta = create_lfsr([exc_cells, inh_cells],
                 [exc_exc_conn, exc_inh_conn, inh_exc_conn, feedforward_exc,
                     feedforward_inh],
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
                             buffer_size=200,
                             decay=150)
    inh_exc_conn.variance_th = np.random.uniform(
            low=inh_exc_conn.variance_th - 0.1,
            high=inh_exc_conn.variance_th + 0.1,
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

# Setting up monitors
spikemon_exc_neurons = SpikeMonitor(exc_cells, name='spikemon_exc_neurons')
spikemon_inh_neurons = SpikeMonitor(inh_cells, name='spikemon_inh_neurons')
spikemon_seq_neurons = SpikeMonitor(seq_cells, name='spikemon_seq_neurons')
statemon_exc_cells = StateMonitor(exc_cells, variables=['Vm'], record=True,
                                  name='statemon_exc_cells')
statemon_inh_cells = StateMonitor(inh_cells, variables=['Vm'], record=True,
                                  name='statemon_inh_cells')
statemon_ei_conns = StateMonitor(exc_inh_conn, variables=['I_syn'], record=True,
                                  name='statemon_ei_conns')
statemon_ie_conns = StateMonitor(inh_exc_conn, variables=['I_syn'], record=True,
                                  name='statemon_ie_conns')
if not simple:
    statemon_rec_conns = StateMonitor(exc_exc_conn, variables=['w_plast', 'I_syn', 'Apre', 'Apost'], record=True,
                                      name='statemon_rec_conns')
if i_plast:
    statemon_inh_conns = StateMonitor(inh_exc_conn, variables=['w_plast', 'delta_w'], record=True,
                                      name='statemon_inh_conns')
statemon_ffe_conns = StateMonitor(feedforward_exc, variables=['w_plast', 'I_syn', 'Apre', 'Apost'], record=True,
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
                statemon_pop_rate_i, statemon_inh_conns, statemon_ei_conns, statemon_ie_conns)
    else:
        net.add(seq_cells, exc_cells, inh_cells, exc_exc_conn, exc_inh_conn, inh_exc_conn,
                feedforward_exc, statemon_exc_cells, statemon_inh_cells, feedforward_inh, 
                statemon_rec_conns, spikemon_exc_neurons, spikemon_inh_neurons,
                spikemon_seq_neurons, statemon_ffe_conns, statemon_pop_rate_e,
                statemon_pop_rate_i, statemon_ei_conns, statemon_ie_conns)
else:
    net.add(seq_cells, exc_cells, inh_cells, exc_inh_conn, inh_exc_conn,
            feedforward_exc, statemon_exc_cells, statemon_inh_cells, feedforward_inh, 
            spikemon_exc_neurons, spikemon_inh_neurons,
            spikemon_seq_neurons, statemon_ffe_conns, statemon_pop_rate_e,
            statemon_pop_rate_i, statemon_ei_conns, statemon_ie_conns)
net.run(training_duration + test_duration, report='stdout', report_period=100*ms)

if not np.array_equal(spk_t, spikemon_seq_neurons.t):
    print('Proxy activity and generated input do not match.')
    sys.exit()

# Save targets of recurrent connections as python object
n_rows = num_exc
recurrent_ids = []
recurrent_weights = []
if not simple:
    for i in range(n_rows):
        recurrent_weights.append(list(exc_exc_conn.w_plast[i, :]))
        recurrent_ids.append(list(exc_exc_conn.j[i, :]))

# Getting permutation indices from firing rates
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
    peak_id = np.where(neuron_rate[key]['rate'] == max(neuron_rate[key]['rate']))[0]
    if neuron_rate[key]['rate'].any():
        peak_instants[key] = neuron_rate[key]['t'][peak_id]
# Remove unspecific cases or choose a single peak
double_peaks = [key for key, val in peak_instants.items() if len(val)>1]
#triple_peaks = [key for key, val in peak_instants.items() if len(val)>2] TODO
#if triple_peaks:
#    import pdb;pdb.set_trace()
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

np.savez(path+f'traces.npz',
         Vm_e=statemon_exc_cells.Vm, Vm_i=statemon_inh_cells.Vm,
         exc_rate_t=np.array(statemon_pop_rate_e.t/ms), exc_rate=np.array(statemon_pop_rate_e.smooth_rate(width=10*ms)/Hz),
         inh_rate_t=np.array(statemon_pop_rate_i.t/ms), inh_rate=np.array(statemon_pop_rate_i.smooth_rate(width=10*ms)/Hz),
        )

np.savez(path+f'matrices.npz',
         rf=statemon_ffe_conns.w_plast,
         rec_ids=recurrent_ids, rec_w=recurrent_weights
        )

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

# Check other variables of the simulation
#from brian2 import *
#figure()
#plot(statemon_ei_conns.I_syn[10])
#plot(statemon_ei_conns.I_syn[100])
#title('ei Isyn 10 and 100')
#xlim([10000, 11000])
#
#figure()
#plot(statemon_ie_conns.I_syn[10])
#plot(statemon_ie_conns.I_syn[100])
#title('ie Isyn 10 and 100')
#xlim([10000, 11000])
#
#figure()
#plot(statemon_rec_conns.I_syn[10])
#plot(statemon_rec_conns.I_syn[200])
#title('rec Isyn 10 and 200')
#xlim([10000, 11000])
#
#figure()
#plot(statemon_rec_conns.Apost[20])
#plot(statemon_rec_conns.Apre[20])
#title('rec Apost 20 and Apre 20')
#xlim([10000, 11000])
#
#figure()
#plot(statemon_ffe_conns.I_syn[10])
#plot(statemon_ffe_conns.I_syn[200])
#title('ffe Isyn 10 and 200')
#xlim([10000, 11000])
#
#figure()
#plot(statemon_ffe_conns.Apost[20])
#plot(statemon_ffe_conns.Apre[20])
#title('ffe Apost 20 and Apre 20')
#xlim([10000, 11000])
#show()
