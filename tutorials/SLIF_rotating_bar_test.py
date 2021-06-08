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
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder

from teili.tools.lfsr import create_lfsr
from teili.tools.misc import neuron_group_from_spikes
from SLIF_utils import neuron_rate, rate_correlations, ensemble_convergence,\
        permutation_from_rate, load_merge_multiple

from orca_wta import ORCA_WTA
from orca_params import excitatory_synapse_soma, excitatory_synapse_dend

import sys
import pickle
import os
from datetime import datetime

# Prepare parameters of the simulation
data_folder = sys.argv[1]

# Initialize simulation preferences
prefs.codegen.target = "numpy"

# Load relevant information
with open(f'{data_folder}metadata', 'rb') as f:
    metadata = pickle.load(f)
num_exc = metadata['num_exc']
num_channels = metadata['num_channels']

# Initialize input with different input pattern or load from metadata files
testbench_stim = OCTA_Testbench()
sequence_repetitions = 6
testbench_stim.rotating_bar(length=10, nrows=10,
                            direction='cw',
                            ts_offset=3, angle_step=10,
                            #noise_probability=0.2,
                            repetitions=sequence_repetitions,
                            debug=False)
test_duration = np.max(testbench_stim.times)*ms
input_indices = testbench_stim.indices
input_times = testbench_stim.times * ms

# Convert input into neuron group (necessary for STDP compatibility)
relay_cells = neuron_group_from_spikes(num_channels,
                                       defaultclock.dt,
                                       test_duration,
                                       spike_indices=input_indices,
                                       spike_times=input_times)

Net = TeiliNetwork()
orca = ORCA_WTA(num_exc_neurons=num_exc)#,
    #ratio_pv=1, ratio_sst=0.02, ratio_vip=0.02)
orca.add_input(relay_cells, 'ff', ['pyr_cells'], 'reinit', 'excitatory',
    sparsity=.3)
orca.add_input(relay_cells, 'ff', ['pv_cells', 'sst_cells', 'vip_cells'],
    'static', 'inhibitory')

orca2 = ORCA_WTA(num_exc_neurons=num_exc, name='top_down_')#,
    #ratio_pv=1, ratio_sst=0.02, ratio_vip=0.02)
orca2.add_input(relay_cells, 'ff', ['pyr_cells'], 'reinit', 'excitatory',
    sparsity=.3)
orca2.add_input(relay_cells, 'ff', ['pv_cells', 'sst_cells', 'vip_cells'],
    'static', 'inhibitory')
orca.add_input(orca2._groups['pyr_cells'], 'fb', ['pyr_cells'], 'reinit',
    'excitatory', exc_params=excitatory_synapse_dend, sparsity=.3)
orca.add_input(orca2._groups['pyr_cells'], 'fb', ['vip_cells'], 'static',
    'inhibitory', exc_params=excitatory_synapse_soma)


# Testing
Net.add(orca, orca2)
Net.restore(filename=f'{data_folder}network')
Net.add(relay_cells)

pyr_mon = SpikeMonitor(orca._groups['pyr_cells'])
pv_mon = SpikeMonitor(orca._groups['pv_cells'])
sst_mon = SpikeMonitor(orca._groups['sst_cells'])
vip_mon = SpikeMonitor(orca._groups['vip_cells'])
state_mon = StateMonitor(orca._groups['pyr_cells'], variables=['Iin0', 'Iin1', 'Iin2', 'Iin3', 'Iin4'], record=True)
Net.add(pyr_mon, pv_mon, sst_mon, vip_mon, state_mon)

# Turn off learning
orca._groups['ff_pyr'].stdp_thres = 0
orca._groups['pyr_pyr'].stdp_thres = 0
orca._groups['pv_pyr'].inh_learning_rate = 0
orca._groups['sst_pyr'].inh_learning_rate = 0
# TODO orca._groups['sst_pv'].stdp_thres = 0

# Normal operation
Net.run(100*ms, report='stdout', report_period=100*ms)

# deactivate top-down only
orca._groups['fb_pyr'].weight = 0
orca._groups['fb_vip'].weight = 0
Net.run(100*ms, report='stdout', report_period=100*ms)

# deactivate bottom-up only
orca._groups['fb_pyr'].weight = 1
orca._groups['fb_vip'].weight = 1
orca._groups['ff_pyr'].weight = 0
orca._groups['ff_pv'].weight = 0
orca._groups['ff_sst'].weight = 0
orca._groups['ff_vip'].weight = 0
Net.run(100*ms, report='stdout', report_period=100*ms)

# No input
orca._groups['fb_pyr'].weight = 0
orca._groups['fb_vip'].weight = 0
Net.run(100*ms, report='stdout', report_period=100*ms)

permutation_file = np.load(f'{data_folder}permutation.npz')
permutation = permutation_file['ids']
sorted_i = np.asarray([np.where(
                np.asarray(permutation) == int(i))[0][0] for i in pyr_mon.i])
import matplotlib.pyplot as plt
plt.plot(pyr_mon.t/second, sorted_i, '.k')
plt.xlabel('Time (second)')
plt.xlim([min(pyr_mon.t/second), Net.t/second])
plt.ylabel('Neuron index')
plt.show()

neu_rates = neuron_rate(pyr_mon, kernel_len=200,
    kernel_var=10, kernel_min=0.001, simulation_dt=defaultclock.dt)

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

#def plot_EI_balance(idx):
#    import matplotlib.pyplot as plt
#    win_len = 100
#    rec_Iin = np.convolve(statemon_net_current.Iin0[idx], np.ones(win_len)/win_len, mode='valid')
#    inh_Iin = np.convolve(statemon_net_current.Iin1[idx], np.ones(win_len)/win_len, mode='valid')
#    ffe_Iin = np.convolve(statemon_net_current.Iin2[idx], np.ones(win_len)/win_len, mode='valid')
#    noise_Iin = np.convolve(statemon_net_current.Iin3[idx], np.ones(win_len)/win_len, mode='valid')
#    total_Iin = np.convolve(statemon_net_current.Iin[idx], np.ones(win_len)/win_len, mode='valid')
#    plt.plot(rec_Iin, 'r', label='rec. current')
#    plt.plot(inh_Iin, 'g', label='inh. current')
#    plt.plot(ffe_Iin, 'b', label='input current')
#    plt.plot(total_Iin, 'k', label='net current')
#    plt.plot(noise_Iin, 'k--', label='spont. activity')
#    plt.legend()
#    plt.ylabel('Current [amp]')
#    plt.xlabel('time [ms]')
#    plt.title('EI balance')
#    plt.show()
