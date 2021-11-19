from brian2 import PoissonGroup
from brian2 import defaultclock, prefs
from brian2 import second, Hz, ms

from teili import TeiliNetwork
from orca_wta import ORCA_WTA

from SLIF_utils import neuron_rate
from orca_params import ConnectionDescriptor, PopulationDescriptor

import numpy as np
import matplotlib.pyplot as plt

defaultclock.dt = 1*ms
prefs.codegen.target = "numpy"
sim_duration = 1*second
Net = TeiliNetwork()

poisson_spikes = PoissonGroup(285, rates=6*Hz)

layer = 'L4'
path = '/Users/Pablo/git/teili/'
conn_desc = ConnectionDescriptor(layer, path)
pop_desc = PopulationDescriptor(layer, path)
conn_desc.intra_prob = {
    'pyr_pyr': 0.1,
    'pyr_pv': 0.1,
    'pyr_sst': 0.1,
    'pyr_vip': 0.1,
    'pv_pyr': 0.1,
    'pv_pv': 0.1,
    'sst_pv': 0.1,
    'sst_pyr': 0.1,
    'sst_vip': 0.1,
    'vip_sst': 0.1}
conn_desc.input_prob = {
    'ff_pyr': 0.25,
    'ff_pv': 0.25,
    'ff_sst': 0.25,
    'ff_vip': 0.25}
for conn in conn_desc.input_plast.keys():
    conn_desc.input_plast[conn] = 'static'
for conn in conn_desc.intra_plast.keys():
    conn_desc.intra_plast[conn] = 'static'
for plast in conn_desc.sample_vars.keys():
    for conn in conn_desc.sample_vars[plast]:
        conn_desc.sample_vars[plast][conn] = []
for conn in pop_desc.sample_vars.keys():
    pop_desc.sample_vars[conn] = []
for pop in pop_desc.group_plast.keys():
    pop_desc.group_plast[pop] = 'static'
pop_desc.group_vals['n_exc'] = 3471
pop_desc.group_vals['num_inputs']['vip_cells'] = 3

pop_desc.update_params()
conn_desc.update_params()

wta = ORCA_WTA(layer=layer,
               conn_params=conn_desc,
               pop_params=pop_desc,
               verbose=True,
               monitor=True)

monitor_params = {'spikemon_pyr_cells': {'group': 'pyr_cells'},
                  'spikemon_pv_cells': {'group': 'pv_cells'},
                  'spikemon_sst_cells': {'group': 'sst_cells'},
                  'spikemon_vip_cells': {'group': 'vip_cells'}}
wta.create_monitors(monitor_params)

# Feedforward weights elicited only a couple of spikes on 
# excitatory neurons
wta.add_input(
    poisson_spikes,
    'ff',
    ['pyr_cells', 'pv_cells', 'sst_cells', 'vip_cells'],
    syn_types=conn_desc.input_plast,
    connectivities=conn_desc.input_prob,
    conn_params=conn_desc)

Net.add(wta, poisson_spikes)
Net.run(sim_duration, report='stdout', report_period=100*ms)

summed_idx = np.cumsum([wta._groups['pyr_cells'].N, wta._groups['pv_cells'].N, wta._groups['sst_cells'].N])
plt.plot(wta.monitors['spikemon_pyr_cells'].t, wta.monitors['spikemon_pyr_cells'].i, 'k.')
plt.plot(wta.monitors['spikemon_pv_cells'].t, wta.monitors['spikemon_pv_cells'].i+summed_idx[0], 'r.')
plt.plot(wta.monitors['spikemon_sst_cells'].t, wta.monitors['spikemon_sst_cells'].i+summed_idx[1], 'g.')
plt.plot(wta.monitors['spikemon_vip_cells'].t, wta.monitors['spikemon_vip_cells'].i+summed_idx[2], 'b.')

neu_rates = neuron_rate(wta.monitors['spikemon_pyr_cells'], kernel_len=100*ms,
    kernel_var=25*ms, simulation_dt=defaultclock.dt, smooth=True)
avg_rates = np.mean(neu_rates['smoothed'], axis=0)
plt.figure()
plt.plot(neu_rates['t'], avg_rates)
plt.ylim([0, 30])
plt.show()
