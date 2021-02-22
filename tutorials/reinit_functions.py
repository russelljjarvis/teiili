"""These functions are supposed to be used only with STDP, in which case
the state variable *weight* is zero or one.
"""
from brian2 import implementation, check_units
from brian2.units import *
import numpy as np
from scipy.stats import gamma

@check_units(weight=1, re_init_counter=1, sim_time=second, result=1)
def get_prune_indices(weight, re_init_counter, sim_time):
    if sim_time > 0:
        counter_th = 1
        connected_weights = np.where(weight==1)[0]
        tmp_indices = np.where(re_init_counter < counter_th)[0]
        # Select weights with low counter values that are connected
        tmp_indices = tmp_indices[np.isin(tmp_indices, connected_weights)]

        # Pruned/spawned synapses are limited by unused synapses
        zero_weights = np.where(weight==0)[0]
        if len(tmp_indices) > len(zero_weights):
            tmp_indices = np.random.choice(tmp_indices, len(zero_weights),
                                             replace=False)

        # Assign to variable with correct size
        prune_indices = np.zeros(len(weight))
        prune_indices[tmp_indices] = 1

        return prune_indices
    else:
        return 0

@check_units(prune_indices=1, weight=1, sim_time=second, result=1)
def get_spawn_indices(prune_indices, weight, sim_time):
    if sim_time > 0:
        # Select indices
        tmp_indices = np.where(weight == 0)[0]
        tmp_indices = np.random.choice(tmp_indices,
                                       len(np.where(prune_indices==1)[0]),
                                       replace=False)

        # Assign to variable with correct size
        spawn_indices = np.zeros(len(weight))
        spawn_indices[tmp_indices] = 1

        return spawn_indices
    else:
        return 0

@check_units(w_plast=1, spawn_indices=1, sim_time=second, result=1)
def wplast_re_init(w_plast, spawn_indices, sim_time):
    if sim_time > 0:
        sampled_weights = gamma.rvs(a=3,
                                    size=len(np.where(spawn_indices==1)[0]))
        sampled_weights = np.clip(sampled_weights.astype(int), 0, 15)
        w_plast[spawn_indices==1] = sampled_weights

    return w_plast

@check_units(tau_syn=second, spawn_indices=1, sim_time=second, result=second)
def tau_re_init(tau_syn, spawn_indices, sim_time):
    if sim_time > 0:
        sampled_taus = np.random.randint(4, 7,
                                         size=len(np.where(spawn_indices==1)[0])) * ms
        tau_syn[spawn_indices==1] = sampled_taus

    return tau_syn

@check_units(delay=second, spawn_indices=1, sim_time=second, result=second)
def delay_re_init(delay, spawn_indices, sim_time):
    if sim_time > 0:
        sampled_delays = np.random.randint(0, 3,
                                           size=len(np.where(spawn_indices==1)[0])) * ms
        delay[spawn_indices==1] = sampled_delays

    return delay

@check_units(weight=1, spawn_indices=1, prune_indices=1, sim_time=second, result=1)
def weight_re_init(weight, spawn_indices, prune_indices, sim_time):
    if sim_time > 0:
        weight[prune_indices==1] = 0
        weight[spawn_indices==1] = 1

    return weight

@check_units(re_init_counter=1, result=1)
def reset_re_init_counter(re_init_counter):
    re_init_counter = np.zeros(len(re_init_counter))

    return re_init_counter
