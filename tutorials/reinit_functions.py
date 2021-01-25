from brian2 import implementation, check_units
from brian2.units import *
import numpy as np
from scipy.stats import gamma

#@implementation('numpy', discard_units=True)
#@check_units(w_plast=1, update_counter=second, update_time=second, sim_time=second, result=1)
#def re_init_weights(w_plast, update_counter, update_time, sim_time):
#    if sim_time > 0:
#        re_init_inds = np.where(update_counter > update_time)[0]
#        re_init_inds = np.delete(re_init_inds, np.where(w_plast[re_init_inds]>7))
#        if np.any(re_init_inds):
#            weights = gamma.rvs(a=2, loc=1, size=len(re_init_inds)).astype(int)
#            weights = np.clip(weights, 0, 15)
#            w_plast[re_init_inds] = weights
#
#    return w_plast
#
#@implementation('numpy', discard_units=True)
#@check_units(delays=1, update_counter=second, update_time=second, result=1)
#def re_init_taus(delays, update_counter, update_time):
#    re_init_inds = np.where(update_counter > update_time)[0]
#    if np.any(re_init_inds):
#        delays[re_init_inds] = np.random.randint(0, 3, size=len(re_init_inds)) * ms
#
#    return delays
#
#@implementation('numpy', discard_units=True)
#@check_units(Vthres=volt, theta=volt, update_counter=second, result=second)
#def activity_tracer(Vthres, theta, update_counter):
#    add_inds = np.where(Vthres < theta)[0]
#    update_counter[add_inds] += 1*ms
#    reset_inds = np.where(Vthres >= theta)[0]
#    update_counter[reset_inds] = 0
#
#    return update_counter

@check_units(prune_indices=1, weight=1, re_init_counter=1, sim_time=second, result=1)
def get_prune_indices(prune_indices, weight, re_init_counter, sim_time):
    if sim_time > 0:
        import pdb;pdb.set_trace()
        zero_weights = np.where(weight==0)
        prune_indices = np.where(re_init_counter < 1)[0]
        # Avoid already inactive weights
        prune_indices = np.delete(prune_indices, zero_weights)

        if len(zero_weights) < len(prune_indices):
            prune_indices = np.random.choice(prune_indices, len(zero_weights),
                                             replace=False)

    return prune_indices

@check_units(spawn_indices=1, prune_indices=1, weight=1, sim_time=second, result=1)
def get_spawn_indices(spawn_indices, prune_indices, weight, sim_time):
    if sim_time > 0:
        # Select indices
        spawn_indices = np.where(weight == 0)[0]
        spawn_indices = np.random.choice(spawn_indices, len(prune_indices),
                                         replace=False)

    return spawn_indices

@check_units(w_plast=1, spawn_indices=1, sim_time=second, result=1)
def wplast_re_init(w_plast, spawn_indices, sim_time):
    if sim_time > 0:
        sampled_weights = gamma.rvs(a=2, size=len(spawn_indices)).astype(int)
        sampled_weights = np.clip(sampled_weights, 0, 15)
        w_plast[spawn_indices] = sampled_weights

    return w_plast

@check_units(tau_syn=second, spawn_indices=1, sim_time=second, result=second)
def tau_re_init(tau_syn, spawn_indices, sim_time):
    if sim_time > 0:
        sampled_taus = np.random.randint(4, 7, size=len(spawn_indices)) * ms
        tau_syn[spawn_indices] = sampled_taus

    return tau_syn

@check_units(delay=second, spawn_indices=1, sim_time=second, result=second)
def delay_re_init(delay, spawn_indices, sim_time):
    if sim_time > 0:
        sampled_delays = np.random.randint(0, 3, size=len(spawn_indices)) * ms
        delay[spawn_indices] = sampled_delays

    return delay

@check_units(weight=1, spawn_indices=1, prune_indices=1, sim_time=second, result=1)
def weight_re_init(weight, spawn_indices, prune_indices, sim_time):
    if sim_time > 0:
        weight[prune_indices] = 0
        weight[spawn_indices] = 1

    return weight

@check_units(re_init_counter=1, result=1)
def reset_re_init_counter(re_init_counter):
    re_init_counter = np.zeros(len(re_init_counter))

    return re_init_counter
