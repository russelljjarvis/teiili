from brian2 import implementation, check_units
from brian2.units import *
import numpy as np
from scipy.stats import gamma

@implementation('numpy', discard_units=True)
@check_units(w_plast=1, update_counter=second, update_time=second, result=1)
def re_init_weights(w_plast, update_counter, update_time):
    re_init_inds = np.where(update_counter > update_time)[0]
    if np.any(re_init_inds):
        weights = gamma.rvs(a=2, loc=1, size=len(re_init_inds)).astype(int)
        weights = np.clip(weights, 0, 15)
        w_plast[re_init_inds] = weights

    return w_plast

@implementation('numpy', discard_units=True)
@check_units(Vthres=volt, theta=volt, update_counter=second, result=second)
def activity_tracer(Vthres, theta, update_counter):
    add_inds = np.where(Vthres < theta)[0]
    update_counter[add_inds] += 1*ms
    reset_inds = np.where(Vthres >= theta)[0]
    update_counter[reset_inds] = 0

    return update_counter
