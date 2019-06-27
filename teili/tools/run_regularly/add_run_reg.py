# -*- coding: utf-8 -*-
""" This set of functions allows the user to easily add the property of certain run_regularly functions,
specified in run_reg_functions.py to neuron/synapse groups.
These function should take a certain neuron/synapse group and add all necessary state_variables
and function calls.
"""
# @Author: mmilde
# @Date:   2018-07-30 14:19:44
from teili.tools.run_regularly.run_reg_functions import re_init_weights, re_init_ipred, get_activity_proxy,\
    max_value_update, normalize_activity_proxy, correlation_coefficient_tracking, weight_regularization
from brian2 import ms, pA, amp, second
import numpy as np


def add_re_init_weights(group, re_init_threshold, dist_param_re_init, scale_re_init, distribution):
    """Adds a re-initialization run_regularly to a synapse group
    """
    group.namespace.update({'re_init_weights': re_init_weights})
    group.namespace['re_init_threshold'] = re_init_threshold
    group.namespace['dist_param'] = dist_param_re_init
    group.namespace['scale'] = scale_re_init
    if distribution == 'normal':
        group.namespace['dist'] = 0
    if distribution == 'gamma':
        group.namespace['dist'] = 1
    group.run_regularly('''w_plast = re_init_weights(w_plast, N_pre,\
                                                     N_post,\
                                                     re_init_threshold,\
                                                     dist_param, scale, dist)''',
                        dt=50 * ms)


def add_re_init_ipred(group, re_init_threshold):
    """Adds a re-initialization run_regularly to a synapse group
    """
    group.namespace.update({'re_init_ipred': re_init_ipred})
    group.namespace['re_init_threshold'] = re_init_threshold

    group.run_regularly('''Ipred_plast = re_init_ipred(Ipred_plast, N_pre,\
                                                       N_post,\
                                                       re_init_threshold)''',
                        dt=50 * ms)


def add_activity_proxy(group, buffer_size, decay):
    """Adds all needed functionality to track normalised Imem variance for
    Variance Dependent Plasticity.
    """
    group.namespace.update({'get_activity_proxy': get_activity_proxy})
    group.namespace.update({'max_value_update': max_value_update})
    group.namespace.update(
        {'normalize_activity_proxy': normalize_activity_proxy})

    group.add_state_variable('buffer_size', shared=True, constant=True)
    group.add_state_variable('buffer_pointer', shared=True, constant=True)

    group.buffer_size = buffer_size
    group.buffer_pointer = -1
    group.variables.add_array('membrane_buffer', size=(group.N, buffer_size))
    group.variables.add_array('kernel', size=(group.N, buffer_size))
    group.variables.add_array('old_max', size=1)

    mask = np.zeros(np.shape(group.kernel)[1]) * np.nan
    for jj in range(np.shape(group.kernel)[1]):
        mask[jj] = np.exp((jj - (np.shape(group.kernel)[1] - 1)) / decay)
    for ii in range(np.shape(group.kernel)[0]):
        ind = (np.ones(np.shape(group.kernel)[1]) * ii).astype(int)
        group.kernel.set_with_index_array(
            item=ind, value=mask, check_units=False)

    group.run_regularly('''buffer_pointer = (buffer_pointer + 1) % buffer_size;\
    activity_proxy = get_activity_proxy(Imem, buffer_pointer, membrane_buffer, kernel)''', dt=1 * ms)
    group.run_regularly(
        '''old_max = max_value_update(activity_proxy, old_max)''', dt=5 * ms)
    group.run_regularly(
        '''normalized_activity_proxy = normalize_activity_proxy(activity_proxy, old_max)''', dt=5 * ms)


def add_weight_decay(group, decay, learning_rate=None):
    group.add_state_variable('decay')
    if learning_rate is None:
        group.decay = decay
    else:
        group.decay = 1 - (learning_rate / 7)
    group.run_regularly('''w_plast *= decay''', dt=100 * ms)


def add_pred_weight_decay(group, decay, learning_rate=None):
    group.add_state_variable('decay')
    if learning_rate is None:
        group.decay = decay
    else:
        group.decay = 1 - (learning_rate / 7)
    group.run_regularly('''Ipred_plast *= decay''', dt=100 * ms)


def add_adaptive_threshold(group, update_step, decay_time):
    group.add_state_variable('update_step', unit=amp)
    group.add_state_variable('decay_time', unit=second)

    group.update_step = update_step
    group.decay_time = decay_time

    group.run_regularly(
        '''Itau = (Itau * ((t-lastspike)<decay_time)) + ((Itau - update_step) * ((t-lastspike)>=decay_time) * (Itau>2*pA)) + 2*pA * (Itau<=2*pA)''', dt=10 * ms)


def add_weight_regularization(group, buffer_size):
    """This functions adds weight regularization to a synapse group.
    However, it depends on the correlation_coefficient_tracking ru_regularly function,
    which is added here automatically.
    """

    group.namespace.update(
        {'correlation_coefficient_tracking': correlation_coefficient_tracking})
    group.add_state_variable('buffer_size', shared=True, constant=True)
    group.add_state_variable('buffer_pointer_norm', shared=True, constant=True)

    group.buffer_size = buffer_size
    group.buffer_pointer_norm = -1
    group.variables.add_array('mean_variance_time', size=buffer_size)
    group.run_regularly('''buffer_pointer_norm = (buffer_pointer_norm + 1) % buffer_size;\
    mean_variance_time = correlation_coefficient_tracking(w_plast,\
                                                          N_pre,\
                                                          N_post,\
                                                          buffer_pointer_norm,\
                                                          mean_variance_time)''',
                        dt=100 * ms)
    group.namespace.update({'weight_regularization': weight_regularization})
    group.run_regularly(''' w_plast = weight_regularization(w_plast, N_pre, N_post, mean_variance_time)''',
                        dt=200 * ms)
