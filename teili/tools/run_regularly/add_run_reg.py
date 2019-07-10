# -*- coding: utf-8 -*-
"""This set of functions allows the user to easily add the property of certain run_regularly functions,
specified in run_reg_functions.py to neuron/synapse groups.
These function should take a certain neuron/synapse group and add all necessary state_variables
and function calls.
"""
# @Author: mmilde
# @Date:   2018-07-30 14:19:44

from teili.tools.run_regularly.run_reg_functions import re_init_weights,\
    re_init_ipred, get_activity_proxy, max_value_update, normalize_activity_proxy,\
    correlation_coefficient_tracking

from brian2 import ms, pA, amp, second
import numpy as np


def add_re_init_weights(group, re_init_threshold, dist_param_re_init, scale_re_init, distribution):
    """Adds a re-initialization run_regularly to a synapse group

    Args:
        group (TYPE): Synapse group
        re_init_threshold (TYPE): Description
        dist_param_re_init (float, optional): Mean of distribution in case of 'gaussian'
            or shape parameter k for 'gamma' distribution
        scale_re_init (float, optional): Standard deviation in case of 'gaussian'
            or scale parameter sigma for 'gamma' distribution
        distribution (str, optional): Parameter to determine the random distribution
            to be used to initialise the weights. Possible 'gaussian' or 'gamma'

    Deleted Parameters:
        dist_param_init (float, optional): Mean of distribution in case of 'gaussian'
            or shape parameter k for 'gamma' distribution
        scale_init (float, optional): Standard deviation in case of 'gaussian'
            or scale parameter sigma for 'gamma' distribution
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

    Args:
        group (Synapse group): Synapse group
        re_init_threshold (float): Value below which ipred is re-initialise
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

    Args:
        group (Synapse group): Synapse group
        buffer_size (int, optional): Parameter to set the size of the buffer for
            adctivity dependent plasticity rule. Meaning how many samples are considered to
            calculate the activity proxy of post synaptic neurons
         decay (int, optional): Time constant for decay of exponentioally weighted activity proxy
        decay (TYPE): Description
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


def add_weight_decay(group, decay_strategy, decay_rate=None):
    """Summary

    Args:
        group (Synapse group): Synapse group
        decay_strategy (str, optional): Weight decay strategy. Either 'global' which decays weight
            based o fixed time interval, or 'local' which performs event-driven weight decay
        decay_rate (float, optional): Value smaller 1 which is multiplied with w_plast every 50 ms

    Raises:
        UserWarning: TBI
    """
    if decay_strategy == 'global':
        group.add_state_variable('decay')

        if decay_rate is not None:
            group.decay = decay_rate
        else:
            group.decay = 1 - (0.001 / 7)

        group.run_regularly('''w_plast *= decay''', dt=100 * ms)

    elif decay_strategy == 'local':
        raise UserWarning('To be implemented')
        """The general idea is to to reduce w_plast on_post proportional to w_plas
        meaning that the on_post statement should look like
        w_plast = w_plast - (eta * k * w_plast)
        where eta is the learning rate aka dApre and k is a scaling factor
        """


def add_pred_weight_decay(group, decay_strategy, decay_rate=None):
    """Summary

    Args:
        group (Synapse group): Synapse group
        decay_strategy (str, optional): Weight decay strategy. Either 'global' which decays weight
            based o fixed time interval, or 'local' which performs event-driven weight decay
        decay_rate (float, optional): Value smaller 1 which is multiplied with w_plast every 100 ms
    """
    group.add_state_variable('decay')
    if decay_strategy == 'global':
        if decay_rate is not None:
            group.decay = decay_rate
        else:
            group.decay = 1 - (0.001 / 7)
        group.run_regularly('''Ipred_plast *= decay''', dt=100 * ms)

    elif decay_strategy == 'local':
        raise UserWarning('To be implemented')
        """The general idea is to to reduce w_plast on_post proportional to w_plas
        meaning that the on_post statement should look like
        w_plast = w_plast - (eta * k * w_plast)
        where eta is the learning rate aka dApre and k is a scaling factor
        """
