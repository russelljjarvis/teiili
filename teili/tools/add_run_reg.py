# -*- coding: utf-8 -*-
"""This set of functions allows the user to easily add the property of
certain run_regularly functions, specified in run_reg_functions.py to
neuron/synapse groups. These function should take a certain neuron/synapse
group and add all necessary state_variables and function calls.
"""
# @Author: mmilde
# @Date:   2018-07-30 14:19:44

import numpy as np
from brian2 import ms, pA, amp, second
from teili.core.groups import Neurons, Connections
from teili.tools.run_reg_functions import re_init_weights,\
    get_activity_proxy_vm, get_activity_proxy_imem,\
    max_value_update_vm, max_value_update_imem,\
    normalize_activity_proxy_vm, normalize_activity_proxy_imem,\
    get_re_init_index, lfsr




def add_re_init_weights(group, re_init_index, re_init_threshold, dist_param_re_init,
                        scale_re_init, distribution):
    """Adds a re-initialization run_regularly to a synapse group

    Args:
        group (Connection group): Synapse group
        re_init_threshold (float, required): Average post-synaptic weight
            which triggers a re-initialisation if below or above
            1 - re_init_threshold.
        dist_param_re_init (float, required): Mean of distribution in case.
            of 'gaussian' or shape parameter k for 'gamma' distribution.
        scale_re_init (float, required): Scale parameter sigma for
            distribution.
        distribution (str, optional): Parameter to determine the random
            distribution to be used to initialise the weights. Possible
            'gaussian' or 'gamma'.
    """
    group.namespace.update({'re_init_weights': re_init_weights})
    if re_init_index is None:
        group.add_state_variable('re_init_index')
        group.namespace['re_init_index'] = re_init_index
    else:
        group.variables.add_array('re_init_index', size=(len(group)))
    group.namespace['re_init_threshold'] = re_init_threshold
    group.namespace['dist_param'] = dist_param_re_init
    group.namespace['scale'] = scale_re_init
    if distribution == 'normal':
        group.namespace['dist'] = 0
    if distribution == 'gamma':
        group.namespace['dist'] = 1
    if re_init_index is not None:
        group.run_regularly('''re_init_index = get_re_init_index(w_plast,\
                                                                 N_pre,\
                                                                 N_post,\
                                                                 re_init_threshold,\
                                                                 lastspike,
                                                                 t)''',
                            dt=10 * ms)
    group.run_regularly('''w_plast = re_init_weights(w_plast,\
                                                     N_pre,\
                                                     N_post,\
                                                     re_init_index,\
                                                     re_init_threshold,\
                                                     dist_param,\
                                                     scale,\
                                                     dist)''',
                        dt=50 * ms)


def add_activity_proxy(group, buffer_size, decay):
    """Adds all needed functionality to track normalised Imem variance for
    Variance Dependent Plasticity.

    Args:
        group (Synapse group): Synapse group
        buffer_size (int, optional): Parameter to set the size of the buffer
            for activity dependent plasticity rule. Meaning how many
            samples are considered to calculate the activity proxy of post
            synaptic neurons
         decay (int, optional): Time constant for decay of exponentioally
            weighted activity proxy
    """
    if 'Imem' in group.equations.names:
        group.namespace.update({'get_activity_proxy': get_activity_proxy_imem})
        group.namespace.update({'max_value_update': max_value_update_imem})
        group.namespace.update(
            {'normalize_activity_proxy': normalize_activity_proxy_imem})
    else:
        group.namespace.update({'get_activity_proxy': get_activity_proxy_vm})
        group.namespace.update({'max_value_update': max_value_update_vm})
        group.namespace.update(
            {'normalize_activity_proxy': normalize_activity_proxy_vm})

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

    if 'Imem' in group.equations.names:
        group.run_regularly('''buffer_pointer = (buffer_pointer + 1) % buffer_size;\
        activity_proxy = get_activity_proxy(Imem, buffer_pointer, membrane_buffer, kernel)''', dt=1 * ms)
    else:
        group.run_regularly('''buffer_pointer = (buffer_pointer + 1) % buffer_size;\
        activity_proxy = get_activity_proxy(Vm, buffer_pointer, membrane_buffer, kernel)''', dt=1 * ms)

    group.run_regularly(
        '''old_max = max_value_update(activity_proxy, old_max)''', dt=5 * ms)
    group.run_regularly(
        '''normalized_activity_proxy = normalize_activity_proxy(activity_proxy, old_max)''', dt=5 * ms)


def add_weight_decay(group, decay_rate, dt):
    """Summary

    Args:
        group (Connection group): Synapse group which weight should
            be decayed.
        decay_rate (float, required): Value smaller 1 which is multiplied
            with w_plast every dt.
        dt (float, second, required): Time interval for the weight decay to
            be executed.
    """
    group.add_state_variable('decay')
    group.decay = decay_rate
    if 'w_plast' in group.equations.names:
        group.run_regularly('''w_plast *= decay''', dt=dt)
    elif 'Ipred_plast' in group.equations.names:
        group.run_regularly('''Ipred_plast *= decay''', dt=dt)

def add_lfsr(group, lfsr_seed, dt):
    """
    Initializes numbers that will be used for each element on the LFSR
    function by iterating on the LFSR num_elements times.

    Parameters
    ----------
    group : Connection group
        Group that the random numbers will be distributed to
    lfsr_seed : int
        The seed of the LFSR
    dt : float
        Time step to run_regularly
    """
    if isinstance(group, Neurons):
        num_bits = int(group.lfsr_num_bits[0])
        num_elements = len(group.lfsr_num_bits)
    else:
        num_bits = int(group.lfsr_num_bits_syn[0])
        num_elements_syn = len(group.lfsr_num_bits_syn)
        if hasattr(group, 'decay_probability_stdp'):
            num_elements_stdp = len(group.lfsr_num_bits_syn)
        else:
            num_elements_stdp = 0
        num_elements = num_elements_syn + num_elements_stdp
    lfsr_out = [0 for _ in range(num_elements)]
    mask = 2**num_bits - 1
    taps = {3: 1, 5: 2, 6: 1, 9: 4, 20: 3}

    for i in range(num_elements):
        lfsr_seed = lfsr_seed << 1
        overflow = True if lfsr_seed & (1 << num_bits) else False

        # Re-introduces 1s beyond last position
        if overflow:
            lfsr_seed |= 1

        # Ensures variable is num_bits long
        lfsr_seed = lfsr_seed & mask

        # Get bits from proper positions
        second_tap = 1 if lfsr_seed & (1 << taps[num_bits]) else 0
        first_tap = 1 if lfsr_seed & (1 << 0) else 0
        # Update bit
        lfsr_seed &=~ (1 << taps[num_bits])
        if bool(second_tap^first_tap):
            lfsr_seed |= (1 << taps[num_bits])
        lfsr_out[i] = lfsr_seed

    if isinstance(group, Neurons):
        group.decay_probability = np.asarray(lfsr_out)/2**num_bits
        group.namespace.update({'lfsr': lfsr})
        group.run_regularly('''decay_probability = lfsr(decay_probability,\
                                                         N,\
                                                         lfsr_num_bits)
                             ''',
                             dt=dt)
    else:
        group.decay_probability_syn = np.asarray(lfsr_out)[0:num_elements_syn]/2**num_bits
        group.namespace.update({'lfsr': lfsr})
        group.run_regularly('''decay_probability_syn = lfsr(decay_probability_syn,\
                                                         N,\
                                                         lfsr_num_bits_syn)
                             ''',
                             dt=dt)
        if hasattr(group, 'decay_probability_stdp'):
            group.decay_probability_stdp = np.asarray(lfsr_out)[num_elements_syn:]/2**num_bits
            group.run_regularly('''decay_probability_stdp = lfsr(decay_probability_stdp,\
                                                             N,\
                                                             lfsr_num_bits_syn)
                                 ''',
                                 dt=dt)
