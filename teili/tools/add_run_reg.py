# -*- coding: utf-8 -*-
"""This set of functions allows the user to easily add the property of
certain run_regularly functions, specified in run_reg_functions.py to
neuron/synapse groups. These function should take a certain neuron/synapse
group and add all necessary state_variables and function calls.
"""
# @Author: mmilde
# @Date:   2018-07-30 14:19:44

from teili.core.groups import Neurons, Connections
import numpy as np
from brian2 import ms, pA, amp, second
from teili.tools.run_reg_functions import re_init_params,\
    get_activity_proxy_vm, get_activity_proxy_imem,\
    max_value_update_vm, max_value_update_imem,\
    normalize_activity_proxy_vm, normalize_activity_proxy_imem,\
    get_re_init_indices, reset_param




def add_re_init_params(group, 
                       variable, 
                       re_init_variable, 
                       re_init_indices, 
                       re_init_threshold, 
                       re_init_dt,
                       dist_param, 
                       scale, 
                       distribution,
                       sparsity,
                       reference,
                       unit,
                       clip_min,
                       clip_max):
    """Adds a re-initialization run_regularly to a synapse group

    Args:
        group (teiligroup, required): Connections or Neurons group
        variable (str, required): Name of the variable to be re-initialised
        re_init_variable (str, optional): Name of the variable to be used to
            calculate re_init_indices.
        re_init_indices (ndarray, optional): Array to indicate which parameters
            need to be re-initialised.
        re_init_threshold (float, required): Threshold value below which
            index is added to re_init_indices.
        re_init_dt (second): Dt of run_regularly.
        dist_param (float, required): Mean of distribution in case.
            of 'gaussian' or shape parameter k for 'gamma' distribution.
        scale (float, required): Scale parameter sigma for
            distribution.
        distribution (str): Parameter to determine the random
            distribution to be used to initialise the weights. Possible
            'gaussian' or 'gamma'.
        sparsity (float): Ratio of zero elements in a set of parameters.
        reference (str, required): Specifies which reference metric is used
            to get indices of parameters to be re-initialised. 'mean_weight', 
            'spike_time', 'synapse_counter' or 'neuron_threshold'.
        unit (brian2.unit):
        clip_min (float, optional): Value to clip distribution at lower bound.
        clip_max (float, optional): Value to clip distribution at upper bound.
    """
    if type(group) == Connections:
            size=len(group)
    elif type(group) == Neurons:
            size=group.N 
    
    # TODO This needs double checking. I believe the name in namespace needs
    # to match the function name itself. So we might need to remove the format.
    group.namespace.update({f're_init_{variable}': re_init_params})
    group.namespace.update({'get_re_init_indices': get_re_init_indices})
    

    if re_init_indices is None:
        group.add_state_variable('re_init_indices')
    else:
        group.variables.add_array('re_init_indices', size=np.int(size))

    group.namespace['re_init_threshold'] = re_init_threshold
    group.namespace['dist_param'] = dist_param
    group.namespace['scale'] = scale
    group.namespace['sparsity'] = sparsity

    # Mapping between keywords to avoid passing strings
    if reference == 'mean_weight':
        group.namespace['reference'] = 0
    elif reference == 'spike_time':
        group.namespace['reference'] = 1
    elif reference == 'synapse_counter':
        group.namespace['reference'] = 2

    if distribution == 'normal':
        group.namespace['dist'] = 0
    if distribution == 'gamma':
        group.namespace['dist'] = 1

    if reference == 'synapse_counter':
        temp_var = np.array(group.__getattr__(re_init_variable))
        temp_var[np.where(group.weight==0)[0]] = np.nan
        group.__setattr__(re_init_variable, temp_var)

        group.run_regularly(f'''re_init_indices = get_re_init_indices(0,\
                                   {re_init_variable},\
                                   0,\
                                   0,\
                                   reference,\
                                   re_init_threshold,\
                                   0*ms,\
                                   t)''',
                            order=0,
                            dt=re_init_dt)
        group.namespace.update({'reset_weight': reset_param})
        group.namespace.update({f'reset_{re_init_variable}': reset_param})
        group.run_regularly(f'''weight = reset_weight(weight,\
                                   {re_init_variable},\
                                   re_init_indices,\
                                   re_init_threshold,\
                                   t)''',
                            when='end',
                            dt=re_init_dt)
        group.run_regularly(f'''{re_init_variable} = reset_{re_init_variable}({re_init_variable},\
                                   0,\
                                   0,\
                                   0,\
                                   t)''',
                            when='end',
                            dt=re_init_dt)
    group.run_regularly(f'''{variable} = re_init_{variable}({variable},\
                                                        {clip_min},\
                                                        {clip_max},\
                                                        re_init_indices,\
                                                        re_init_threshold,\
                                                        dist_param,\
                                                        scale,\
                                                        dist,\
                                                        1)''',
                        order=1,
                        dt=re_init_dt)


def add_activity_proxy(group, buffer_size, decay):
    """Adds all needed functionality to track normalised Vm/Imem activity proxy 
    for Activity Dependent Plasticity.

    Args:
        group (Neuron group): Neuron group
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
    group.membrane_buffer = np.nan

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
