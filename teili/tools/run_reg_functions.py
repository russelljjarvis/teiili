# -*- coding: utf-8 -*-
""" A collection of run_regular functions including weight_initialization,
weight_normalization, adaptive thresholds etc.

For more details see below.
"""
# @Author: Moritz Milde
# @Date:   2018-06-22 10:05:24

import numpy as np
from brian2 import implementation, check_units,\
    amp, pA, second, ms, volt, mV


@implementation('numpy', discard_units=True)
@check_units(weights=1,
             source_N=1,
             target_N=1,
             re_init_index=1,
             re_init_threshold=1,
             dist_param=1,
             scale=1,
             dist=1,
             result=1)
def re_init_weights(weights,
                    source_N,
                    target_N,
                    re_init_index=None,
                    re_init_threshold=0.2,
                    dist_param=0.4,
                    scale=0.2,
                    dist=0):
    """Re-initializes a given weight matrix using a normal distribution with
    a specified mean and standard deviation if the mean is below a user
    defined threshold.

    Example:
        >>> from teili.core.groups import Neurons, Connections
        >>> from teili.models.neuron_models import DPI as neuron_model
        >>> from teili.models.synapse_models import DPISyn as syn_model

        >>> neuron_obj = Neurons(2, equation_builder=neuron_model(num_inputs=2),
                                 name="neuron_obj")
        >>> syn_obj = Connections(neuron_obj,
                                  neuron_obj,
                                  equation_builder=syn_model(),
                                  method='euler',
                                  name='syn_obj')

        >>> # Define parameters such as mean and stamdard deviation
        >>> dist_param_init = 0.5
        >>> scale_init = 1.0
        >>> sdist_param_re_init = 0.5
        >>> scale_re_init = 1.0
        >>> re_init_threshold = 0.2
        >>> re_init_index = None

        >>> # Now we can connect and initialize the weight matrix
        >>> syn_obj.connect('True')
        >>> syn_obj.weight = wtaParams['weInpWTA']
        >>> syn_obj.namespace.update({'dist_param': dist_param})
        >>> syn_obj.namespace.update({'scale': scale_init})
        >>> syn_obj.namespace.update({'re_init_index: re_init_index})

        >>> # Now we can add the run_regularly function
        >>> syn_obj.namespace.update({'re_init_weights': re_init_weights})
        >>> syn_obj.namespace.update({'re_init_threshold': re_init_threshold})
        >>> syn_obj.namespace['dist_param'] = dist_param_re_init
        >>> syn_obj.namespace['scale'] = scale_re_init
        >>> syn_obj.run_regularly('''w_plast = re_init_weights(w_plast,\
                                                               N_pre,\
                                                               N_post,\
                                                               re_init_index,\
                                                               re_init_threshold,\
                                                               dist_param,\
                                                               scale)''',
                                                               dt=2000*ms)

    Args:
        weights (np.ndarray. required): Flattened weight matrix.
        source_N (int, required): Number of units in the source population.
        target_N (int, required): Number of units in the target population.
        re_init_index (array, bool, optional): Boolean index array
            indicating which weights need to be re-initilized. If None
            weights are updated based on average post-synaptic weight.
            re_init_index can be obtained using get_re_init_index
            run_regularly.
        re_init_threshold (float, optional): Re-initialization threshold.
        dist_param (float, optional): Shape factor of gamma, or mean of
            normal distribution from which weights are sampled.
        scale (float, optional): Scale factor of gamma distribution from
            which weights are sampled.

    Returns:
        ndarray: Flatten re-initialized weight matrix
    """
    data = np.zeros((source_N, target_N)) * np.nan
    data = np.reshape(weights, (source_N, target_N))
    # Thresholding post-synaptic weights
    if re_init_index is None or len(re_init_index) != target_N:
        re_init_index = np.logical_or(np.mean(data, 0) < re_init_threshold,
                                      np.mean(data, 0) > (1 - re_init_threshold),
                                      dtype=bool)
    # Re-initializing weights with normal distribution
    if dist == 1:
        data[:, re_init_index.astype(bool)] = np.reshape(np.random.gamma(
            shape=dist_param,
            scale=scale,
            size=np.int(source_N * np.sum(re_init_index))),
            (np.int(source_N), np.sum(re_init_index).astype(int)))
    if dist == 0:
        data[:, re_init_index.astype(bool)] = np.reshape(np.random.normal(
            loc=dist_param,
            scale=scale,
            size=int(source_N * np.sum(re_init_index))),
            (np.int(source_N), np.sum(re_init_index).astype(int)))
    data = np.clip(data, 0, 1)
    return data.flatten()


@implementation('numpy', discard_units=True)
@check_units(taus=second,
             re_init_index=1,
             source_N=1,
             target_N=1,
             dist_param=1,
             scale=1,
             dist=1,
             result=second)
def re_init_taus(taus,
                 re_init_index,
                 source_N, target_N,
                 dist_param=0.4,
                 scale=0.2,
                 dist=0):
    """Re-initializes a given synaptic time constants matrix using a normal
    distribution with a specified mean and standard deviation given
    a re_init_index vector.

    Args:
        taus (np.ndarray. required): Flattened synaptic time constant
            matrix.
        re_init_index (np.ndarray, required): Boolean array of rows to be
            updated in matrix
        source_N (int, required): Number of units in the source population.
        target_N (int, required): Number of units in the target population.
        dist_param (float, optional): Shape factor of gamma, or mean of
            normal distribution from which weights are sampled.
        scale (float, optional): Scale factor of gamma distribution from
            which weights are sampled.

    Returns:
        ndarray: Flatten re-initialized tau matrix
    """
    data = np.zeros((source_N, target_N)) * np.nan
    data = np.reshape(taus / ms, (source_N, target_N))
    # Re-initializing weights with normal distribution
    if dist == 1:
        data[:, re_init_index] = np.reshape(np.random.gamma(
            shape=dist_param,
            scale=scale,
            size=source_N * np.sum(re_init_index)),
            (source_N, np.sum(re_init_index)))
    if dist == 0:
        data[:, re_init_index] = np.reshape(np.random.normal(
            loc=dist_param,
            scale=scale,
            size=source_N * np.sum(re_init_index)),
            (source_N, np.sum(re_init_index)))
    data = np.clip(data, 0.05, 200)
    return data.flatten() * ms


@implementation('numpy', discard_units=True)
@check_units(Imem=amp,
             buffer_pointer=1,
             membrane_buffer=1,
             kernel=1,
             result=amp)
def get_activity_proxy_imem(Imem,
                            buffer_pointer,
                            membrane_buffer,
                            kernel):
    """This function calculates an activity proxy using an integrated,
    exponentially weighted estimate of Imem of the N last time steps and
    stores it in membrane_buffer.

    This is needed for Variance Dependent Plasticity of inhibitory
    synapse.

    Example:
        >>> from teili.core.groups import Neurons, Connections
        >>> from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
        >>> from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder

        >>> DPIvar = NeuronEquationBuilder.import_eq('octa/models/equations/DPIvar',
                                                         num_inputs=2)
        >>> DPIvdp = SynapseEquationBuilder.import_eq('octa/models/equations/DPIvdp')

        >>> neuron_obj = Neurons(2, equation_builder=DPIvar_obj,
                                 name="neuron_obj")
        >>> syn_obj = Connections(neuron_obj,
                                  neuron_obj,
                                  equation_builder=DPIvdp,
                                  method='euler',
                                  name='syn_obj')

        >>> neuron_obj.namespace.update({'get_membrane_variance': get_membrane_variance})
        >>> neuron_obj.namespace.update({'normalize_membrane_variance': normalize_membrane_variance})
        >>> neuron_obj.add_state_variable('buffer_size', shared=True, constant=True)
        >>> neuron_obj.add_state_variable('buffer_pointer', shared=True, constant=True)
        >>> neuron_obj.namespace.update({'decay': decay})

        >>> neuron_obj.buffer_size = octaParams['buffer_size_plast']
        >>> neuron_obj.buffer_pointer = -1
        >>> neuron_obj.variables.add_array('membrane_buffer', size=(neuron_obj.N,
                                        octaParams['buffer_size_plast']))
        >>> # Now we can initialize the buffer_pointer and add the run regularly function
        >>> neuron_obj.run_regularly('''buffer_pointer = (buffer_pointer + 1) % buffer_size
            activity_proxy = get_activity_proxy(Imem, buffer_pointer, membrane_buffer, decay)''', dt=1*ms)

    Args:
        Imem (float): The membrane current of the DPI neuron model.
        buffer_pointer (int): Pointer to keep track of the buffer
        membrane_buffer (numpy.ndarray): Ring buffer for membrane potential
        decay (int): Time constant of exp. weighted decay

    Returns:
        neuron_obj.array: An array which holds the integral of the
            activity_proxy over the N time steps of the
    """
    buffer_pointer = int(buffer_pointer)
    if np.sum(membrane_buffer == 0) > 0:
        membrane_buffer[:, buffer_pointer] = Imem
        # kernel = np.zeros(np.shape(membrane_buffer)) * np.nan
    else:
        membrane_buffer[:, :-1] = membrane_buffer[:, 1:]
        membrane_buffer[:, -1] = Imem

    '''Exponential weighing the membrane buffer to reflect more recent
    fluctuations in Imem. The exponential kernel is choosen to weight
    the most recent activity with a weight of 1, so we can normalize using
    the Ispkthr variable.
    '''
    exp_weighted_membrane_buffer = np.array(membrane_buffer, copy=True)
    exp_weighted_membrane_buffer *= kernel[:, :np.shape(exp_weighted_membrane_buffer)[1]]

    activity_proxy = np.sum(exp_weighted_membrane_buffer, axis=1)
    return activity_proxy


@implementation('numpy', discard_units=True)
@check_units(Vm=volt,
             buffer_pointer=1,
             membrane_buffer=1,
             kernel=1,
             result=volt)
def get_activity_proxy_vm(Vm,
                          buffer_pointer,
                          membrane_buffer,
                          kernel):
    """This function calculates an activity proxy using an integrated,
    exponentially weighted estimate of Imem of the N last time steps and
    stores it in membrane_buffer.

    This is needed for Variance Dependent Plasticity of inhibitory
    synapse.

     Args:
        Vm (float): The membrane potential of the LIF neuron model.
        buffer_pointer (int): Pointer to keep track of the buffer
        membrane_buffer (numpy.ndarray): Ring buffer for membrane potential.
        decay (int): Time constant of exp. weighted decay

    Returns:
        neuron_obj.array: An array which holds the integral of the
            activity_proxy over the N time steps of the
    """
    buffer_pointer = int(buffer_pointer)
    if np.sum(membrane_buffer == 0) > 0:
        membrane_buffer[:, buffer_pointer] = Vm
        # kernel = np.zeros(np.shape(membrane_buffer)) * np.nan
    else:
        membrane_buffer[:, :-1] = membrane_buffer[:, 1:]
        membrane_buffer[:, -1] = Vm

    '''Exponential weighing the membrane buffer to reflect more recent
    fluctuations in Imem. The exponential kernel is choosen to weight
    the most recent activity with a weight of 1, so we can normalize using
    the Ispkthr variable.
    '''
    exp_weighted_membrane_buffer = np.array(membrane_buffer, copy=True)
    exp_weighted_membrane_buffer *= kernel[:, :np.shape(exp_weighted_membrane_buffer)[1]]

    activity_proxy = np.sum(exp_weighted_membrane_buffer, axis=1)
    return activity_proxy



@implementation('numpy', discard_units=True)
@check_units(activity_proxy=volt, old_max=1, result=1)
def max_value_update_vm(activity_proxy, old_max):
    """ This run_regularly simply calculates the maximum value of the
    activity proxy to normalize it accordingly.

    Args:
        activity_proxy(numpy.ndarray): Array containing the exponentially
            weights membrane potential traces.
        old_max(float): Maximum value of activity_proxy since the start
            of the simulation.

    Returns:
        old_max (float): Updated maximum value

    """
    if (np.max(activity_proxy / mV) > old_max[0]):
        old_max[0] = np.array(np.max(activity_proxy / mV), copy=True)
    return old_max[0]


@implementation('numpy', discard_units=True)
@check_units(activity_proxy=amp, old_max=1, result=1)
def max_value_update_imem(activity_proxy, old_max):
    """ This run_regularly simply calculates the maximum value of the
    activity proxy to normalize it accordingly.

    Args:
        activity_proxy(numpy.ndarray): Array containing the exponentially
            weights membrane potential traces.
        old_max(float): Maximum value of activity_proxy since the start
            of the simulation.

    Returns:
        old_max (float): Updated maximum value
    """
    if (np.max(activity_proxy / pA) > old_max[0]):
        old_max[0] = np.array(np.max(activity_proxy / pA), copy=True)
    return old_max[0]


@implementation('numpy', discard_units=True)
@check_units(activity_proxy=amp, old_max=1, result=1)
def normalize_activity_proxy_imem(activity_proxy, old_max):
    """This function normalized the activity proxy of Imem, as calculated
    by get_activity_proxy_imem.

    Example:
        >>> from teili.core.groups import Neurons, Connections
        >>> from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
        >>> from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder

        >>> DPIvar = NeuronEquationBuilder.import_eq('octa/models/equations/DPIvar',
                                                         num_inputs=2)
        >>> DPIvdp = SynapseEquationBuilder.import_eq('octa/models/equations/DPIvdp')

        >>> neuron_obj = Neurons(2, equation_builder=DPIvar_obj,
                                 name="neuron_obj")
        >>> syn_obj = Connections(neuron_obj,
                                  neuron_obj,
                                  equation_builder=DPIvdp,
                                  method='euler',
                                  name='syn_obj')

        >>> neuron_obj.run_regularly('''normalized_activity_proxy=normalize_activity_proxy(activity_proxy, old_max)''',
                                     dt=5*ms)
    Args:
        activity_proxy (float): exponentially weighted Imem fluctuations
        old_max (float, amps): Value used to normalize the
            activity proxy variable

    Returns:
        float: Normalized activity proxy
    """
    if old_max == 0.0:
        normalized_activity_proxy = np.zeros(np.shape(activity_proxy))
    else:
        normalized_activity_proxy = (activity_proxy / pA) / old_max

    return normalized_activity_proxy


@implementation('numpy', discard_units=True)
@check_units(activity_proxy=volt, old_max=1, result=1)
def normalize_activity_proxy_vm(activity_proxy, old_max):
    """This function normalized the variance of Vm, as calculated
    by get_activity_proxy_vm.

    Args:
        activity_proxy (float): exponentially weighted Imem fluctuations
        old_max (float, volt): Value used to normalize the
            activity proxy variable

    Returns:
        float: Normalized activity proxy
    """
    if old_max == 0.0:
        normalized_activity_proxy = np.zeros(np.shape(activity_proxy))
    else:
        normalized_activity_proxy = (activity_proxy / mV) / old_max

    return normalized_activity_proxy



@implementation('numpy', discard_units=True)
@check_units(weights=1,
             source_N=1,
             target_N=1,
             normalization_factor=1,
             upper_bound=1,
             lower_bound=1,
             result=1)
def weight_normalization(weights,
                         source_N,
                         target_N,
                         normalization_factor=0.1,
                         upper_bound=.75,
                         lower_bound=.25):
    """This run regular function calculates the sum of the weights and
    scales all weights up or down if the sum is larger or smaller than the
    75 % or 25 % of the total possiblr maximum weight.

    Conceptually this function can be understood as stumulatin all synapses
    at the same time and performing a global operation on the weights if the
    total sum of synaptic weights fall out of the boundaries.

    Args:

    Returns:
    """
    data = np.reshape(weights, (source_N, target_N))
    sum_of_weights = np.sum(data, 1) / source_N
    try:
        data[:, sum_of_weights < lower_bound] *= (1 + normalization_factor)
        data[:, sum_of_weights > upper_bound] *= (1 - normalization_factor)
    except IndexError:
        data[sum_of_weights < lower_bound] *= (1 + normalization_factor)
        data[sum_of_weights > upper_bound] *= (1 - normalization_factor)
    return data.flatten()


@implementation('numpy', discard_units=True)
@check_units(weights=1,
             source_N=1,
             target_N=1,
             re_init_threshold=1,
             lastspike=second,
             t=second,
             result=1)
def get_re_init_index(weights,
                      source_N,
                      target_N,
                      re_init_threshold,
                      lastspike,
                      t):
    """ This function provides a boolean vector to indicate if the parameters
    of a given neuron need to be updated. This is required if two or more
    variables of a neuron needs to be updated according to the same condition
    as e.g. in structural plasticity, where the weight values trigger the
    respawning of synapses. To ensure that the synaptic time constants are
    resampled accordingly a boolean vector is required to indicate the update,
    as brian2 does not support the return of more than one variable from a
    run_regularly function.

    Args:
        weights (numpy.ndarray):
        source_N (int, required):
        target_N (in_requiredm):
        re_init_threshold (float, required):
        lastspike (float, second, required):
        t (float, second, required):

    Returns:
        re_init_index (numpy.ndarray): Boolean vector indicating which post-
            synaptic neuron will be subject to re-sampling of its free
            parameters.

    """
    data = np.zeros((source_N, target_N))
    data = np.reshape(weights, (source_N, target_N))
    re_init_index = np.mean(data, 0) < re_init_threshold
    lastspike_tmp = np.reshape(lastspike, (source_N, target_N))
    if (lastspike < 0*second).any() and (np.sum(lastspike_tmp[0, :] < 0 * second) > 2):
        re_init_index = np.any(lastspike_tmp < 0 * second, axis=0)
    elif ((t - np.abs(lastspike_tmp[0, :])) > (1 * second)).any():
        re_init_index = np.any((t - lastspike_tmp) > (1 * second), axis=0)

    return re_init_index


@implementation('numpy', discard_units=True)
@check_units(Vthr=volt,
             Vm=volt,
             EL=volt,
             VT=volt,
             sigma_membrane=volt,
             not_refractory=1,
             result=volt)
def update_threshold(Vthr, Vm, EL, VT, sigma_membrane, not_refractory):
    """ A function whoch based on the deviation of the membrane potential
    from its equilibrium potential adjusts the firing threshold.
    The idea behind this run_regular function was published by
    Afshar et al. (2019): Event-based Feature Extraction Using Adaptive
    Selection Thresholds. https://arxiv.org/pdf/1907.07853.pdf

    Args:
        Vthr (array, volt): Spike thresholds.
        Vm (array, volt): Membrane potential
        VT (float, volt): Trigger voltage for Iexp.
        sigma_membrane (float, volt): Increment/Decrement of the spike
            threshold.
        not_refractory (array, bool): Boolean vector indicating if neuron
            is currently in its refractory period.

    Returns:
        data (array, volt): Updated spiking thresholds.

    Important note:
        DOES NOT WRK FOR DPI NEURON
    """
    data = np.zeros((len(Vthr))) * np.nan
    data = np.asarray(Vthr / mV)

    data += (((sigma_membrane / mV) * -1) * (Vm > (EL+1*mV)) +\
             ((sigma_membrane / mV)* 5 * (Vm < (EL-1*mV))) * not_refractory)

    return data * mV
