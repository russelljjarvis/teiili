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
@check_units(params=1,
             clip_min=1,
             clip_max=1,
             re_init_indices=1,
             re_init_threshold=1,
             dist_param=1,
             scale=1,
             dist=1,
             unit=1,
             result=1)
def re_init_params(params,
                   clip_min=None,
                   clip_max=None,
                   re_init_indices=None,
                   re_init_threshold=None,
                   dist_param=0.4,
                   scale=0.2,
                   dist=0,
                   unit=None):
    """Re-initializes a given parameter, e.g. weights or time constants,
    using a normal or gamma distribution with a specified mean and standard 
    deviation re-initilaisation indices.

    Example:
        >>> from teili.core.groups import Neurons, Connections
        >>> from teili.models.neuron_models import DPI as neuron_model
        >>> from teili.models.synapse_models import DPISyn as syn_model
        >>> from teili.tools.run_regfunctions import re_init_params

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
        >>> re_init_indices = None
        >>> clip_min = 0
        >>> clip_max = 1
        >>> variable = "w_plast"
        >>> re_init_dt= 2000 * ms

        >>> # Now we can connect and initialize the weight matrix
        >>> syn_obj.connect('True')
        >>> syn_obj.weight = wtaParams['weInpWTA']
        >>> syn_obj.namespace.update({'dist_param': dist_param})
        >>> syn_obj.namespace.update({'scale': scale_init})
        >>> syn_obj.namespace.update({'clip_min': clip_min})
        >>> syn_obj.namespace.update({'clip_max': clip_max})
        >>> syn_obj.namespace.update({'re_init_indices: re_init_indices})

        >>> # Now we can add the run_regularly function
        >>> syn_obj.namespace.update({'re_init_weights': re_init_params})
        >>> syn_obj.namespace.update({'re_init_threshold': re_init_threshold})
        >>> syn_obj.namespace['dist_param'] = dist_param_re_init
        >>> syn_obj.namespace['scale'] = scale_re_init
        >>> syn_obj.run_regularly('''syn_obj.__setattr__(variable, re_init_params(syn_obj.__getattr__(variable),\
                                                               clip_min,\
                                                               clip_max,\
                                                               re_init_indices,\
                                                               re_init_threshold,\
                                                               dist_param,\
                                                               scale,
                                                               dist)''',
                                                               dt=re_init_dt)

    Args:
        params (np.ndarray. required): Flattened parameter vector.
        clip_min (float, optional): Value to clip distrubtion at lower bound.
        clip_max (float, optional): Value to clip distribution at upper bound.
        re_init_indices (vector, bool, optional): Boolean index array
            indicating which parameters need to be re-initilised. If None
            parameters are updated based on average lower and upper 20 %.
            re_init_indices can be obtained using get_re_init_indices
            run_regularly.
        re_init_threshold (float, optional): Re-initialisation threshold. 
            Default is 0.2, i.e. 20 %
        dist_param (float, optional): Shape factor of gamma, or mean of
            normal distribution from which weights are sampled.
        scale (float, optional): Scale factor of gamma distribution from
            which weights are sampled.
        dist (int, required): Flag to use either normal distribution (0)
            or gamma distribution (1).
        unit (brain2.unit, optional): Unit of parameter to re-initialise

    Returns:
        ndarray: Flatten re-initialized weight matrix
    """
    data = np.zeros(len(re_init_indices)) * np.nan

    if re_init_indices is None:
        re_init_indices = np.logical_or(np.mean(params, 0) < re_init_threshold,
                                        np.mean(params, 0) > (1 - re_init_threshold),
                                        dtype=bool)


    if dist == 1:
        data[re_init_indices.astype(bool)] = np.random.gamma(
            shape=dist_param,
            scale=scale,
            size=np.int(len(re_init_indices)))
    elif dist == 0:
        data[re_init_indices.astype(bool)] = np.random.normal(
            loc=dist_param,
            scale=scale,
            size=np.int(len(re_init_indices)))

    if clip_min is not None and clip_max is not None:
        data = np.clip(data, clip_min, clip_max)

    if unit is None:
        return data.flatten()
    else:
        return data.flatten() * unit


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
@check_units(params=1,
             source_N=1,
             target_N=1,
             reference=1,
             re_init_threshold=1,
             lastspike=second,
             t=second,
             result=1)
def get_re_init_indices(params,
                       source_N,
                       target_N,
                       reference,
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
        params (numpy.ndarray):
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
    data = params
    re_init_indices = np.zeros(len(params))


    if reference == 'mean_weight':
        re_init_indices[np.mean(data, 0) < re_init_threshold] = 1
    elif reference == 'spike_time':
        lastspike_tmp = np.reshape(lastspike, (source_N, target_N))
        if (lastspike < 0*second).any() and (np.sum(lastspike_tmp[0, :] < 0 * second) > 2):
            re_init_indices[np.any(lastspike_tmp < 0 * second, axis=0)] = 1
        elif ((t - np.abs(lastspike_tmp[0, :])) > (1 * second)).any():
            re_init_indices[np.any((t - lastspike_tmp) > (1 * second), axis=0)] = 1
    elif reference == 'synapse_counter':
        # @pablo, please add your code here
        pass
    elif reference == 'neuron_threshold':
        # @pablo add your code here
        pass

    return re_init_indices


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

@implementation('numpy', discard_units=True)
@check_units(decay_probability=1, num_elements=1, lfsr_num_bits=1, mask=1, result=1)
def lfsr(decay_probability, num_elements, lfsr_num_bits, mask):
    """
    Generate a pseudorandom number between 0 and 1 with a Linear
    Feedback Shift Register (LFSR), which is equivalent to generating random
    numbers from an uniform distribution. This is a Galois or many-to-one
    implementation.

    This function receives a given number and performs num_elements iterations
    of the LFSR. This is done when a given neuron needs another random number.
    The LFSR does a circular shift (i.e. all the values are shifted left while
    the previous MSB becomes the new LSB) and ensures the variable is no bigger
    than the specified number of bits. Note that, for convenience,
    the input and outputs are normalized, i.e. value/2**lfsr_num_bits.

    In practice, when there is not overflow, the operation is a normal shift.
    Otherwise, a XOR operation between the mask and the shifted value is
    necessary to send MSB to LSB and update relative taps.

    Parameters
    ----------
    decay_probability : float
        Value between 0 and 1 that will be the input to the LFSR
    num_elements : int
        Number of neurons in the group
    lfsr_num_bits : int
        Number of bits of the LFSR
    mask : int
        Value to be used in XOR operation, depending on number of bits used

    Returns
    -------
    float
        A random number between 0 and 1

    Examples
    --------
    >>> from teili.tools.run_reg_functions import lfsr
    >>> number = 2**4 + 2**3 + 2**1
    >>> n_bits = 5
    >>> bin(number)
    '0b11010'
    >>> bin(int(lfsr([number/2**n_bits], 1, [n_bits], 0b100101)*2**n_bits))
    '0b10001'
    """
    try:
        lfsr_num_bits = int(lfsr_num_bits[0])
        seed = int(decay_probability[-1]*2**lfsr_num_bits)
    except:
        print('Please use numpy target')
        quit()

    updated_probabilities = [0 for _ in range(num_elements)]
    mask = int(mask)

    for i in range(num_elements):
        seed = seed << 1
        overflow = seed >> lfsr_num_bits
        if overflow:
            seed ^= mask
        updated_probabilities[i] = seed

    return np.array(updated_probabilities)/2**lfsr_num_bits
