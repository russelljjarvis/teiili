# -*- coding: utf-8 -*-
"""To test different learning strategies and calculate proxy parameters we developed
different run_regularly functions to track the weights and update them.
The functions are all currently in numpy, but potentially can be converted into cpp.

These function include:
    re-initializing of weights
    If the summed weight per post-synatpic neuron is below a user defined threshold.

    weight regularization:
    As in conventional ANN's we need regularizer on the weights to ensure stability.
        L1 norm:
        The idea of L1 norm is favour a few weights, which has the consequence of
        most weights being zero and a few weights are large, e.g. 1.

        L2 norm:
        The ideao of L2 norm is to have many small weights. In standard neural networks
        synapses can be both, excitatory and inhibitory, which means that weights tend to be centered
        around 0 if L2 regularization is used. In the spiking neural network case synapses are either
        inhibitory or excitatory, which has the consequence that the weights tend to be centered
        around the mean, which is non-zero, e.g. 0.3.

    Correlation tracking
    Correlation tracking is used to determine if L1 or L2 weight regularization should be used.
    The idea is to estimate the slope of the cross-correlation coefficient for each post-synaptic
    neuron in a  small time window, which is set by the buffer. If the slope tends to become zero
    the weight regularization switches from L2 (exploration) to L1 (facilitation).

    Membrane variance
    The membrane variance is used to determine if the the inhibitory weight should be increased or decreased.
    The intuition behind this is that the, since neurons are recurrently connected each post-syaptic membrane
    variance signals a high correlated activity. In order to diversify the firing pattern inh. synapses connected
    to neurons with a high membrane variance is increased, lower their variance. On the contrary, if the membrane
    variance is to low, the inhibitory weights should be decreased to allow those neurons to pick up uncovered
    input statistics.

    normalization of the membrane variance
    The normalization is used to make the comparison between neurons easier and set a threshold independent
    of the unit and between 0 and 1.

Example:


"""
# @Author: Moritz Milde
# @Date:   2018-06-22 10:05:24

import numpy as np
from brian2 import implementation, check_units, amp, pA


@implementation('numpy', discard_units=True)
@check_units(weights=1, source_N=1, target_N=1, re_init_threshold=1, dist_param=1, scale=1, dist=1, result=1)
def re_init_weights(weights, source_N, target_N, re_init_threshold=0.2, dist_param=0.4, scale=0.2, dist=0):
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

        >>> # Now we can connect and initialize the weight matrix
        >>> syn_obj.connect('True')
        >>> syn_obj.weight = wtaParams['weInpWTA']
        >>> syn_obj.namespace.update({'dist_param': dist_param})
        >>> syn_obj.namespace.update({'scale': scale_init})

        >>> # Now we can add the run_regularly function
        >>> syn_obj.namespace.update({'re_init_weights': re_init_weights})
        >>> syn_obj.namespace.update({'re_init_threshold': re_init_threshold})
        >>> syn_obj.namespace['dist_param'] = dist_param_re_init
        >>> syn_obj.namespace['scale'] = scale_re_init
        >>> syn_obj.run_regularly('''w_plast = re_init_weights(w_plast, N_pre,
                                                               N_post,
                                                               re_init_threshold,
                                                               dist_param, scale)''',
                                                               dt=2000*ms)

    Args:
        weights (np.ndarray. required): Flattened weight matrix.
        source_N (int, required): Number of units in the source population.
        target_N (int, required): Number of units in the target population.
        re_init_threshold (float, optional): Re-initialization threshold.
        dist_param (float, optional): Shape factor of gamma, or mean of
            normal distribution from which weights are sampled.
        scale (float, optional): Scale factor of gamma distribution from which weights are sampled.

    Returns:
        ndarray: Flatten re-initialized weight matrix
    """
    data = np.zeros((source_N, target_N)) * np.nan
    data = np.reshape(weights, (source_N, target_N))
    # Summing weights
    # summed_weights = np.sum(data, 0)
    # Thresholding post-synaptic weights
    reinit_index = np.logical_or(np.mean(data, 0) < re_init_threshold,
                                 np.mean(data, 0) > (1 - re_init_threshold))

    # Re-initializing weights with normal distribution
    if dist == 1:
        data[:, reinit_index] = np.reshape(np.random.gamma(shape=dist_param, scale=scale,
                                                           size=source_N * np.sum(reinit_index)),
                                           (source_N, np.sum(reinit_index)))
    if dist == 0:
        data[:, reinit_index] = np.reshape(np.random.normal(loc=dist_param, scale=scale,
                                                            size=source_N * np.sum(reinit_index)),
                                           (source_N, np.sum(reinit_index)))
    data = np.clip(data, 0, 1)
    return data.flatten()


@implementation('numpy', discard_units=True)
@check_units(Ipred_plast=1, source_N=1, target_N=1, re_init_threshold=1, result=1)
def re_init_ipred(Ipred_plast, source_N, target_N, re_init_threshold=0.2):
    data = np.zeros((source_N, target_N)) * np.nan
    data = np.reshape(Ipred_plast, (source_N, target_N))

    reinit_index = np.mean(data, 1) > (1 - re_init_threshold)

    data[reinit_index, :] = np.zeros((np.sum(reinit_index), target_N))
    data = np.clip(data, 0, 1)
    return data.flatten()


@implementation('numpy', discard_units=True)
@check_units(Imem=amp, buffer_pointer=1, membrane_buffer=1, kernel=1, result=amp)
def get_activity_proxy(Imem, buffer_pointer, membrane_buffer, kernel):
    """This function calculates an activity proxy using an integrated, exponentially
    weighted estimate of Imem of the N last time steps and stores it in membrane_buffer.

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
        buffer_pointer (TYPE): Description.
        membrane_buffer (TYPE): Description.
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

    # membrane_variance = np.sum(
    #     np.diff(membrane_buffer, axis=1)**2, axis=1)
    # membrane_variance = np.var(np.diff(membrane_buffer, axis=1)**2, axis=1)
    # membrane_variance = np.var(membrane_buffer, axis=1)

    '''Exponential weighing the membrane buffer to reflect more recent
    fluctuations in Imem. The exponential kernel is choosen to weight
    the most recent activity with a weight of 1, so we can normalize using
    the Ispkthr variable.
    '''
    # exp_weighted_membrane_buffer = np.zeros(np.shape(membrane_buffer)) * np.nan
    # exp_weighted_membrane_buffer = np.diff(np.array(membrane_buffer, copy=True), axis=1)**2
    exp_weighted_membrane_buffer = np.array(membrane_buffer, copy=True)
    exp_weighted_membrane_buffer *= kernel[:, :np.shape(exp_weighted_membrane_buffer)[1]]

    activity_proxy = np.sum(exp_weighted_membrane_buffer, axis=1)
    return activity_proxy


@implementation('numpy', discard_units=True)
@check_units(activity_proxy=amp, old_max=1, result=1)
def max_value_update(activity_proxy, old_max):
    if (np.max(activity_proxy / pA) > old_max[0]):
        old_max[0] = np.array(np.max(activity_proxy / pA), copy=True)
    return old_max[0]


@implementation('numpy', discard_units=True)
@check_units(activity_proxy=amp, old_max=1, result=1)
def normalize_activity_proxy(activity_proxy, old_max):
    """This function normalized the variance of Imem, as calculated
    by get_membrane_variance.

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

        >>> neuron_obj.run_regularly('''normalized_membrane_var = normalize_membrane_variance(membrane_variance, Ispkthr)''',
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
    # if np.max((Imem_var)) == 0 or np.max((Imem_var / pA)) == np.min((Imem_var / pA)):
    #     normalized_Imem_var = np.zeros(np.shape(Imem_var))
    # else:
    #     normalized_Imem_var = ((Imem_var / pA) - np.min((Imem_var / pA))) / (np.max((Imem_var / pA)) - np.min((Imem_var / pA)))
    #     # normalized_Imem_var = (Imem_var / pA) / np.max((Imem_var / pA))

    return normalized_activity_proxy


@implementation('numpy', discard_units=True)
@check_units(weights=1, source_N=1, target_N=1, buffer_pointer_norm=1, mean_variance_time=1, result=1)
def correlation_coefficient_tracking(weights, source_N, target_N, buffer_pointer_norm, mean_variance_time):
    """This function keeps track of the average variance of the average correlation coefficient
    analysis of a given matrix.

    This function fills a buffer in a sliding window fashion, which is needed by
    the weight_regularization function.

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

        >>> syn_obj.namespace.update({'correlation_coefficient_tracking': correlation_coefficient_tracking})
        >>> syn_obj.add_state_variable('buffer_size', shared=True, constant=True)
        >>> syn_obj.add_state_variable('buffer_pointer_norm', shared=True, constant=True)

        >>> syn_obj.buffer_size = octaParams['buffer_size']
        >>> syn_obj.buffer_pointer_norm = -1
        >>> syn_obj.variables.add_array('mean_variance_time', size=octaParams['buffer_size'])
        >>> syn_obj.run_regularly('''buffer_pointer_norm = (buffer_pointer_norm + 1) % buffer_size
                                  mean_variance_time = correlation_coefficient_tracking(w_plast,
                                                                                        N_pre,
                                                                                        N_post,
                                                                                        buffer_pointer_norm,
                                                                                        mean_variance_time)''',
                                  dt=100*ms)
    Args:
        weights (np.ndarray. required): Flattened weight matrix.
        source_N (int, required): Number of units in the source population.
        target_N (int, required): Number of units in the target population.
        buffer_pointer_norm (int): The index/position of the mean_variance_time buffer.
        mean_variance_time (array): With size buffer_size, which needs to be
            specified outside this function (see example above). This buffer stores
            mean of the varaince of the sorted correlation coefficients of a given
            weight matrix, which is needed to switch between
            L1 & L2 regularization.

    Returns:
        array: Array which stores N=buffer_size average Imem variances.
    """
    data = np.zeros((source_N, target_N)) * np.nan
    data = np.reshape(weights, (source_N, target_N))
    R = np.corrcoef(data.T)
    indices = np.shape(data.T)[0]
    R_sorted = np.zeros((indices, indices)) * np.nan
    for index in range(indices):
        R_sorted[index, :] = sorted(R[index])

    R_variance = np.var(R_sorted, axis=1)
    buffer_pointer_norm = int(buffer_pointer_norm)
    if np.sum(mean_variance_time == 0) > 0:
        mean_variance_time[buffer_pointer_norm] = np.mean(R_variance)
    else:
        mean_variance_time[:-1] = mean_variance_time[1:]
        mean_variance_time[-1] = np.mean(R_variance)

    return mean_variance_time
