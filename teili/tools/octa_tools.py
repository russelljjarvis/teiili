#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:19:57 2019

@author: matteo, mmilde
"""
import time
import numpy as np
import os
from brian2 import ms
from teili.models.parameters.octa_params import *

from teili.tools.run_regularly.add_run_reg import add_weight_decay,\
    add_pred_weight_decay, add_re_init_weights,\
    add_re_init_ipred, add_activity_proxy
"""
this file contains:
    -wrapper functions for the run regular functions
    -saving and loading functions for monitors and weights
    -weight initialization

all these functions are linked the the octa building block.
"""


def add_bb_mismatch(bb, seed=42):
    """This allows to add mismatch to all the neuron and connection groups
    present in a building block.

    args:
        bb (type): building block object to which mismatch should be added
        seed (int, optional): random seed to sample the mismatch from

    returns:
        none
    """
    for i in bb.groups:
        if bb.groups[i]._tags['group_type'] == 'neuron':
            bb.groups[i].add_mismatch(mismatch_neuron_param, seed=seed)
            bb.groups[i]._tags['mismatch'] = true
        elif bb.groups[i]._tags['group_type'] == 'Connection':
            bb.groups[i].add_mismatch(mismatch_synap_param, seed=seed)
            bb.groups[i]._tags['mismatch'] = True
        else:
            pass
    return None


def add_decay_weight(group, decay_strategy, decay_rate):
    """This allows to add a weight decay run regular function following a
    pre-defined decay strategy.

    Args:
        group (list): List of Synapse group which should be subject to weight decay
        decay_strategy (str): Weight decay strategy. Either 'global' which decays weight
                based o fixed time interval, or 'local' which performs event-driven weight decay
        decay_rate (float): Amount of weight decay per time step

    Returns:
        None
    """
    for grp in group:
        add_weight_decay(grp, decay_strategy, decay_rate)
        dict_append = {'weight decay': decay_strategy}
        if hasattr(grp, "_tags"):
            grp._tags.update(dict_append)
        else:
            self._groups[target_group]._tags = {}
            self._groups[target_group]._tags.update(dict_append)

    return None


def add_weight_pred_decay(group, decay_strategy, decay_rate):
    """ This allows to add a weight decay run regular function following a
    pre defined decay strategay.

    Args:
        group (list): List of Synapse group which should be subject to weight decay
        decay_strategy (str): Weight decay strategy. Either 'global' which decays weight
                based o fixed time interval, or 'local' which performs event-driven weight decay
        decay_rate (float): Amount of weight decay per time step

    Returns:
        None
    """
    for grp in group:
        add_pred_weight_decay(grp, decay_strategy, decay_rate)
        dict_append = {'weight decay (pred)': decay_strategy}
        if hasattr(grp, "_tags"):
            grp._tags.update(dict_append)
        else:
            self._groups[target_group]._tags = {}
            self._groups[target_group]._tags.update(dict_append)
    return None


def add_weight_re_init(group, re_init_threshold, dist_param_re_init,
                       scale_re_init, distribution):
    """This allows adding a weight re-initialization run-regular function
    specifying the distribution parameters from which to sample.

    Args:
        group (list): List of groups which are subject to weight initialization
        re_init_threshold (float): Parameter between 0 and 0.5. Threshold which
            triggers re-initialization.
        dist_param_re_init (bool): Shape of gamma distribution or mean of normal
            distribution used.
        scale_re_init (int): Scale for gamma distribution or std of normal
            distribution used.
        distribution (bool): Distribution from which to initialize the weights.
            Gamma (1) or normal (0) distributions.

    Returns:
        None
    """
    for grp in group:
        add_re_init_weights(grp,
                            re_init_threshold=re_init_threshold,
                            dist_param_re_init=dist_param_re_init,
                            scale_re_init=scale_re_init,
                            distribution=distribution)

        dict_append = {'re initializes weights' : distribution}
        if hasattr(grp, "_tags"):
            grp._tags.update(dict_append)
        else:

            self._groups[target_group]._tags = {}
            self._groups[target_group]._tags.update(dict_append)

    return None


def add_weight_re_init_ipred(group, re_init_threshold):
    """This allows to add a weight re initialization run regular function
    specifying the distribution parameters from which to sample.

    Args:
        group (list): List of groups which are subject to weight initialiazion
        re_init_threshold (float): Parameter between 0 and 0.5. Threshold which
            triggers reinitialization.

    Returns:
        None
    """
    for grp in group:
        add_re_init_ipred(grp, re_init_threshold=re_init_threshold)
        dict_append = {'re initializes weights (ipred)' : True}
        if hasattr(grp, "_tags"):
            grp._tags.update(dict_append)
        else:
            self._groups[target_group]._tags = {}
            self._groups[target_group]._tags.update(dict_append)
    return None


def add_proxy_activity(group, buffer_size, decay):
    """This allows to add an activity proxy run regular function.


    Args:
        group (list): List of neuron groups which are subject to weight initialiazion
        buffer_size (int): Size of the buffer which serves to calculate the activty
        decay (TYPE): Decay
    """
    for grp in group:
        add_activity_proxy(grp,
                           buffer_size=buffer_size,
                           decay=decay)
        dict_append = {'activity proxy' : True}
        if hasattr(grp, "_tags"):
            grp._tags.update(dict_append)
        else:
            self._groups[target_group]._tags = {}
            self._groups[target_group]._tags.update(dict_append)


def add_weight_init(group , dist_param, scale, distribution):
    """Function to add the weight initialisation to a given
    `Connections` group.

    Args:
        group (teili object): Connection group whose weights are intialised
        dist_param (float): Parameter between 0 and 0.5. Threshold which
            triggers re-initialization.
        scale (float): Scale for gamma distribution or std of normal
            distribution used.
        distribution (bool): Distribution from which to initialize the weights.
            Gamma (1) or normal (0) distributions.

    """
    for grp in group:
        grp.w_plast = weight_init(grp,
                                  dist_param=dist_param,
                                  scale=scale,
                                  distribution=distribution)
        dict_append = {'weight initialization' : distribution}
        if hasattr(grp, "_tags"):
            grp._tags.update(dict_append)
        else:
            self._groups[target_group]._tags = {}
            self._groups[target_group]._tags.update(dict_append)

class monitor_init():
    """Proxy class to initialise monitor in order to visualise activity
    using Visualizer

    Attributes:
        i (np.ndarray): Array of neuron ids.
        t (np.ndarray): Array of time stamps.
    """

    def __init__(self):
        """Creates an object monitor

        Returns:
            TYPE: Description
        """
        self.i = None
        self.t = None
        return None


def save_monitor(monitor, filename, path, variable=None):
    """Save SpikeMonitor using numpy.save()

    Args:
        monitor (brian2 monitor): Spikemonitor of brian2
        filename (str, required): String specifying the name
        path (str, required): Path/where/monitor/should/be/stored/
        variable (None, optional): Description
    """
    date = time.strftime("%d_%m_%Y")
    if not os.path.exists(path + date):
        os.mkdir(path + date + "/")
    path = path + date + "/"
    filename = time.strftime("%d_%m_%H_%M") + '_' + filename

    if variable is None:
        toSave = np.zeros((2, len(monitor.t))) * np.nan
        toSave[0, :] = np.asarray(monitor.i)
        toSave[1, :] = np.asarray(monitor.t / ms)
    else:
        toSave = np.asarray(getattr(monitor, variable))

    np.save(path + filename, toSave)


def load_monitor(filename):
    """Load a saved spikemonitor using numpy.load()

    Args:
        filename (str, required): String specifying the name

    Returns:
        monitor obj.: A monitor with i and t attributes, reflecting
            neuron index and time of spikes
    """
    data = np.load(filename)
    monitor = monitor_init()
    monitor.i = data[0, :]
    monitor.t = data[1, :] * ms
    return monitor


def save_weights(weights, filename, path):
    """Save weight matrix between to populations into .npy file

    Args:
        weights (TYPE): Description
        filename (str, required): String specifying the name
        path (str, required): Path/where/weights/should/be/stored/
    """
    date = time.strftime("%d_%m_%Y")
    if not os.path.exists(path + date):
        os.mkdir(path + date + "/")
    path = path + date + "/"
    filename = time.strftime("%d_%m_%H_%M") + '_' + filename
    toSave = np.zeros(np.shape(weights)) * np.nan
    toSave = weights
    np.save(path + filename, toSave)


def save_params(params, filename, path):
    """Save dictionary containing neuron/synapse paramters
    or simulation parameters.

    Args:
        params (dict, required): Dictionary containing parameters
            as keywords and associated values, which were needed
            for this simulation.
        filename (str, required): String specifying the name
        path (str, required): Path/where/params/should/be/stored/
    """
    date = time.strftime("%d_%m_%Y")
    if not os.path.exists(path + date):
        os.mkdir(path + date + "/")
    path = path + date + "/"
    filename = time.strftime("%d_%m_%H_%M") + '_' + filename
    np.save(path + filename, params)


def load_weights(filename=None, nrows=None, ncols=None):
    """Load weights from .npy file.

    Args:
        filename (str, optional): Absolute path from which /stored/weights.npy are loaded
        nrows (None, optional): Number of rows of the original non-flattened matrix
        ncols (None, optional): Number of columns of the original non-flattened matrix

    Returns:
        ndarray: Array containing the loaded weight matrix

    Raises:
        UserWarning: You need specify either nrows, ncols or both, otherwise reshape
            is not possible
    """
    if filename is not None:
        weight_matrix = np.load(filename)
    else:
        raise UserWarning('Please specify a filename')

    if nrows is None and ncols is None:
        raise UserWarning(
            'Please specify the dimension of the matrix, since it not squared.')
    if ncols is None:
        weight_matrix = weight_matrix.reshape((nrows, -1))
    elif nrows is None:
        weight_matrix = weight_matrix.reshape((-1, ncols))
    else:
        weight_matrix = weight_matrix.reshape((nrows, ncols))
    return weight_matrix


def weight_init(synapse_group, dist_param, scale, variable='w_plast', distribution='normal'):
    """This function initializes the weight matrix of a given synapse group
    sampled from a gamma distribution.

    Args:
        synapse_group (teili.group.Connections): A teili connection group
        dist_param (float, required): Hyperparameter specifying shape of gamma
            distribution. Also known as k. In case of normal distribution
            dist_param encodes the mean.
        scale (float, required): Hyperparameter specifying scale of gamma
            distribution. Also know as sigma.
        variable (str, optional): Description
        distribution (str, optional): Description

    Returns:
        TYPE: Description
    """

    synapse_group.namespace.update({'dist_param': dist_param})
    synapse_group.namespace.update({'scale': scale})
    weights = getattr(synapse_group, variable)
    if distribution == 'gamma':
        weights = np.random.gamma(shape=dist_param,
                                  scale=scale,
                                  size=len(synapse_group))
    if distribution == 'normal':
        weights = np.random.normal(loc=dist_param,
                                   scale=scale,
                                   size=len(synapse_group))
    weights = np.clip(weights, 0, 1)
    return weights.flatten()
