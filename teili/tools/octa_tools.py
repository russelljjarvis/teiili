#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:19:57 2019

@author: matteo
"""
import time
import numpy as np
from tkinter import filedialog
import os
from brian2 import ms
from teili.models.parameters.octa_params import *

from teili.tools.run_regularly.add_run_reg import add_weight_decay,\
add_pred_weight_decay, add_re_init_weights, add_weight_regularization,\
add_re_init_ipred, add_activity_proxy
'''
This file contains:
    -wrapper functions for the run regular functions
    -saving and loading functions for monitors and weights
    -weight initialization

All these functions are linked the the OCTA building block
'''


def add_bb_mismatch(bb):
    '''
    This allows to add mismatch to all the neuron and connection groups present in a building block

    Args:
        bb (Building Block Object, required)

    Returns:
        None
    '''
    for i in bb.groups:
        if bb.groups[i]._tags['group_type'] == 'Neuron':
            bb.groups[i].add_mismatch(mismatch_neuron_param, seed=42)
            bb.groups[i]._tags['mismatch'] = 1
        elif bb.groups[i]._tags['group_type'] == 'Connection':
            bb.groups[i].add_mismatch(mismatch_synap_param, seed= 42)
            bb.groups[i]._tags['mismatch'] = 1
        else:
            pass
    return None

def add_decay_weight(group, decay_strategy, learning_rate):
        '''
    This allows to add a weight decay run regular function following a pre defined
    decay strategay

    Args:
        groups (list, required) : list of groups that implemet

    Returns:
        None
        '''

        for grp in group:
            add_weight_decay(grp, decay_strategy, learning_rate)

        return None

def add_weight_pred_decay(group, decay_strategy, learning_rate):

        for grp in group:
            add_pred_weight_decay(grp, decay_strategy, learning_rate)
        return None

def add_weight_re_init(group, re_init_threshold, dist_param_re_init, scale_re_init,
                       distribution):

    for grp in group:
        add_re_init_weights(grp,
                            re_init_threshold=re_init_threshold,
                            dist_param_re_init=dist_param_re_init,
                            scale_re_init=scale_re_init,
                            distribution=distribution)

    return None

def add_weight_re_init_ipred(group, re_init_threshold):
    for grp in group:
        add_re_init_ipred(grp,  re_init_threshold=re_init_threshold)
    return None

def add_regulatization_weight(group, buffer_size):
    for grp in group:
        add_weight_regularization(grp,buffer_size=buffer_size)

def add_proxy_activity(group, buffer_size, decay):
    for grp in group:
        add_activity_proxy(grp,
                   buffer_size=buffer_size,
                   decay=decay)

def add_weight_init (group , dist_param, scale, distribution):
    for grp in group:
        grp.w_plast = weight_init(grp,
                                   dist_param=dist_param,
                                   scale=scale,
                                   distribution=distribution)

class monitor_init():

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
        filename = filedialog.askopenfilename()
        weight_matrix = np.load(filename)

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
