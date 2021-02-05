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
from numpy.core.fromnumeric import var
from numpy.linalg.linalg import _raise_linalgerror_eigenvalues_nonconvergence

from teili import Neurons, Connections
from teili.tools.add_run_reg import add_weight_decay,\
    add_re_init_params, add_activity_proxy
"""
this file contains:
    -wrapper functions for the run regular functions
    -saving and loading functions for monitors and weights
    -weight initialization

all these functions are linked the the octa building block.
"""

def add_group_weight_decay(groups, decay_rate, dt):
    """This allows to add a weight decay run regular function following a
    pre-defined decay strategy.

    Args:
        group (list): List of Synapse group which should be subject to
            weight decay
        decay_rate (float): Amount of weight decay per time step.
        dt (float, second): Time step of run regularly.

    Returns:
        None
    """
    for group in groups:
        add_weight_decay(group, decay_rate, dt)
        dict_append = {'weight decay': 'clock-driven'}
        group._tags.update(dict_append)


def add_group_params_re_init(groups,
                             variable,
                             re_init_variable,
                             re_init_indices,
                             re_init_threshold,
                             re_init_dt,
                             dist_param,
                             scale,
                             distribution,
                             unit):
    """This allows adding a weight re-initialization run-regular function
    specifying the distribution parameters from which to sample.

    Args:
        group (list): List of groups which are subject to weight
            initialization
        re_init_varaiable (str, optional): Name of the variabe to be used to
            calculate re_init_indices.
        re_init_indices (ndarray, optional): Array to indicate which parameters
            need to be re-initialised. re_init_threshold (float): Parameter between 0 and 0.5. Threshold
            which triggers re-initialization.
        re_init_dt (second): Dt of run_regularly.
        dist_param (float): Shape of gamma distribution or mean of
            normal distribution used.
        scale (float): Scale for gamma distribution or std of normal
            distribution used.
        distribution (bool): Distribution from which to initialize the
            weights. Gamma (1) or normal (0) distributions.
        unit (brian.unit, optional): Unit of the parameter.
    """
    for group in groups:
        add_re_init_params(group,
                           variable=variable,
                           re_init_variable=re_init_variable,
                           re_init_indices=re_init_indices,
                           re_init_threshold=re_init_threshold,
                           re_init_dt=re_init_dt,
                           dist_paramt=dist_param,
                           scale=scale,
                           distribution=distribution,
                           unit=unit)

        if distribution == 0:
            group._tags.update({'re_init_{}'.format(variable) : "Normal"})
        elif distribution == 1:
            group._tags.update({'re_init_{}'.format(variable) : "Gamma"})


def add_group_activity_proxy(groups, buffer_size, decay):
    """This warpper function allows to add an activity proxy 
    run regular function.

    Args:
        group (list): List of neuron groups which are subject to
            weight initialiazion
        buffer_size (int): Size of the buffer which serves to calculate
            the activty
        decay (TYPE): Width of the running window.
    """
    for group in groups:
        add_activity_proxy(group,
                           buffer_size=buffer_size,
                           decay=decay)
        dict_append = {'activity_proxy' : True}
        group._tags.update(dict_append)


def add_group_param_init(groups, variable, dist_param, scale, 
                         distribution, clip_min=None, clip_max=None):
    """Function to add the parameter initialisation to a given
    group to be sampled from a specified distribution.

    Args:
        group (teili object): Connection or Neuron group whose specified 
            parameters are intialised
        paramter (str): Name of parameter to be initialised.
        dist_param (float): Mean of the distribution. In case of gamma 
            this paramter refers to shape paramter. 
        scale (float): Scale for gamma distribution or std of normal
            distribution used.
        distribution (bool): Distribution from which to initialize the
            weights. Gamma (1) or normal (0) distributions.

    """
    for group in groups:
        group.namespace.update({'dist_param': dist_param})
        group.namespace.update({'scale': scale})
        params = group.__getattr__(variable)

        if type(group) == Connections:
            size=len(group)
        elif type(group) == Neurons:
            size=group.N
        
        if distribution == 'gamma':
            params = np.random.gamma(shape=dist_param,
                                         scale=scale,
                                         size=size)
                                        
        elif distribution == 'normal':
            params = np.random.normal(loc=dist_param,
                                          scale=scale,
                                          size=size)

        if clip_min is not None and clip_max is not None:
            params = np.clip(params, clip_min, clip_max)

        group.__setattr__(variable, params)

        if distribution == 0:
            group._tags.update({'{}_distribution'.format(variable): "Normal"})
        elif distribution == 1:
            group._tags.update({'{}_distribution'.format(variable): "Gamma"})

