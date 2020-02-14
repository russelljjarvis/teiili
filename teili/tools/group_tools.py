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

from teili.tools.add_run_reg import add_weight_decay,\
    add_re_init_weights, add_activity_proxy
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


def add_group_weight_re_init(groups,
                             re_init_index,
                             re_init_threshold,
                             dist_param_re_init,
                             scale_re_init,
                             distribution):
    """This allows adding a weight re-initialization run-regular function
    specifying the distribution parameters from which to sample.

    Args:
        group (list): List of groups which are subject to weight
            initialization
        re_init_threshold (float): Parameter between 0 and 0.5. Threshold
            which triggers re-initialization.
        dist_param_re_init (bool): Shape of gamma distribution or mean of
            normal distribution used.
        scale_re_init (int): Scale for gamma distribution or std of normal
            distribution used.
        distribution (bool): Distribution from which to initialize the
            weights. Gamma (1) or normal (0) distributions.
    """
    for group in groups:
        add_re_init_weights(group,
                            re_init_index=re_init_index,
                            re_init_threshold=re_init_threshold,
                            dist_param_re_init=dist_param_re_init,
                            scale_re_init=scale_re_init,
                            distribution=distribution)

        if distribution == 0:
            group._tags.update({'re_init_weights' : "Normal"})
        elif distribution == 1:
            group._tags.update({'re_init_weights' : "Gamma"})


def add_group_activity_proxy(groups, buffer_size, decay):
    """This allows to add an activity proxy run regular function.

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


def add_goup_weight_init(groups, dist_param, scale, distribution):
    """Function to add the weight initialisation to a given
    `Connections` group.

    Args:
        group (teili object): Connection group whose weights are intialised
        dist_param (float): Parameter between 0 and 0.5. Threshold which
            triggers re-initialization.
        scale (float): Scale for gamma distribution or std of normal
            distribution used.
        distribution (bool): Distribution from which to initialize the
            weights. Gamma (1) or normal (0) distributions.

    """
    for group in groups:
        group.namespace.update({'dist_param': dist_param})
        group.namespace.update({'scale': scale})
        weights = group.w_plast
        if distribution == 'gamma':
            weights = np.random.gamma(shape=dist_param,
                                      scale=scale,
                                      size=len(group))
        if distribution == 'normal':
            weights = np.random.normal(loc=dist_param,
                                       scale=scale,
                                       size=len(group))
        weights = np.clip(weights, 0, 1)

        group.w_plast = weights
        if distribution == 0:
            group._tags.update({'init_weights' : "Normal"})
        elif distribution == 1:
            group._tags.update({'init_weights' : "Gamma"})

