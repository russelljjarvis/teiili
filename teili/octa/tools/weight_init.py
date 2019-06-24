# -*- coding: utf-8 -*-
"""A collection of functions to easily initialize
weight matrices of provided Connections/Synapses.

"""
# @Author: mmilde
# @Date:   2018-08-10 10:19:18

import numpy as np


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
