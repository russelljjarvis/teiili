#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 13:43:45 2018

@author: alpha

This module provides generic functions that are not yet provided by brian2 including a cpp
implementation.
This is not to be confused with the synaptic kernels, that are for conectivity matrix generation.
"""
from brian2 import implementation, check_units, declare_types
import numpy as np

# TODO: Add cpp implementation
@implementation('cpp', '''
                float gaussian(int square_size, int sigma, float mu) {
                return None;
                }
                ''')
@declare_types(square_size='integer', sigma='integer', mu='float', result='float')
@check_units(square_size=1, sigma=1, mu=1, result=1)
def gaussian(square_size, sigma=1, mu=None):
    """Makes a square gaussian kernel

    Args:
        square_size (int):  Size of the square, i.e. amplitude? (not sure!)
        sigma (int, optional): Standard deviation of gaussian distribution
        mu (None, optional): Mean of gaussian distribution

    Returns:
        TYPE: Description
    """

    x = np.arange(0, square_size)
    y = x[:, np.newaxis]

    if mu is None:
        x0 = y0 = square_size // 2
    else:
        x0 = mu[0]
        y0 = mu[1]

    return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
