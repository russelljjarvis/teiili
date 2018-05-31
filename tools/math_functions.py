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
                float gaussian2d(int square_size, int sigma, float mu) {
                return None;
                }
                ''')
@declare_types(nrows='integer', ncols='integer', sigma='integer', mu_x='float', mu_y='float', result='float')
@check_units(nrows=1, ncols=1, sigma=1, mu_x=1, mu_y=1, result=1)
def gaussian2d(nrows, ncols, sigma=1, mu_x=None, mu_y=None):
    """Makes a gaussian

    Args:
        square_size (int):  Size of the square, i.e. amplitude? (not sure!)
        sigma (int, optional): Standard deviation of gaussian distribution
        mu (None, optional): Mean of gaussian distribution

    Returns:
        TYPE: Description
    """

    x = np.arange(0, nrows)
    y = np.reshape(np.arange(0, ncols),(ncols,1))
    #y = x[:, np.newaxis]

    if mu_x is None:
        mu_x = nrows // 2
    if mu_y is None:
        mu_y = ncols // 2

    gaussian = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2))

    return gaussian





if __name__ == '__main__':
    # just plot all the functions

    import matplotlib.pyplot as plt
    img = gaussian2d(50,60,10)
    plt.imshow(img)
