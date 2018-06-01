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
@declare_types(nrows='integer', ncols='integer', sigma_x='float', sigma_y='float', rho='float', mu_x='float', mu_y='float', result='float')
@check_units(nrows=1, ncols=1, sigma_x=1, sigma_y=1, mu_x=1, mu_y=1, rho = 1, result=1)
def normal2d_density(nrows, ncols, sigma_x=1, sigma_y=1, rho = 0, mu_x=None, mu_y=None):
    """returns a 2d normal density distributuion array of size (nrows, ncols)


    Args:
        ncols, nrows (int): size of the output array
        sigma_x and _y (float, optional): Standard deviations of gaussian distribution
        mu_x and _y (float, optional): Means of gaussian distribution
        rho (float, optional): correlation coefficient of the 2 variables

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

    #gaussian = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2))
    f1 = (1 / (2*np.pi*sigma_x*sigma_y*np.sqrt(1-rho**2)))
    f2 = -(1 / (2*(1-rho**2)))
    fx = (x - mu_x)/sigma_x
    fy = (y - mu_y)/sigma_y
    fxy = 2*fx*fy*rho
    density = f1*np.exp(f2*(fx**2+fy**2-fxy))

    return density

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img = normal2d_density(100, 100, 20, 10, 0.5)
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    #the sum is almost one, if the sigmas are not much larger than
    print(np.sum(img))
