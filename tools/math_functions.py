#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 13:43:45 2018

@author: alpha

This module provides generic functions that are not yet provided by brian2 including a cpp
implementation.
This is not to be confused with the synaptic kernels, that are for conectivity matrix generation (they could/should be used by those!).

the suffix "_cpp" avoids that variables are string replaced by brian2 if the same name
is used in the network

"""
from brian2 import implementation, check_units, declare_types
import numpy as np

@implementation('cpp', '''
    float normal2d_density(float x_cpp, float y_cpp, float mu_x_cpp, float mu_y_cpp,
                           float sigma_x_cpp, float sigma_y_cpp, float rho_cpp, bool normalized_cpp) {
        float dist_x = x_cpp-mu_x_cpp
        float dist_y = y_cpp-mu_y_cpp
        float f1;
        if (normalized_cpp)
            f1 = (1.0 / (2.0 * M_PI * sigma_x_cpp * sigma_y_cpp * sqrt(1 - pow(rho_cpp,2))));
        else
            f1 = 1.0;

            float f2 = -(1.0 / (2.0 * (1.0 - pow(rho_cpp,2))));
            float fxy = 2 * (dist_x / sigma_x_cpp) * (dist_y / sigma_y_cpp) * rho_cpp;
            float density = f1 * exp(f2 * (pow((dist_x / sigma_x_cpp),2) + pow((dist_y / sigma_y_cpp),2) - fxy));

            return density;
    }
                ''')
@declare_types(x='float', y='float', mu_x='float', mu_y='float', sigma_x='float', sigma_y='float',
               rho='float', result='float', normalized = 'boolean')
@check_units(x=1, y=1, mu_x = 1, mu_y = 1, sigma_x=1, sigma_y=1, rho=1, normalized = 1, result=1)
def normal2d_density(x, y, mu_x = 0, mu_y = 0, sigma_x=1, sigma_y=1, rho=0, normalized = True):
    """
    Args:
        x and y (float):  x and y coordinates
        mu_x and y (float):  Means of gaussian in x and y dimension
        ncols, nrows (int): size of the output array
        sigma_x and _y (float, optional): Standard deviations of gaussian distribution
        rho (float, optional): correlation coefficient of the 2 variables
    Returns:
        TYPE: float
        normal (probability) density at a specific distance to the mean (dist_x,dist_y) of a 2d distribution
    """
    #print(dist_x, dist_y, sigma_x, sigma_y, rho, normalized )
    dist_x = x-mu_x
    dist_y = y-mu_y
    if normalized:
        f1 = (1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2)))
    else:
        f1 = 1
    f2 = -(1 / (2 * (1 - rho**2)))
    fx = dist_x / sigma_x
    fy = dist_y / sigma_y
    fxy = 2 * fx * fy * rho
    density = f1 * np.exp(f2 * (fx**2 + fy**2 - fxy))
    #print(density)
    return density





# Did not add a cpp implementation, as we usually can't work with arrays anyway
@implementation('numpy', discard_units=True)
@declare_types(nrows='integer', ncols='integer', sigma_x='float', sigma_y='float', rho='float',
               mu_x='float', mu_y='float', result='float', normalized = 'boolean')
@check_units(nrows=1, ncols=1, sigma_x=1, sigma_y=1, mu_x=1, mu_y=1, rho=1, normalized = 1, result=1)
def normal2d_density_array(nrows, ncols, sigma_x=1, sigma_y=1, rho=0, mu_x=None, mu_y=None, normalized = True):
    """returns a 2d normal density distributuion array of size (nrows, ncols)

    Args:
        ncols, nrows (int): size of the output array
        sigma_x and _y (float, optional): Standard deviations of gaussian distribution
        mu_x and _y (float, optional): Means of gaussian distribution
        rho (float, optional): correlation coefficient of the 2 variables
        normalized (boolean, optional): If you set this to False, it will no longer be a
            probability density with integral of one, but the max amplitude (in the middle of the bump) will be 1

    Returns:
        TYPE: Description
    """
    x = np.arange(0, nrows)
    y = np.reshape(np.arange(0, ncols), (ncols, 1))
    #y = x[:, np.newaxis]

    if mu_x is None:
        mu_x = nrows // 2
    if mu_y is None:
        mu_y = ncols // 2

    density = normal2d_density(x, y, mu_x, mu_y, sigma_x, sigma_y, rho, normalized)

##as the function is vectorized, this is the same as:
#    density = np.zeros((nrows+1, ncols+1))
#    i = -1
#    for dx in dist_x:
#        i+= 1
#        j = -1
#        for dy in dist_y:
#            j+=1
#            density[j,i] = normal2d_density(dx, dy, sigma_x, sigma_y, rho, normalized)

    return density



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img = normal2d_density_array(100, 100, 20, 10, 0.5)
    #img = normal2d_density_array(100, 100, 20, 20, 0.5, 50,50,0)
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    # the sum is almost one, if the sigmas are not much larger than
    print(np.sum(img))
