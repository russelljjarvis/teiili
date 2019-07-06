#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module provides generic functions that are not yet provided by brian2 including a cpp
implementation.

This is not to be confused with the synaptic kernels that are for conectivity matrix generation (they could/should be used by those!).

the suffix "_cpp" avoids variables being string-replaced by brian2 if the same name
is used in the network
"""
# Created on Wed May 30 13:43:45 2018
# @author: alpha

import os
from brian2 import implementation, check_units, declare_types, set_device, run, ms
import numpy as np
from scipy.stats import gamma


# For alpha = 1, gamma reduces to the exponential distribution.
# For large alpha the gamma distribution converges to normal distribution
# with mean μ = alpha/beta and variance σ2 = alpha/beta**2.

# TODO
@implementation('cpp', '''
    float gamm(float x) {
    float ret = (1.000000000190015 + 76.18009172947146 / (x + 1) +  
                -86.50532032941677 / (x + 2) + 24.01409824083091 / (x + 3) +  
                -1.231739572450155 / (x + 4) + 1.208650973866179e-3 / (x + 5) + 
                -5.395239384953e-6 / (x + 6));
    return ret * sqrt(2*M_PI)/x * pow(x + 5.5, x+.5) * exp(-x-5.5);}
    float gamma1d_density(float x_cpp, float alpha_cpp, float beta_cpp, bool normalized_cpp) {
        float f;
        if (normalized_cpp)
            f = pow(beta_cpp, alpha_cpp) / gamm(alpha_cpp);
        else
            f = 1.0;
        float density = exp(-x_cpp * beta_cpp) * pow(x_cpp, alpha_cpp - 1.0) * f;
        return density;}
                ''')
@declare_types(x='float', alpha='float', beta='float', result='float', normalized='boolean')
@check_units(x=1, alpha=1, beta=1, normalized=1, result=1)
def gamma1d_density(x, alpha=1, beta=1, normalized=True):
    """
    This function is supposed to be used in brian2 strings e.g. to initialize weights.
    In python, this just wraps scipy.stats.gamma, in c++  it manually
     calculates an approximation of the gamma density.

    :param x: distance to the mean
    :param alpha:
    :param beta:
    :param normalized: boolean
    :return: float: (probability) density at a specific distance to the mean of a Gaussian distribution.

    >>> import matplotlib.pyplot as plt
    >>> from teili import Neurons
    >>>
    >>> standalone_dir = os.path.expanduser('~/mismatch_standalone')
    >>> set_device('cpp_standalone', directory=standalone_dir)
    >>>
    >>> dx = 0.01
    >>> x = np.arange(0, 10, 0.01)
    >>> gamma_pdf = gamma1d_density(x, 2, 1)
    >>>
    >>> n = Neurons(1000, model='x : 1')
    >>> n.namespace.update({'gamma1d_density': gamma1d_density})
    >>> n.x = "gamma1d_density(i/100.0,2.0,1.0,False)"
    >>>
    >>> run(1 * ms)
    >>> plt.figure()
    >>> plt.plot(np.arange(0, 10, 1 / 100), n.x)
    >>> plt.plot(x, gamma_pdf)
    >>> plt.show()
    >>> sum(n.x * 1 / 100)
    >>> sum(gamma_pdf * dx)
    """

    # TODO:
    # if normalized:
    density = gamma.pdf(x, alpha, scale=beta)
    #    else:
    #        f = 1
    #    density = f * np.exp(-(1/2)*(dist_x / sigma)**2)
    # print(density)
    return density


@implementation('cpp', '''
    float normal1d_density(float x_cpp, float mu_cpp, float sigma_cpp, bool normalized_cpp) {
        float dist_x = x_cpp-mu_cpp;
        float f;
        if (normalized_cpp)
            f = 1.0 / sqrt(2.0 * M_PI * pow(sigma_cpp,2));
        else
            f = 1.0;
        float density = f * exp(-0.5*pow((dist_x / sigma_cpp),2));
        return density;
    }
                ''')
@declare_types(x='float', mu='float', sigma='float', result='float', normalized='boolean')
@check_units(x=1, mu=1, sigma=1, normalized=1, result=1)
def normal1d_density(x, mu=0, sigma=1, normalized=True):
    """
    Calculates gaussian density in 1d
    Args:
        x, (float): x values at which density is calculated.
        mu (float): Mean of Gaussian.
        sigma (float, optional): Standard deviation Gaussian distribution.
        normalized (bool, optional): Description

    Returns:
        float: (probability) density at a specific distance to the mean of a Gaussian distribution.

    >>> import matplotlib.pyplot as plt
    >>> dx = 0.1
    >>> normal1drange = np.arange(-10, 10, dx)
    >>> # thanks to the brian2 decorators, keyword arguments don't work, but you can add all args as positional arguments
    >>> gaussian = [normal1d_density(x, 0, 1, True) for x in normal1drange]
    >>> print(np.sum(gaussian) * dx)
    >>>
    >>> plt.figure()
    >>> plt.plot(normal1drange, gaussian)
    >>> plt.show()
    """
    dist_x = x - mu
    if normalized:
        f = 1 / np.sqrt(2 * np.pi * sigma ** 2)
    else:
        f = 1
    density = f * np.exp(-(1 / 2) * (dist_x / sigma) ** 2)
    # print(density)
    return density


@implementation('cpp', '''
    float normal2d_density(float x_cpp, float y_cpp, float mu_x_cpp, float mu_y_cpp,
                           float sigma_x_cpp, float sigma_y_cpp, float rho_cpp, bool normalized_cpp) {
        float dist_x = x_cpp-mu_x_cpp;
        float dist_y = y_cpp-mu_y_cpp;
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
               rho='float', result='float', normalized='boolean')
@check_units(x=1, y=1, mu_x=1, mu_y=1, sigma_x=1, sigma_y=1, rho=1, normalized=1, result=1)
def normal2d_density(x, y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1, rho=0, normalized=True):
    """
        Calculates gaussian density in 2d

    Args:
        x, y (float): x and y values at which density is calculated.
        mu_x, mu_y (float): Means of Gaussian in x and y dimension.
        sigma_x, sigma_y (float, optional): Standard deviations of Gaussian distribution.
        rho (float, optional): correlation coefficient of the 2 variables.
        normalized (bool, optional): Description

    Returns:
        float: normal (probability) density at a specific distance to the mean (dist_x,dist_y) of a 2d distribution.
    """
    dist_x = x - mu_x
    dist_y = y - mu_y
    if normalized:
        f1 = (1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho ** 2)))
    else:
        f1 = 1
    f2 = -(1 / (2 * (1 - rho ** 2)))
    fx = dist_x / sigma_x
    fy = dist_y / sigma_y
    fxy = 2 * fx * fy * rho
    density = f1 * np.exp(f2 * (fx ** 2 + fy ** 2 - fxy))
    # print(density)
    return density


# Did not add a cpp implementation, as we usually can't work with arrays anyway
@implementation('numpy', discard_units=True)
@declare_types(nrows='integer', ncols='integer', sigma_x='float', sigma_y='float', rho='float',
               mu_x='float', mu_y='float', result='float', normalized='boolean')
@check_units(nrows=1, ncols=1, sigma_x=1, sigma_y=1, mu_x=1, mu_y=1, rho=1, normalized=1, result=1)
def normal2d_density_array(nrows, ncols, sigma_x=1, sigma_y=1, rho=0, mu_x=None, mu_y=None, normalized=True):
    """Returns a 2d normal density distributuion array of size (nrows, ncols).

    Args:
        ncols, nrows (int): size of the output array.
        sigma_x (int, optional): Description
        sigma_y (int, optional): Description
        rho (float, optional): correlation coefficient of the 2 variables.
        mu_x (None, optional): Description
        mu_y (None, optional): Description
        normalized (boolean, optional): If you set this to False, it will no longer be a
            probability density with an integral of one, but the maximum amplitude (in the middle of the bump) will be 1.
        sigma_x and sigma_y (float, optional): Standard deviations of Gaussian distribution.
        mu_x and mu_y (float, optional): Means of Gaussian distribution.

    Returns:
        ndarray: Description


    >>> import matplotlib.pyplot as plt
    >>>
    >>> img = normal2d_density_array(100, 100, 20, 10, 0.5)
    >>> # img = normal2d_density_array(100, 100, 20, 20, 0.5, 50,50,0)
    >>> plt.figure()
    >>> plt.imshow(img)
    >>> plt.colorbar()
    >>> # the sum is almost one, if the sigmas are much smaller than the range
    >>> print(np.sum(img))


    Note:
        as the function is vectorized, this is the same as:

        >>> density = np.zeros((nrows+1, ncols+1))
        >>>     i = -1
        >>>     for dx in dist_x:
        >>>         i+= 1
        >>>         j = -1
        >>>         for dy in dist_y:
        >>>         j+=1
        >>>         density[j,i] = normal2d_density(dx, dy, sigma_x, sigma_y, rho, normalized)
    """
    x = np.arange(0, nrows)
    y = np.reshape(np.arange(0, ncols), (ncols, 1))
    # y = x[:, np.newaxis]

    if mu_x is None:
        mu_x = nrows // 2
    if mu_y is None:
        mu_y = ncols // 2

    density = normal2d_density(
        x, y, mu_x, mu_y, sigma_x, sigma_y, rho, normalized)

    return density

