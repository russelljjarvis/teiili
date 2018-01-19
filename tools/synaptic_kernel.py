# -*- coding: utf-8 -*-
# @Author: alpren, mmilde
# @Date:   2018-01-09 17:25:21
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-18 17:12:59


"""
This module provides functions, that can be used for synaptic connectivity kernels (generalte weight matrices).
In order to also use them with c++ code generation all functions that are added here need to have a cpp implementation given by the @implementation decorator.
"""
import numpy as np
from brian2 import implementation, check_units, ms, exp, mean, diff, declare_types
from NCSBrian2Lib.tools.indexing import ind2xy


@implementation('cpp', '''
    float kernel_mexican_1d(int i, int j, float gsigma) {
    x = i - j
    exponent = -pow(x,2) / (2 * pow(gsigma,2))
    res = (1 + 2 * exponent) * exp(exponent)
    return res;
    }
     ''')
@declare_types(i='integer', j='integer', gsigma='float', result='float')
@check_units(i=1, j=1, gsigma=1, result=1)
def kernel_mexican_1d(i, j, gsigma):
    """Summary:function that calculates mexican hat 1D kernel

    Args:
        i (int): presynaptic index
        j (int): postsynaptic index
        gsigma (float): sigma (sd) of kernel

    Returns:
        float: value of kernel, that can be set as a weight
    """
    x = i - j
    exponent = -(x**2) / (2 * gsigma**2)
    res = (1 + 2 * exponent) * exp(exponent)  # mexican hat, not normalized
    return res


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


@implementation('cpp', '''
    float kernel_gauss_1d(int i, int j, float gsigma) {
    return exp(-(pow((i - j),2)) / (2 * pow(gsigma,2)));
    }
     ''')
@declare_types(i='integer', j='integer', gsigma='float', result='float')
@check_units(i=1, j=1, gsigma=1, result=1)
def kernel_gauss_1d(i, j, gsigma):
    """Summary: function that calculates 1D kernel

    Args:
        i (int): presynaptic index
        j (int): postsynaptic index
        gsigma (float): sigma (sd) of kernel

    Returns:
        float: value of kernel, that can be set as a weight
    """
    res = exp(-((i - j)**2) / (2 * gsigma**2))  # gaussian, not normalized
    return res


# TODO: Make this general for non square groups
@implementation('cpp', '''
    float kernel_mexican_2d(int i, int j, float gsigma, int n2d_neurons) {
    int ix = i / n2d_neurons;
    int iy = i % n2d_neurons;
    int jx = j / n2d_neurons;
    int jy = j % n2d_neurons;
    int x = ix - jx;
    int y = iy - jy;
    float exponent = -(pow(x,2) + pow(y,2)) / (2 * pow(gsigma,2));
    return ((1 + exponent) * exp(exponent));
    }
     ''')
@declare_types(i='integer', j='integer', gsigma='float', n2d_neurons='integer', result='float')
@check_units(i=1, j=1, gsigma=1, n2d_neurons=1, result=1)
def kernel_mexican_2d(i, j, gsigma, n2d_neurons):
    """Summary: function that calculates 2D kernel

    Args:
        i (int): presynaptic index
        j (int): postsynaptic index
        gsigma (float): sigma (sd) of kernel
        n2d_neurons (int): number of neurons sqrt(toal_group_neurons), needed to calcualte 1d index (Group has to be square)

    Returns:
        float: value of kernel, that can be set as a weight
    """
    # exponent = -(fdist(i,j,n2d_neurons)**2)/(2*gsigma**2) #alternative
    (ix, iy) = ind2xy(i, n2d_neurons)
    (jx, jy) = ind2xy(j, n2d_neurons)
    x = ix - jx
    y = iy - jy
    exponent = -(x**2 + y**2) / (2 * gsigma**2)
    res = (1 + exponent) * exp(exponent)  # mexican hat / negative Laplacian of Gaussian #not normalized
    return res


@implementation('cpp', '''
    float kernel_gauss_2d(int i, int j, float gsigma, int n2d_neurons) {
    int ix = i / n2d_neurons;
    int iy = i % n2d_neurons;
    int jx = j / n2d_neurons;
    int jy = j % n2d_neurons;
    int x = ix - jx;
    int y = iy - jy;
    float exponent = -(pow(x,2) + pow(y,2)) / (2 * pow(gsigma,2));
    return exp(exponent);
    }
     ''')
@declare_types(i='integer', j='integer', gsigma='float', n2d_neurons='integer', result='float')
@check_units(i=1, j=1, gsigma=1, n2d_neurons=1, result=1)
def kernel_gauss_2d(i, j, gsigma, n2d_neurons):
    """Summary: function that calculates symmetrical gaussian 2D kernel

    Args:
        i (int): presynaptic index
        j (int): postsynaptic index
        gsigma (float): sigma (sd) of kernel
        n2d_neurons (int, required): Description

    Returns:
        float: value of kernel, that can be set as a weight
    """
    (ix, iy) = ind2xy(i, n2d_neurons)
    (jx, jy) = ind2xy(j, n2d_neurons)
    x = ix - jx
    y = iy - jy
    exponent = -(x**2 + y**2) / (2 * gsigma**2)
    res = exp(exponent)
    return res


# TODO: Add cpp implementation
# TODO: Add docstring type declaration and description
@implementation('cpp', '''

     ''')
@declare_types(i='integer', j='integer', offx='integer', offy='integer', theta='float', sigmax='float', sigmay='float', freq='float', InputSizeX='integer', InputSizeY='integer', WindowSizeX='integer', WindowSizeY='integer', RFSize='integer', result='float')
@check_units(i=1, j=1, offx=1, offy=1, theta=1, sigmax=1, sigmay=1, freq=1, InputSizeX=1, InputSizeY=1, WindowSizeX=1, WindowSizeY=1, RFSize=1, result=1)
def fkernelGabor2d(i, j, offx, offy, theta, sigmax, sigmay, freq, InputSizeX, InputSizeY, WindowSizeX, WindowSizeY, RFSize):
    """Summary: function that calculates Gabor 2D kernel, only works with odd square Receptive Fields

    Args:
        i (TYPE): Description
        j (TYPE): Description
        offx (TYPE): Description
        offy (TYPE): Description
        theta (TYPE): Description
        sigmax (TYPE): Description
        sigmay (TYPE): Description
        freq (TYPE): Description
        InputSizeX (TYPE): Description
        InputSizeY (TYPE): Description
        WindowSizeX (TYPE): Description
        WindowSizeY (TYPE): Description
        RFSize (TYPE): Description

    Returns:
        TYPE: Description
    """

    (ix, iy) = np.unravel_index(i, (InputSizeX, InputSizeY))
    if (WindowSizeX + abs(offx) <= (InputSizeX - (RFSize - 1))) & (WindowSizeY + abs(offy) <= (InputSizeY - (RFSize - 1))):
        (x0, y0) = np.unravel_index(j, (WindowSizeX, WindowSizeY))
        x0 = x0 + int((InputSizeX - WindowSizeX + 1) / 2) + offx
        y0 = y0 + int((InputSizeY - WindowSizeY + 1) / 2) - offy
        x = (ix - x0) * np.cos(theta) + (iy - y0) * np.sin(theta)
        y = -(ix - x0) * np.sin(theta) + (iy - y0) * np.cos(theta)
        exponent = -((x**2) / 2 * sigmax**2 + (y**2) / 2 * sigmay**2)
        res = exp(exponent) * np.cos(2 * np.pi * x / freq)
        res = res * (abs(ix - x0) < RFSize / 2) * (abs(iy - y0) < RFSize / 2)
        return res
    else:
        print("The kernel window it's bigger than InputSize-(RFSize-1)")
        return 0
