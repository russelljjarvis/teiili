# -*- coding: utf-8 -*-
# @Author: alpren, mmilde
# @Date:   2018-01-09 17:25:21
# @Last Modified by:   Moritz Milde
# @Last Modified time: 2018-06-01 16:56:34

"""
This module provides functions that can be used for synaptic connectivity kernels (generate weight matrices).
In order to also use them with C++ code generation, all functions that are added
here need to have a cpp implementation given by the @implementation decorator.

TODO: It would be good, if one could easily use different distance functions like
on a ring or torus (1d/2d periodic boundary conditions).
For numpy, this would be easy to implement by just using the respective function (from tools.distance) and add a selector as a parameter.
For cpp, we would have to make sure that the functions used are known.

"""

import numpy as np
from brian2 import implementation, check_units, exp, declare_types
from teili.tools.indexing import ind2xy  # , xy2ind, ind2x, ind2y


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
    """Summary: function that calculates mexican hat 1D kernel.

    Args:
        i (int): presynaptic index.
        j (int): postsynaptic index.
        gsigma (float): sigma (sd) of kernel.

    Returns:
        float: value of kernel that can be set as a weight.
    """
    x = i - j
    exponent = -(x**2) / (2 * gsigma**2)
    res = (1 + 2 * exponent) * exp(exponent)  # mexican hat, not normalized
    return res


@implementation('cpp', '''
    float kernel_gauss_1d(int i, int j, float gsigma) {
    return exp(-(pow((i - j),2)) / (2 * pow(gsigma,2)));
    }
     ''')
@declare_types(i='integer', j='integer', gsigma='float', result='float')
@check_units(i=1, j=1, gsigma=1, result=1)
def kernel_gauss_1d(i, j, gsigma):
    """Summary: function that calculates 1D kernel.

    Args:
        i (int): presynaptic index.
        j (int): postsynaptic index.
        gsigma (float): sigma (SD) of kernel.

    Returns:
        float: value of kernel that can be set as a weight.
    """
    res = exp(-((i - j)**2) / (2 * gsigma**2))  # gaussian, not normalized
    return res


@implementation('cpp', '''
    float kernel_mexican_2d(int i, int j, float gsigma, int nrows, int ncols) {
    int ix = i / ncols;
    int iy = i % ncols;
    int jx = j / ncols;
    int jy = j % ncols;
    int x = ix - jx;
    int y = iy - jy;
    float exponent = -(pow(x,2) + pow(y,2)) / (2 * pow(gsigma,2));
    return ((1 + exponent) * exp(exponent));
    }
     ''')
@declare_types(i='integer', j='integer', gsigma='float', nrows='integer', ncols='integer', result='float')
@check_units(i=1, j=1, gsigma=1, nrows=1, ncols=1, result=1)
def kernel_mexican_2d(i, j, gsigma, nrows, ncols):
    """Summary: function that calculates 2D kernel.

    Args:
        i (int): presynaptic index.
        j (int): postsynaptic index.
        gsigma (float): sigma (SD) of kernel.
        ncols (int): number of cols in 2d array of neurons, needed to calcualate 1d index.
        nrows (int): number of rows in 2d array of neurons, only needed to check
                    if index is in array (this check is skipped in cpp version).

    Returns:
        float: value of kernel that can be set as a weight.
    """
    # exponent = -(fdist(i,j,ncols)**2)/(2*gsigma**2) #alternative
    (ix, iy) = ind2xy(i, nrows, ncols)
    (jx, jy) = ind2xy(j, nrows, ncols)
    x = ix - jx
    y = iy - jy
    exponent = -(x**2 + y**2) / (2 * gsigma**2)
    # mexican hat / negative Laplacian of Gaussian #not normalized
    res = (1 + exponent) * exp(exponent)
    return res


@implementation('cpp', '''
    float kernel_gauss_2d(int i, int j, float gsigma, int nrows, int ncols) {
    int ix = i / ncols;
    int iy = i % ncols;
    int jx = j / ncols;
    int jy = j % ncols;
    int x = ix - jx;
    int y = iy - jy;
    float exponent = -(pow(x,2) + pow(y,2)) / (2 * pow(gsigma,2));
    return exp(exponent);
    }
     ''')
@declare_types(i='integer', j='integer', gsigma='float', nrows='integer', ncols='integer', result='float')
@check_units(i=1, j=1, gsigma=1, nrows=1, ncols=1, result=1)
def kernel_gauss_2d(i, j, gsigma, nrows, ncols):
    """Summary: function that calculates symmetrical gaussian 2D kernel.

    Args:
        i (int): presynaptic index.
        j (int): postsynaptic index.
        gsigma (float): sigma (SD) of kernel.
        ncols (int): number of cols in 2d array of neurons, needed to calcualate 1d index.
        nrows (int): number of rows in 2d array of neurons, only needed to check
                    if index is in array (this check is skipped in cpp version).


    Returns:
        float: value of kernel that can be set as a weight
    """
    (ix, iy) = ind2xy(i, nrows, ncols)
    (jx, jy) = ind2xy(j, nrows, ncols)
    x = ix - jx
    y = iy - jy
    exponent = -(x**2 + y**2) / (2 * gsigma**2)
    res = exp(exponent)
    return res


# TODO: Make this consistent with the other functions in this module
# TODO: Add phase parameter
@implementation('cpp', '''
    float kernel_gabor_2d(int i, int j, int offx, int offy, float theta, float sigmax, float sigmay, float freq, int input_size_x, int input_size_y, int window_size_x, int window_size_y, int rf_size){
    int ix = i % input_size_x;
    int iy = i / input_size_x;
    if(((window_size_x + abs(offx)) <= (input_size_x-(rf_size-1))) & ((window_size_y + abs(offy)) <= (input_size_y-(rf_size-1)))){
        int x0 = j % window_size_x;
        int y0 = j / window_size_x;
        x0 += int((input_size_x-window_size_x+1)/2) + offx;
        y0 += int((input_size_y-window_size_y+1)/2) + offy;
        float x =  (ix - x0)*cos(theta-M_PI/2) + (iy - y0)*sin(theta-M_PI/2);
        float y = -(ix - x0)*sin(theta-M_PI/2) + (iy - y0)*cos(theta-M_PI/2);
        float exponent = -((pow(x,2)/(2*pow(sigmax,2))) + (pow(y,1)/(2*pow(sigmay,2))));
        float res = exp(exponent)*cos(M_PI*x/freq);
        res = res*(abs(ix - x0)<rf_size/2) *(abs(iy - y0)<rf_size/2);
        return res;}
    else{
        return 0;}
        }
     ''')
@declare_types(i='integer', j='integer', offx='integer', offy='integer', theta='float', sigmax='float', sigmay='float', freq='float', input_size_x='integer', input_size_y='integer', window_size_x='integer', window_size_y='integer', rf_size='integer', result='float')
@check_units(i=1, j=1, offx=1, offy=1, theta=1, sigmax=1, sigmay=1, freq=1, input_size_x=1, input_size_y=1, window_size_x=1, window_size_y=1, rf_size=1, result=1)
def kernel_gabor_2d(i, j, offx, offy, theta, sigmax, sigmay, freq, input_size_x, input_size_y, window_size_x, window_size_y, rf_size):
    """Summary: function that calculates Gabor 2D kernel.
    Only works with odd-sized square Receptive Fields.
    It prints the weight values for a couple of neurons.
    To spare computation, this connectivity kernel allows the use of a
    smaller output layer which only accounts for a small portion of the input
    layer. The output layer can be displaced w.r.t. the input layer using offx
    and offy.


    Args:
        i (int): First layer neuron's index.
        j (int): Second layer neuron's index.
        offx (int): x offset of the the second layer's center respective to
            first layer's center.
        offy (int): y offset of the the second layer's center respective to
            first layer's center.
        theta (float): orientation of the gabor filter.
        sigmax (float): variance along x.
        sigmay (float): variance along y
        freq (float): frequency of the filter.
        input_size_x (int): x size of the input layer.
        input_size_y (int): y size of the input layer.
        window_size_x (int): x size of the output layer.
        window_size_y (int): y size of the output layer.
        rf_size (int): side size of the Receptive Field.

    Returns:
        float: The weight between the ith and jth neuron.
    """

    (iy, ix) = np.unravel_index(i, (input_size_y, input_size_x))
    if (window_size_x + abs(offx) <= (input_size_x - (rf_size - 1))) & (window_size_y + abs(offy) <= (input_size_y - (rf_size - 1))):
        (y0, x0) = np.unravel_index(j, (window_size_y, window_size_x))
        x0 = x0 + int((input_size_x - window_size_x + 1) / 2) + offx
        y0 = y0 + int((input_size_y - window_size_y + 1) / 2) + offy
        x = (ix - x0) * np.cos(theta - np.pi / 2) + \
            (iy - y0) * np.sin(theta - np.pi / 2)
        y = -(ix - x0) * np.sin(theta - np.pi / 2) + \
            (iy - y0) * np.cos(theta - np.pi / 2)
        exponent = -(((x**2) / (2 * sigmax**2)) + ((y**2) / (2 * sigmay**2)))
        res = exp(exponent) * np.cos(np.pi * x / freq)
        res = res * (abs(ix - x0) < rf_size / 2) * (abs(iy - y0) < rf_size / 2)
        return res
    else:
        print("The kernel window is bigger than InputSize-(rf_size-1)")
        return 0
