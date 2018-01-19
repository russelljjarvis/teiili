#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: mmilde, alpren
# @Date:   2017-12-27 11:54:09
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-18 17:11:13
"""A collection of helpful miscellaneous functions when working with brian2

"""

from brian2 import implementation, check_units, ms, exp, mean, diff, declare_types,\
    figure, subplot, plot, xlim, ylim, ones, zeros, xticks, xlabel, ylabel, device
import numpy as np
from NCSBrian2Lib.tools.indexing import ind2xy


#===============================================================================
# def setParams(neurongroup, params, debug=False):
#     for par in params:
#         if hasattr(neurongroup, par):
#             setattr(neurongroup, par, params[par])
#     if debug:
#         states = neurongroup.get_states()
#         print ('\n')
#         print ('-_-_-_-_-_-_-_', '\n', 'Parameters set')
#===============================================================================


def print_states(briangroup):
    """Wrapper function to print states of a brian2 groups such as NeuronGroup or Synapses

    Args:
        briangroup (brian2.group): Brain object/group which states/statevariables
            should be printed
    """
    states = briangroup.get_states()
    print ('\n')
    print ('-_-_-_-_-_-_-_')
    print(briangroup.name)
    print('list of states and first value:')
    for key in states.keys():
        if states[key].size > 1:
            print (key, states[key][1])
        else:
            print (key, states[key])
    print ('----------')


# This function is a workaround to allow if statements in run_regularly code
# It is e.g. necessary in order to set values conditional on the current time
@implementation('cpp', '''
float returnValueIf(float test_val, float greater_than_val, float smaller_than_val, float return_val_true, float return_val_false) {
    if ((test_val > greater_than_val) && (test_val < smaller_than_val))
        return return_val_true;
    else
        return return_val_false;
}
''')
@declare_types(test_val='float', greater_than_val='float', smaller_than_val='float',
               return_val_true='float', return_val_false='float', result='float')
@check_units(test_val=1, greater_than_val=1, smaller_than_val=1, return_val_true=1, return_val_false=1, result=1)
def return_value_if(test_val, greater_than_val, smaller_than_val, return_val_true, return_val_false):
    """Summary
    This function is a workaround to allow if statements in run_regularly code
    It is e.g. necessary in order to set values conditional on the current time
    it returns a value (return_val_true or return_val_false) depending on whether test_val is between
    smaller_than_val and greater_than_val or not
    Args:
        test_val (TYPE): the value that is tested
        greater_than_val (TYPE): upper bound of the value
        smaller_than_val (TYPE): lower bound of the value
        return_val_true (TYPE): value returned if test_val is in bounds
        return_val_false (TYPE): value returned if test_val is out of bounds

    Returns:
        float: returns a specified value (return_val_true or return_val_false) depending on whether test_val is between
    smaller_than_val and greater_than_val
    """
    if (test_val > greater_than_val and test_val < smaller_than_val):
        return return_val_true
    else:
        return return_val_false


@implementation('cpp', '''
    float dist1d2dfloat(float i, float j, int n2d_neurons) {
    int ix = i / n2d_neurons;
    int iy = i % n2d_neurons;
    int jx = j / n2d_neurons;
    int jy = j % n2d_neurons;
    return sqrt(pow((ix - jx),2) + pow((iy - jy),2));
    }
     ''')
@declare_types(i='float', j='float', n2d_neurons='integer', result='float')
@check_units(i=1, j=1, n2d_neurons=1, result=1)
def dist1d2dfloat(i, j, n2d_neurons):
    """function that calculates distance in 2D field from 2 1D indices

    Args:
        i (float, required): 1D index of source neuron
        j (float, required): 1D index of target neuron
        n2d_neurons (int, required): Size of neuron population

    Returns:
        float: Distance in 2D field
    """
    (ix, iy) = ind2xy(i, n2d_neurons)
    (jx, jy) = ind2xy(j, n2d_neurons)
    return np.sqrt((ix - jx)**2 + (iy - jy)**2)


@implementation('cpp', '''
    float dist1d2dint(int i, int j, int n2d_neurons) {
    int ix = i / n2d_neurons;
    int iy = i % n2d_neurons;
    int jx = j / n2d_neurons;
    int jy = j % n2d_neurons;
    return sqrt(pow((ix - jx),2) + pow((iy - jy),2));
    }
     ''')
@declare_types(i='integer', j='integer', n2d_neurons='integer', result='float')
@check_units(i=1, j=1, n2d_neurons=1, result=1)
def dist1d2dint(i, j, n2d_neurons):
    """function that calculates distance in 2D field from 2 1D indices

    Args:
        i (int, required): 1D index of source neuron
        j (int, required): 1D index of target neuron
        n2d_neurons (int, required): Size of neuron population

    Returns:
        int: Distance in 2D field
    """
    (ix, iy) = ind2xy(i, n2d_neurons)
    (jx, jy) = ind2xy(j, n2d_neurons)
    return np.sqrt((ix - jx)**2 + (iy - jy)**2)


@implementation('cpp', '''
    float dist2dind(int ix, int iy,int jx, int jy) {
    return sqrt(pow((ix - jx),2) + pow((iy - jy),2));
    }
     ''')
@declare_types(ix='integer', iy='integer', jx='integer', jy='integer', result='float')
@check_units(ix=1, iy=1, jx=1, jy=1, result=1)
def dist2d2dint(ix, iy, jx, jy):
    """Summary: function that calculates distance in 2D field from 4 integer 2D indices

    Args:
        ix (int, required): x component of 2D source neuron coordinate
        iy (int, required): y component of 2D source neuron coordinate
        jx (int, required): x component of 2D target neuron coordinate
        jy (int, required): y component of 2D target neuron coordinate

    Returns:
        int: Distance in 2D field
    """
    return np.sqrt((ix - jx)**2 + (iy - jy)**2)


@implementation('cpp', '''
    float dist2d(float ix, float iy,float jx, float jy) {
    return sqrt(pow((ix - jx),2) + pow((iy - jy),2));
    }
     ''')
@declare_types(ix='float', iy='float', jx='float', jy='float', result='float')
@check_units(ix=1, iy=1, jx=1, jy=1, result=1)
def dist2d2dfloat(ix, iy, jx, jy):
    """Summary: function that calculates distance in 2D field from 4 2D position values

    Args:
        ix (float, required): x component of 2D source neuron coordinate
        iy (float, required): y component of 2D source neuron coordinate
        jx (float, required): x component of 2D target neuron coordinate
        jy (float, required): y component of 2D target neuron coordinate

    Returns:
        float: Distance in 2D field
    """
    return np.sqrt((ix - jx)**2 + (iy - jy)**2)


def spikemon2firingRate(spikemon, start_time=0 * ms, end_time="max"):
    """Calculates the firing rate within a window of interest from a SpikeMonitor

    Args:
        spikemon (brian2.SpikeMonitor): Brian2 SpikeMoitor object
        start_time (brain2.unit.ms, optional): Starting point for window to calculate
            the firing rate. Must be provided as desired time in ms, e.g. 5 * ms
        end_time (str, optional): End point for window to calculate
            the firing rate. Must be provided as desired time in ms, e.g. 5 * ms

    Returns:
        int: Firing rate in Hz
    """
    spiketimes = (spikemon.t / ms)
    if len(spiketimes) == 0:
        return 0
    if end_time == "max":
        end_time = max(spikemon.t / ms)
    spiketimes = spiketimes[spiketimes <= end_time]
    spiketimes = spiketimes[spiketimes >= start_time / ms]
    spiketimes = spiketimes / 1000
    if len(spiketimes) == 0:
        return 0
    return(mean(1 / diff(spiketimes)))
