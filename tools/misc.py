#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: mmilde, alpren
# @Date:   2017-12-27 11:54:09
# @Last Modified by:   mmilde
# @Last Modified time: 2017-12-27 12:13:11
"""A collection of helpful functions when working with brian2

"""

from brian2 import implementation, check_units, ms, exp, mean, diff, declare_types,\
    figure, subplot, plot, xlim, ylim, ones, zeros, xticks, xlabel, ylabel, device
import numpy as np


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


def printStates(briangroup):
    """Summary

    Args:
        briangroup (TYPE): Description
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
float returnValueIf(float testVal, float greaterThanVal, float smallerThanVal, float returnValTrue, float returnValFalse) {
    if ((testVal > greaterThanVal) && (testVal < smallerThanVal))
        return returnValTrue;
    else
        return returnValFalse;
}
''')
@declare_types(testVal='float', greaterThanVal='float', smallerThanVal='float',
               returnValTrue='float', returnValFalse='float', result='float')
@check_units(testVal=1, greaterThanVal=1, smallerThanVal=1, returnValTrue=1, returnValFalse=1, result=1)
def returnValueIf(testVal, greaterThanVal, smallerThanVal, returnValTrue, returnValFalse):
    """Summary
    This function is a workaround to allow if statements in run_regularly code
    It is e.g. necessary in order to set values conditional on the current time
    it returns a value (returnValTrue or returnValFalse) depending on whether testVal is between
    smallerThanVal and greaterThanVal or not
    Args:
        testVal (TYPE): the value that is tested
        greaterThanVal (TYPE): upper bound of the value
        smallerThanVal (TYPE): lower bound of the value
        returnValTrue (TYPE): value returned if testVal is in bounds
        returnValFalse (TYPE): value returned if testVal is out of bounds

    Returns:
        float: returns a specified value (returnValTrue or returnValFalse) depending on whether testVal is between
    smallerThanVal and greaterThanVal
    """
    if (testVal > greaterThanVal and testVal < smallerThanVal):
        return returnValTrue
    else:
        return returnValFalse


@implementation('cpp', '''
    float dist2dind(int ix, int iy,int jx, int jy) {
    return sqrt(pow((ix - jx),2) + pow((iy - jy),2));
    }
     ''')
@declare_types(ix='integer', iy='integer', jx='integer', jy='integer', result='float')
@check_units(ix=1, iy=1, jx=1, jy=1, result=1)
def dist2dint(ix, iy, jx, jy):
    """Summary: function that calculates distance in 2D field from 4 integer 2D indices

    Args:
        ix (TYPE): Description
        iy (TYPE): Description
        jx (TYPE): Description
        jy (TYPE): Description

    Returns:
        TYPE: Description
    """
    return np.sqrt((ix - jx)**2 + (iy - jy)**2)


@implementation('cpp', '''
    float dist2d(float ix, float iy,float jx, float jy) {
    return sqrt(pow((ix - jx),2) + pow((iy - jy),2));
    }
     ''')
@declare_types(ix='float', iy='float', jx='float', jy='float', result='float')
@check_units(ix=1, iy=1, jx=1, jy=1, result=1)
def dist2d(ix, iy, jx, jy):
    """Summary: function that calculates distance in 2D field from 4 2D position values

    Args:
        ix (TYPE): Description
        iy (TYPE): Description
        jx (TYPE): Description
        jy (TYPE): Description

    Returns:
        TYPE: Description
    """
    return np.sqrt((ix - jx)**2 + (iy - jy)**2)


def spikemon2firingRate(spikemon, fromT=0 * ms, toT="max"):
    """Summary

    Args:
        spikemon (TYPE): Description
        fromT (TYPE, optional): Description
        toT (str, optional): Description

    Returns:
        TYPE: Description
    """
    spiketimes = (spikemon.t / ms)
    if len(spiketimes) == 0:
        return 0
    if toT == "max":
        toT = max(spikemon.t / ms)
    spiketimes = spiketimes[spiketimes <= toT]
    spiketimes = spiketimes[spiketimes >= fromT / ms]
    spiketimes = spiketimes / 1000
    if len(spiketimes) == 0:
        return 0
    return(mean(1 / diff(spiketimes)))


def gaussian(squareSize, sigma=1, mu=None):
    """Make a square gaussian kernel

    Args:
        squareSize (TYPE): Description
        sigma (int, optional): Description
        mu (None, optional): Description

    Returns:
        TYPE: Description
    """

    x = np.arange(0, squareSize)
    y = x[:, np.newaxis]

    if mu is None:
        x0 = y0 = squareSize // 2
    else:
        x0 = mu[0]
        y0 = mu[1]

    return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
