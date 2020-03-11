#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A collection of helpful miscellaneous functions when working with brian2
"""
# @Author: mmilde, alpren
# @Date:   2017-12-27 11:54:09

from brian2 import implementation, check_units, ms, declare_types
import numpy as np


# This function is a workaround to allow if statements in run_regularly code.
# It is necessary for example in order to set values conditional on the current time.
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
    This function is a workaround to allow if statements in run_regularly code.
    It is necessary for example in order to set values conditional on the current time.
    It returns a value (return_val_true or return_val_false) depending on whether test_val is between
    smaller_than_val and greater_than_val or not.
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


def print_states(briangroup):
    """Wrapper function to print states of a brian2 groups such as NeuronGroup or Synapses

    Args:
        briangroup (brian2.group): Brain object/group which states/statevariables
            should be printed
    """
    states = briangroup.get_states()
    print('\n')
    print('-_-_-_-_-_-_-_')
    print(briangroup.name)
    print('list of states and first value:')
    for key in states.keys():
        if states[key].size > 1:
            print(key, states[key][1])
        else:
            print(key, states[key])
    print('----------')


def spikemon2firing_rate(spikemon, start_time=0 * ms, end_time="max"):
    """Calculates the instantaneous firing rate within a window of interest from a SpikeMonitor

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
    return (np.mean(1 / np.diff(spiketimes)))
