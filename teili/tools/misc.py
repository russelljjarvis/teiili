#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A collection of helpful miscellaneous functions when working with brian2
"""
# @Author: mmilde, alpren
# @Date:   2017-12-27 11:54:09

from brian2 import implementation, check_units, ms, declare_types,\
        SpikeMonitor, Network, NeuronGroup, TimedArray
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


def neuron_group_from_spikes(num_inputs, simulation_dt, duration,
                             poisson_group=None, spike_indices=None,
                             spike_times=None):
    """Converts spike activity in a neuron poisson_group with the same
    activity.

    Args:
        num_inputs (int): Number of input channels from source.
        simulation_dt (brian2.unit.ms): Time step of simulation.
        duration (int): Duration of simulation in brian2.ms.
        poisson_group (brian2.poissonGroup): Poisson poisson_group that is
            passed instead of spike times and indices.
        spike_indices (numpy.array): Indices of the original source.
        spike_times (numpy.array): Time stamps with unit of original spikes in
            ms.

    Returns:
        neu_group (brian2 object): Neuron poisson_group with mimicked activity.
    """
    if spike_indices is None and spike_indices is None:
        net = Network()
        try:
            monitor = SpikeMonitor(poisson_group)
        except TypeError:
            raise
        net.add(poisson_group, monitor)
        net.run(duration)
        spike_times, spike_indices = monitor.t, monitor.i
    spike_times = [spike_times[np.where(spike_indices == i)[0]]
                   for i in range(num_inputs)]
    # Create matrix where each row (neuron id) is associated with time when
    # there is a spike or -1 when there is not
    converted_input = (np.zeros((num_inputs,
                                 np.around(duration/simulation_dt).astype(int)+1
                                 ))
                       - 1)*simulation_dt
    for ind, val in enumerate(spike_times):
        # Prevents floating point errors
        int_values = np.around(val/simulation_dt).astype(int)

        converted_input[ind, int_values] = int_values * simulation_dt
    converted_input = np.transpose(converted_input)
    converted_input = TimedArray(converted_input, dt=simulation_dt)
    # t is simulation time, and will be equal to tspike when there is a spike.
    # Cell remains refractory when there is no spike, i.e. when tspike=-1
    neu_group = NeuronGroup(num_inputs,
                            model='tspike=converted_input(t, i): second',
                            threshold='t==tspike',
                            refractory='tspike < 0*ms')
    neu_group.namespace.update({'converted_input': converted_input})

    return neu_group
