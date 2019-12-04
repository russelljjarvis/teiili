#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This is a collection of input-output functions.
Primary it provides wrapper functions to save and load
monitors and statevariables.
"""

import time
import numpy as np
import os
from brian2 import ms

class monitor_init():

    def __init__(self):
        """Creates an object monitor

        Returns:
            TYPE: Description
        """
        self.i = None
        self.t = None
        self.I_syn = None
        self.Vm = None
        self.Vthr = None
        self.Imem = None
        return None


def save_monitor(monitor, filename, path, unit=ms, variable=None):
    """Save Monitor using numpy.save()

    Args:
        monitor (brian2 monitor): Spikemonitor of brian2
        filename (str, required): String specifying the name
        path (str, required): Path/where/monitor/should/be/stored/
        unit (float, brain2.unit): Time unit of the monitor.
        variable (str, optional): Name of the variable to be loaded.
    """
    date = time.strftime("%d_%m_%Y")
    if not os.path.exists(path + date):
        os.mkdir(path + date + "/")
    path = path + date + "/"
    filename = time.strftime("%d_%m_%H_%M") + '_' + filename

    if variable is None:
        toSave = np.zeros((2, len(monitor.t))) * np.nan
        toSave[0, :] = np.asarray(monitor.i)
        toSave[1, :] = np.asarray(monitor.t / unit)
    else:
        toSave = np.asarray(getattr(monitor, variable))

    np.save(path + filename, toSave)


def load_monitor(filename, monitor_dt=1*ms, variable=None):
    """Load a saved spikemonitor using numpy.load()

    Args:
        filename (str, required): String specifying the name
        monitor_dt (float, second): Time step of Monitor.
        variable (str, optional): Name of the variable to be loaded.

    Returns:
        monitor obj.: A monitor with i and t attributes, reflecting
            neuron index and time of spikes
    """
    data = np.load(filename)
    monitor = monitor_init()
    if variable is None:
        monitor.i = data[0, :]
        monitor.t = data[1, :] * monitor_dt
    elif variable == 'Vthr':
        monitor.Vthr= data
        monitor.t = np.arange(0, len(data), monitor_dt)
    elif variable == 'I_syn':
        monitor.I_syn = data
        monitor.t = np.arange(0, len(data), monitor_dt)
    elif variable == 'Vm':
        monitor.Vm = data
        monitor.t = np.arange(0, len(data), monitor_dt)
    elif variable == 'Imem':
        monitor.Imem = data
        monitor.t = np.arange(0, len(data), monitor_dt)
    else:
        UserWarning("{} is currently not supported.".format(variable))

    return monitor

def save_weights(weights, filename, path):
    """Save weight matrix between to populations into .npy file

    Args:
        weights (TYPE): Description
        filename (str, required): String specifying the name
        path (str, required): Path/where/weights/should/be/stored/
    """
    date = time.strftime("%d_%m_%Y")
    if not os.path.exists(path + date):
        os.mkdir(path + date + "/")
    path = path + date + "/"
    filename = time.strftime("%d_%m_%H_%M") + '_' + filename
    toSave = np.zeros(np.shape(weights)) * np.nan
    toSave = weights
    np.save(path + filename, toSave)


def save_params(params, filename, path):
    """Save dictionary containing neuron/synapse paramters
    or simulation parameters.

    Args:
        params (dict, required): Dictionary containing parameters
            as keywords and associated values, which were needed
            for this simulation.
        filename (str, required): String specifying the name
        path (str, required): Path/where/params/should/be/stored/
    """
    date = time.strftime("%d_%m_%Y")
    if not os.path.exists(path + date):
        os.mkdir(path + date + "/")
    path = path + date + "/"
    filename = time.strftime("%d_%m_%H_%M") + '_' + filename
    np.save(path + filename, params)


def load_weights(filename=None, nrows=None, ncols=None):
    """Load weights from .npy file.

    Args:
        filename (str, optional): Absolute path from which /stored/weights.npy are loaded
        nrows (None, optional): Number of rows of the original non-flattened matrix
        ncols (None, optional): Number of columns of the original non-flattened matrix

    Returns:
        ndarray: Array containing the loaded weight matrix

    Raises:
        UserWarning: You need specify either nrows, ncols or both, otherwise reshape
            is not possible
    """
    if filename is not None:
        weight_matrix = np.load(filename)
    else:
        filename = filedialog.askopenfilename()
        weight_matrix = np.load(filename)

    if nrows is None and ncols is None:
        raise UserWarning(
            'Please specify the dimension of the matrix, since it not squared.')
    if ncols is None:
        weight_matrix = weight_matrix.reshape((nrows, -1))
    elif nrows is None:
        weight_matrix = weight_matrix.reshape((-1, ncols))
    else:
        weight_matrix = weight_matrix.reshape((nrows, ncols))
    return weight_matrix
