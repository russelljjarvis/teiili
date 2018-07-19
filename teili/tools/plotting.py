#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summary

Attributes:
    colors (TYPE): Description
    labelStyle (dict): Description
"""
# @Author: alpha, mmilde
# @Date:   2017-07-31 16:13:59
# @Last Modified by:   Moritz Milde
# @Last Modified time: 2018-06-20 08:42:47

"""
Plotting tools for different spike and state monitors
"""
import numpy as np
from brian2 import ms, mV, pA
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, xlim, ylim
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg


colors = [(255, 0, 0), (89, 198, 118), (0, 0, 255), (247, 0, 255),
          (0, 0, 0), (255, 128, 0), (120, 120, 120), (0, 171, 255)]
labelStyle = {'color': '#FFF', 'font-size': '12pt'}

# this is to make plotting that starts at a certain time easier


def plot_spikemon(start_time, end_time, monitor, num_neurons, ylab='ind'):
    """Summary

    Args:
        start_time (TYPE): Time from which spikes should be visualized
        end_time (TYPE): Time until which spikes should be visualized
        monitor (brian2.monitoritor): Monitor which serve as basis for plotting
        num_neurons (int): Number of neurons to visualize
        ylab (str, optional): Description
    """
    if len(monitor.t) > 1:
        indstart = abs(monitor.t - start_time).argmin()
        indend = abs(monitor.t - end_time).argmin()
        plot(np.asarray(monitor.t / ms)
             [indstart:indend], np.asarray(monitor.i)[indstart:indend], '.k')
        xlabel('Time [ms]')
        ylabel(ylab)
        xlim([start_time / ms, end_time / ms])
        if num_neurons is not None:
            ylim([0, num_neurons])


def plot_spikemon_qt(monitor, start_time=None, end_time=None, num_neurons=16, window=None, unit=None):
    """Generic plotting function to plot spikemonitors using pyqtgraph

    Args:
        start_time (int, optional): Time from which spikes should be visualized
        end_time (int, optional): Time until which spikes should be visualized
        monitor (brian2.obj): Monitor which serve as basis for plotting
        num_neurons (int): Number of neurons to be plotted
        window (TYPE): PyQtGraph window to which the plot should be added

    Raises:
        UserWarning: Description
    """
    if unit is None:
        try:  # .dim raises an Error if monitor.t is not a brian2 Quantity
            if str(monitor.t.dim) == 's':
                unit = ms
            else:
                pass  # this should not happen
        except AttributeError:
            unit = 1

    if len(monitor.t) > 1:
        if start_time is None:
            start_time = 0 * unit
        else:
            try:
                start_time.dim
            except AttributeError:
                start_time = start_time * unit
        if end_time is None:
            end_time = monitor.t[-1]
        else:
            try:
                start_time.dim
            except AttributeError:
                start_time = start_time * unit
        if window is None:
            raise UserWarning(
                "Please provide plot_statemon_qt with pyqtgraph window.")
        if monitor is None:
            raise UserWarning("No statemonitor provided. Abort plotting")
        else:
            start_ind = np.argmin(abs(monitor.t - start_time))
            end_ind = np.argmin(abs(monitor.t - end_time))

            window.setXRange(0, end_time / unit, padding=0)
            window.setYRange(0, num_neurons)

            window.plot(x=np.asarray(monitor.t / unit)[start_ind:end_ind] - np.array(monitor.t / unit)[start_ind],
                        y=np.asarray(monitor.i)[start_ind:end_ind],
                        pen=None, symbol='o', symbolPen=None,
                        symbolSize=7, symbolBrush=colors[1])
            window.setLabel('left', "Neuron ID", **labelStyle)
            window.setLabel('bottom', 'Time ({})'.format(
                str(unit)), **labelStyle)
            b = QtGui.QFont("Sans Serif", 10)
            window.getAxis('bottom').tickFont = b
            window.getAxis('left').tickFont = b

    else:
        print("monitor is empty")

    return window


def plot_statemon(start_time, end_time, monitor, neuron_id, variable='Vm', unit=mV, name=''):
    """Summary

    Args:
        start_time (int, optional): Time from which spikes should be visualized
        end_time (int, optional): Time until which spikes should be visualized
        monitor (brian2.obj): Monitor which serve as basis for plotting
        neuron_id (int): ID of neuron to be visualized
        variable (str): State variable to visualize
        unit (brian2.unit, optional): Unit of state variable
        name (str, optional): Description
    """
    indstart = abs(monitor.t - start_time).argmin()
    indend = abs(monitor.t - end_time).argmin()
    plot(monitor.t[indstart:indend] / ms,
         monitor[neuron_id].__getattr__(variable)[indstart:indend] / unit)
    xlabel('Time [ms]')
    ylabel(name + '_' + variable + ' (' + str(unit) + ')')
    xlim([start_time / ms, end_time / ms])


def plot_statemon_qt(start_time=None, end_time=None, monitor=None, neuron_id=True,
                     variable="Imem", unit=pA, window=None, name=''):
    """Generic plotting function to plot statemonitors using pyqtgraph

    Args:
        start_time (int, optional): Time from which spikes should be visualized
        end_time (int, optional): Time until which spikes should be visualized
        monitor (brian2.obj): Monitor which serve as basis for plotting
        neuron_id (int): ID of neuron to be visualized
        variable (str): State variable to visualize
        unit (brian2.unit, optional): Unit of state variable
        window (pyqtgraph.window): PyQtGraph window to which the plot should be added
        name (str, optional): Name of window

    Raises:
        UserWarning: Description
    """
    if start_time is None:
        start_time = 0 * ms
    if end_time is None:
        end_time = monitor.t[-1]
    if window is None:
        raise UserWarning(
            "Please provide plot_statemon_qt with pyqtgraph window.")
    if monitor is None:
        raise UserWarning("No statemonitor provided. Abort plotting")

    start_ind = abs(monitor.t - start_time).argmin()
    end_ind = abs(monitor.t - end_time).argmin()

    window.setXRange(0, end_time / ms, padding=0)
    for i, data in enumerate(np.asarray(monitor.__getattr__(variable) / unit)[start_ind:end_ind]):
        window.plot(x=np.asarray(monitor.t / ms)[start_ind:end_ind], y=data[:-1],
                    pen=pg.mkPen(colors[6], width=2))

    window.setLabel('left', name + '_' + variable +
                    ' (' + str(unit) + ')', **labelStyle)
    window.setLabel('bottom', "Time (ms)", **labelStyle)
    b = QtGui.QFont("Sans Serif", 10)
    window.getAxis('bottom').tickFont = b
    window.getAxis('left').tickFont = b


def plot_weights_wta2group(name, n_wta2d_neurons, syn_g_wta, n_col):
    """Summary

    Args:
        name (str): Name of the plot to be saved
        n_wta2d_neurons (TYPE): Number of 2d WTA population
        syn_g_wta (TYPE): Synapse group which weights should be plotted
        n_col (TYPE): Number of column to visualize
    """
    mat = np.reshape(syn_g_wta.w, (n_wta2d_neurons, n_wta2d_neurons, -1))
    imgShape = np.shape(mat)
    print(imgShape)
    nPlots = imgShape[2]
    #n_col = 10
    # ,sharex=True,sharey=True)
    fig, axarr = plt.subplots(nPlots // n_col, n_col)
    for i in range(nPlots):
        axarr[i // n_col, np.mod(i, n_col)].set_xlim(0, n_wta2d_neurons)
        axarr[i // n_col, np.mod(i, n_col)].set_ylim(0, n_wta2d_neurons)
        axarr[i // n_col, np.mod(i, n_col)].axes.get_xaxis().set_visible(False)
        axarr[i // n_col, np.mod(i, n_col)].axes.get_yaxis().set_visible(False)
        axarr[i // n_col, np.mod(i, n_col)].set_xticklabels([])
        axarr[i // n_col, np.mod(i, n_col)].set_yticklabels([])
        axarr[i // n_col, np.mod(i, n_col)].autoscale(False)
        # axarr[i//n_col,np.mod(i,n_col)].set(aspect='equal')#adjustable='box-forced',
        img = axarr[i // n_col, np.mod(i, n_col)].imshow(
            mat[:, :, i], cmap=plt.cm.binary, vmin=0, vmax=1)
        # img.set_aspect(1)
        # print(np.shape(mat[i,:,:]))
    #plt.colorbar(img, ax=axarr[i//n_col,np.mod(i,n_col)],ticks=[0,0.5,1])
    # fig.savefig('fig/'+name+'.png',dpi=400)


def plot_weights_group2wta(name, n_wta2d_neurons, syn_g_wta, n_col):
    """Summary

    Args:
        name (str): Name of the plot to be saved
        n_wta2d_neurons (int): Number of 2d WTA population
        syn_g_wta (brian2.synapses): Synapse group which weights should be plotted
        n_col (int):  Number of column to visualize
    """
    mat = np.reshape(syn_g_wta.w, (-1, n_wta2d_neurons, n_wta2d_neurons))
    imgShape = np.shape(mat)
    print(imgShape)
    nPlots = imgShape[0]
    #n_col = 10
    # ,sharex=True,sharey=True)
    fig, axarr = plt.subplots(nPlots // n_col, n_col)
    for i in range(nPlots):
        axarr[i // n_col, np.mod(i, n_col)].set_xlim(0, n_wta2d_neurons)
        axarr[i // n_col, np.mod(i, n_col)].set_ylim(0, n_wta2d_neurons)
        axarr[i // n_col, np.mod(i, n_col)].axes.get_xaxis().set_visible(False)
        axarr[i // n_col, np.mod(i, n_col)].axes.get_yaxis().set_visible(False)
        axarr[i // n_col, np.mod(i, n_col)].set_xticklabels([])
        axarr[i // n_col, np.mod(i, n_col)].set_yticklabels([])
        axarr[i // n_col, np.mod(i, n_col)].autoscale(False)
        # axarr[i//n_col,np.mod(i,n_col)].set(aspect='equal')#adjustable='box-forced',
        img = axarr[i // n_col, np.mod(i, n_col)].imshow(
            mat[i, :, :], cmap=plt.cm.binary, vmin=0, vmax=1)
        # img.set_aspect(1)
        # print(np.shape(mat[i,:,:]))
    #plt.colorbar(img, ax=axarr[i//n_col,np.mod(i,n_col)],ticks=[0,0.5,1])
    # fig.savefig('fig/'+name+'.png',dpi=400)
