#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: alpha, mmilde
# @Date:   2017-07-31 16:13:59
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-12 15:20:18

"""
Plotting tools for different spike and state monitors
"""
import numpy as np
from brian2 import ms, mV, pA, plot, xlabel, ylabel, xlim, ylim
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg


colors = [(255, 0, 0), (89, 198, 118), (0, 0, 255), (247, 0, 255),
          (0, 0, 0), (255, 128, 0), (120, 120, 120), (0, 171, 255)]
labelStyle = {'color': '#FFF', 'font-size': '12pt'}

# this is to make plotting that starts at a certain time easier


def plotSpikemon(startTime, endTime, SpikeMon, nNeurons, ylab='ind'):
    if len(SpikeMon.t) > 1:
        indstart = abs(SpikeMon.t - startTime).argmin()
        indend = abs(SpikeMon.t - endTime).argmin()
        plot(np.asarray(SpikeMon.t / ms)[indstart:indend], np.asarray(SpikeMon.i)[indstart:indend], '.k')
        xlabel('Time [ms]')
        ylabel(ylab)
        xlim([startTime / ms, endTime / ms])
        if nNeurons is not None:
            ylim([0, nNeurons])


def plot_spikemon_qt(start_time=None, end_time=None, monitor=None, num_neurons=16, window=None):
    """Generic plotting function to plot spikemonitors using pyqtgraph

    Args:
        start_time (int, optional): Description
        end_time (int, optional): Description
        monitor (brian2.obj): Monitor which serve as basis for plotting
        num_neurons (int): Number of neurons to be plotted
        window (TYPE): PyQtGraph window to which the plot should be added
    """
    if len(monitor.t) > 1:
        if start_time is None:
            start_time = 0 * ms
        if end_time is None:
            end_time = monitor.t.argmax()
        if window is None:
            raise UserWarning("Please provide plot_statemon_qt with pyqtgraph window.")
        if monitor is None:
            raise UserWarning("No statemonitor provided. Abort plotting")

        start_ind = abs(monitor.t - start_time).argmin()
        end_ind = abs(monitor.t - end_time).argmin()

        window.setXRange(0, end_time / ms, padding=0)
        window.plot(x=np.asarray(monitor.t / ms)[start_ind:end_ind], y=np.asarray(monitor.i)[start_ind:end_ind],
                    pen=None, symbol='o', symbolPen=None,
                    symbolSize=7, symbolBrush=colors[1])
        window.setLabel('left', "Neuron ID", **labelStyle)
        window.setLabel('bottom', "Time (ms)", **labelStyle)
        b = QtGui.QFont("Sans Serif", 10)
        window.getAxis('bottom').tickFont = b
        window.getAxis('left').tickFont = b


def plotStatemon(startTime, endTime, StateMon, neuronInd, variable='Vm', unit=mV, name=''):
    indstart = abs(StateMon.t - startTime).argmin()
    indend = abs(StateMon.t - endTime).argmin()
    plot(StateMon.t[indstart:indend] / ms, StateMon[neuronInd].__getattr__(variable)[indstart:indend] / unit)
    xlabel('Time [ms]')
    ylabel(name + '_' + variable + ' (' + str(unit) + ')')
    xlim([startTime / ms, endTime / ms])


def plot_statemon_qt(start_time=None, end_time=None, monitor=None, neuron_id=True,
                     variable="Imem", unit=pA, window=None, name=''):
    """Generic plotting function to plot statemonitors using pyqtgraph

    Args:
        start_time (int, optional): Description
        end_time (int, optional): Description
        monitor (brian2.obj): Monitor which serve as basis for plotting
        neuron_id (TYPE): Description
        variable (TYPE): Description
        window (TYPE): PyQtGraph window to which the plot should be added
        name (str, optional): Description
    """
    if start_time is None:
        start_time = 0 * ms
    if end_time is None:
        end_time = monitor.t.argmax()
    if window is None:
        raise UserWarning("Please provide plot_statemon_qt with pyqtgraph window.")
    if monitor is None:
        raise UserWarning("No statemonitor provided. Abort plotting")

    start_ind = abs(monitor.t - start_time).argmin()
    end_ind = abs(monitor.t - end_time).argmin()

    window.setXRange(0, end_time / ms, padding=0)
    for i, data in enumerate(np.asarray(monitor.__getattr__(variable) / unit)[start_ind:end_ind]):
        window.plot(x=np.asarray(monitor.t / ms)[start_ind:end_ind], y=data[:-1],
                pen=pg.mkPen(colors[6], width=2))

    window.setLabel('left', name + '_' + variable + ' (' + str(unit) + ')', **labelStyle)
    window.setLabel('bottom', "Time (ms)", **labelStyle)
    b = QtGui.QFont("Sans Serif", 10)
    window.getAxis('bottom').tickFont = b
    window.getAxis('left').tickFont = b


def plotWeightsWtatoGroup(name, nWTA2dNeurons, synWtaG, nCol):

    mat = np.reshape(synWtaG.w, (nWTA2dNeurons, nWTA2dNeurons, -1))
    imgShape = np.shape(mat)
    print(imgShape)
    nPlots = imgShape[2]
    #nCol = 10
    fig, axarr = plt.subplots(nPlots // nCol, nCol)  # ,sharex=True,sharey=True)
    for i in range(nPlots):
        axarr[i // nCol, np.mod(i, nCol)].set_xlim(0, nWTA2dNeurons)
        axarr[i // nCol, np.mod(i, nCol)].set_ylim(0, nWTA2dNeurons)
        axarr[i // nCol, np.mod(i, nCol)].axes.get_xaxis().set_visible(False)
        axarr[i // nCol, np.mod(i, nCol)].axes.get_yaxis().set_visible(False)
        axarr[i // nCol, np.mod(i, nCol)].set_xticklabels([])
        axarr[i // nCol, np.mod(i, nCol)].set_yticklabels([])
        axarr[i // nCol, np.mod(i, nCol)].autoscale(False)
        # axarr[i//nCol,np.mod(i,nCol)].set(aspect='equal')#adjustable='box-forced',
        img = axarr[i // nCol, np.mod(i, nCol)].imshow(mat[:, :, i], cmap=plt.cm.binary, vmin=0, vmax=1)
        # img.set_aspect(1)
        # print(np.shape(mat[i,:,:]))
    #plt.colorbar(img, ax=axarr[i//nCol,np.mod(i,nCol)],ticks=[0,0.5,1])
    # fig.savefig('fig/'+name+'.png',dpi=400)


def plotWeightsGrouptoWta(name, nWTA2dNeurons, synGWta, nCol):

    mat = np.reshape(synGWta.w, (-1, nWTA2dNeurons, nWTA2dNeurons))
    imgShape = np.shape(mat)
    print(imgShape)
    nPlots = imgShape[0]
    #nCol = 10
    fig, axarr = plt.subplots(nPlots // nCol, nCol)  # ,sharex=True,sharey=True)
    for i in range(nPlots):
        axarr[i // nCol, np.mod(i, nCol)].set_xlim(0, nWTA2dNeurons)
        axarr[i // nCol, np.mod(i, nCol)].set_ylim(0, nWTA2dNeurons)
        axarr[i // nCol, np.mod(i, nCol)].axes.get_xaxis().set_visible(False)
        axarr[i // nCol, np.mod(i, nCol)].axes.get_yaxis().set_visible(False)
        axarr[i // nCol, np.mod(i, nCol)].set_xticklabels([])
        axarr[i // nCol, np.mod(i, nCol)].set_yticklabels([])
        axarr[i // nCol, np.mod(i, nCol)].autoscale(False)
        # axarr[i//nCol,np.mod(i,nCol)].set(aspect='equal')#adjustable='box-forced',
        img = axarr[i // nCol, np.mod(i, nCol)].imshow(mat[i, :, :], cmap=plt.cm.binary, vmin=0, vmax=1)
        # img.set_aspect(1)
        # print(np.shape(mat[i,:,:]))
    #plt.colorbar(img, ax=axarr[i//nCol,np.mod(i,nCol)],ticks=[0,0.5,1])
    # fig.savefig('fig/'+name+'.png',dpi=400)
