# -*- coding: utf-8 -*-
"""Summary
"""
# @Author: schlowmo
# @Date:   2018-05-06 13:27:12
# @Last Modified by:   Moritz Milde
# @Last Modified time: 2018-06-22 10:34:34

import numpy as np
import matplotlib.pyplot as plt

from teili.tools.indexing import ind2xy

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg


def plot_adjacency_matrix(synapse_group, variable='weight', window=None, statemon=None, debug=False):
    """This function will plot the adjacency matrix between to population of neurons.


    Args:
        synapse_group (Brian2 synapses obj.): The synapse object betweeen two neuron populations
        variable (str, optional): The variable which should be visualized, e.g. the static weight or
            the pastic weight ('w_plast')
        statemon (Monitor, optional): If no synapse object is provided a statemonitor object can
            be provided where only the last recorded instance of th variable will be displayed
        debug (bool, optional): To get more information of the shape and sizes of the adjacency matrix
    """
    num_source_neurons = synapse_group.source.N
    num_target_neurons = synapse_group.target.N

    x = np.arange(0, num_target_neurons, 1)
    y = np.arange(0, num_source_neurons, 1)
    X, Y = np.meshgrid(x, y)
    data = np.zeros((num_source_neurons, num_target_neurons)) * np.nan
    # Getting sparse weights
    if debug:
        print("Shape of statemon: {}".format(np.shape(statemon.w_plast)))
    if statemon is None:
        data[synapse_group.i, synapse_group.j] = synapse_group.__getattr__(
            variable)
    else:
        w_plast = statemon.__getattr__(variable)[:, -2:-1]
    if debug:
        print("Shape of plastic weight matrix: {}".format(np.shape(w_plast)))
        print("Shape of data to be plotted: {}".format(np.shape(data)))
        print("Shape of meshgrid. X: {}, Y: {}".format(np.shape(X), np.shape(Y)))
        print("Shape of source population: {}".format(np.shape(synapse_group.i)))
        print("Shape of target population: {}".format(synapse_group.j))
        print("Shape of weight matrix: {}".format(np.shape(w_plast)))
    try:
        if statemon is not None:
            data[synapse_group.i, synapse_group.j] = w_plast[:, 0]
        if debug:
            print("Shape of plastic weight matrix to be plotted {}".format(
                np.shape(w_plast[:, 0])))
    except IndexError:
        if debug:
            print("W plast: {}".format(w_plast))
            print("Squeezed weight matrix: {}".format(
                np.shape(np.squeeze(w_plast))))
        data[synapse_group.i, synapse_group.j] = synapse_group.__getattr__(
            variable)
    data[np.isnan(data)] = 0
    if debug:
        print(data)

    if window is None:
        cm = plt.cm.get_cmap('jet')
        fig = plt.figure()
        plt.pcolor(X, Y, data, cmap=cm, vmin=0, vmax=1)
        plt.colorbar()
        plt.xlim((0, np.max(x)))
        plt.ylim((0, np.max(y)))
        plt.ylabel('Source neuron index')
        plt.xlabel('Target neuron index')
        plt.title('Adjacency matrix between {} and {}'.format(synapse_group.source.name,
                                                              synapse_group.target.name))
        plt.draw()

    else:
        data = np.zeros((1, num_source_neurons, num_target_neurons)) * np.nan
        data[:, synapse_group.i, synapse_group.j] = synapse_group.w_plast
        pos = np.array([0.0, 0.5, 1.0])
        # color needs as many colors as postions are specified
        color = np.array([[0, 0, 255, 255],
                         [255, 255, 255, 255],
                         [255, 102, 0, 255]], dtype=np.ubyte)
        cm = pg.ColorMap(pos, color)
        imv1 = pg.ImageView()
        imv1.setImage(data)
        imv1.setColorMap(cm)

        return imv1


def plot_receptive_fields_sorted(synapse_group, variable='weight', index=0):
    """Summary

    Args:
        synapse_group (TYPE): Description
        variable (str, optional): Description
        index (TYPE): Description
    """
    cm = plt.cm.get_cmap('jet')
    f, axarr = plt.subplots(np.sqrt(synapse_group.target.N).astype(int),
                            np.sqrt(synapse_group.target.N).astype(int))
    data = np.reshape(synapse_group.__getattr__(variable),
                      (synapse_group.source.N, synapse_group.target.N))
    R = np.corrcoef(data.T)
    R_sorted = sorted(R[index])
    sorted_indices = np.argsort(R[index])

    for ind, target in enumerate(sorted_indices):
        cInd_post = synapse_group.j == target
        cInd_pre = synapse_group.i[cInd_post]

        weights = np.asarray(synapse_group.w_plast)[cInd_post]

        x_pre, y_pre = ind2xy(np.asarray(synapse_group.i)[cInd_post],
                              np.sqrt(synapse_group.source.N).astype(int))

        x_post, y_post = ind2xy(np.asarray(synapse_group.j)[cInd_post],
                                np.sqrt(synapse_group.target.N).astype(int))

        sub_x, sub_y = ind2xy(ind, np.sqrt(synapse_group.target.N).astype(int))

        x = np.arange(0, np.max(x_pre) + 1, 1)
        y = np.arange(0, np.max(y_pre) + 1, 1)
        X, Y = np.meshgrid(x, y)
        data = np.zeros((np.max(x_pre) + 1, np.max(y_pre) + 1)) * np.nan
        data[x_pre, y_pre] = weights

        im = axarr[sub_x, sub_y].pcolor(X, Y, data, cmap=cm, vmin=0, vmax=1)

        axarr[sub_x, sub_y].set_xlim((0, np.max(x_pre)))
        axarr[sub_x, sub_y].set_ylim((0, np.max(y_pre)))
        axarr[sub_x, sub_y].get_xaxis().set_visible(False)
        axarr[sub_x, sub_y].get_yaxis().set_visible(False)

    cbar_ax = f.add_axes([0.91, 0.13, 0.05, 0.75])
    f.colorbar(im, cax=cbar_ax)
    plt.draw()
    plt.savefig(
        '/home/schlowmo/Repositories/OCTA/plots/receptive_fields_sorted_{}.pdf'.format(index))


def plot_receptive_fields(synapse_group, variable='weight'):
    """Summary

    Args:
        synapse_group (TYPE): Description
        variable (str, optional): Description
    """
    cm = plt.cm.get_cmap('jet')
    f, axarr = plt.subplots(np.sqrt(synapse_group.target.N).astype(int),
                            np.sqrt(synapse_group.target.N).astype(int))
    for target in range(np.max(synapse_group.j) + 1):
        cInd_post = synapse_group.j == target
        cInd_pre = synapse_group.i[cInd_post]

        weights = np.asarray(synapse_group.__getattr__(variable))[cInd_post]

        x_pre, y_pre = ind2xy(np.asarray(synapse_group.i)[cInd_post],
                              np.sqrt(synapse_group.source.N).astype(int))

        x_post, y_post = ind2xy(np.asarray(synapse_group.j)[cInd_post],
                                np.sqrt(synapse_group.target.N).astype(int))

        sub_x, sub_y = ind2xy(target, np.sqrt(
            synapse_group.target.N).astype(int))
        x = np.arange(0, np.max(x_pre) + 1, 1)
        y = np.arange(0, np.max(y_pre) + 1, 1)
        X, Y = np.meshgrid(x, y)
        data = np.zeros((np.max(x_pre) + 1, np.max(y_pre) + 1)) * np.nan
        data[x_pre, y_pre] = weights

        im = axarr[sub_x, sub_y].pcolor(X, Y, data, cmap=cm, vmin=0, vmax=1)

        axarr[sub_x, sub_y].set_xlim((0, np.max(x_pre)))
        axarr[sub_x, sub_y].set_ylim((0, np.max(y_pre)))
        axarr[sub_x, sub_y].get_xaxis().set_visible(False)
        axarr[sub_x, sub_y].get_yaxis().set_visible(False)

    cbar_ax = f.add_axes([0.91, 0.13, 0.05, 0.75])
    f.colorbar(im, cax=cbar_ax)
    plt.draw()
