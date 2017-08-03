'''
Created 03.2017
This file contains plot functions for WTA circuits
@author: Alpha
'''

from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second, Network, prefs, SpikeGeneratorGroup, NeuronGroup,\
    Synapses, SpikeMonitor, StateMonitor, figure, plot, show, xlabel, ylabel,\
    seed, xlim, ylim, subplot
from brian2 import *
import matplotlib.pyplot as plt
from NCSBrian2Lib.Tools.tools import xy2ind, ind2xy
import numpy as np


def plotWTATiles(name, startTime, endTime, nWTA2dNeurons, spikemonWTA, interval=10 * ms, nCol=10, showfig=False, savepath=False, maxFiringRate=300, tilecolors=[]):
    'Plot a 2d WTA as tiles over time'
    duration = endTime - startTime

    indstart = np.abs(spikemonWTA.t - startTime).argmin()
    indend = np.abs(spikemonWTA.t - endTime).argmin()
    spikemonWTA_t = (spikemonWTA.t[indstart:indend] - startTime) / ms
    spikemonWTA_i = spikemonWTA.i[indstart:indend]

    interval = interval / ms
    duration = duration / ms
    # division by ms is done here, as brian2's unit division is not precise (?) and it might give wrong results e.g. with np.ceil()

    nPlots = int(np.ceil(duration / interval))
    nRow = int(np.ceil(nPlots / nCol))
    #print('nPlots: '+str(nPlots))
    #print('nCol: '+str(nCol))
    #print('nRow: '+str(nRow))
    if nRow < 2:
        nRow = 2  # otherwise axarr is 1 dimensional and cannot be indexed with 2 indices

    if 0:
        fig, axarr = plt.subplots(nRow, nCol, sharex=True, sharey=True, figsize=(nCol, max(1, nPlots // nCol) + 1))
        for i in range(nPlots):
            start = i * interval
            inds1d = spikemonWTA_i[np.logical_and(start < spikemonWTA_t, spikemonWTA_t < (start + interval))]
            inds2d = ind2xy(inds1d, nWTA2dNeurons)
            # print(i//nCol)
            # print(np.mod(i,nCol))
            axarr[i // nCol, np.mod(i, nCol)].set_xlim(0, nWTA2dNeurons)
            axarr[i // nCol, np.mod(i, nCol)].set_ylim(0, nWTA2dNeurons)
            axarr[i // nCol, np.mod(i, nCol)].plot(inds2d[0], inds2d[1], '.k')
        # fig.savefig('fig/'+name+'_tiles.png')

    fig, axarr = plt.subplots(nRow, nCol, sharex=True, sharey=True, figsize=(nCol, max(1, nPlots // nCol) + 1))
    for i in range(nPlots):
        start = i * interval
        inds1d = spikemonWTA_i[np.logical_and(start < spikemonWTA_t, spikemonWTA_t < (start + interval))]
        hist1d = np.histogram(inds1d, bins=range(nWTA2dNeurons**2 + 1))[0]
        hist1d = hist1d / (interval / (1000))
        hist2d = np.reshape(hist1d, (nWTA2dNeurons, nWTA2dNeurons))
        inds2d = ind2xy(inds1d, nWTA2dNeurons)
        axarr[i // nCol, np.mod(i, nCol)].set_xlim(0, nWTA2dNeurons)
        axarr[i // nCol, np.mod(i, nCol)].set_ylim(0, nWTA2dNeurons)
        # axarr[i//nCol,np.mod(i,nCol)].set_xticks(np.arange(nWTA2dNeurons)+0.5)
        # axarr[i//nCol,np.mod(i,nCol)].set_yticks(np.arange(nWTA2dNeurons)+0.5)
        # axarr[i//nCol,np.mod(i,nCol)].set_xticklabels([])
        # axarr[i//nCol,np.mod(i,nCol)].set_yticklabels([])

        # axarr[i//nCol,np.mod(i,nCol)].grid(True,linestyle='-')
        axarr[i // nCol, np.mod(i, nCol)].autoscale(False)
        if len(tilecolors) > 1:
            axarr[i // nCol, np.mod(i, nCol)].set_facecolor(tilecolors[i])
        axarr[i // nCol, np.mod(i, nCol)].imshow(hist2d, cmap=plt.cm.jet, clim=(0, maxFiringRate))
    if not savepath:
        pass
    else:
        fig.savefig(savepath + '/' + name + '_tiles_col.png', dpi=500)
    if showfig:
        plt.show()
    return
