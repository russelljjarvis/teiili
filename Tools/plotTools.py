#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:13:59 2017

@author: alpha
"""
import numpy as np
from brian2 import ms, mV, plot, xlabel, ylabel, xlim, ylim
import matplotlib.pyplot as plt

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


def plotStatemon(startTime, endTime, StateMon, neuronInd, variable='Vm', unit=mV, name = ''):
    indstart = abs(StateMon.t - startTime).argmin()
    indend = abs(StateMon.t - endTime).argmin()
    plot(StateMon.t[indstart:indend] / ms, StateMon[neuronInd].__getattr__(variable)[indstart:indend] / unit)
    xlabel('Time [ms]')
    ylabel(name + '_' + variable + ' (' + str(unit) + ')')
    xlim([startTime / ms, endTime / ms])



def plotWeightsWtatoGroup(name,nWTA2dNeurons,synWtaG,nCol):

    mat = np.reshape(synWtaG.w,(nWTA2dNeurons,nWTA2dNeurons,-1))
    imgShape = np.shape(mat)
    print(imgShape)
    nPlots = imgShape[2]
    #nCol = 10
    fig, axarr = plt.subplots(nPlots//nCol,nCol)#,sharex=True,sharey=True)
    for i in range(nPlots):
        axarr[i//nCol,np.mod(i,nCol)].set_xlim(0,nWTA2dNeurons)
        axarr[i//nCol,np.mod(i,nCol)].set_ylim(0,nWTA2dNeurons)
        axarr[i//nCol,np.mod(i,nCol)].axes.get_xaxis().set_visible(False)
        axarr[i//nCol,np.mod(i,nCol)].axes.get_yaxis().set_visible(False)
        axarr[i//nCol,np.mod(i,nCol)].set_xticklabels([])
        axarr[i//nCol,np.mod(i,nCol)].set_yticklabels([])
        axarr[i//nCol,np.mod(i,nCol)].autoscale(False)
        #axarr[i//nCol,np.mod(i,nCol)].set(aspect='equal')#adjustable='box-forced',
        img = axarr[i//nCol,np.mod(i,nCol)].imshow(mat[:,:,i],cmap=plt.cm.binary, vmin=0, vmax=1)
        #img.set_aspect(1)
        #print(np.shape(mat[i,:,:]))
    #plt.colorbar(img, ax=axarr[i//nCol,np.mod(i,nCol)],ticks=[0,0.5,1])
    #fig.savefig('fig/'+name+'.png',dpi=400)

def plotWeightsGrouptoWta(name,nWTA2dNeurons,synGWta,nCol ):

    mat = np.reshape(synGWta.w,(-1,nWTA2dNeurons,nWTA2dNeurons))
    imgShape = np.shape(mat)
    print(imgShape)
    nPlots = imgShape[0]
    #nCol = 10
    fig, axarr = plt.subplots(nPlots//nCol,nCol)#,sharex=True,sharey=True)
    for i in range(nPlots):
        axarr[i//nCol,np.mod(i,nCol)].set_xlim(0,nWTA2dNeurons)
        axarr[i//nCol,np.mod(i,nCol)].set_ylim(0,nWTA2dNeurons)
        axarr[i//nCol,np.mod(i,nCol)].axes.get_xaxis().set_visible(False)
        axarr[i//nCol,np.mod(i,nCol)].axes.get_yaxis().set_visible(False)
        axarr[i//nCol,np.mod(i,nCol)].set_xticklabels([])
        axarr[i//nCol,np.mod(i,nCol)].set_yticklabels([])
        axarr[i//nCol,np.mod(i,nCol)].autoscale(False)
        #axarr[i//nCol,np.mod(i,nCol)].set(aspect='equal')#adjustable='box-forced',
        img = axarr[i//nCol,np.mod(i,nCol)].imshow(mat[i,:,:],cmap=plt.cm.binary, vmin=0, vmax=1)
        #img.set_aspect(1)
        #print(np.shape(mat[i,:,:]))
    #plt.colorbar(img, ax=axarr[i//nCol,np.mod(i,nCol)],ticks=[0,0.5,1])
    #fig.savefig('fig/'+name+'.png',dpi=400)