'''
Created 03.2017
This file contains plot functions for WTA circuits
@author: Alpha
'''

from brian2 import *
import matplotlib
import matplotlib.pyplot as plt
from NCSBrian2Lib.tools import xy2ind, ind2xy

def plotWTA(name,duration,nWTANeurons,plot2d,spikemonWTA,spikemonWTAInh,spikemonWTAInp,statemonWTA):
    
    nnWTANeurons = nWTANeurons
    if plot2d:
        nnWTANeurons = nnWTANeurons**2
      
    fig = figure(figsize=(8,10))
    nPlots=3*100
    subplot(nPlots+11)
    plot(spikemonWTA.t/ms, spikemonWTA.i, '.k')
    xlabel('Time [ms]')
    ylabel('i_WTA')
    xlim([0,duration/ms])
    ylim([0,nnWTANeurons])
    subplot(nPlots+12)
    plot(spikemonWTAInp.t/ms, spikemonWTAInp.i, '.k')
    xlabel('Time [ms]')
    ylabel('i_WTAInp')
    xlim([0,duration/ms])
    ylim([0,nnWTANeurons])
    subplot(nPlots+13)
    plot(spikemonWTAInh.t/ms, spikemonWTAInh.i, '.k')
    xlabel('Time [ms]')
    ylabel('i_WTAInh')
    xlim([0,duration/ms])
    #fig.savefig('fig/'+name+'_Spikes.png')
    
    if statemonWTA is not False:  
        fig = figure(figsize=(8,10))
        nPlots=3*100
        subplot(nPlots+11)
        for ii in range(nnWTANeurons):
            plot(statemonWTA.t/ms, statemonWTA.Vm[ii]/mV, label='v')       
        #ylim([Vr/mV-30,Vt/mV+10])
        xlabel('Time [ms]')
        ylabel('V (mV)')
        subplot(nPlots+12)
        for ii in range(nnWTANeurons):
            plot(statemonWTA.t/ms, statemonWTA.Ii[ii]/pA, label='Ii')
        xlabel('Time [ms]')
        ylabel('Ii (pA)')
        subplot(nPlots+13)
        for ii in range(nnWTANeurons):
            plot(statemonWTA.t/ms, statemonWTA.Ie[ii]/pA, label='Ie')
        xlabel('Time [ms]')
        ylabel('Ie (pA)')
        #fig.savefig('fig/'+name+'_States.png', dpi=300)
    
    return


def plotWTATiles(name,duration,nWTA2dNeurons, spikemonWTA, interval = 10*ms, nCol = 10, showfig = False,savepath=False, tilecolors = []):
    'Plot a 2d WTA as tiles over time'
    nPlots = int(np.ceil((duration/ms)/(interval/ms))) #division by ms is necessary as brian2's unit division is not precise and it might round up 
    nRow = int(np.ceil(nPlots/nCol))
    #print('nPlots: '+str(nPlots))
    #print('nCol: '+str(nCol))
    #print('nRow: '+str(nRow))
    if nRow < 2:
        nRow = 2 #otherwise axarr is 1 dimensional and cannot be indexed with 2 indices
    
    if 0:            
        fig, axarr = plt.subplots(nRow,nCol,sharex=True,sharey=True,figsize=(nCol,max(1,nPlots//nCol)+1))
        for i in range(nPlots):
            start = i*interval
            inds1d = spikemonWTA.i[np.logical_and(start<spikemonWTA.t,spikemonWTA.t<(start+interval))]
            inds2d = ind2xy(inds1d,nWTA2dNeurons)
            #print(i//nCol)
            #print(np.mod(i,nCol))
            axarr[i//nCol,np.mod(i,nCol)].set_xlim(0,nWTA2dNeurons)
            axarr[i//nCol,np.mod(i,nCol)].set_ylim(0,nWTA2dNeurons)  
            axarr[i//nCol,np.mod(i,nCol)].plot(inds2d[0],inds2d[1],'.k')
        #fig.savefig('fig/'+name+'_tiles.png')

    fig, axarr = plt.subplots(nRow,nCol,sharex=True,sharey=True,figsize=(nCol,max(1,nPlots//nCol)+1))
    for i in range(nPlots):
        start = i*interval
        inds1d = spikemonWTA.i[np.logical_and(start<spikemonWTA.t,spikemonWTA.t<(start+interval))]
        hist1d = np.histogram(inds1d,bins=range(nWTA2dNeurons**2+1))[0]
        hist1d = hist1d/(interval/(1000*ms))
        hist2d = np.reshape(hist1d,(nWTA2dNeurons,nWTA2dNeurons))
        inds2d = ind2xy(inds1d,nWTA2dNeurons)
        #print(i//nCol)
        #print(np.mod(i,nCol))
        #print(col[i])
        axarr[i//nCol,np.mod(i,nCol)].set_xlim(0,nWTA2dNeurons)
        axarr[i//nCol,np.mod(i,nCol)].set_ylim(0,nWTA2dNeurons) 
        #axarr[i//nCol,np.mod(i,nCol)].axes.get_xaxis().set_visible(False)
        #axarr[i//nCol,np.mod(i,nCol)].axes.get_yaxis().set_visible(False)
        axarr[i//nCol,np.mod(i,nCol)].set_xticks(np.arange(nWTA2dNeurons)+0.5)
        axarr[i//nCol,np.mod(i,nCol)].set_yticks(np.arange(nWTA2dNeurons)+0.5)
        axarr[i//nCol,np.mod(i,nCol)].set_xticklabels([])
        axarr[i//nCol,np.mod(i,nCol)].set_yticklabels([])
        axarr[i//nCol,np.mod(i,nCol)].grid(True,linestyle='-')
        axarr[i//nCol,np.mod(i,nCol)].autoscale(False)
        if len(tilecolors) > 1:
            axarr[i//nCol,np.mod(i,nCol)].set_facecolor(tilecolors[i])
            #axarr[i//nCol,np.mod(i,nCol)].set_axis_bgcolor(tilecolors[i])
        #axarr[i//nCol,np.mod(i,nCol)].imshow(log10(hist2d),cmap=plt.cm.hot,clim=(0,log10(1000)))
        axarr[i//nCol,np.mod(i,nCol)].imshow(hist2d,cmap=plt.cm.jet,clim=(0,300))
    if not savepath:
        pass
    else:
        fig.savefig(savepath+'/'+name+'_tiles_col.png', dpi = 500)
    if showfig:
        plt.show()
    return
