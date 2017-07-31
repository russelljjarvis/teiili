#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:13:59 2017

@author: alpha
"""
from brian2 import ms,mV,plot,xlabel,ylabel,xlim,ylim

# this is to make plotting that starts at a certain time easier
def plotSpikemon(startTime,endTime,SpikeMon,nNeurons,ylab='ind'):
    if len(SpikeMon.t)>1:  
        indstart = abs(SpikeMon.t-startTime).argmin()
        indend = abs(SpikeMon.t-endTime).argmin()
        plot(SpikeMon.t[indstart:indend]/ms, SpikeMon.i[indstart:indend], '.k')
        xlabel('Time [ms]')
        ylabel(ylab)
        xlim([startTime/ms,endTime/ms])
        if nNeurons is not None:
            ylim([0,nNeurons])

def plotStatemon(startTime,endTime,StateMon,neuronInd,variable='Vm', unit=mV):
    indstart = abs(StateMon.t-startTime).argmin()
    indend = abs(StateMon.t-endTime).argmin()
    plot(StateMon.t[indstart:indend]/ms, StateMon[neuronInd].__getattr__(variable)[indstart:indend]/unit)
    xlabel('Time [ms]')
    ylabel(variable+' ['+str(unit)+']')
    xlim([startTime/ms,endTime/ms])
