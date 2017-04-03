'''
Created 03.2017
@author: Alpha

This is not a building block jet, but it should become one, when it works

This is an attempt to implement a spiking SOM inspired by Rumbell et al. 2014
'''

#===============================================================================
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime as dtime
from random import randrange, random
from random import seed as rnseed

from brian2 import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

import os

from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from NCSBrian2Lib.neuronEquations import ExpAdaptIFrev as neuronEq
from NCSBrian2Lib.synapseEquations import fusiSynV as fusiSynapseEq
from NCSBrian2Lib.synapseEquations import reversalSynV as synapseEq
from NCSBrian2Lib.tools import setParams
from NCSBrian2Lib.Parameters.neuronParams import gerstnerExpAIFdefaultregular as neuronPar
from NCSBrian2Lib.Parameters.synapseParams import fusiDefault
from NCSBrian2Lib.Parameters.synapseParams import revSyndefault as synapsePar
from NCSBrian2Lib.tools import setParams, fkernelgauss1d, printStates, fkernel2d
from NCSBrian2Lib.BasicBuildingBlocks.WTA import gen2dWTA,gen1dWTA
from NCSBrian2Lib.Plotting.WTAplot import plotWTA,plotWTATiles
from tools import xy2ind

prefs.codegen.target = "numpy" 

seed(42)
rnseed(42)

plotweights = True

#defaultclock.dt = 100 *us

somNet = Network()


nWTA1dNeurons = 64
nWTA2dNeurons = 16
nPerGroup = 1


duration = 400 * ms #needs to be a multiple of interval # and needs to be much longer for SOM training, but for debugging and looking at input, keep it short 

gerstnerExpAIFmodified = {"C"      : 281   *pF,
                "gL"     : 35    *nS,
                "EL"     : -70.6 *mV,
                "VT"     : -50.4 *mV,
                "DeltaT" : 2     *mV,
                "tauwad" : 144   *ms,
                "a"      : 4     *nS,
                "b"      : 0.0805*nA,
                "Vr"     : -70.6 *mV,
                "Vm"     : -70.6 *mV,
                "Iconst" : 0 * pA,
                "sigma" : 0 *pA,
                "taue"  : 5 *ms,
                "taui"  : 10 *ms,
                "gIe"   : 0 *nS,
                "gIi"   : 0 *nS,
                "taugIe": 5 *ms,
                "taugIi": 4 *ms,
                "EIe"   : 60.0 *mV,
                "EIi"   : -90.0  *mV}


#===============================================================================
## create wtaSOM
sigmnbi = 2.6
(gwtaSOMGroup,gwtaSOMInhGroup,gwtaSOMInpGroup,synInpwtaSOM1e,synwtaSOMwtaSOM1e,synwtaSOMInh1e,synInhwtaSOM1i,
            spikemonwtaSOM,spikemonwtaSOMInh,spikemonwtaSOMInp,statemonwtaSOM) = gen2dWTA('wtaSOM',
             neuronParameters = neuronPar, synParameters = synapsePar,
             weInpWTA = 0, weWTAInh =2.4, wiInhWTA = -1.4, w_lat=0.04,
             rpWTA = 2*ms, rpInh = 1*ms,
             sigm = sigmnbi, nNeurons = nWTA2dNeurons, nInhNeurons = 8, cutoff = 9, monitor = True, debug=False)

somNet.add((gwtaSOMGroup,gwtaSOMInhGroup,gwtaSOMInpGroup,synInpwtaSOM1e,synwtaSOMwtaSOM1e,synwtaSOMInh1e,synInhwtaSOM1i,
            spikemonwtaSOM,spikemonwtaSOMInh,spikemonwtaSOMInp)) #,statemonwtaSOM


#gwtaSOMGroup.gIi = - 50 * nS * rand()


## color example
red = (1.0,0.0,0.0)
or1 = (1.0,0.3,0.1)
or2 = (1.0,0.7,0.0)
yel = (1.0,0.9,0.0)
gr1 = (0.2,0.8,0.2)
gr2 = (0.2,0.7,0.5)
blu = (0.0,0.0,1.0)
bl2 = (0.0,0.3,1.0)
bl3 = (0.3,0.2,0.9)

collist = [red,or1,or2,yel,gr1,gr2,blu,bl2,bl3]
#collist = [(random(),random(),random()) for ii in range(8)]
interval = 20
#pause = 0
nRepet = int((duration/ms) /interval)
randind = [randrange(0,len(collist)) for ii in range(nRepet)]
col = [collist[ii] for ii in randind]

#col = [(random(),random(),random()) for ii in range(nRepet)]
#print(col)

#===============================================================================
## create Input u
ninputs = 3
rpInp = 2*ms
nInpNeurons = nWTA1dNeurons

neuronEqDict, args = neuronEq(debug = False)
gInpGroup = NeuronGroup(ninputs*nInpNeurons, refractory = rpInp , method='euler',name = 'gInpGroup', **neuronEqDict)

setParams(gInpGroup, gerstnerExpAIFmodified, debug = False)

spikemonInp = SpikeMonitor(gInpGroup)
statemonInp = StateMonitor(gInpGroup, ('Vm','Iin','Iconst'), record=True)

somNet.add((gInpGroup,spikemonInp)) #,statemonInp

#===============================================================================
## create Input Inhibition Inh_u

nInhNeurons = 4
rpInpInh = 2*ms
gInpInhGroup = NeuronGroup(nInhNeurons, refractory = rpInpInh , method='euler',name = 'gInpInhGroup', **neuronEqDict)

setParams(gInpInhGroup, gerstnerExpAIFmodified, debug = False)

spikemonInpInh = SpikeMonitor(gInpInhGroup)
statemonInpInh = StateMonitor(gInpInhGroup, ('Vm','Iin','Iconst'), record=True)

somNet.add((gInpInhGroup,spikemonInpInh,statemonInpInh))


#===============================================================================
## create inhibition synapses

synDict = synapseEq()
synInpInh1e = Synapses(gInpGroup, gInpInhGroup, method = "euler", name = 'sInpInh1e', **synDict)
synInpInh1i = Synapses(gInpGroup, gInpInhGroup, method = "euler", name = 'sInpInh1e', **synDict)      
synInhInp1i = Synapses(gInpInhGroup, gInpGroup, method = "euler", name = 'sInhInp1i', **synDict)

synInpInh1e.connect(True)
synInpInh1i.connect(True)
synInhInp1i.connect(True)

synapseParInpInh1i=synapsePar
synapseParInpInh1i["taugIi"] = 5 * ms # This only makes sense, if the synapse actually has a tau!
synapseParInpInh1e=synapsePar
synapseParInpInh1e["taugIe"] = 15 * ms

setParams(synInpInh1e, synapseParInpInh1e, debug = False)
setParams(synInpInh1i, synapseParInpInh1i, debug = False)
setParams(synInhInp1i, synapsePar, debug = False)

synInpInh1e.weight = 0.6
synInpInh1i.weight = -0.2           
synInhInp1i.weight = -1.5

somNet.add((synInpInh1e,synInhInp1i))
#===============================================================================
## create plastic synapses

#fusiSynapsePar = fusiDefault
fusiSynapsePar = { "w_plus" : 0.2,
                "w_minus": 0.2, 
                "theta_upl" : 180 *mV,
                "theta_uph" : 1*volt,  
                "theta_downh" : 90 *mV,
                "theta_downl" : 50 *mV, 
                "theta_V" : -59 *mV, 
                "alpha" : 0.0001/second, 
                "beta" : 0.0001/second, 
                "tau_ca" : 8*ms,
                "w_ca" : 250*mV,
                "w_min" : 0,
                "w_max" : 1,
                "theta_w" : 0.5,
                "w" : 0,
                "gWe" : 7 *nS,
                "gIe" : 0 *nS,
                "taugIe" : 5 *ms,
                "EIe" : 60.0 *mV}

fusiSynDict = fusiSynapseEq()
   
synInpSom1e = Synapses(gInpGroup, gwtaSOMGroup, method = "euler", name = 'sInpSom1e', **fusiSynDict)    

synInpSom1e.connect(True)
setParams(synInpSom1e, fusiSynapsePar, debug = False)
statemonSynInpSom1e = StateMonitor(synInpSom1e,('w','Ca'), record=True )

synInpSom1e.weight = '0.1+0.6*rand()' #0.3 # making these random might be a way to get around the binary fusi synapse issue but it might also stop the SOM from working (think about it)  
synInpSom1e.w = '0.99*rand()'

somNet.add((synInpSom1e))


#===============================================================================
# This is just a first idea of a decay of learning rate
# @network_operation(dt=2*interval*ms)
# def decreaseWplus(ti):
#     synInpSom1e.w_plus = synInpSom1e.w_plus*0.995
#     synInpSom1e.w_minus = synInpSom1e.w_minus*0.995
# 
# somNet.add((decreaseWplus))
#===============================================================================

# This is just a first idea for neighborhood width decay, better function needs to be implemented
# neighborhood width decay
@network_operation(dt=20*ms)
def decreaseNBW(ti):
    print('simulation at ' + str(int(ti/ms))+ 'ms')   
    sigmnb = sigmnbi*(1.1-ti/duration)
    #print(sigmnb)
    synwtaSOMwtaSOM1e.weight = '0.5 * fkernel2d(i,j,sigmnb,nWTA2dNeurons)'  

somNet.add((decreaseNBW))


sigm = 6.0

@network_operation(dt=interval*ms)
def inputCurrent(ti):
    for inp in range(ninputs):
        #print(col[int(floor(ti/interval))][inp])
        meangaussind=int(sigm+(nInpNeurons-2*sigm)*col[int(floor(ti/(interval*ms)))][inp])
        #print(ti/ms)
        #print(meangaussind)
        #print(gInpGroup[(inp*nInpNeurons):((inp+1)*nInpNeurons)].Iconst)
        for j in range(nInpNeurons):
            gInpGroup[(inp*nInpNeurons):((inp+1)*nInpNeurons)].Iconst[j] = fkernelgauss1d(meangaussind,j,sigm)*pA*2000
        #print(gInpGroup[(inp*nInpNeurons):((inp+1)*nInpNeurons)].Iconst)
somNet.add((inputCurrent))


#===============================================================================
## simulation
start = time.clock()
somNet.run(duration-interval*ms)

if plotweights: #this saves memor
    somNet.add((statemonSynInpSom1e))
somNet.run(interval*ms)
    
end = time.clock()
print ('simulation took ' + str(end - start) + ' sec')
print('simulation done!')


#===============================================================================
## plots

fig = figure(figsize=(8,10))
nPlots=ninputs*100
for ii in range(ninputs):
    subplot(nPlots+11+ii)
    spikeinds = spikemonInp.i[((nInpNeurons*ii)<spikemonInp.i)&(spikemonInp.i<(nInpNeurons*(ii+1)))]
    spiketimes = spikemonInp.t[((nInpNeurons*ii)<spikemonInp.i)&(spikemonInp.i<(nInpNeurons*(ii+1)))]
    plot(spiketimes/ms, spikeinds, '.k')
    xlabel('Time [ms]')
    ylabel('i_In')
    xlim([0,duration/ms])
    ylim([nInpNeurons*ii,nInpNeurons*(ii+1)])


# fig = figure(figsize=(8,10))
# for ii in range(ninputs*nInpNeurons):
#     plot(statemonInp.t/ms, statemonInp.Vm[ii]/mV, label='v')       
# xlabel('Time [ms]')
# ylabel('V (mV)')
# 
# fig = figure(figsize=(8,10))
# for ii in range(ninputs*nInpNeurons):
#     plot(statemonInp.t/ms, statemonInp.Iin[ii]/pA, label='I')       
# xlabel('Time [ms]')
# ylabel('Iin (pA)')




if plotweights:
    end=np.shape(statemonSynInpSom1e.w)[1]-1
    mat = reshape(statemonSynInpSom1e.w[:,end],(-1,nWTA2dNeurons,nWTA2dNeurons))
    imgShape = np.shape(mat) 
    #print(imgShape)
    #nPlots = imgShape[0]
    nPlots = nInpNeurons
    nCol = 8
    nRow = int(np.ceil(nPlots/nCol))
    #for i in range(imgShape[0]):
    for ii in range(ninputs):
        fig, axarr = plt.subplots(nRow,nCol,sharex=True,sharey=True,figsize=(nCol,nPlots//nCol))
        for i in range(nInpNeurons):
            axarr[i//nCol,np.mod(i,nCol)].set_xlim(0,nWTA2dNeurons)
            axarr[i//nCol,np.mod(i,nCol)].set_ylim(0,nWTA2dNeurons)
            axarr[i//nCol,np.mod(i,nCol)].set_xticks(np.arange(nWTA2dNeurons)+0.5)
            axarr[i//nCol,np.mod(i,nCol)].set_yticks(np.arange(nWTA2dNeurons)+0.5)
            axarr[i//nCol,np.mod(i,nCol)].set_xticklabels([])
            axarr[i//nCol,np.mod(i,nCol)].set_yticklabels([])
            axarr[i//nCol,np.mod(i,nCol)].grid(True,linestyle='-')
            axarr[i//nCol,np.mod(i,nCol)].autoscale(False)
            img = axarr[i//nCol,np.mod(i,nCol)].imshow(mat[i+nInpNeurons*ii,:,:],cmap=plt.cm.binary, vmin=0, vmax=1)
    


plotWTA('wtaSOM',duration,nWTA2dNeurons,True,spikemonwtaSOM,spikemonwtaSOMInh,spikemonwtaSOMInp,False)

## WTA plot tiles over time
plotWTATiles('wtaSOM',duration,nWTA2dNeurons, spikemonwtaSOM,interval=interval*ms, showfig = False, tilecolors=col )
show()

print("done!")