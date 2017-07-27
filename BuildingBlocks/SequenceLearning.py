#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:45:09 2017

@author: alpha
"""

import numpy as np

from brian2 import *
from brian2 import ms,mV,pA,nS,nA,pF,us,volt,second,Network,prefs,SpikeGeneratorGroup,NeuronGroup,\
                   Synapses,SpikeMonitor,StateMonitor,figure, plot,show,xlabel,ylabel,\
                   seed,xlim,ylim,subplot,network_operation,set_device,device,TimedArray,\
                   defaultclock,profiling_summary,codegen,floor

from NCSBrian2Lib.Equations.neuronEquations import ExpAdaptIF
from NCSBrian2Lib.Equations.synapseEquations import reversalSynV

from NCSBrian2Lib.Parameters.neuronParams import gerstnerExpAIFdefaultregular
from NCSBrian2Lib.Parameters.synapseParams import revSyn_default

from NCSBrian2Lib.Tools.tools import setParams

#%%
#===============================================================================

def genSequenceLearning(groupname = 'Seq',
             neuronEq  = ExpAdaptIF,    neuronPar  = gerstnerExpAIFdefaultregular,
             synapseEq = reversalSynV,  synapsePar = revSyn_default,    
             numElements = 4,           numNeuronsPerGroup = 8,
             synInpOrd1e_weight   = 1.3,
             synOrdMem1e_weight   = 1.1,
             synMemOrd1e_weight   = 0.16,
             #local
             synOrdOrd1e_weight   = 1.04,
             synMemMem1e_weight   = 1.54,
             #inhibitory
             synOrdOrd1i_weight   = -1.95,
             synMemOrd1i_weight   = -0.384,
             synCoSOrd1i_weight   = -1.14,
             synResetOrd1i_weight = -1.44,
             synResetMem1i_weight = -2.6,
             #refractory
             gOrdGroups_refP = 1.7*ms,
             gMemGroups_refP = 2.3*ms,
             #
             additionalInputs = 0,
             debug = False):
    """create Sequence Learning Network after the model from Sandamirskaya and Schoener (2010)"""


    nOrdNeurons = numNeuronsPerGroup*numElements
    nMemNeurons = numNeuronsPerGroup*numElements
    
    # Input to start sequence manually
    tsInput  = np.asarray([]) * ms
    indInput = np.asarray([])
    gInputGroup = SpikeGeneratorGroup(numNeuronsPerGroup, indices=indInput, times=tsInput, name='spikegenInp_'+ groupname)
    # CoS Input #TODO: CoS as NeuronGroup 
    tsCoS  = np.asarray([]) * ms
    indCoS = np.asarray([])
    gCoSGroup = SpikeGeneratorGroup(numNeuronsPerGroup, indices=indCoS, times=tsCoS, name='spikegenCoS_'+ groupname)
    # reset group #TODO: Reset as NeuronGroup 
    tsReset  = np.asarray([]) * ms
    indReset = np.asarray([])
    gResetGroup = SpikeGeneratorGroup(numNeuronsPerGroup, indices=indReset, times=tsReset, name='spikegenReset_'+ groupname)

    neuronEqsDict_Ord = neuronEq (numInputs   = 7+additionalInputs, debug = debug)
    neuronEqsDict_Mem = neuronEq (numInputs   = 3, debug = debug)
    synEqsDict1     = synapseEq(inputNumber = 1, debug = debug)
    synEqsDict2     = synapseEq(inputNumber = 2, debug = debug)
    synEqsDict3     = synapseEq(inputNumber = 3, debug = debug)
    synEqsDict4     = synapseEq(inputNumber = 4, debug = debug)
    synEqsDict5     = synapseEq(inputNumber = 5, debug = debug)
    synEqsDict6     = synapseEq(inputNumber = 6, debug = debug)
    synEqsDict7     = synapseEq(inputNumber = 7, debug = debug)

    # Neurongoups
    gOrdGroups = NeuronGroup(nOrdNeurons,  name='g' +'Ord_'+ groupname, **neuronEqsDict_Ord)
    gMemGroups = NeuronGroup(nMemNeurons,  name='g' +'Mem_'+ groupname, **neuronEqsDict_Mem)
    
    # Synapses 
    #excitatory
    synInpOrd1e   = Synapses(gInputGroup, gOrdGroups, method="euler", name='sInpOrd1e_'+ groupname, **synEqsDict1)
    synOrdMem1e   = Synapses(gOrdGroups,  gMemGroups, method="euler", name='sOrdMem1e_'+ groupname, **synEqsDict1)
    synMemOrd1e   = Synapses(gMemGroups,  gOrdGroups, method="euler", name='sMemOrd1e_'+ groupname, **synEqsDict2)
    #local
    synOrdOrd1e   = Synapses(gOrdGroups,  gOrdGroups, method="euler", name='sOrdOrd1e_'+ groupname, **synEqsDict3)
    synMemMem1e   = Synapses(gMemGroups,  gMemGroups, method="euler", name='sMemMem1e_'+ groupname, **synEqsDict2)
    #inhibitory
    synOrdOrd1i   = Synapses(gOrdGroups,  gOrdGroups, method="euler", name='sOrdOrd1i_'+ groupname, **synEqsDict4)
    synMemOrd1i   = Synapses(gMemGroups,  gOrdGroups, method="euler", name='sMemOrd1i_'+ groupname, **synEqsDict5)
    synCoSOrd1i   = Synapses(gCoSGroup,   gOrdGroups, method="euler", name='sCoSOrd1i_'+ groupname, **synEqsDict6)  
    synResetOrd1i = Synapses(gResetGroup, gOrdGroups, method="euler", name='sResOrd1i_'+ groupname, **synEqsDict7)
    synResetMem1i = Synapses(gResetGroup, gMemGroups, method="euler", name='sResMem1i_'+ groupname, **synEqsDict3)
    
    
    # predefine connections for efficiency reasons
    iii = []
    jjj= []
    for ii in range (nOrdNeurons):
        for jj in range (nMemNeurons):
            if ( (floor(ii/numNeuronsPerGroup)==floor(jj/numNeuronsPerGroup)) & (ii!=jj) ):
                iii.append(ii)
                jjj.append(jj)
                
    iiMemOrd = []
    jjMemOrd = []
    for ii in range (nOrdNeurons):
        for jj in range (nMemNeurons):
            if (floor(ii/numNeuronsPerGroup)==floor(jj/numNeuronsPerGroup)-1) & ((floor(jj/numNeuronsPerGroup)-1)>=0) :
                iiMemOrd.append(ii)
                jjMemOrd.append(jj)
                
    synInpOrd1e.connect(i=np.arange(numNeuronsPerGroup), j=np.arange(numNeuronsPerGroup)) # Input to first OG       
    synOrdMem1e.connect(i=iii ,j=jjj)
    synMemOrd1e.connect(i=iiMemOrd ,j=jjMemOrd)
    synOrdOrd1e.connect(i=iii ,j=jjj) # i=j is also connected, exclude if necessary
    synMemMem1e.connect(i=iii ,j=jjj) # i=j is also connected, exclude if necessary
    synOrdOrd1i.connect('floor(i/numNeuronsPerGroup)!=floor(j/numNeuronsPerGroup) and (i!=j)')
    synMemOrd1i.connect(i=iii,j=jjj)
    #CoS
    synCoSOrd1i.connect(True)
    #Reset
    synResetOrd1i.connect(True)
    synResetMem1i.connect(True)

    # set parameters of neuron groups
    setParams(gOrdGroups, neuronPar, debug=debug)
    setParams(gMemGroups, neuronPar, debug=debug)
    # set parameters of synapses
    setParams(synInpOrd1e,   synapsePar, debug=debug)
    setParams(synOrdMem1e,   synapsePar, debug=debug)
    setParams(synMemOrd1e,   synapsePar, debug=debug)
    setParams(synOrdOrd1e,   synapsePar, debug=debug)
    setParams(synMemMem1e,   synapsePar, debug=debug)
    setParams(synOrdOrd1i,   synapsePar, debug=debug)
    setParams(synMemOrd1i,   synapsePar, debug=debug)
    setParams(synCoSOrd1i,   synapsePar, debug=debug)
    setParams(synResetOrd1i, synapsePar, debug=debug)
    setParams(synResetMem1i, synapsePar, debug=debug)  
    
    # change some parameters
    # weights
    #excitatory
    synInpOrd1e.weight   = synInpOrd1e_weight 
    synOrdMem1e.weight   = synOrdMem1e_weight
    synMemOrd1e.weight   = synMemOrd1e_weight
    #local
    synOrdOrd1e.weight   = synOrdOrd1e_weight
    synMemMem1e.weight   = synMemMem1e_weight
    #inhibitory
    synOrdOrd1i.weight   = synOrdOrd1i_weight
    synMemOrd1i.weight   = synMemOrd1i_weight 
    synCoSOrd1i.weight   = synCoSOrd1i_weight
    synResetOrd1i.weight = synResetOrd1i_weight
    synResetMem1i.weight = synResetMem1i_weight
    # refractory periods
    gOrdGroups.refP = gOrdGroups_refP
    gMemGroups.refP = gMemGroups_refP
    
    SLGroups = {
                'gOrdGroups'  : gOrdGroups,
                'gMemGroups'  : gMemGroups,
                'gInputGroup' : gInputGroup,
                'gCoSGroup'   : gCoSGroup,
                'gResetGroup' : gResetGroup,
                'synInpOrd1e' : synInpOrd1e,
                'synOrdMem1e' : synOrdMem1e,
                'synMemOrd1e' : synMemOrd1e,
                'synOrdOrd1e' : synOrdOrd1e,
                'synMemMem1e' : synMemMem1e,
                'synOrdOrd1i' : synOrdOrd1i,
                'synMemOrd1i' : synMemOrd1i,
                'synCoSOrd1i' : synCoSOrd1i,
                'synResetOrd1i' : synResetOrd1i,
                'synResetMem1i' : synResetMem1i}
    
    # Monitors
    spikemonOrd = SpikeMonitor(gOrdGroups, name='spikemonOrd_'+ groupname)
    spikemonMem = SpikeMonitor(gMemGroups, name='spikemonMem_'+ groupname)
    spikemonCoS = SpikeMonitor(gCoSGroup, name='spikemonCoS_'+ groupname)
    spikemonInp = SpikeMonitor(gInputGroup, name='spikemonInp_'+ groupname)
    spikemonReset = SpikeMonitor(gResetGroup, name='spikemonReset_'+ groupname)
        
    SLMonitors = {
                'spikemonOrd' : spikemonOrd,
                'spikemonMem'  : spikemonMem,
                'spikemonCoS'  : spikemonCoS,
                'spikemonInp'  : spikemonInp,
                'spikemonReset': spikemonReset}
    
    replaceVars = ['sInpOrd1e_'+ groupname +'_weight',
                   'sOrdMem1e_'+ groupname +'_weight',
                   'sMemOrd1e_'+ groupname +'_weight',
                   #local
                   'sOrdOrd1e_'+ groupname +'_weight',
                   'sMemMem1e_'+ groupname +'_weight',
                   #inhibitory
                   'sOrdOrd1i_'+ groupname +'_weight',
                   'sMemOrd1i_'+ groupname +'_weight',
                   'sCoSOrd1i_'+ groupname +'_weight',
                   'sResetOrd1i_'+ groupname +'_weight',
                   'sResetMem1i_'+ groupname +'_weight',
                   #refractory
                   'gOrd_'+ groupname +'_refP',
                   'gMem_'+ groupname +'_refP']
                 
    return SLGroups,SLMonitors,replaceVars



def plotSequenceLearning(SLMonitors):
    
    spikemonOrd = SLGroups['spikemonOrd']
    spikemonMem = SLGroups['spikemonMem']
    spikemonInp = SLGroups['spikemonInp']
    spikemonCoS = SLGroups['spikemonCoS']
    spikemonReset = SLGroups['spikemonReset']
    duration = max(spikemonOrd.t)+10*ms
    print ('plot...')
    figure(figsize=(8,12))
    title('sequence learning')
    nPlots=5*100
    subplot(nPlots+11)
    plot(spikemonOrd.t/ms, spikemonOrd.i, '.k')
    xlabel('Time [ms]')
    ylabel('i_ord')
    #ylim([0,nOrdNeurons])
    xlim([0,duration/ms])
    subplot(nPlots+12)
    plot(spikemonMem.t/ms, spikemonMem.i, '.k')
    xlabel('Time [ms]')
    ylabel('i_mem')
    #ylim([0,nOrdNeurons])
    xlim([0,duration/ms])
    subplot(nPlots+13)
    plot(spikemonInp.t/ms, spikemonInp.i, '.k')
    xlabel('Time [ms]')
    ylabel('i_in')
    #ylim([0,nPerGroup])
    xlim([0,duration/ms])
    subplot(nPlots+14)
    plot(spikemonCoS.t/ms, spikemonCoS.i, '.k')
    xlabel('Time [ms]')
    ylabel('i_CoS')
    #ylim([0,nPerGroup])
    xlim([0,duration/ms])
    subplot(nPlots+15)
    plot(spikemonReset.t/ms, spikemonReset.i, '.k')
    xlabel('Time [ms]')
    ylabel('i_Reset')
    #ylim([0,nPerGroup])
    xlim([0,duration/ms])