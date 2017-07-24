# coding: utf-8
#===============================================================================
#from brian2 import *
from brian2 import ms,mV,pA,nS,nA,pF,us,volt,second,Network,prefs,SpikeGeneratorGroup,NeuronGroup,\
                   Synapses,SpikeMonitor,StateMonitor,figure, plot,show,xlabel,ylabel,\
                   seed,xlim,ylim,subplot,network_operation,set_device,device,TimedArray,\
                   defaultclock,codegen
import numpy as np

from NCSBrian2Lib.neuronEquations import ExpAdaptIF
from NCSBrian2Lib.synapseEquations import reversalSynV

from NCSBrian2Lib.Parameters.neuronParams import gerstnerExpAIFdefaultregular
from NCSBrian2Lib.Parameters.synapseParams import revSyn_default

from NCSBrian2Lib.tools import setParams

#===============================================================================

def genChain(groupname = 'Cha',
             neuronEq  = ExpAdaptIF,    neuronPar  = gerstnerExpAIFdefaultregular,
             synapseEq = reversalSynV,  synapsePar = revSyn_default,    
             numChains = 4,             numNeuronsPerChain = 15,
             synChaCha1e_weight = 4,    synInpCha1e_weight = 1,
             gChaGroup_refP     = 1*ms,
             monitor   = True,          debug = False):
    """create chains of neurons"""
    
    neuronEqDict = neuronEq (numInputs   = 2, debug = debug)
    synDict1     = synapseEq(inputNumber = 1, debug = debug)
    synDict2     = synapseEq(inputNumber = 2, debug = debug)

    # empty input SpikeGenerator
    tsChaInp = np.asarray([]) * ms
    indChaInp = np.asarray([]) 
    gChaInpGroup = SpikeGeneratorGroup(numChains, indices = indChaInp, times=tsChaInp, name = 'g' + groupname + 'Inp')
    
    gChaGroup = NeuronGroup(numNeuronsPerChain*numChains,name = 'g' + groupname,**neuronEqDict)
    
    synChaCha1e = Synapses(gChaGroup, gChaGroup, method = 'euler', name = 's' + groupname + '' + groupname + '1e',**synDict1)
    synChaCha1e.connect('i+1==j and (j%numNeuronsPerChain)!=0')
    
    synInpCha1e = Synapses(gChaInpGroup,gChaGroup, method = 'euler', name = 'sInp' + groupname + '1e',**synDict2)
    for i_cha in range(numChains):
        synInpCha1e.connect(i=i_cha,j=i_cha*numNeuronsPerChain)
    
    # set parameters of neuron groups
    setParams(gChaGroup, neuronPar, debug=debug)
    # set parameters of synapses
    setParams(synChaCha1e, synapsePar, debug=debug)
    setParams(synInpCha1e, synapsePar, debug=debug)
    # change some parameters
    synChaCha1e.weight = synChaCha1e_weight
    synInpCha1e.weight = synInpCha1e_weight
    gChaGroup.refP = gChaGroup_refP
    
    if monitor:
        spikemonChaInp = SpikeMonitor(gChaInpGroup)
        spikemonCha    = SpikeMonitor(gChaGroup)
    else:
        spikemonChaInp = False
        spikemonCha    = False        

    return ((gChaGroup,gChaInpGroup,synChaCha1e,synInpCha1e,spikemonChaInp,spikemonCha))
