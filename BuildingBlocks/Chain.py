# coding: utf-8
#===============================================================================
#from brian2 import *
from brian2 import ms,mV,pA,nS,nA,pF,us,volt,second,Network,prefs,SpikeGeneratorGroup,NeuronGroup,\
                   Synapses,SpikeMonitor,StateMonitor,figure, plot,show,xlabel,ylabel,\
                   seed,xlim,ylim,subplot,network_operation,set_device,device,TimedArray,\
                   defaultclock,codegen
import numpy as np
import os
from datetime import datetime

from NCSBrian2Lib.Equations.neuronEquations import ExpAdaptIF
from NCSBrian2Lib.Equations.synapseEquations import reversalSynV

from NCSBrian2Lib.Parameters.neuronParams import gerstnerExpAIFdefaultregular
from NCSBrian2Lib.Parameters.synapseParams import revSyn_default

from NCSBrian2Lib.Tools.tools import setParams

from NCSBrian2Lib.BuildingBlocks.BuildingBlock import BuildingBlock
#===============================================================================


chainParams = {'numChains' : 4,
               'numNeuronsPerChain' : 15,
               'synChaCha1e_weight' : 4,
               'synInpCha1e_weight' : 1,
               'gChaGroup_refP'     : 1*ms}


class Chain(BuildingBlock):
    def __init__(self,blockName,neuronEq=ExpAdaptIF,synapseEq=reversalSynV,
             neuronParams=gerstnerExpAIFdefaultregular,synapseParams=revSyn_default,
             blockParams=chainParams,debug=False):
        self.numChains          = blockParams['numChains']
        self.numNeuronsPerChain = blockParams['numNeuronsPerChain']
        self.synChaCha1e_weight = blockParams['synChaCha1e_weight']
        self.synInpCha1e_weight = blockParams['synInpCha1e_weight']
        self.gChaGroup_refP     = blockParams['gChaGroup_refP']
        self.inputGroup         = None
        BuildingBlock.__init__(self,blockName,neuronEq,synapseEq,neuronParams,synapseParams,blockParams,debug)
        
        self.Groups,self.Monitors,self.replaceVars = self.genChain(blockName,
                 neuronEq,neuronParams,synapseEq,synapseParams,    
                 self.numChains,self.numNeuronsPerChain,
                 self.synChaCha1e_weight,self.synInpCha1e_weight,
                 self.gChaGroup_refP,self.debug)
        
        self.allBrian = {**self.Groups,**self.Monitors}
    
    
    # this allows us to iterate over the BrianObjects and directly add the Block to a Network
    def __iter__(self):
        return iter([self.allBrian[key] for key in self.allBrian])
    
    
    def genChain(self, groupname = 'Cha',
                 neuronEq  = ExpAdaptIF,    neuronPar  = gerstnerExpAIFdefaultregular,
                 synapseEq = reversalSynV,  synapsePar = revSyn_default,    
                 numChains = 4,             numNeuronsPerChain = 15,
                 synChaCha1e_weight = 4,    synInpCha1e_weight = 1,
                 gChaGroup_refP     = 1*ms,
                 debug = False):
        """create chains of neurons"""
        
        neuronEqDict = neuronEq (numInputs   = 2, debug = debug)
        synDict1     = synapseEq(inputNumber = 1, debug = debug)
        synDict2     = synapseEq(inputNumber = 2, debug = debug)
    
        # empty input SpikeGenerator
        tsChaInp = np.asarray([]) * ms
        indChaInp = np.asarray([]) 
        gChaInpGroup = SpikeGeneratorGroup(numChains, indices = indChaInp, times=tsChaInp, name = 'g' + groupname + 'Inp')
        self.inputGroup = gChaInpGroup
        
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
        
        self.spikemonChaInp = SpikeMonitor(gChaInpGroup)
        self.spikemonCha    = SpikeMonitor(gChaGroup)
       
        ChaMonitors = {
               'spikemonChaInp' : self.spikemonChaInp, 
               'spikemonCha'    : self.spikemonCha
               }
        
        ChaGroups = {
                'gChaGroup'   : gChaGroup,
                'gChaInpGroup': gChaInpGroup,
                'synChaCha1e' : synChaCha1e,
                'synInpCha1e' : synInpCha1e
                }
        
        replaceVars = [synChaCha1e.name+'_weight']
        
        return ChaGroups,ChaMonitors,replaceVars

    def plot(self,savedir=None):
        
        if len(self.spikemonCha.t) < 1:
            print('Monitor is empty, have you run the Network?')
            return
        
        duration = max(self.spikemonCha.t + 10*ms)
        # Cha plots
        fig = figure()
        subplot(211)
        plot(self.spikemonCha.t/ms, self.spikemonCha.i, '.k')
        xlabel('Time [ms]')
        ylabel('i_Cha')
        #ylim([0,0])
        xlim([0,duration/ms])
        subplot(212)
        plot(self.spikemonChaInp.t/ms, self.spikemonChaInp.i, '.k')
        xlabel('Time [ms]')
        ylabel('i_Cha_Inp')
        #ylim([0,0])
        xlim([0,duration/ms])
        if savedir is not None:
            fig.savefig(os.path.join(savedir, self.blockName + '_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.png'))
