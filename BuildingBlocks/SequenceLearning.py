#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:45:09 2017

@author: alpha
"""

import numpy as np

from brian2 import *
from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second, Network, prefs, SpikeGeneratorGroup, NeuronGroup,\
    SpikeMonitor, StateMonitor, figure, plot, show, xlabel, ylabel,\
    seed, xlim, ylim, subplot, network_operation, set_device, device, TimedArray,\
    defaultclock, profiling_summary, codegen, floor, title

from NCSBrian2Lib.Equations.neuronEquations import ExpAdaptIF
from NCSBrian2Lib.Equations.neuronEquations import Silicon

from NCSBrian2Lib.Equations.synapseEquations import reversalSynV
from NCSBrian2Lib.Equations.synapseEquations import BraderFusiSynapses, SiliconSynapses

from NCSBrian2Lib.Parameters.neuronParams import gerstnerExpAIFdefaultregular
from NCSBrian2Lib.Parameters.neuronParams import SiliconNeuronP

from NCSBrian2Lib.Parameters.synapseParams import revSyn_default
from NCSBrian2Lib.Parameters.synapseParams import Braderfusi, SiliconSynP

from NCSBrian2Lib.BuildingBlocks.BuildingBlock import BuildingBlock
from NCSBrian2Lib.Groups.Groups import Neurons, Connections

#%%
#===============================================================================

slParams = {'synInpOrd1e_weight': 1.3,
            'synOrdMem1e_weight': 1.1,
            'synMemOrd1e_weight': 0.16,
            # local
            'synOrdOrd1e_weight': 1.04,
            'synMemMem1e_weight': 1.54,
            # inhibitory
            'synOrdOrd1i_weight': -1.95,
            'synMemOrd1i_weight': -0.384,
            'synCoSOrd1i_weight': -1.14,
            'synResetOrd1i_weight': -1.44,
            'synResetMem1i_weight': -2.6,
            # refractory
            'gOrdGroups_refP': 1.7 * ms,
            'gMemGroups_refP': 2.3 * ms
            }


class SequenceLearning(BuildingBlock):
    '''a 1 or 2D square WTA'''

    def __init__(self, name, neuronEq=ExpAdaptIF, synapseEq=reversalSynV,
                 neuronParams=gerstnerExpAIFdefaultregular, synapseParams=revSyn_default,
                 blockParams=slParams, numElements=3, numNeuronsPerGroup=6, additionalInputs=0, debug=False):

        BuildingBlock.__init__(self, name, neuronEq, synapseEq, neuronParams, synapseParams, blockParams, debug)

        self.Groups, self.Monitors, self.standaloneParams = genSequenceLearning(name,
                                                                                neuronEq, neuronParams, synapseEq, synapseParams,
                                                                                numElements=numElements, numNeuronsPerGroup=numNeuronsPerGroup,
                                                                                additionalInputs=additionalInputs, debug=debug, **blockParams)

        self.inputGroup = self.Groups['gInputGroup']
        self.cosGroup = self.Groups['gCoSGroup']
        self.resetGroup = self.Groups['gResetGroup']

    def plot(self):

        plotSequenceLearning(self.Monitors)


def genSequenceLearning(groupname='Seq',
                        neuronEq=ExpAdaptIF, neuronPar=gerstnerExpAIFdefaultregular,
                        synapseEq=reversalSynV, synapsePar=revSyn_default,
                        numElements=4, numNeuronsPerGroup=8,
                        synInpOrd1e_weight=1.3,
                        synOrdMem1e_weight=1.1,
                        synMemOrd1e_weight=0.16,
                        # local
                        synOrdOrd1e_weight=1.04,
                        synMemMem1e_weight=1.54,
                        # inhibitory
                        synOrdOrd1i_weight=-1.95,
                        synMemOrd1i_weight=-0.384,
                        synCoSOrd1i_weight=-1.14,
                        synResetOrd1i_weight=-1.44,
                        synResetMem1i_weight=-2.6,
                        # refractory
                        gOrdGroups_refP=1.7 * ms,
                        gMemGroups_refP=2.3 * ms,
                        #
                        additionalInputs=0,
                        debug=False):
    """create Sequence Learning Network after the model from Sandamirskaya and Schoener (2010)"""

    nOrdNeurons = numNeuronsPerGroup * numElements
    nMemNeurons = numNeuronsPerGroup * numElements

    # Input to start sequence manually
    tsInput = np.asarray([]) * ms
    indInput = np.asarray([])
    gInputGroup = SpikeGeneratorGroup(numNeuronsPerGroup, indices=indInput, times=tsInput, name='spikegenInp_' + groupname)
    # CoS Input #TODO: CoS as NeuronGroup
    tsCoS = np.asarray([]) * ms
    indCoS = np.asarray([])
    gCoSGroup = SpikeGeneratorGroup(numNeuronsPerGroup, indices=indCoS, times=tsCoS, name='spikegenCoS_' + groupname)
    # reset group #TODO: Reset as NeuronGroup
    tsReset = np.asarray([]) * ms
    indReset = np.asarray([])
    gResetGroup = SpikeGeneratorGroup(numNeuronsPerGroup, indices=indReset, times=tsReset, name='spikegenReset_' + groupname)

    # Neurongoups
    gOrdGroups = Neurons(nOrdNeurons, neuronEq, neuronPar, refractory=gOrdGroups_refP, name='g' + 'Ord_' + groupname, numInputs=7 + additionalInputs, debug=debug)
    gMemGroups = Neurons(nMemNeurons, neuronEq, neuronPar, refractory=gMemGroups_refP, name='g' + 'Mem_' + groupname, numInputs=3, debug=debug)

    # Synapses
    # excitatory
    synInpOrd1e = Connections(gInputGroup, gOrdGroups, synapseEq, synapsePar, method="euler", debug=debug, name='sInpOrd1e_' + groupname)
    synOrdMem1e = Connections(gOrdGroups, gMemGroups, synapseEq, synapsePar, method="euler", debug=debug, name='sOrdMem1e_' + groupname)
    synMemOrd1e = Connections(gMemGroups, gOrdGroups, synapseEq, synapsePar, method="euler", debug=debug, name='sMemOrd1e_' + groupname)
    # local
    synOrdOrd1e = Connections(gOrdGroups, gOrdGroups, synapseEq, synapsePar, method="euler", debug=debug, name='sOrdOrd1e_' + groupname)
    synMemMem1e = Connections(gMemGroups, gMemGroups, synapseEq, synapsePar, method="euler", debug=debug, name='sMemMem1e_' + groupname)
    # inhibitory
    synOrdOrd1i = Connections(gOrdGroups, gOrdGroups, synapseEq, synapsePar, method="euler", debug=debug, name='sOrdOrd1i_' + groupname)
    synMemOrd1i = Connections(gMemGroups, gOrdGroups, synapseEq, synapsePar, method="euler", debug=debug, name='sMemOrd1i_' + groupname)
    synCoSOrd1i = Connections(gCoSGroup, gOrdGroups, synapseEq, synapsePar, method="euler", debug=debug, name='sCoSOrd1i_' + groupname)
    synResetOrd1i = Connections(gResetGroup, gOrdGroups, synapseEq, synapsePar, method="euler", debug=debug, name='sResOrd1i_' + groupname)
    synResetMem1i = Connections(gResetGroup, gMemGroups, synapseEq, synapsePar, method="euler", debug=debug, name='sResMem1i_' + groupname)

    # predefine connections for efficiency reasons
    iii = []
    jjj = []
    for ii in range(nOrdNeurons):
        for jj in range(nMemNeurons):
            if ((floor(ii / numNeuronsPerGroup) == floor(jj / numNeuronsPerGroup)) & (ii != jj)):
                iii.append(ii)
                jjj.append(jj)

    iiMemOrd = []
    jjMemOrd = []
    for ii in range(nOrdNeurons):
        for jj in range(nMemNeurons):
            if (floor(ii / numNeuronsPerGroup) == floor(jj / numNeuronsPerGroup) - 1) & ((floor(jj / numNeuronsPerGroup) - 1) >= 0):
                iiMemOrd.append(ii)
                jjMemOrd.append(jj)

    synInpOrd1e.connect(i=np.arange(numNeuronsPerGroup), j=np.arange(numNeuronsPerGroup))  # Input to first OG
    synOrdMem1e.connect(i=iii, j=jjj)
    synMemOrd1e.connect(i=iiMemOrd, j=jjMemOrd)
    synOrdOrd1e.connect(i=iii, j=jjj)  # i=j is also connected, exclude if necessary
    synMemMem1e.connect(i=iii, j=jjj)  # i=j is also connected, exclude if necessary
    synOrdOrd1i.connect('floor(i/numNeuronsPerGroup)!=floor(j/numNeuronsPerGroup) and (i!=j)')
    synMemOrd1i.connect(i=iii, j=jjj)
    # CoS
    synCoSOrd1i.connect(True)
    # Reset
    synResetOrd1i.connect(True)
    synResetMem1i.connect(True)

    # change some parameters
    # weights
    # excitatory
    synInpOrd1e.weight = synInpOrd1e_weight
    synOrdMem1e.weight = synOrdMem1e_weight
    synMemOrd1e.weight = synMemOrd1e_weight
    # local
    synOrdOrd1e.weight = synOrdOrd1e_weight
    synMemMem1e.weight = synMemMem1e_weight
    # inhibitory
    synOrdOrd1i.weight = synOrdOrd1i_weight
    synMemOrd1i.weight = synMemOrd1i_weight
    synCoSOrd1i.weight = synCoSOrd1i_weight
    synResetOrd1i.weight = synResetOrd1i_weight
    synResetMem1i.weight = synResetMem1i_weight

    SLGroups = {
        'gOrdGroups': gOrdGroups,
        'gMemGroups': gMemGroups,
        'gInputGroup': gInputGroup,
        'gCoSGroup': gCoSGroup,
        'gResetGroup': gResetGroup,
        'synInpOrd1e': synInpOrd1e,
        'synOrdMem1e': synOrdMem1e,
        'synMemOrd1e': synMemOrd1e,
        'synOrdOrd1e': synOrdOrd1e,
        'synMemMem1e': synMemMem1e,
        'synOrdOrd1i': synOrdOrd1i,
        'synMemOrd1i': synMemOrd1i,
        'synCoSOrd1i': synCoSOrd1i,
        'synResetOrd1i': synResetOrd1i,
        'synResetMem1i': synResetMem1i}

    # Monitors
    spikemonOrd = SpikeMonitor(gOrdGroups, name='spikemonOrd_' + groupname)
    spikemonMem = SpikeMonitor(gMemGroups, name='spikemonMem_' + groupname)
    spikemonCoS = SpikeMonitor(gCoSGroup, name='spikemonCoS_' + groupname)
    spikemonInp = SpikeMonitor(gInputGroup, name='spikemonInp_' + groupname)
    spikemonReset = SpikeMonitor(gResetGroup, name='spikemonReset_' + groupname)

    SLMonitors = {
        'spikemonOrd': spikemonOrd,
        'spikemonMem': spikemonMem,
        'spikemonCoS': spikemonCoS,
        'spikemonInp': spikemonInp,
        'spikemonReset': spikemonReset}

    standaloneParams = {synInpOrd1e.name + '_weight': synInpOrd1e_weight,
                        synOrdMem1e.name + '_weight': synOrdMem1e_weight,
                        synMemOrd1e.name + '_weight': synMemOrd1e_weight,
                        # local
                        synOrdOrd1e.name + '_weight': synOrdOrd1e_weight,
                        synMemMem1e.name + '_weight': synMemMem1e_weight,
                        # inhibitory
                        synOrdOrd1i.name + '_weight': synOrdOrd1i_weight,
                        synMemOrd1i.name + '_weight': synMemOrd1i_weight,
                        synCoSOrd1i.name + '_weight': synCoSOrd1i_weight,
                        synResetOrd1i.name + '_weight': synResetOrd1i_weight,
                        synResetMem1i.name + '_weight': synResetMem1i_weight,
                        # refractory
                        gOrdGroups.name + '_refP': gOrdGroups_refP,
                        gMemGroups.name + '_refP': gMemGroups_refP
                        }

    return SLGroups, SLMonitors, standaloneParams


def plotSequenceLearning(SLMonitors):

    spikemonOrd = SLMonitors['spikemonOrd']
    spikemonMem = SLMonitors['spikemonMem']
    spikemonInp = SLMonitors['spikemonInp']
    spikemonCoS = SLMonitors['spikemonCoS']
    spikemonReset = SLMonitors['spikemonReset']
    duration = max(spikemonOrd.t) + 10 * ms
    print ('plot...')
    figure(figsize=(8, 12))
    title('sequence learning')
    nPlots = 5 * 100
    subplot(nPlots + 11)
    plot(spikemonOrd.t / ms, spikemonOrd.i, '.k')
    xlabel('Time [ms]')
    ylabel('i_ord')
    # ylim([0,nOrdNeurons])
    xlim([0, duration / ms])
    subplot(nPlots + 12)
    plot(spikemonMem.t / ms, spikemonMem.i, '.k')
    xlabel('Time [ms]')
    ylabel('i_mem')
    # ylim([0,nOrdNeurons])
    xlim([0, duration / ms])
    subplot(nPlots + 13)
    plot(spikemonInp.t / ms, spikemonInp.i, '.k')
    xlabel('Time [ms]')
    ylabel('i_in')
    # ylim([0,nPerGroup])
    xlim([0, duration / ms])
    subplot(nPlots + 14)
    plot(spikemonCoS.t / ms, spikemonCoS.i, '.k')
    xlabel('Time [ms]')
    ylabel('i_CoS')
    # ylim([0,nPerGroup])
    xlim([0, duration / ms])
    subplot(nPlots + 15)
    plot(spikemonReset.t / ms, spikemonReset.i, '.k')
    xlabel('Time [ms]')
    ylabel('i_Reset')
    # ylim([0,nPerGroup])
    xlim([0, duration / ms])
