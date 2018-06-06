#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:45:09 2017

@author: alpha
"""

import numpy as np

#from brian2 import *
from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second, Network, prefs,\
    SpikeGeneratorGroup, NeuronGroup, SpikeMonitor, StateMonitor, figure, plot,\
    seed, xlim, ylim, subplot, network_operation, set_device, device, TimedArray,\
    defaultclock, profiling_summary, floor, title, xlabel, ylabel

from teili.models.neuron_models import ExpAdaptIF
from teili.models.synapse_models import ReversalSynV

from teili.building_blocks.building_block import BuildingBlock
from teili.core.groups import Neurons, Connections

#%%
#=========================================================================

sl_params = {'synInpOrd1e_weight': 1.3,
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

    def __init__(self, name,
                 neuron_eq_builder=ExpAdaptIF,
                 synapse_eq_builder=ReversalSynV,
                 block_params=sl_params, num_elements=3, num_neurons_per_group=6,
                 num_inputs=1, debug=False):

        BuildingBlock.__init__(self, name, neuron_eq_builder, synapse_eq_builder,
                               block_params, debug)

        self.Groups, self.Monitors,\
            self.standalone_params = gen_sequence_learning(name,
                                                        neuron_eq_builder,
                                                        synapse_eq_builder,
                                                        num_elements=num_elements,
                                                        num_neurons_per_group=num_neurons_per_group,
                                                        num_inputs=num_inputs,
                                                        debug=debug, **block_params)
        self.group = self.Groups['gOrdGroups']
        self.inputGroup = self.Groups['gInputGroup']
        self.cosGroup = self.Groups['gCoSGroup']
        self.resetGroup = self.Groups['gResetGroup']

    def plot(self):

        return plot_sequence_learning(self.Monitors)


def gen_sequence_learning(groupname='Seq',
                        neuron_eq_builder=ExpAdaptIF,
                        synapse_eq_builder=ReversalSynV,
                        num_elements=4, num_neurons_per_group=8,
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
                        num_inputs=1,
                        debug=False):
    """create Sequence Learning Network after the model from Sandamirskaya and Schoener (2010)"""

    nOrdNeurons = num_neurons_per_group * num_elements
    nMemNeurons = num_neurons_per_group * num_elements

    # Input to start sequence manually
    ts_input = np.asarray([]) * ms
    ind_input = np.asarray([])
    gInputGroup = SpikeGeneratorGroup(
        num_neurons_per_group, indices=ind_input, times=ts_input, name='spikegenInp_' + groupname)
    # CoS Input #TODO: CoS as NeuronGroup
    ts_cos = np.asarray([]) * ms
    ind_cos = np.asarray([])
    gCoSGroup = SpikeGeneratorGroup(
        num_neurons_per_group, indices=ind_cos, times=ts_cos, name='spikegenCoS_' + groupname)
    # reset group #TODO: Reset as NeuronGroup
    ts_reset = np.asarray([]) * ms
    ind_reset = np.asarray([])
    gResetGroup = SpikeGeneratorGroup(
        num_neurons_per_group, indices=ind_reset, times=ts_reset, name='spikegenReset_' + groupname)

    # NeuronGroups
    gOrdGroups = Neurons(nOrdNeurons,
                         equation_builder=neuron_eq_builder(num_inputs=7 + num_inputs),
                         refractory=gOrdGroups_refP,
                         name='g' + 'Ord_' + groupname)
    gMemGroups = Neurons(nMemNeurons,
                         equation_builder=neuron_eq_builder(num_inputs=3),
                         refractory=gMemGroups_refP,
                         name='g' + 'Mem_' + groupname)

    # Synapses
    # excitatory
    synInpOrd1e = Connections(gInputGroup, gOrdGroups, equation_builder=synapse_eq_builder(),
                              method="euler", name='sInpOrd1e_' + groupname)
    synOrdMem1e = Connections(gOrdGroups, gMemGroups, equation_builder=synapse_eq_builder(),
                              method="euler", name='sOrdMem1e_' + groupname)
    synMemOrd1e = Connections(gMemGroups, gOrdGroups, synapse_eq_builder(),
                              method="euler", name='sMemOrd1e_' + groupname)
    # local
    synOrdOrd1e = Connections(gOrdGroups, gOrdGroups, equation_builder=synapse_eq_builder(),
                              method="euler", name='sOrdOrd1e_' + groupname)
    synMemMem1e = Connections(gMemGroups, gMemGroups, equation_builder=synapse_eq_builder(),
                              method="euler", name='sMemMem1e_' + groupname)
    # inhibitory
    synOrdOrd1i = Connections(gOrdGroups, gOrdGroups, equation_builder=synapse_eq_builder(),
                              method="euler", name='sOrdOrd1i_' + groupname)
    synMemOrd1i = Connections(gMemGroups, gOrdGroups, equation_builder=synapse_eq_builder(),
                              method="euler", name='sMemOrd1i_' + groupname)
    synCoSOrd1i = Connections(gCoSGroup, gOrdGroups, equation_builder=synapse_eq_builder(),
                              method="euler", name='sCoSOrd1i_' + groupname)
    synResetOrd1i = Connections(gResetGroup, gOrdGroups, equation_builder=synapse_eq_builder(),
                                method="euler", name='sResOrd1i_' + groupname)
    synResetMem1i = Connections(gResetGroup, gMemGroups, equation_builder=synapse_eq_builder(),
                                method="euler", name='sResMem1i_' + groupname)

    # predefine connections for efficiency reasons
    iii = []
    jjj = []
    for ii in range(nOrdNeurons):
        for jj in range(nMemNeurons):
            if (floor(ii / num_neurons_per_group) == floor(jj / num_neurons_per_group)) & (ii != jj):
                iii.append(ii)
                jjj.append(jj)

    iiMemOrd = []
    jjMemOrd = []
    for ii in range(nOrdNeurons):
        for jj in range(nMemNeurons):
            if (floor(ii / num_neurons_per_group) == floor(jj / num_neurons_per_group) - 1) & ((floor(jj / num_neurons_per_group) - 1) >= 0):
                iiMemOrd.append(ii)
                jjMemOrd.append(jj)

    synInpOrd1e.connect(i=np.arange(num_neurons_per_group), j=np.arange(
        num_neurons_per_group))  # Input to first OG
    synOrdMem1e.connect(i=iii, j=jjj)
    synMemOrd1e.connect(i=iiMemOrd, j=jjMemOrd)
    # i=j is also connected, exclude if necessary
    synOrdOrd1e.connect(i=iii, j=jjj)
    # i=j is also connected, exclude if necessary
    synMemMem1e.connect(i=iii, j=jjj)
    synOrdOrd1i.connect(
        'floor(i/num_neurons_per_group)!=floor(j/num_neurons_per_group) and (i!=j)')
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

    Groups = {
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
    spikemonReset = SpikeMonitor(
        gResetGroup, name='spikemonReset_' + groupname)

    Monitors = {
        'spikemonOrd': spikemonOrd,
        'spikemonMem': spikemonMem,
        'spikemonCoS': spikemonCoS,
        'spikemonInp': spikemonInp,
        'spikemonReset': spikemonReset}

    standalone_params = {  # synInpOrd1e.name + '_weight': synInpOrd1e_weight,
        # synOrdMem1e.name + '_weight': synOrdMem1e_weight,
        # synMemOrd1e.name + '_weight': synMemOrd1e_weight,
        # local
        # synOrdOrd1e.name + '_weight': synOrdOrd1e_weight,
        # synMemMem1e.name + '_weight': synMemMem1e_weight,
        # inhibitory
        # synOrdOrd1i.name + '_weight': synOrdOrd1i_weight,
        # synMemOrd1i.name + '_weight': synMemOrd1i_weight,
        # synCoSOrd1i.name + '_weight': synCoSOrd1i_weight,
        # synResetOrd1i.name + '_weight': synResetOrd1i_weight,
        # synResetMem1i.name + '_weight': synResetMem1i_weight,
        # refractory
        # gOrdGroups.name + '_refP': gOrdGroups_refP,
        # gMemGroups.name + '_refP': gMemGroups_refP
    }

    return Groups, Monitors, standalone_params


def plot_sequence_learning(Monitors):

    spikemonOrd = Monitors['spikemonOrd']
    spikemonMem = Monitors['spikemonMem']
    spikemonInp = Monitors['spikemonInp']
    spikemonCoS = Monitors['spikemonCoS']
    spikemonReset = Monitors['spikemonReset']
    duration = max(spikemonOrd.t) + 10 * ms
    print('plot...')
    fig = figure(figsize=(8, 12))
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

    return fig
