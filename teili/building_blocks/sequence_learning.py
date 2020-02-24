#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module provides a sequence learning building block.
This building block can be used to learn sequences of items

Attributes:
    sl_params (dict): Dictionary of default parameters for reservoir.

Example:
    To use the Reservoir building block in your simulation you need
    to create an object of the class by doing:

    >>> from teili.building_blocks.reservoir import Reservoir
    >>> my_bb = Reservoir(name='my_sequence')

    If you want to change the underlying neuron and synapse model you need to
    provide a different equation_builder class:

    >>> from teili.models.neuron_models import DPI
    >>> from teili.models.synapse_models import DPISyn
    >>> my_bb = Reservoir(name='my_sequence',
                      neuron_eq_builder=DPI,
                      synapse_eq_builder=DPISyn)

    If you want to change the default parameters of your building block
    you need to define a dictionary, which you pass to the building_block:

    >>> sl_params = {'synInpOrd1e_weight': 1.3,
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
    >>> my_bb = Reservoir(name='my_sequence', block_params=sl_params)
"""
# @Author: Alpha Renner, mmilde
# @Date:   2018-06-01 18:45:19

import numpy as np
from numpy import floor
from brian2 import ms, SpikeGeneratorGroup, SpikeMonitor
from matplotlib.pyplot import xlim, figure, xlabel, \
    ylabel, plot, subplot, title

from teili.building_blocks.building_block import BuildingBlock
from teili.core.groups import Neurons, Connections

from teili.models.neuron_models import ExpAdaptIF
from teili.models.synapse_models import ReversalSynV

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
    '''Sequence Learning Network.

    Attributes:
        cos_group (neuron group): Condition of Satisfaction group.
        group (dict): List of keys of neuron population.
        input_group (SpikeGenerator): SpikeGenerator object to stimulate Reservoir.
        reset_group (neuron group): Reset group, to reset network after CoS is met.
        standalone_params (dict): Keys for all standalone parameters necessary for cpp code generation.
    '''

    def __init__(self, name,
                 neuron_eq_builder=ExpAdaptIF,
                 synapse_eq_builder=ReversalSynV,
                 block_params=sl_params, num_elements=3, num_neurons_per_group=6,
                 num_inputs=1, verbose=False):
        """Summary

        Args:
            name (str, required): Base name for building block.
            neuron_eq_builder (teili.models.builder obj, optional): Neuron equation builder object.
            synapse_eq_builder (teili.models.builder obj, optional): Synapse equation builder object.
            block_params (dict, optional): Dictionary of parameters such as synChaCha1e_weight or gChaGroup_refP.
            num_elements (int, optional): Number of elements in the sequence.
            num_neurons_per_group (int, optional): Number of neurons used to remember each item.
            num_inputs (int, optional): Number of inputs from different source populations.
            verbose (bool, optional): Debug flag.
        """
        BuildingBlock.__init__(self, name, neuron_eq_builder, synapse_eq_builder,
                               block_params, verbose)

        self._groups, self.monitors,\
            self.standalone_params = gen_sequence_learning(name,
                                                           neuron_eq_builder=neuron_eq_builder,
                                                           synapse_eq_builder=synapse_eq_builder,
                                                           num_elements=num_elements,
                                                           num_neurons_per_group=num_neurons_per_group,
                                                           num_inputs=num_inputs,
                                                           verbose=verbose, **block_params)
        self.group = self.groups['gOrdGroups']
        self.input_group = self.groups['gInputGroup']
        self.cos_group = self.groups['gCoSGroup']
        self.reset_group = self.groups['gResetGroup']

    def plot(self, duration = None):
        """Simple plot for sequence learning network.

        Returns:
            pyqtgraph window: The window containing the plot.
        """
        return plot_sequence_learning(self.monitors, duration)


def gen_sequence_learning(groupname='Seq',
                        neuron_eq_builder=ExpAdaptIF,
                        synapse_eq_builder=ReversalSynV,
                        num_elements=4, num_neurons_per_group=8,
                        synInpOrd1e_weight=1.3,
                        synOrdMem1e_weight=1.1,
                        synMemOrd1e_weight=0.16,
                        synOrdOrd1e_weight=1.04,
                        synMemMem1e_weight=1.54,
                        synOrdOrd1i_weight=-1.95,
                        synMemOrd1i_weight=-0.384,
                        synCoSOrd1i_weight=-1.14,
                        synResetOrd1i_weight=-1.44,
                        synResetMem1i_weight=-2.6,
                        gOrdGroups_refP=1.7 * ms,
                        gMemGroups_refP=2.3 * ms,
                        num_inputs=1,
                        verbose=False):
    """Create Sequence Learning Network after the model from Sandamirskaya and Schoener (2010).

    Args:
        groupname (str, optional): Base name for building block.
        neuron_eq_builder (teili.models.builder obj, optional): Neuron equation builder object.
        synapse_eq_builder (teili.models.builder obj, optional): Synapse equation builder object.
        num_elements (int, optional): Number of elements in the sequence.
        num_neurons_per_group (int, optional): Number of neurons used to remember each item.
        synInpOrd1e_weight (float, optional): Parameter specifying the input weight.
        synOrdMem1e_weight (float, optional): Parameter specifying the ordinary to memory weight.
        synMemOrd1e_weight (float, optional): Parameter specifying the memory to ordinary weight.
        synOrdOrd1e_weight (float, optional): Parameter specifying the recurrent weight (ord).
        synMemMem1e_weight (float, optional): Parameter specifying the recurrent weight (memory).
        synOrdOrd1i_weight (TYPE, optional): Parameter specifying the recurrent inhibitory weight.
        synMemOrd1i_weight (TYPE, optional): Parameter specifying the memory to ordinary inhibitory weight.
        synCoSOrd1i_weight (TYPE, optional): Parameter specifying the inhibitory weight from cos to ord.
        synResetOrd1i_weight (TYPE, optional): Parameter specifying the the inhibitory weight from reset to ord.
        synResetMem1i_weight (TYPE, optional): Parameter specifying the the inhibitory weight from reset cos to memory.
        gOrdGroups_refP (TYPE, optional): Parameter specifying the refractory period.
        gMemGroups_refP (TYPE, optional): Parameter specifying the refractory period.
        num_inputs (int, optional): Number of inputs from different source populations.
        debug (bool, optional): Debug flag.

    Returns:
        Groups (dictionary): Keys to all neuron and synapse groups.
        Monitors (dictionary): Keys to all spike- and statemonitors.
        standalone_params (dictionary): Dictionary which holds all parameters to create a standalone network.
    """

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


def plot_sequence_learning(Monitors, duration=None):
    """A simple matplotlib wrapper function to plot network activity.

    Args:
        Monitors (building_block.monitors): Dictionary containing all monitors
            created by gen_sequence_learning().

    Returns:
        plt.fig: Matplotlib figure.
    """
    spikemonOrd = Monitors['spikemonOrd']
    spikemonMem = Monitors['spikemonMem']
    spikemonInp = Monitors['spikemonInp']
    spikemonCoS = Monitors['spikemonCoS']
    spikemonReset = Monitors['spikemonReset']
    if duration is None:
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
