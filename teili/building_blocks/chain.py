# -*- coding: utf-8 -*-
"""This is a simple syn-fire chain of neurons

Attributes:
    chain_params (dict): Dictionary of default parameters for syn-fire
        chain

Todo:
    * Update docstrings
        Not all Description of attributes are set. Please provide meaningful
            docstrings

Example:
    To use the syn-fire chain building block in your simulation you need
    to create an object of the class by:

    >>> from teili.building_blocks.chain import Chain
    >>> my_bb = Chain(name='my_chain')

    If you want to change the underlying neuron and synapse model you need to
    provide a different equation_builder class:

    >>> from teili.models.neuron_models import DPI
    >>> from teili.models.synapse_models import DPISyn
    >>> my_bb = Chain(name='my_chain',
                      neuron_eq_builder=DPI,
                      synapse_eq_builder=DPISyn)

    If you want to change the default parameters of your building block
    you need to define a dictionary, which you pass to the building_block

    >>> chain_params = {'num_chains': 4,
                        'num_neurons_per_chain': 15,
                        'synChaCha1e_weight': 4,
                        'synInpCha1e_weight': 1,
                        'gChaGroup_refP': 1 * ms}
    >>> my_bb = Chain(name='my_chain', block_params=chain_params)
"""
# @Author: Alpha Renner
# @Date:   2018-06-01 18:45:19

import os
import numpy as np
from datetime import datetime

from brian2 import ms, SpikeGeneratorGroup, SpikeMonitor

from matplotlib.pyplot import xlim, figure, xlabel, \
    ylabel, plot, subplot

from teili.models.neuron_models import ExpAdaptIF
from teili.models.synapse_models import ReversalSynV

from teili.building_blocks.building_block import BuildingBlock
from teili.core.groups import Neurons, Connections
#=========================================================================


chain_params = {'num_chains': 4,
                'num_neurons_per_chain': 15,
                'synChaCha1e_weight': 4,
                'synInpCha1e_weight': 1,
                'gChaGroup_refP': 1 * ms}


class Chain(BuildingBlock):

    """This is a simple syn-fire chain of neurons.

    Attributes:
        gChaGroup_refP (str, optional): Parameter specifying the refractory period.
        group (dict): List of keys of neuron population.
        inputGroup (SpikeGenerator): SpikeGenerator obj. to stimulate syn-fire chain.
        num_chains (int, optional): Number of chains to generate.
        num_neurons_per_chain (int, optional): Number of neurons within one chain.
        spikemon_cha (brian2 SpikeMonitor obj.): Description.
        spikemon_cha_inp (brian2 SpikeMonitor obj.): Description.
        standalone_params (dict): Keys for all standalone parameters necessary for cpp code generation.
        synapse (TYPE): Description.
        synChaCha1e_weight (int, optional): Parameter specifying the recurrent weight.
        synInpCha1e_weight (int, optional): Parameter specifying the input weight.
    """

    def __init__(self, name, neuron_eq_builder=ExpAdaptIF,
                 synapse_eq_builder=ReversalSynV,
                 block_params=chain_params,
                 num_inputs=1, verbose=False):
        """Summary

        Args:
            name (str, required): Base name for building block.
            neuron_eq_builder (teili.models.builder obj, optional): Neuron equation builder object.
            synapse_eq_builder (teili.models.builder obj, optional): Synapse equation builder object.
            block_params (dict, optional): Dictionary of parameters such as synChaCha1e_weight or gChaGroup_refP.
            num_inputs (int, optional): Number of inputs from different source populations.
            verbose (bool, optional): Debug flag.
        """
        self.num_chains = block_params['num_chains']
        self.num_neurons_per_chain = block_params['num_neurons_per_chain']
        self.synChaCha1e_weight = block_params['synChaCha1e_weight']
        self.synInpCha1e_weight = block_params['synInpCha1e_weight']
        self.gChaGroup_refP = block_params['gChaGroup_refP']

        BuildingBlock.__init__(self, name, neuron_eq_builder, synapse_eq_builder,
                               block_params, verbose)

        self.Groups, self.Monitors,\
            self.standalone_params = gen_chain(name,
                                               neuron_eq_builder,
                                               synapse_eq_builder,
                                               self.numChains,
                                               self.num_neurons_per_chain,
                                               num_inputs,
                                               self.synChaCha1e_weight,
                                               self.synInpCha1e_weight,
                                               self.gChaGroup_refP,
                                               debug=self.debug)

        self.input_group = self.Groups['gChaInpGroup']
        self.group = self.Groups['gChaGroup']
        self.synapse = self.Groups['synChaCha1e']

        self.spikemon_cha = self.Monitors['spikemon_cha']
        self.spikemon_cha_inp = self.Monitors['spikemon_cha_inp']

    def plot(self, savedir=None):
        """Simple function to plot recorded state and spikemonitors.

        Args:
            savedir (str, optional): Path to directory to save plot.

        Returns:
            matplotlib.pyplot object: Returns figure.
        """
        if len(self.spikemon_cha.t) < 1:
            print(
                'Monitor is empty, have you run the Network and added the monitor to the Network?')
            return

        duration = max(self.spikemon_cha.t + 10 * ms)
        # Cha plots
        fig = figure()
        subplot(211)
        plot(self.spikemon_cha.t / ms, self.spikemon_cha.i, '.k')
        xlabel('Time [ms]')
        ylabel('i_Cha')
        # ylim([0,0])
        xlim([0, duration / ms])
        subplot(212)
        plot(self.spikemon_cha_inp.t / ms, self.spikemon_cha_inp.i, '.k')
        xlabel('Time [ms]')
        ylabel('i_Cha_Inp')

        xlim([0, duration / ms])
        if savedir is not None:
            fig.savefig(os.path.join(savedir, self.name + '_' +
                                     datetime.now().strftime('%Y%m%d_%H%M%S') + '.png'))
        return fig


def gen_chain(groupname='Cha',
              neuron_eq_builder=ExpAdaptIF(1),
              synapse_eq_builder=ReversalSynV(),
              num_chains=4,
              num_neurons_per_chain=15,
              num_inputs=1,
              synChaCha1e_weight=4,
              synInpCha1e_weight=1,
              gChaGroup_refP=1 * ms,
              debug=False):
    """Creates chains of neurons

    Args:
        groupname (str, optional): Base name for building block.
        neuron_eq_builder (TYPE, optional): Neuron equation builder object.
        synapse_eq_builder (TYPE, optional): Synapse equation builder object.
        num_chains (int, optional): Number of chains to generate.
        num_neurons_per_chain (int, optional): Number of neurons within one chain.
        num_inputs (int, optional): Number of inputs from different source populations.
        synChaCha1e_weight (int, optional): Parameter specifying the recurrent weight.
        synInpCha1e_weight (int, optional): Parameter specifying the input weight.
        gChaGroup_refP (TYPE, optional): Parameter specifying the refractory period.
        debug (bool, optional): Debug flag.

    Returns:
        Groups (dictionary): Keys to all neuron and synapse groups.
        Monitors (dictionary): Keys to all spike- and statemonitors.
        standalone_params (dictionary): Dictionary which holds all parameters to create a standalone network.
    """

    # empty input SpikeGenerator
    ts_cha_inp = np.asarray([]) * ms
    ind_cha_inp = np.asarray([])
    gChaInpGroup = SpikeGeneratorGroup(num_chains, indices=ind_cha_inp,
                                       times=ts_cha_inp, name='g' + groupname + 'Inp')

    gChaGroup = Neurons(num_neurons_per_chain * num_chains, equation_builder=neuron_eq_builder(num_inputs),
                        refractory=gChaGroup_refP, name='g' + groupname,
                        debug=debug)

    synChaCha1e = Connections(gChaGroup, gChaGroup,
                              equation_builder=synapse_eq_builder,
                              method='euler', name='s' + groupname + '' + groupname + '1e')
    synChaCha1e.connect('i+1==j and (j%numNeuronsPerChain)!=0')

    synInpCha1e = Connections(gChaInpGroup, gChaGroup,
                              equation_builder=synapse_eq_builder,
                              method='euler', name='sInp' + groupname + '1e')

    for i_cha in range(num_chains):
        synInpCha1e.connect(i=i_cha, j=i_cha * num_neurons_per_chain)

    # change some parameters
    synChaCha1e.weight = synChaCha1e_weight
    synInpCha1e.weight = synInpCha1e_weight

    spikemon_cha_inp = SpikeMonitor(
        gChaInpGroup, name='spikemon' + groupname + '_cha_inp')
    spikemon_cha = SpikeMonitor(gChaGroup, name='spikemon' + groupname + '_cha')

    Monitors = {
        'spikemon_cha_inp': spikemon_cha_inp,
        'spikemon_cha': spikemon_cha
    }

    Groups = {
        'gChaGroup': gChaGroup,
        'gChaInpGroup': gChaInpGroup,
        'synChaCha1e': synChaCha1e,
        'synInpCha1e': synInpCha1e
    }

    standalone_params = [synChaCha1e.name + '_weight']

    return Groups, Monitors, standalone_params
