# coding: utf-8
#===============================================================================
"""
This is a simple chain of neurons
"""
import os
import numpy as np
from datetime import datetime

from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second, Network, prefs,\
    SpikeMonitor, StateMonitor, figure, plot, show, xlabel, ylabel,\
    seed, xlim, ylim, subplot, network_operation, TimedArray,\
    defaultclock, SpikeGeneratorGroup

from NCSBrian2Lib.models.neuron_models import ExpAdaptIF
from NCSBrian2Lib.models.synapse_models import reversalSynV

from NCSBrian2Lib.BuildingBlocks.BuildingBlock import BuildingBlock
from NCSBrian2Lib.Groups.Groups import Neurons, Connections
#===============================================================================


chain_params = {'num_chains': 4,
               'num_neurons_per_chain': 15,
               'synChaCha1e_weight': 4,
               'synInpCha1e_weight': 1,
               'gChaGroup_refP': 1 * ms}


class Chain(BuildingBlock):
    def __init__(self, name, neuron_eq_builder=ExpAdaptIF(1),
                 synapse_eq_builder=reversalSynV(),
                 block_params=chain_params,
                 num_inputs=1, debug=False):
        self.num_chains = block_params['num_chains']
        self.num_neurons_per_chain = block_params['num_neurons_per_chain']
        self.synChaCha1e_weight = block_params['synChaCha1e_weight']
        self.synInpCha1e_weight = block_params['synInpCha1e_weight']
        self.gChaGroup_refP = block_params['gChaGroup_refP']

        BuildingBlock.__init__(self, name, neuron_eq_builder, synapse_eq_builder,
                               block_params, debug)

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

        self.inputGroup = self.Groups['gChaInpGroup']
        self.group = self.Groups['gChaGroup']
        self.synapse = self.Groups['synChaCha1e']

        self.spikemonCha = self.Monitors['spikemonCha']
        self.spikemonChaInp = self.Monitors['spikemonChaInp']

    def plot(self, savedir=None):

        if len(self.spikemonCha.t) < 1:
            print(
                'Monitor is empty, have you run the Network and added the monitor to the Network?')
            return

        duration = max(self.spikemonCha.t + 10 * ms)
        # Cha plots
        fig = figure()
        subplot(211)
        plot(self.spikemonCha.t / ms, self.spikemonCha.i, '.k')
        xlabel('Time [ms]')
        ylabel('i_Cha')
        # ylim([0,0])
        xlim([0, duration / ms])
        subplot(212)
        plot(self.spikemonChaInp.t / ms, self.spikemonChaInp.i, '.k')
        xlabel('Time [ms]')
        ylabel('i_Cha_Inp')

        xlim([0, duration / ms])
        if savedir is not None:
            fig.savefig(os.path.join(savedir, self.name + '_' +
                                     datetime.now().strftime('%Y%m%d_%H%M%S') + '.png'))
        return fig


def gen_chain(groupname='Cha',
             neuron_eq_builder=ExpAdaptIF(1),
             synapse_eq_builder=reversalSynV(),
             num_chains=4,
             num_neurons_per_chain=15,
             num_inputs=1,
             synChaCha1e_weight=4,
             synInpCha1e_weight=1,
             gChaGroup_refP=1 * ms,
             debug=False):
    """create chains of neurons"""

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

    spikemonChaInp = SpikeMonitor(gChaInpGroup, name='spikemon' + groupname + 'ChaInp')
    spikemonCha = SpikeMonitor(gChaGroup, name='spikemon' + groupname + 'Cha')

    Monitors = {
        'spikemonChaInp': spikemonChaInp,
        'spikemonCha': spikemonCha
    }

    Groups = {
        'gChaGroup': gChaGroup,
        'gChaInpGroup': gChaInpGroup,
        'synChaCha1e': synChaCha1e,
        'synInpCha1e': synInpCha1e
    }

    standalone_params = [synChaCha1e.name + '_weight']

    return Groups, Monitors, standalone_params
