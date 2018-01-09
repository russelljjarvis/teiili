# -*- coding: utf-8 -*-
# @Author: mmilde, alpren
# @Date:   2017-12-27 10:46:44
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-09 15:09:09

"""
This files contains different WTA circuits
1dWTA
2dWTA
"""

import time
import numpy as np

from brian2 import ms, SpikeGeneratorGroup, SpikeMonitor,\
    StateMonitor, figure, subplot, mV, pA

# from NCSBrian2Lib.tools.tools import fkernel1d, fkernel2d, fdist2d,
from NCSBrian2Lib.tools.misc import printStates
from NCSBrian2Lib.tools.indexing import ind2xy, ind2x, ind2y
from NCSBrian2Lib.tools.plotting import plotSpikemon, plotStatemon

from NCSBrian2Lib.building_blocks.building_block import BuildingBlock
from NCSBrian2Lib.core.groups import Neurons, Connections

from NCSBrian2Lib.models.neuron_models import DPI
from NCSBrian2Lib.models.synapse_models import DPISyn


wtaParams = {'weInpWTA': 1.5,
             'weWTAInh': 1,
             'wiInhWTA': -1,
             'weWTAWTA': 0.5,
             'sigm': 3,
             'rpWTA': 3 * ms,
             'rpInh': 1 * ms
             }


class WTA(BuildingBlock):
    '''a 1 or 2D square WTA

    Attributes:
        dimensions (TYPE): Description
        group (TYPE): Description
        inputGroup (TYPE): Description
        numNeurons (TYPE): Description
        spikemonWTA (TYPE): Description
        standaloneParams (TYPE): Description
    '''

    def __init__(self, name,
                 dimensions=1,
                 neuron_eq_builder=DPI,
                 synapse_eq_builder=DPISyn,
                 block_params=wtaParams,
                 num_inp_neurons=10,
                 num_neurons=16,
                 num_inh_neurons=2,
                 cutoff=10,
                 additional_statevars=[],
                 num_inputs=1,
                 monitor=True,
                 debug=False):
        """Summary

        Args:
            name (TYPE): Description
            dimensions (int, optional): Description
            neuron_eq_builder (TYPE, optional): Description
            synapse_eq_builder (TYPE, optional): Description
            block_params (TYPE, optional): Description
            num_inp_neurons (int, optional): Description
            num_neurons (int, optional): Description
            num_inh_neurons (int, optional): Description
            cutoff (int, optional): Description
            additional_statevars (list, optional): Description
            num_inputs (int, optional): Description
            monitor (bool, optional): Description
            debug (bool, optional): Description

        Raises:
            NotImplementedError: Description
        """
        self.num_neurons = num_neurons
        self.dimensions = dimensions
        BuildingBlock.__init__(self, name,
                               neuron_eq_builder,
                               synapse_eq_builder,
                               block_params,
                               debug,
                               monitor)

        if dimensions == 1:
            self.Groups, self.Monitors,
            self.standaloneParams = gen1dWTA(name,
                                             neuron_eq_builder,
                                             synapse_eq_builder,
                                             num_neurons=num_neurons,
                                             num_inh_neurons=num_inh_neurons,
                                             additional_statevars=additional_statevars,
                                             cutoff=cutoff,
                                             num_inputs=num_inputs,
                                             monitor=True,
                                             debug=debug,
                                             **block_params)
        elif dimensions == 2:
            self.Groups, self.Monitors,
            self.standaloneParams = gen2dWTA(name,
                                             neuron_eq_builder,
                                             synapse_eq_builder,
                                             num_neurons=num_neurons,
                                             num_inh_neurons=num_inh_neurons,
                                             additional_statevars=additional_statevars,
                                             cutoff=cutoff,
                                             num_inputs=num_inputs,
                                             monitor=True,
                                             debug=debug,
                                             **block_params)

        else:
            raise NotImplementedError("only 1 and 2 d WTA available, sorry")

        self.inputGroup = self.Groups['gWTAInpGroup']
        self.group = self.Groups['gWTAGroup']

        self.spikemonWTA = self.Monitors['spikemonWTA']

    def plot(self, startTime=0 * ms, endTime=None):
        """Simple plot for WTA

        Args:
            startTime (int, optional): Description
            endTime (int, optional): Description
        """

        if endTime is None:
            if len(self.spikemonWTA.t) > 0:
                endTime = max(self.spikemonWTA.t)
            else:
                endTime = 0 * ms
        plotWTA(self.name, startTime, endTime, self.numNeurons **
                self.dimensions, self.Monitors)

# TODO: Generalize for n dimensions


def gen1dWTA(groupname,
             neuron_eq_builder=DPI,
             synapse_eq_builder=DPISyn,
             weInpWTA=1.5, weWTAInh=1, wiInhWTA=-1, weWTAWTA=0.5, sigm=3,
             rpWTA=3 * ms, rpInh=1 * ms,
             num_neurons=64, num_inh_neurons=5, cutoff=10, num_inputs=1,
             monitor=True, additional_statevars=[], debug=False):
    """Summary

    Args:
        groupname (TYPE): Description
        neuron_eq_builder (TYPE, optional): Description
        synapse_eq_builder (TYPE, optional): Description
        weInpWTA (float, optional): Description
        weWTAInh (int, optional): Description
        wiInhWTA (TYPE, optional): Description
        weWTAWTA (float, optional): Description
        sigm (int, optional): Description
        rpWTA (TYPE, optional): Description
        rpInh (TYPE, optional): Description
        num_neurons (int, optional): Description
        num_inh_neurons (int, optional): Description
        cutoff (int, optional): Description
        num_inputs (int, optional): Description
        monitor (bool, optional): Description
        additional_statevars (list, optional): Description
        debug (bool, optional): Description

    Returns:
        TYPE: Description
    """
    # time measurement
    start = time.clock()

    # create neuron groups
    gWTAGroup = Neurons(num_neurons, equation_builder=neuron_eq_builder(),
                        refractory=rpWTA, name='g' + groupname,
                        num_inputs=3 + num_inputs)
    gWTAInhGroup = Neurons(num_inh_neurons, equation_builder=neuron_eq_builder(),
                           refractory=rpInh, name='g' + groupname + '_Inh',
                           num_inputs=1)

    # empty input for WTA group
    tsWTA = np.asarray([]) * ms
    indWTA = np.asarray([])
    gWTAInpGroup = SpikeGeneratorGroup(
        num_neurons, indices=indWTA, times=tsWTA, name='g' + groupname + '_Inp')

    # create synapses
    synInpWTA1e = Connections(gWTAInpGroup, gWTAGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_Inpe')
    synWTAWTA1e = Connections(gWTAGroup, gWTAGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_e')  # ,
                              # additionalStatevars=["latWeight : 1 (shared, constant)",
                              #                      "latSigma : 1 (shared,constant)"] +
                              # additional_statevars)  # kernel function
    synInhWTA1i = Connections(gWTAInhGroup, gWTAGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_Inhi')
    synWTAInh1e = Connections(gWTAGroup, gWTAInhGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_Inhe')

    # connect synapses
    synInpWTA1e.connect('i==j')
    # connect the nearest neighbors including itself
    synWTAWTA1e.connect('abs(i-j)<=cutoff')
    synWTAInh1e.connect('True')  # Generates all to all connectivity
    synInhWTA1i.connect('True')

    synWTAWTA1e.addStateVariable(name='latWeight', shared=False, constant=True)
    synWTAWTA1e.addStateVariable(name='latSigma', shared=False, constant=True)

    # set weights
    synInpWTA1e.weight = weInpWTA
    synWTAInh1e.weight = weWTAInh
    synInhWTA1i.weight = wiInhWTA
    # lateral excitation kernel
    # we add an additional attribute to that synapse, which allows us to change
    # and retrieve that value more easily
    synWTAWTA1e.latWeight = weWTAWTA
    synWTAWTA1e.latSigma = sigm
    synWTAWTA1e.namespace['fkernel1d'] = fkernel1d
    synWTAWTA1e.weight = 'latWeight * fkernel1d(i,j,latSigma)'

    # print(synWTAWTA1e.weight)

    Groups = {
        'gWTAGroup': gWTAGroup,
        'gWTAInhGroup': gWTAInhGroup,
        'gWTAInpGroup': gWTAInpGroup,
        'synInpWTA1e': synInpWTA1e,
        'synWTAWTA1e': synWTAWTA1e,
        'synWTAInh1e': synWTAInh1e,
        'synInhWTA1i': synInhWTA1i}

    # spikemons
    if monitor:
        spikemonWTA = SpikeMonitor(gWTAGroup, name='spikemon' + groupname + '_WTA')
        spikemonWTAInh = SpikeMonitor(gWTAInhGroup, name='spikemon' + groupname + '_WTAInh')
        spikemonWTAInp = SpikeMonitor(gWTAInpGroup, name='spikemon' + groupname + '_WTAInp')
        statemonWTA = StateMonitor(gWTAGroup, ('Vm', 'Ie', 'Ii'), record=True, name='statemon' + groupname + '_WTA')
        Monitors = {
            'spikemonWTA': spikemonWTA,
            'spikemonWTAInh': spikemonWTAInh,
            'spikemonWTAInp': spikemonWTAInp,
            'statemonWTA': statemonWTA}

    # replacevars should be the 'real' names of the parameters, that can be
    # changed by the arguments of this function:
    # in this case: weInpWTA, weWTAInh, wiInhWTA, weWTAWTA,rpWTA, rpInh,sigm
    standaloneParams = {
        synInpWTA1e.name + '_weight': weInpWTA,
        synWTAInh1e.name + '_weight': weWTAInh,
        synInhWTA1i.name + '_weight': wiInhWTA,
        synWTAWTA1e.name + '_latWeight': weWTAWTA,
        synWTAWTA1e.name + '_latSigma': sigm,
        gWTAGroup.name + '_refP': rpWTA,
        gWTAInhGroup.name + '_refP': rpInh,
    }

    end = time.clock()
    print('creating WTA of ' + str(num_neurons) + ' neurons with name ' +
          groupname + ' took ' + str(end - start) + ' sec')

    return Groups, Monitors, standaloneParams


def gen2dWTA(groupname,
             neuron_eq_builder=DPI,
             synapse_eq_builder=DPISyn,
             weInpWTA=1.5, weWTAInh=1, wiInhWTA=-1, weWTAWTA=2, sigm=2.5,
             rpWTA=2.5 * ms, rpInh=1 * ms,
             num_neurons=20, num_inh_neurons=3, cutoff=9, num_inputs=1,
             monitor=True, additional_statevars=[], debug=False):
    '''generates a new square 2d WTA

    Args:
        groupname (TYPE): Description
        neuron_eq_builder (TYPE, optional): Description
        synapse_eq_builder (TYPE, optional): Description
        weInpWTA (float, optional): Description
        weWTAInh (int, optional): Description
        wiInhWTA (TYPE, optional): Description
        weWTAWTA (int, optional): Description
        sigm (float, optional): Description
        rpWTA (TYPE, optional): Description
        rpInh (TYPE, optional): Description
        num_neurons (int, optional): Description
        num_inh_neurons (int, optional): Description
        cutoff (int, optional): Description
        num_inputs (int, optional): Description
        monitor (bool, optional): Description
        additional_statevars (list, optional): Description
        debug (bool, optional): Description

    Returns:
        TYPE: Description
    '''

    # time measurement
    start = time.clock()

    # create neuron groups
    num2dNeurons = num_neurons**2
    # gWTAGroup = Neurons(num2dNeurons, neuronEquation, neuronParameters, refractory=rpWTA, name='g' + groupname,
    #                     numInputs=3 + numWtaInputs, debug=debug)
    # gWTAInhGroup = Neurons(numInhNeurons, neuronEquation, neuronParameters, refractory=rpInh, name='g' + groupname + '_Inh',
    #                        numInputs=1, debug=debug)
    gWTAGroup = Neurons(num2dNeurons, equation_builder=neuron_eq_builder(),
                        refractory=rpWTA, name='g' + groupname,
                        num_inputs=3 + num_inputs)
    gWTAInhGroup = Neurons(num_inh_neurons, equation_builder=neuron_eq_builder(),
                           refractory=rpInh, name='g' + groupname + '_Inh',
                           num_inputs=1)

    gWTAGroup.namespace['numNeurons'] = num_neurons
    gWTAGroup.namespace['ind2x'] = ind2x
    gWTAGroup.namespace['ind2y'] = ind2y
    gWTAGroup.x = "ind2x(i, numNeurons)"
    gWTAGroup.y = "ind2y(i, numNeurons)"

    # empty input for WTA group
    tsWTA = np.asarray([]) * ms
    indWTA = np.asarray([])
    gWTAInpGroup = SpikeGeneratorGroup(
        num2dNeurons, indices=indWTA, times=tsWTA, name='g' + groupname + '_Inp')

    # printStates(gWTAInpGroup)
    # create synapses
    synInpWTA1e = Connections(gWTAInpGroup, gWTAGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_Inpe')
    synWTAWTA1e = Connections(gWTAGroup, gWTAGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_e')  # ,
                              #additional_statevars=["latWeight : 1 (constant)", "latSigma : 1"] + additional_statevars)  # kernel function
    synInhWTA1i = Connections(gWTAInhGroup, gWTAGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_Inhi')
    synWTAInh1e = Connections(gWTAGroup, gWTAInhGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_Inhe')

    # connect synapses
    synInpWTA1e.connect('i==j')
    # connect the nearest neighbors including itself
    synWTAWTA1e.connect('fdist2d(i,j,numNeurons)<=cutoff')
    synWTAInh1e.connect('True')  # Generates all to all connectivity
    synInhWTA1i.connect('True')

    synWTAWTA1e.addStateVariable(name='latWeight', shared=True, constant=True)
    synWTAWTA1e.addStateVariable(name='latSigma', shared=True, constant=True)

    # set weights
    synInpWTA1e.weight = weInpWTA
    synWTAInh1e.weight = weWTAInh
    synInhWTA1i.weight = wiInhWTA

    # lateral excitation kernel
    # we add an additional attribute to that synapse, which allows us to change
    # and retrieve that value more easily
    synWTAWTA1e.latWeight = weWTAWTA
    synWTAWTA1e.latSigma = sigm
    synWTAWTA1e.namespace['fkernel2d'] = fkernel2d
    synWTAWTA1e.namespace['numNeurons'] = numNeurons
    synWTAWTA1e.weight = 'latWeight * fkernel2d(i,j,latSigma,numNeurons)'
    # print(synWTAWTA1e.weight)

    Groups = {
        'gWTAGroup': gWTAGroup,
        'gWTAInhGroup': gWTAInhGroup,
        'gWTAInpGroup': gWTAInpGroup,
        'synInpWTA1e': synInpWTA1e,
        'synWTAWTA1e': synWTAWTA1e,
        'synWTAInh1e': synWTAInh1e,
        'synInhWTA1i': synInhWTA1i}

    # spikemons
    spikemonWTA = SpikeMonitor(gWTAGroup, name='spikemon' + groupname + '_WTA')
    spikemonWTAInh = SpikeMonitor(gWTAInhGroup, name='spikemon' + groupname + '_WTAInh')
    spikemonWTAInp = SpikeMonitor(gWTAInpGroup, name='spikemon' + groupname + '_WTAInp')
    statemonWTA = StateMonitor(gWTAGroup, ('Vm', 'Ie', 'Ii'), record=True, name='statemon' + groupname + '_WTA')
    Monitors = {
        'spikemonWTA': spikemonWTA,
        'spikemonWTAInh': spikemonWTAInh,
        'spikemonWTAInp': spikemonWTAInp,
        'statemonWTA': statemonWTA}

    # replacevars should be the real names of the parameters,
    # that can be changed by the arguments of this function:
    # in this case: weInpWTA, weWTAInh, wiInhWTA, weWTAWTA,rpWTA, rpInh,sigm
    standaloneParams = {
        synInpWTA1e.name + '_weight': weInpWTA,
        synWTAInh1e.name + '_weight': weWTAInh,
        synInhWTA1i.name + '_weight': wiInhWTA,
        synWTAWTA1e.name + '_latWeight': weWTAWTA,
        synWTAWTA1e.name + '_latSigma': sigm,
        gWTAGroup.name + '_refP': rpWTA,
        gWTAInhGroup.name + '_refP': rpInh,
    }

    end = time.clock()
    print ('creating WTA of ' + str(num_neurons) + ' x ' + str(num_neurons) + ' neurons with name ' +
           groupname + ' took ' + str(end - start) + ' sec')

    if True:
        print('The keys of the output dict are:')
        for key in Groups:
            print(key)

    return Groups, Monitors, standaloneParams


def plotWTA(name, startTime, endTime, numNeurons, WTAMonitors):
    """Summary

    Args:
        name (TYPE): Description
        startTime (TYPE): Description
        endTime (TYPE): Description
        numNeurons (TYPE): Description
        WTAMonitors (TYPE): Description
    """
    fig = figure(figsize=(8, 3))
    plotSpikemon(startTime, endTime,
                 WTAMonitors['spikemonWTA'], numNeurons, ylab='ind WTA_' + name)
    fig = figure(figsize=(8, 3))
    plotSpikemon(startTime, endTime,
                 WTAMonitors['spikemonWTAInp'], None, ylab='ind WTAInp_' + name)
    fig = figure(figsize=(8, 3))
    plotSpikemon(startTime, endTime,
                 WTAMonitors['spikemonWTAInh'], None, ylab='ind WTAInh_' + name)
    # fig.savefig('fig/'+name+'_Spikes.png')

    if numNeurons > 20:
        plotStateNeurons = range(20)
    else:
        plotStateNeurons = numNeurons

    statemonWTA = WTAMonitors['statemonWTA']
    if len(statemonWTA.t) > 0:
        fig = figure(figsize=(8, 10))
        nPlots = 3 * 100
        subplot(nPlots + 11)
        for ii in plotStateNeurons:
            plotStatemon(startTime, endTime, statemonWTA,
                         ii, variable='Vm', unit=mV, name=name)
        subplot(nPlots + 12)
        for ii in plotStateNeurons:
            plotStatemon(startTime, endTime, statemonWTA,
                         ii, variable='Ii', unit=pA, name=name)
        subplot(nPlots + 13)
        for ii in plotStateNeurons:
            plotStatemon(startTime, endTime, statemonWTA,
                         ii, variable='Ie', unit=pA, name=name)
        # fig.savefig('fig/'+name+'_States.png', dpi=300)
