'''
Created 03.2017
This files contains different WTA circuits
1dWTA
2dWTA

@author: Alpha
'''
import time
import numpy as np

from brian2 import ms, SpikeGeneratorGroup, SpikeMonitor,\
    StateMonitor, figure, subplot, mV, pA
#from brian2 import *

from NCSBrian2Lib.Tools.tools import fkernel1d, fkernel2d, fdist2d, printStates, ind2xy, ind2x, ind2y
from NCSBrian2Lib.Tools.plotTools import plotSpikemon, plotStatemon

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

wtaParams = {'weInpWTA': 1.5,
             'weWTAInh': 1,
             'wiInhWTA': -1,
             'weWTAWTA': 0.5,
             'sigm': 3,
             'rpWTA': 3 * ms,
             'rpInh': 1 * ms
            }


class WTA(BuildingBlock):
    '''a 1 or 2D square WTA'''

    def __init__(self, name, dimensions=1, neuronEq=ExpAdaptIF, synapseEq=reversalSynV,
                 neuronParams=gerstnerExpAIFdefaultregular, synapseParams=revSyn_default,
                 blockParams=wtaParams, numNeurons=16, numInhNeurons=2, cutoff=10,
                 additionalStatevars=[], numInputs=1, debug=False):

        self.numNeurons = numNeurons
        self.dimensions = dimensions

        BuildingBlock.__init__(self, name, neuronEq, synapseEq,
                               neuronParams, synapseParams, blockParams, debug)

        if dimensions == 1:
            self.Groups, self.Monitors,\
                self.standaloneParams = gen1dWTA(name,
                                                 neuronEq, neuronParams,
                                                 synapseEq, synapseParams,
                                                 numNeurons=numNeurons,
                                                 numInhNeurons=numInhNeurons,
                                                 additionalStatevars=additionalStatevars,
                                                 cutoff=cutoff, numInputs=numInputs,
                                                 monitor=True, debug=debug,
                                                 **blockParams)
        elif dimensions == 2:
            self.Groups, self.Monitors,\
                self.standaloneParams = gen2dWTA(name,
                                                 neuronEq, neuronParams,
                                                 synapseEq, synapseParams,
                                                 numNeurons=numNeurons,
                                                 numInhNeurons=numInhNeurons,
                                                 additionalStatevars=additionalStatevars,
                                                 cutoff=cutoff, numInputs=numInputs,
                                                 monitor=True, debug=debug,
                                                 **blockParams)
        else:
            raise NotImplementedError("only 1 and 2 d WTA available, sorry")

        self.inputGroup = self.Groups['gWTAInpGroup']
        self.group = self.Groups['gWTAGroup']

        self.spikemonWTA = self.Monitors['spikemonWTA']

    def plot(self, startTime=0 * ms, endTime=None):
        "Simple plot for WTA"

        if endTime is None:
            if len(self.spikemonWTA.t) > 0:
                endTime = max(self.spikemonWTA.t)
            else:
                endTime = 0 *ms
        plotWTA(self.name, startTime, endTime, self.numNeurons **
                self.dimensions, self.Monitors)

# TODO: Generalize for n dimensions


def gen1dWTA(groupname, neuronEquation=ExpAdaptIF,
             neuronParameters=gerstnerExpAIFdefaultregular,
             synEquation=reversalSynV, synParameters=revSyn_default,
             weInpWTA=1.5, weWTAInh=1, wiInhWTA=-1, weWTAWTA=0.5, sigm=3,
             rpWTA=3 * ms, rpInh=1 * ms,
             numNeurons=64, numInhNeurons=5, cutoff=10, numInputs=1,
             monitor=True, additionalStatevars=[], debug=False):
    '''generates a new WTA'''

    # time measurement
    start = time.clock()

    # create neuron groups
    gWTAGroup = Neurons(numNeurons, neuronEquation, neuronParameters, refractory=rpWTA,
                        name='g' + groupname, numInputs=3 + numInputs, debug=debug)
    gWTAInhGroup = Neurons(numInhNeurons, neuronEquation, neuronParameters,
                           refractory=rpInh, name='g' + groupname + '_Inh',
                           numInputs=1, debug=debug)

    # empty input for WTA group
    tsWTA = np.asarray([]) * ms
    indWTA = np.asarray([])
    gWTAInpGroup = SpikeGeneratorGroup(
        numNeurons, indices=indWTA, times=tsWTA, name='g' + groupname + '_Inp')

    # printStates(gWTAInpGroup)
    # create synapses
    synInpWTA1e = Connections(gWTAInpGroup, gWTAGroup, synEquation, synParameters,
                              method="euler", debug=debug, name='s' + groupname + '_Inpe')
    synWTAWTA1e = Connections(gWTAGroup, gWTAGroup, synEquation, synParameters,
                              method="euler", debug=debug, name='s' + groupname + '_e',
                              additionalStatevars=["latWeight : 1 (constant)", "latSigma : 1"] + additionalStatevars)  # kernel function
    synInhWTA1i = Connections(gWTAInhGroup, gWTAGroup, synEquation, synParameters,
                              method="euler", debug=debug, name='s' + groupname + '_Inhi')
    synWTAInh1e = Connections(gWTAGroup, gWTAInhGroup, synEquation, synParameters,
                              method="euler", debug=debug, name='s' + groupname + '_Inhe')

    # connect synapses
    synInpWTA1e.connect('i==j')
    # connect the nearest neighbors including itself
    synWTAWTA1e.connect('abs(i-j)<=cutoff')
    synWTAInh1e.connect('True')  # Generates all to all connectivity
    synInhWTA1i.connect('True')

    # set weights
    synInpWTA1e.weight = weInpWTA
    synWTAInh1e.weight = weWTAInh
    synInhWTA1i.weight = wiInhWTA
    # lateral excitation kernel
    # we add an additional attribute to that synapse, which allows us to change
    # and retrieve that value more easily
    synWTAWTA1e.latWeight = weWTAWTA
    synWTAWTA1e.latSigma = sigm
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
        statemonWTA = StateMonitor(gWTAGroup, ('Vm', 'Ie', 'Ii'), record=True, name = 'statemon' + groupname + '_WTA')
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
    print('creating WTA of ' + str(numNeurons) + ' neurons with name ' +
          groupname + ' took ' + str(end - start) + ' sec')

    return Groups, Monitors, standaloneParams


def gen2dWTA(groupname, neuronEquation=ExpAdaptIF,
             neuronParameters=gerstnerExpAIFdefaultregular,
             synEquation=reversalSynV, synParameters=revSyn_default,
             weInpWTA=1.5, weWTAInh=1, wiInhWTA=-1, weWTAWTA=2, sigm=2.5,
             rpWTA=2.5 * ms, rpInh=1 * ms,
             numNeurons=20, numInhNeurons=3, cutoff=9, numInputs=1,
             monitor=True, additionalStatevars=[], debug=False):
    '''generates a new square 2d WTA
    3 inputs to the gWTAGroup are used, so start with 4 for additional inputs'''

    # time measurement
    start = time.clock()

    # create neuron groups
    num2dNeurons = numNeurons**2
    gWTAGroup = Neurons(num2dNeurons, neuronEquation, neuronParameters, refractory=rpWTA,
                        name='g' + groupname, numInputs=3 + numInputs, debug=debug)
    gWTAInhGroup = Neurons(numInhNeurons, neuronEquation, neuronParameters,
                           refractory=rpInh, name='g' + groupname + '_Inh', numInputs=1, debug=debug)


    gWTAGroup.addAttr('numNeurons',numNeurons)
    gWTAGroup.addAttr('ind2x',ind2x)
    gWTAGroup.addAttr('ind2y',ind2y)
    gWTAGroup.x = "ind2x(i, numNeurons)"
    gWTAGroup.y = "ind2y(i, numNeurons)"

    # empty input for WTA group
    tsWTA = np.asarray([]) * ms
    indWTA = np.asarray([])
    gWTAInpGroup = SpikeGeneratorGroup(
        num2dNeurons, indices=indWTA, times=tsWTA, name='g' + groupname + '_Inp')

    # printStates(gWTAInpGroup)
    # create synapses
    synInpWTA1e = Connections(gWTAInpGroup, gWTAGroup, synEquation, synParameters,
                              method="euler", debug=debug, name='s' + groupname + '_Inpe')
    synWTAWTA1e = Connections(gWTAGroup, gWTAGroup, synEquation, synParameters,
                              method="euler", debug=debug, name='s' + groupname + '_e',
                              additionalStatevars=["latWeight : 1 (constant)", "latSigma : 1"] + additionalStatevars)  # kernel function
    synInhWTA1i = Connections(gWTAInhGroup, gWTAGroup, synEquation, synParameters,
                              method="euler", debug=debug, name='s' + groupname + '_Inhi')
    synWTAInh1e = Connections(gWTAGroup, gWTAInhGroup, synEquation, synParameters,
                              method="euler", debug=debug, name='s' + groupname + '_Inhe')

    # connect synapses
    synInpWTA1e.connect('i==j')
    # connect the nearest neighbors including itself
    synWTAWTA1e.connect('fdist2d(i,j,numNeurons)<=cutoff')
    synWTAInh1e.connect('True')  # Generates all to all connectivity
    synInhWTA1i.connect('True')

    # set weights
    synInpWTA1e.weight = weInpWTA
    synWTAInh1e.weight = weWTAInh
    synInhWTA1i.weight = wiInhWTA

    # lateral excitation kernel
    # we add an additional attribute to that synapse, which allows us to change
    # and retrieve that value more easily
    synWTAWTA1e.latWeight = weWTAWTA
    synWTAWTA1e.latSigma = sigm
    synWTAWTA1e.addAttr('fkernel2d',fkernel2d)
    synWTAWTA1e.addAttr('numNeurons',numNeurons)
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
    statemonWTA = StateMonitor(gWTAGroup, ('Vm', 'Ie', 'Ii'), record=True, name = 'statemon' + groupname + '_WTA')
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
    print('creating WTA of ' + str(numNeurons) + ' x ' + str(numNeurons) +
          ' neurons with name ' + groupname + ' took ' + str(end - start) + ' sec')

    return Groups, Monitors, standaloneParams


def plotWTA(name, startTime, endTime, numNeurons, WTAMonitors):

    fig = figure(figsize=(8, 3))
    plotSpikemon(startTime, endTime,
                 WTAMonitors['spikemonWTA'], numNeurons, ylab='ind WTA_'+name)
    fig = figure(figsize=(8, 3))
    plotSpikemon(startTime, endTime,
                 WTAMonitors['spikemonWTAInp'], None, ylab='ind WTAInp_'+name)
    fig = figure(figsize=(8, 3))
    plotSpikemon(startTime, endTime,
                 WTAMonitors['spikemonWTAInh'], None, ylab='ind WTAInp_'+name)
    # fig.savefig('fig/'+name+'_Spikes.png')

    if numNeurons > 20:
        plotStateNeurons = range(20)
    else:
        plotStateNeurons = numNeurons

    statemonWTA = WTAMonitors['statemonWTA']
    if len(statemonWTA.t)>0:
        fig = figure(figsize=(8, 10))
        nPlots = 3 * 100
        subplot(nPlots + 11)
        for ii in plotStateNeurons:
            plotStatemon(startTime, endTime, statemonWTA,
                         ii, variable='Vm', unit=mV)
        subplot(nPlots + 12)
        for ii in plotStateNeurons:
            plotStatemon(startTime, endTime, statemonWTA,
                         ii, variable='Ii', unit=pA)
        subplot(nPlots + 13)
        for ii in plotStateNeurons:
            plotStatemon(startTime, endTime, statemonWTA,
                         ii, variable='Ie', unit=pA)
        #fig.savefig('fig/'+name+'_States.png', dpi=300)
