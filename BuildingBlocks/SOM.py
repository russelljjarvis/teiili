'''
Created 03.2017
@author: Alpha

This is an attempt to implement a spiking SOM inspired by Rumbell et al. 2014
'''

#===============================================================================
#import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from random import randrange, random
from random import seed as rnseed
from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second, Network, prefs,\
    SpikeMonitor, StateMonitor, figure, plot, show, xlabel, ylabel,\
    seed, xlim, ylim, subplot, network_operation, set_device, device, TimedArray,\
    defaultclock, profiling_summary, codegen
from brian2 import *

from NCSBrian2Lib.Equations.neuronEquations import ExpAdaptIF
#from NCSBrian2Lib.Equations.synapseEquations import fusiSynV as fusiSynapseEq
from NCSBrian2Lib.Equations.synapseEquations import StdpSynV as StdpSynapseEq
from NCSBrian2Lib.Equations.synapseEquations import reversalSynV
from NCSBrian2Lib.Parameters.neuronParams import gerstnerExpAIFdefaultregular
#from NCSBrian2Lib.Parameters.synapseParams import fusiDefault
from NCSBrian2Lib.Parameters.synapseParams import revSyn_default
from NCSBrian2Lib.Parameters.synapseParams import StdpSyn_default as plasticSynapsePar
from NCSBrian2Lib.Tools.tools import fkernelgauss1d, printStates, fkernel2d, xy2ind
from NCSBrian2Lib.BuildingBlocks.WTA import WTA
from NCSBrian2Lib.Plotting.WTAplot import plotWTATiles

from NCSBrian2Lib.BuildingBlocks.BuildingBlock import BuildingBlock
from NCSBrian2Lib.Groups.Groups import Neurons, Connections

somParams = {  # input params
    'sigmGaussInput': 6.0,  # this is the tau of the gaussian input to the input fields
    'inputWeight': 2000,  # this is the weight of the gaussian input to the input fields
    # input inh params
    'synInpInh1e_weight': 0.6,
    'synInpInh1i_weight': -0.2,
    'synInhInp1i_weight': -1.5,
    'synInpInh1e_taugIe': 14 * ms,
    'synInpInh1i_taugIi': 5 * ms,
    # plastic
    'plasticSynapse_taupre': 10 * ms,
    'plasticSynapse_taupost': 10 * ms,
    'plasticSynapse_weight': 124,
    'plasticSynapse_LRdecay': 0.99,
    # wtaSOM params
    'synWTAInh1e_weight': 0.5,
    'synInhWTA1i_weight': -1.8,
    'synWTAWTA1e_weight': 0.18,  # lateral excitation
    'rpWTA': 2 * ms,
    'rpInh': 1 * ms,
    'rpInp': 2 * ms,
    # sigm SOM connectivity kernel params
    'sigmSOMlateralExc': 2.6,
    'sigmSOMlateralExc_DecayTau': 2 * second}


class SOM(BuildingBlock):
    def __init__(self, name,
                 SOMinput,
                 interval,
                 nWTA1dNeurons=64,
                 nWTA2dNeurons=10,  # 16
                 ninputs=3,  # number of Input groups
                 cutoff=6,
                 # Input inhibition:
                 nInhNeuronsSOM=2,
                 nInhNeuronsInp=2,
                 neuronEq=ExpAdaptIF, synapseEq=reversalSynV,
                 neuronParams=gerstnerExpAIFdefaultregular, synapseParams=revSyn_default,
                 blockParams=somParams, debug=False):

        BuildingBlock.__init__(self, name, neuronEq, synapseEq, neuronParams, synapseParams, blockParams, debug)

        self.Groups, self.Monitors, self.replaceVars = genSOM(name, SOMinput,
                                                              interval, nWTA1dNeurons=nWTA1dNeurons, nWTA2dNeurons=nWTA2dNeurons,
                                                              ninputs=ninputs, cutoff=cutoff, nInhNeuronsSOM=nInhNeuronsSOM,
                                                              nInhNeuronsInp=nInhNeuronsInp,
                                                              neuronEq=neuronEq, neuronPar=neuronParams,
                                                              synapseEq=synapseEq, synapsePar=synapseParams,
                                                              monitor=True, debug=debug,
                                                              **blockParams)


def genSOM(name,  # not used at the moment
           # network parameters
           SOMinput,
           interval,
           nWTA1dNeurons=64,
           nWTA2dNeurons=10,  # 16
           ninputs=3,  # number of Input groups
           cutoff=6,
           # Input inhibition:
           nInhNeuronsSOM=2,
           nInhNeuronsInp=2,
           # input params
           sigmGaussInput=6.0,  # this is the tau of the gaussian input to the input fields
           inputWeight=2000,  # this is the weight of the gaussian input to the input fields
           # input inh params
           synInpInh1e_weight=0.6,
           synInpInh1i_weight=-0.2,
           synInhInp1i_weight=-1.5,
           synInpInh1e_taugIe=14 * ms,
           synInpInh1i_taugIi=5 * ms,
           # plastic
           plasticSynapse_taupre=10 * ms,
           plasticSynapse_taupost=10 * ms,
           plasticSynapse_weight=124,
           plasticSynapse_LRdecay=0.99,
           # wtaSOM params
           synWTAInh1e_weight=0.5,
           synInhWTA1i_weight=-1.8,
           synWTAWTA1e_weight=0.18,  # lateral excitation
           rpWTA=2 * ms,
           rpInh=1 * ms,
           rpInp=2 * ms,
           # sigm SOM connectivity kernel params
           sigmSOMlateralExc=2.6,
           sigmSOMlateralExc_DecayTau=2 * second,
           neuronEq=ExpAdaptIF, synapseEq=reversalSynV,
           neuronPar=gerstnerExpAIFdefaultregular, synapsePar=revSyn_default,
           monitor=True, debug=False):

    #SOMinput = []

    plasticSynapsePar["taupre"] = plasticSynapse_taupre
    plasticSynapsePar["taupost"] = plasticSynapse_taupost
    plasticSynapsePar["weight"] = plasticSynapse_weight

    neuronPar['a'] = 0 * nS  # 4*nS #disable adaptation
    neuronPar['b'] = 0 * nA  # 0.0805*nA

    start = time.time()

    #===============================================================================
    # create wtaSOM
    wtaParams = {'weInpWTA': 0,
                 'weWTAInh': synWTAInh1e_weight,
                 'wiInhWTA': synInhWTA1i_weight,
                 'weWTAWTA': synWTAWTA1e_weight,
                 'rpWTA': rpWTA,
                 'rpInh': rpInh,
                 'sigm': sigmSOMlateralExc
                 }
    wtaSOM = WTA('wtaSOM', dimensions=2,
                 neuronParams=neuronPar, synapseParams=synapsePar, blockParams=wtaParams,
                 numNeurons=nWTA2dNeurons, numInhNeurons=nInhNeuronsSOM,
                 cutoff=cutoff, numWtaInputs=1,
                 additionalStatevars=["latSigmaTau : second (constant)"], debug=debug)

    # decay of kernel sigma ???
    synwtaSOMwtaSOM1e = wtaSOM.Groups['synWTAWTA1e']
    synwtaSOMwtaSOM1e.latSigmaTau = sigmSOMlateralExc_DecayTau
    synwtaSOMwtaSOM1e.run_regularly('''weight = latWeight * fkernel2d(i,j,latSigma*exp(-t/latSigmaTau),nWTA2dNeurons)''', dt=100 * ms)

    #===============================================================================
    # create Input groups u

    nInpNeurons = nWTA1dNeurons

    gInpGroup = Neurons(ninputs * nInpNeurons, neuronEq, neuronPar, refractory=rpInp, numInputs=2,
                        additionalStatevars=["inputWeight : 1 (constant)"], debug=debug, name='gInpGroup')
    gInpGroup.inputWeight = inputWeight
    gInpSubgroups = [gInpGroup[(inp * nInpNeurons):((inp + 1) * nInpNeurons)] for inp in range(ninputs)]
    #spikemonInp = SpikeMonitor(gInpGroup)
    #statemonInp = StateMonitor(gInpGroup, ('Vm','Iin','Iconst'), record=range(ninputs*nInpNeurons))

    #===============================================================================
    # create Input Inhibition Inh_u
    gInpInhGroup = Neurons(nInhNeuronsInp, neuronEq, neuronPar, refractory=rpInh, debug=debug, numInputs=2, name='gInpInhGroup')
    #spikemonInpInh = SpikeMonitor(gInpInhGroup)
    #statemonInpInh = StateMonitor(gInpInhGroup, ('Vm','Iin','Iconst'), record=range(nInhNeuronsInp))

    #===============================================================================
    # create input inhibition synapses

    synInpInh1e = Connections(gInpGroup, gInpInhGroup, synapseEq, synapsePar, method="euler", debug=debug, name='sInpInh1e')
    synInpInh1i = Connections(gInpGroup, gInpInhGroup, synapseEq, synapsePar, method="euler", debug=debug, name='sInpInh1i')
    synInhInp1i = Connections(gInpInhGroup, gInpGroup, synapseEq, synapsePar, method="euler", debug=debug, name='sInhInp1i')

    synInpInh1e.connect(True)
    synInpInh1i.connect(True)
    synInhInp1i.connect(True)

    synapseParInpInh1i = synapsePar
    synapseParInpInh1i["taugIi"] = synInpInh1i_taugIi
    synapseParInpInh1e = synapsePar
    synapseParInpInh1e["taugIe"] = synInpInh1e_taugIe

    synInpInh1e.weight = synInpInh1e_weight
    synInpInh1i.weight = synInpInh1i_weight
    synInhInp1i.weight = synInhInp1i_weight

    #===============================================================================
    # create plastic synapses

    synInpSom1e = Connections(gInpGroup, wtaSOM.Groups['gWTAGroup'], StdpSynapseEq, plasticSynapsePar, debug=debug,
                              additionalStatevars=["LRdecay: 1 (constant)"], name='sInpSom1e')

    synInpSom1e.connect(True)

    #statemonSynInpSom1e = StateMonitor(synInpSom1e,('w'), record=range(ninputs*nInpNeurons*nWTA2dNeurons*nWTA2dNeurons))

    synInpSom1e.w = 'w_max*rand()'

    synInpSom1e.LRdecay = plasticSynapse_LRdecay
    synInpSom1e.run_regularly('''diffApre = diffApre*LRdecay''', dt=100 * ms)
    # :TODO decrease of learning rate

    stimTimes = range(0, len(SOMinput) * interval, interval)
    meangaussind = [TimedArray([int(sigmGaussInput + (nInpNeurons - 2 * sigmGaussInput) *
                                    SOMinput[int(np.floor(stimTime / interval))][inp]) for stimTime in stimTimes], dt=interval * ms) for inp in range(ninputs)]

    for ii in range(len(meangaussind)):
        exec('meangaussind' + str(ii) + '=meangaussind[ii]')  # sorry for that. Please tell me, if you find a better way

    for inp in range(ninputs):
        gInpSubgroups[inp].run_regularly('''Iconst = inputWeight * fkernelgauss1d(int(meangaussind{inp}(t)),i,sigmGaussInput)*pA'''.format(inp=inp), dt=interval * ms)

    #===============================================================================

    # if plotInp:
    #    somNet.add((spikemonInp,statemonInp))#,statemonInp
    #    somNet.add((spikemonInpInh,statemonInpInh))
    #

    # make dictionnaries
    SOMGroups = {
        'gInpGroup': gInpGroup,
        'gInpSubgroups': gInpSubgroups,

        'synwtaSOMwtaSOM1e': synwtaSOMwtaSOM1e,

        'synInpSom1e': synInpSom1e,
        'gInpInhGroup': gInpInhGroup,
        'synInpInh1e': synInpInh1e,
        'synInpInh1i': synInpInh1i,
        'synInhInp1i': synInhInp1i}

    SOMGroups.update(wtaSOM.Groups)

    # spikemons
    if monitor:
        SOMMonitors = wtaSOM.Monitors

    # replacevars should be the real names of the parameters, that can be changed by the arguments of this function:
    # in this case: weInpWTA, weWTAInh, wiInhWTA, weWTAWTA,rpWTA, rpInh,sigm
    replaceVars = wtaSOM.replaceVars
    replaceVars += [
        gInpGroup.name + '_inputWeight',
        gInpGroup.name + '_refP',
        gInpInhGroup.name + '_refP',

        synInpSom1e.name + '_weight',
        synInpSom1e.name + '_taupre',
        synInpSom1e.name + '_taupost',
        synInpSom1e.name + '_diffApre',
        synInpSom1e.name + '_Q_diffAPrePost',
        synInpSom1e.name + '_LRdecay',
        synInpInh1e.name + '_weight',
        synInpInh1i.name + '_weight',
        synInhInp1i.name + '_weight',

        synwtaSOMwtaSOM1e.name + '_weight',
        synwtaSOMwtaSOM1e.name + '_latSigmaTau',
    ]

    if True:
        print('The keys of the output dict are:')
        for key in SOMGroups:
            if key not in wtaSOM.Groups:
                print(key)

    end = time.time()
    print ('setting up took ' + str(end - start) + ' sec')

    return SOMGroups, SOMMonitors, replaceVars
