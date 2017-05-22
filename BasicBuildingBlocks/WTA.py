'''
Created 03.2017
This files contains different WTA circuits
1dWTA
2dWTA

@author: Alpha
'''

from brian2 import *
import time

from NCSBrian2Lib.tools import setParams, fkernel1d, fkernel2d, fdist2d, printStates
from NCSBrian2Lib.neuronEquations import ExpAdaptIFrev as neuronEq
from NCSBrian2Lib.neuronEquations import Silicon
from NCSBrian2Lib.synapseEquations import reversalSynV as synapseEq
from NCSBrian2Lib.synapseEquations import BraderFusiSynapses
from NCSBrian2Lib.Parameters.neuronParams import gerstnerExpAIFdefaultregular as neuronPar
from NCSBrian2Lib.Parameters.synapseParams import revSyndefault as synapsePar
from NCSBrian2Lib.Parameters.neuronParams import SiliconNeuronP
from NCSBrian2Lib.Parameters.synapseParams import Braderfusi


defaultneuronEqDict, args = neuronEq(debug=True)
defaultsynapseEqDict = synapseEq(debug=True)
siliconneuronEqDict, args = Silicon(debug=True)
siliconsynapseEqDict = SiliconSynapses(debug=True)
learningsynapseEqDict = BraderFusiSynapses()


def gen1dWTA(groupname, neuronEqsDict=defaultneuronEqDict, neuronParameters=neuronPar,
             synEqsDict=defaultsynapseEqDict, synParameters=synapsePar,
             weInpWTA=1.5, weWTAInh=1, wiInhWTA=-1, w_lat=0.5,
             rpWTA=3 * ms, rpInh=1 * ms,
             sigm=3, nNeurons=64, nInhNeurons=5, cutoff=10, monitor=True, debug=False):
    "generates a new WTA"

    # time measurement
    start = time.clock()

    # create neuron groups
    gWTAGroup = NeuronGroup(nNeurons, refractory=rpWTA, method='euler', name='g' + groupname, **neuronEqsDict)
    gWTAInhGroup = NeuronGroup(nInhNeurons, refractory=rpInh, method='euler', name='g' + groupname + '_Inh', **neuronEqsDict)

    # empty input for WTA group
    tsWTA = np.asarray([]) * ms
    indWTA = np.asarray([])
    gWTAInpGroup = SpikeGeneratorGroup(nNeurons, indices=indWTA, times=tsWTA, name='g' + groupname + '_Inp')
    #gWTAInpGroup = PoissonGroup(nNeurons, 100 * Hz,name = 'g'+groupname +'_Inp')

    # printStates(gWTAInpGroup)
    # create synapses
    synInpWTA1e = Synapses(gWTAInpGroup, gWTAGroup, method="euler", name='s' + groupname + '_Inpe', **synEqsDict)
    synWTAWTA1e = Synapses(gWTAGroup, gWTAGroup, method="euler", name='s' + groupname + '_e', **synEqsDict)  # kernel function
    synWTAInh1e = Synapses(gWTAGroup, gWTAInhGroup, method="euler", name='s' + groupname + '_Inhe', **synEqsDict)
    synInhWTA1i = Synapses(gWTAInhGroup, gWTAGroup, method="euler", name='s' + groupname + '_Inhi', **synEqsDict)

    # connect synapses
    synInpWTA1e.connect('i==j')
    synWTAWTA1e.connect('abs(i-j)<=cutoff')  # connect the nearest neighbors including itself
    synWTAInh1e.connect('True')  # Generates all to all connectivity
    synInhWTA1i.connect('True')

    # set weights
    synInpWTA1e.weight = weInpWTA
    synWTAInh1e.weight = weWTAInh
    synInhWTA1i.weight = wiInhWTA
    # lateral excitation kernel
    synWTAWTA1e.weight = 'w_lat * fkernel1d(i,j,sigm)'
    # print(synWTAWTA1e.weight)

    # set parameters of neuron groups
    setParams(gWTAGroup, neuronParameters, debug=True)
    setParams(gWTAInhGroup, neuronParameters, debug=True)
    # printStates(gWTAGroup)

    # set parameters of synapses
    setParams(synInpWTA1e, synParameters, debug=False)
    setParams(synWTAWTA1e, synParameters, debug=True)
    setParams(synWTAInh1e, synParameters, debug=True)
    setParams(synInhWTA1i, synParameters, debug=True)

    # spikemons
    if monitor:
        spikemonWTA = SpikeMonitor(gWTAGroup)
        spikemonWTAInh = SpikeMonitor(gWTAInhGroup)
        spikemonWTAInp = SpikeMonitor(gWTAInpGroup)
        statemonWTA = StateMonitor(gWTAGroup, ('Vm', 'Ie', 'Ii'), record=True)
    else:
        spikemonWTA = 'not monitored'
        spikemonWTAInh = 'not monitored'
        spikemonWTAInp = 'not monitored'
        statemonWTA = 'not monitored'

    end = time.clock()
    print ('creating WTA of ' + str(nNeurons) + ' neurons with name ' + groupname + ' took ' + str(end - start) + ' sec')

    return((gWTAGroup, gWTAInhGroup, gWTAInpGroup,
            synInpWTA1e, synWTAWTA1e, synWTAInh1e, synInhWTA1i,
            spikemonWTA, spikemonWTAInh, spikemonWTAInp, statemonWTA))


def gen2dWTA(groupname, neuronEqsDict=defaultneuronEqDict, neuronParameters=neuronPar,
             synEqsDict=defaultsynapseEqDict, synParameters=synapsePar,
             weInpWTA=1.5, weWTAInh=1, wiInhWTA=-1, w_lat=1,
             rpWTA=2.5 * ms, rpInh=1 * ms,
             sigm=2.5, nNeurons=20, nInhNeurons=3, cutoff=9, monitor=True, debug=False):
    "generates a new WTA"

    # time measurement
    start = time.clock()

    # create neuron groups
    gWTAGroup = NeuronGroup(nNeurons**2, refractory=rpWTA, method='euler', name='g' + groupname, **neuronEqsDict)
    gWTAInhGroup = NeuronGroup(nInhNeurons, refractory=rpInh, method='euler', name='g' + groupname + '_Inh', **neuronEqsDict)

    # empty input for WTA group
    tsWTA = np.asarray([]) * ms
    indWTA = np.asarray([])
    gWTAInpGroup = SpikeGeneratorGroup(nNeurons**2, indices=indWTA, times=tsWTA, name='g' + groupname + '_Inp')
    #gWTAInpGroup = PoissonGroup(nNeurons, 100 * Hz,name = 'g'+groupname +'_Inp')

    # printStates(gWTAInpGroup)
    # create synapses
    synInpWTA1e = Synapses(gWTAInpGroup, gWTAGroup, method="euler", name='s' + groupname + '_Inpe', **synEqsDict)
    synWTAWTA1e = Synapses(gWTAGroup, gWTAGroup, method="euler", name='s' + groupname + '_e', **synEqsDict)  # kernel function
    synWTAInh1e = Synapses(gWTAGroup, gWTAInhGroup, method="euler", name='s' + groupname + '_Inhe', **synEqsDict)
    synInhWTA1i = Synapses(gWTAInhGroup, gWTAGroup, method="euler", name='s' + groupname + '_Inhi', **synEqsDict)

    # connect synapses
    synInpWTA1e.connect('i==j')
    synWTAWTA1e.connect('fdist2d(i,j,nNeurons)<=cutoff')  # connect the nearest neighbors including itself
    synWTAInh1e.connect('True')  # Generates all to all connectivity
    synInhWTA1i.connect('True')

    # set weights
    synInpWTA1e.weight = weInpWTA
    synWTAInh1e.weight = weWTAInh
    synInhWTA1i.weight = wiInhWTA
    # lateral excitation kernel
    synWTAWTA1e.weight = 'w_lat * fkernel2d(i,j,sigm,nNeurons)'
    # print(synWTAWTA1e.weight)

    # set parameters of neuron groups
    setParams(gWTAGroup, neuronParameters, debug=debug)
    setParams(gWTAInhGroup, neuronParameters, debug=debug)
    # printStates(gWTAGroup)

    # set parameters of synapses
    setParams(synInpWTA1e, synParameters, debug=debug)
    setParams(synWTAWTA1e, synParameters, debug=debug)
    setParams(synWTAInh1e, synParameters, debug=debug)
    setParams(synInhWTA1i, synParameters, debug=debug)

    # spikemons
    if monitor:
        spikemonWTA = SpikeMonitor(gWTAGroup)
        spikemonWTAInh = SpikeMonitor(gWTAInhGroup)
        spikemonWTAInp = SpikeMonitor(gWTAInpGroup)
        statemonWTA = StateMonitor(gWTAGroup, ('Vm', 'Ie', 'Ii'), record=True)
    else:
        spikemonWTA = False
        spikemonWTAInh = False
        spikemonWTAInp = False
        statemonWTA = False

    end = time.clock()
    print ('creating ' + str(nNeurons) + 'x' + str(nNeurons) + ' WTA of ' + str(nNeurons**2) + ' neurons with name ' + groupname + ' took ' + str(end - start) + ' sec')

    return((gWTAGroup, gWTAInhGroup, gWTAInpGroup,
            synInpWTA1e, synWTAWTA1e, synWTAInh1e, synInhWTA1i,
            spikemonWTA, spikemonWTAInh, spikemonWTAInp, statemonWTA))


def gen2dWTA_plasticSyn(groupname, neuronEqsDict=siliconneuronEqDict, neuronParameters=SiliconNeuronP,
                        synEqsDict=siliconsynapseEqDict, synParameters=SiliconSynP,
                        synEqsDict_learning=learningsynapseEqDict, synParameters_learning=Braderfusi,
                        weInpWTA=1.5, weWTAInh=1, wiInhWTA=-1, w_lat=1,
                        rpWTA=2.5 * ms, rpInh=1 * ms,
                        sigm=2.5, nNeurons=20, nInpNeurons=10, nInhNeurons=3, cutoff=9, monitor=True, debug=False):
    "generates a new WTA"

    # time measurement
    start = time.clock()

    # create neuron groups
    gWTAGroup = NeuronGroup(nNeurons**2, refractory=rpWTA, method='euler', name='g' + groupname, **neuronEqsDict)
    gWTAInhGroup = NeuronGroup(nInhNeurons, refractory=rpInh, method='euler', name='g' + groupname + '_Inh', **neuronEqsDict)

    # empty input for WTA group
    tsWTA = np.asarray([]) * ms
    indWTA = np.asarray([])
    gWTAInpGroup = SpikeGeneratorGroup(nInpNeurons**2, indices=indWTA, times=tsWTA, name='g' + groupname + '_Inp')

    # create synapses
    synInpWTA1e = Synapses(gWTAInpGroup, gWTAGroup, method="euler", name='s' + groupname + '_Inpe', **synEqsDict_learning)
    synWTAWTA1e = Synapses(gWTAGroup, gWTAGroup, method="euler", name='s' + groupname + '_e', **synEqsDict)  # kernel function
    synWTAInh1e = Synapses(gWTAGroup, gWTAInhGroup, method="euler", name='s' + groupname + '_Inhe', **synEqsDict)
    synInhWTA1i = Synapses(gWTAInhGroup, gWTAGroup, method="euler", name='s' + groupname + '_Inhi', **synEqsDict)

    # connect synapses
    synInpWTA1e.connect('True')  # Starting from weak All-To-All connectivity with STDP synapses
    synWTAWTA1e.connect('fdist2d(i,j,nNeurons)<=cutoff')  # connect the nearest neighbors including itself
    synWTAInh1e.connect('True')  # Generates all to all connectivity
    synInhWTA1i.connect('True')

    # set weights
    synInpWTA1e.weight = weInpWTA  # CHANGE TO BE RANDOMLY INITIALIZED!!!!
    synWTAInh1e.weight = weWTAInh
    synInhWTA1i.weight = wiInhWTA
    # lateral excitation kernel
    synWTAWTA1e.weight = 'w_lat * fkernel2d(i,j,sigm,nNeurons)'
    # print(synWTAWTA1e.weight)

    # set parameters of neuron groups
    setParams(gWTAGroup, neuronParameters, debug=debug)
    setParams(gWTAInhGroup, neuronParameters, debug=debug)
    # printStates(gWTAGroup)

    # set parameters of synapses
    setParams(synInpWTA1e, synParameters_learning, debug=debug)
    setParams(synWTAWTA1e, synParameters, debug=debug)
    setParams(synWTAInh1e, synParameters, debug=debug)
    setParams(synInhWTA1i, synParameters, debug=debug)

    # spikemons
    if monitor:
        spikemonWTA = SpikeMonitor(gWTAGroup)
        spikemonWTAInh = SpikeMonitor(gWTAInhGroup)
        spikemonWTAInp = SpikeMonitor(gWTAInpGroup)
        statemonWTA = StateMonitor(gWTAGroup, ('Vm', 'Ie', 'Ii'), record=True)
    else:
        spikemonWTA = False
        spikemonWTAInh = False
        spikemonWTAInp = False
        statemonWTA = False

    end = time.clock()
    print ('creating ' + str(nNeurons) + 'x' + str(nNeurons) + ' WTA of ' +
           str(nNeurons**2) + ' neurons with name ' + groupname + ' took ' +
           str(end - start) + ' sec')

    return((gWTAGroup, gWTAInhGroup, gWTAInpGroup,
            synInpWTA1e, synWTAWTA1e, synWTAInh1e, synInhWTA1i,
            spikemonWTA, spikemonWTAInh, spikemonWTAInp, statemonWTA))
