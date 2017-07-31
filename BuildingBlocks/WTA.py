'''
Created 03.2017
This files contains different WTA circuits
1dWTA
2dWTA

@author: Alpha
'''

from brian2 import NeuronGroup,ms,SpikeGeneratorGroup,Synapses,SpikeMonitor,StateMonitor,figure, subplot
from brian2 import *

import time
import numpy as np

from NCSBrian2Lib.Tools.tools import fkernel1d, fkernel2d, fdist2d, printStates,ind2xy,ind2x,ind2y
from NCSBrian2Lib.Tools.plotTools import plotSpikemon, plotStatemon

from NCSBrian2Lib.Equations.neuronEquations import ExpAdaptIF 
from NCSBrian2Lib.Equations.neuronEquations import Silicon

from NCSBrian2Lib.Equations.synapseEquations import reversalSynV 
from NCSBrian2Lib.Equations.synapseEquations import BraderFusiSynapses,SiliconSynapses

from NCSBrian2Lib.Parameters.neuronParams import gerstnerExpAIFdefaultregular
from NCSBrian2Lib.Parameters.neuronParams import SiliconNeuronP

from NCSBrian2Lib.Parameters.synapseParams import revSyn_default
from NCSBrian2Lib.Parameters.synapseParams import Braderfusi,SiliconSynP

from NCSBrian2Lib.BuildingBlocks.BuildingBlock import BuildingBlock
from NCSBrian2Lib.Groups.Groups import Neurons, Connections

wtaParams = {'weInpWTA' : 1.5,
             'weWTAInh' : 1,
             'wiInhWTA' : -1,
             'weWTAWTA' : 0.5,
             'rpWTA'    : 3 * ms,
             'rpInh'    : 1 * ms,
             'sigm'     : 3}

class WTA(BuildingBlock):
    '''a 1 or 2D square WTA'''
    def __init__(self,name,dimensions = 1, neuronEq=ExpAdaptIF,synapseEq=reversalSynV,
             neuronParams=gerstnerExpAIFdefaultregular,synapseParams=revSyn_default,
             blockParams=wtaParams, numNeurons = 16, numInhNeurons = 2, cutoff=10,
             additionalStatevars = [], numWtaInputs = 1, debug=False):
        
        self.numNeurons = numNeurons
        self.dimensions = dimensions
        
        BuildingBlock.__init__(self,name,neuronEq,synapseEq,neuronParams,synapseParams,blockParams,debug)
        
        if dimensions == 1:
            self.Groups,self.Monitors,self.replaceVars = gen1dWTA(name,
                     neuronEq,neuronParams,synapseEq,synapseParams,**blockParams,
                     numNeurons = numNeurons, numInhNeurons = numInhNeurons,
                     additionalStatevars = additionalStatevars,
                     cutoff=cutoff, numWtaInputs = numWtaInputs, monitor=True, debug=debug)
        elif dimensions==2:
            self.Groups,self.Monitors,self.replaceVars = gen2dWTA(name,
                     neuronEq,neuronParams,synapseEq,synapseParams,**blockParams,
                     numNeurons = numNeurons, numInhNeurons = numInhNeurons,
                     additionalStatevars = additionalStatevars,
                     cutoff=cutoff, numWtaInputs = numWtaInputs, monitor=True, debug=debug)
        else:
            raise NotImplementedError("only 1 and 2 d WTA available, sorry")
        
        
        self.inputGroup = self.Groups['gWTAInpGroup']
        self.wtaGroup = self.Groups['gWTAGroup']
        
        self.spikemonWTA = self.Monitors['spikemonWTA']
        
    def plot(self,startTime=0*ms,endTime=None):
        
        if endTime is None:
            endTime = max(self.spikemonWTA.t)
        plotWTA(self.name,startTime,endTime,self.numNeurons**self.dimensions,self.Monitors)
        
# TODO: Generalize for n dimensions

def gen1dWTA(groupname, neuronEquation=ExpAdaptIF, neuronParameters=gerstnerExpAIFdefaultregular,
             synEquation=reversalSynV, synParameters=revSyn_default,
             weInpWTA=1.5, weWTAInh=1, wiInhWTA=-1, weWTAWTA=0.5,
             rpWTA=3 * ms, rpInh=1 * ms,
             sigm=3, numNeurons=64, numInhNeurons=5, cutoff=10, numWtaInputs = 1, monitor=True, additionalStatevars = [], debug=False):
    '''generates a new WTA
    3 inputs to the gWTAGroup are used, so start with startInputNumbering+4 for additional inputs'''

    # time measurement
    start = time.clock()

    # create neuron groups
    gWTAGroup       = Neurons(numNeurons,     neuronEquation, neuronParameters, refractory = rpWTA, name='g' + groupname, numInputs = 3+numWtaInputs,debug=debug)
    gWTAInhGroup    = Neurons(numInhNeurons,  neuronEquation, neuronParameters, refractory = rpInh, name='g' + groupname + '_Inh', numInputs = 1,debug=debug)

    # empty input for WTA group
    tsWTA = np.asarray([]) * ms
    indWTA = np.asarray([])
    gWTAInpGroup = SpikeGeneratorGroup(numNeurons, indices=indWTA, times=tsWTA, name='g' + groupname + '_Inp')

    # printStates(gWTAInpGroup)
    # create synapses
    synInpWTA1e = Connections(gWTAInpGroup, gWTAGroup,     synEquation, synParameters, method="euler", debug=debug, name='s' + groupname + '_Inpe')
    synWTAWTA1e = Connections(gWTAGroup,    gWTAGroup,     synEquation, synParameters, method="euler", debug=debug, name='s' + groupname + '_e',
                              additionalStatevars = ["latWeight : 1 (constant)","latSigma : 1"]+additionalStatevars) # kernel function
    synInhWTA1i = Connections(gWTAInhGroup, gWTAGroup,     synEquation, synParameters, method="euler", debug=debug, name='s' + groupname + '_Inhi')
    synWTAInh1e = Connections(gWTAGroup,    gWTAInhGroup,  synEquation, synParameters, method="euler", debug=debug, name='s' + groupname + '_Inhe')

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
    # we add an additional attribute to that synapse, which allows us to change and retrieve that value more easily
    synWTAWTA1e.latWeight = weWTAWTA
    synWTAWTA1e.latSigma = sigm
    synWTAWTA1e.weight = 'latWeight * fkernel1d(i,j,latSigma)'

    # print(synWTAWTA1e.weight)

    Groups = {
            'gWTAGroup'     : gWTAGroup,
            'gWTAInhGroup'  : gWTAInhGroup,
            'gWTAInpGroup'  : gWTAInpGroup,
            'synInpWTA1e'   : synInpWTA1e,
            'synWTAWTA1e'   : synWTAWTA1e, 
            'synWTAInh1e'   : synWTAInh1e, 
            'synInhWTA1i'   : synInhWTA1i}
    
    # spikemons
    if monitor:
        spikemonWTA = SpikeMonitor(gWTAGroup)
        spikemonWTAInh = SpikeMonitor(gWTAInhGroup)
        spikemonWTAInp = SpikeMonitor(gWTAInpGroup)
        statemonWTA = StateMonitor(gWTAGroup, ('Vm', 'Ie', 'Ii'), record=True)
        Monitors = {
               'spikemonWTA'   : spikemonWTA, 
               'spikemonWTAInh': spikemonWTAInh, 
               'spikemonWTAInp': spikemonWTAInp, 
               'statemonWTA'   : statemonWTA}
                
    #replacevars should be the 'real' names of the parameters, that can be changed by the arguments of this function:
    # in this case: weInpWTA, weWTAInh, wiInhWTA, weWTAWTA,rpWTA, rpInh,sigm
    replaceVars = [
                synInpWTA1e.name +'_weight',
                synWTAInh1e.name +'_weight',
                synInhWTA1i.name +'_weight',
                synWTAWTA1e.name +'_latWeight',
                synWTAWTA1e.name +'_latSigma',
                gWTAGroup.name   +'_refP',
                gWTAInhGroup.name+'_refP',
                 ]     
    
    end = time.clock()
    print('creating WTA of ' + str(numNeurons) + ' neurons with name ' + groupname + ' took ' + str(end - start) + ' sec')
    print('The keys of the output dict are:')
    for key in Groups: print(key)
            
    return Groups,Monitors,replaceVars


def gen2dWTA(groupname, neuronEquation=ExpAdaptIF, neuronParameters=gerstnerExpAIFdefaultregular,
             synEquation=reversalSynV, synParameters=revSyn_default,
             weInpWTA=1.5, weWTAInh=1, wiInhWTA=-1, weWTAWTA=2,
             rpWTA=2.5 * ms, rpInh=1 * ms,
             sigm=2.5, numNeurons=20, numInhNeurons=3, cutoff=9, numWtaInputs = 1, monitor=True, additionalStatevars = [], debug=False):
    '''generates a new square 2d WTA
    3 inputs to the gWTAGroup are used, so start with 4 for additional inputs'''

    # time measurement
    start = time.clock()
    
    # create neuron groups
    num2dNeurons = numNeurons**2
    gWTAGroup       = Neurons(num2dNeurons,   neuronEquation, neuronParameters, refractory = rpWTA, name='g' + groupname, numInputs = 3+numWtaInputs,debug=debug)
    gWTAInhGroup    = Neurons(numInhNeurons,  neuronEquation, neuronParameters, refractory = rpInh, name='g' + groupname + '_Inh', numInputs = 1,debug=debug)

    gWTAGroup.x = "ind2x(i, numNeurons)"
    gWTAGroup.y = "ind2y(i, numNeurons)"
    
    # empty input for WTA group
    tsWTA = np.asarray([]) * ms
    indWTA = np.asarray([])
    gWTAInpGroup = SpikeGeneratorGroup(num2dNeurons, indices=indWTA, times=tsWTA, name='g' + groupname + '_Inp')

    # printStates(gWTAInpGroup)
    # create synapses
    synInpWTA1e = Connections(gWTAInpGroup, gWTAGroup,     synEquation, synParameters, method="euler", debug=debug, name='s' + groupname + '_Inpe')
    synWTAWTA1e = Connections(gWTAGroup,    gWTAGroup,     synEquation, synParameters, method="euler", debug=debug, name='s' + groupname + '_e',
                              additionalStatevars = ["latWeight : 1 (constant)","latSigma : 1"]+additionalStatevars)  # kernel function
    synInhWTA1i = Connections(gWTAInhGroup, gWTAGroup,     synEquation, synParameters, method="euler", debug=debug, name='s' + groupname + '_Inhi')
    synWTAInh1e = Connections(gWTAGroup,    gWTAInhGroup,  synEquation, synParameters, method="euler", debug=debug, name='s' + groupname + '_Inhe')

    # connect synapses
    synInpWTA1e.connect('i==j')
    synWTAWTA1e.connect('fdist2d(i,j,numNeurons)<=cutoff')  # connect the nearest neighbors including itself
    synWTAInh1e.connect('True')  # Generates all to all connectivity
    synInhWTA1i.connect('True')
    
    # set weights
    synInpWTA1e.weight = weInpWTA
    synWTAInh1e.weight = weWTAInh
    synInhWTA1i.weight = wiInhWTA
    
    # lateral excitation kernel
    # we add an additional attribute to that synapse, which allows us to change and retrieve that value more easily
    synWTAWTA1e.latWeight = weWTAWTA
    synWTAWTA1e.latSigma = sigm
    synWTAWTA1e.weight = 'latWeight * fkernel2d(i,j,latSigma,numNeurons)'
    # print(synWTAWTA1e.weight)


    Groups = {
            'gWTAGroup'     : gWTAGroup,
            'gWTAInhGroup'  : gWTAInhGroup,
            'gWTAInpGroup'  : gWTAInpGroup,
            'synInpWTA1e'   : synInpWTA1e,
            'synWTAWTA1e'   : synWTAWTA1e, 
            'synWTAInh1e'   : synWTAInh1e, 
            'synInhWTA1i'   : synInhWTA1i}
    
    # spikemons
    spikemonWTA = SpikeMonitor(gWTAGroup)
    spikemonWTAInh = SpikeMonitor(gWTAInhGroup)
    spikemonWTAInp = SpikeMonitor(gWTAInpGroup)
    statemonWTA = StateMonitor(gWTAGroup, ('Vm', 'Ie', 'Ii'), record=True)
    Monitors = {
           'spikemonWTA'   : spikemonWTA, 
           'spikemonWTAInh': spikemonWTAInh, 
           'spikemonWTAInp': spikemonWTAInp, 
           'statemonWTA'   : statemonWTA}
    
    #replacevars should be the real names of the parameters, that can be changed by the arguments of this function:
    # in this case: weInpWTA, weWTAInh, wiInhWTA, weWTAWTA,rpWTA, rpInh,sigm  

    replaceVars = [
                synInpWTA1e.name +'_weight',
                synWTAInh1e.name +'_weight',
                synInhWTA1i.name +'_weight',
                synWTAWTA1e.name +'_latWeight',
                synWTAWTA1e.name +'_latSigma',
                gWTAGroup.name   +'_refP',
                gWTAInhGroup.name+'_refP',
                 ]
                    
    end = time.clock()
    print ('creating WTA of ' + str(numNeurons) + ' x ' + str(numNeurons) + ' neurons with name ' + groupname + ' took ' + str(end - start) + ' sec')
    
    if True:
        print('The keys of the output dict are:')
        for key in Groups: print(key)
            
    return Groups,Monitors,replaceVars


def plotWTA(name,startTime,endTime,numNeurons,WTAMonitors):
      
    fig = figure(figsize=(8,3))
    plotSpikemon(startTime,endTime,WTAMonitors['spikemonWTA'],numNeurons,ylab='ind WTA')
    fig = figure(figsize=(8,3))
    plotSpikemon(startTime,endTime,WTAMonitors['spikemonWTAInp'],None,ylab='ind WTA')
    fig = figure(figsize=(8,3))
    plotSpikemon(startTime,endTime,WTAMonitors['spikemonWTAInh'],None,ylab='ind WTA')
    #fig.savefig('fig/'+name+'_Spikes.png')
    
    if numNeurons > 20:
        plotStateNeurons = range(20)
    else:
        plotStateNeurons = numNeurons
    
    statemonWTA = WTAMonitors['statemonWTA']    
    if statemonWTA is not False:  
        fig = figure(figsize=(8,10))
        nPlots=3*100
        subplot(nPlots+11)
        for ii in plotStateNeurons:
            plotStatemon(startTime,endTime,statemonWTA,ii,variable='Vm',unit=mV)
        subplot(nPlots+12)
        for ii in plotStateNeurons:
            plotStatemon(startTime,endTime,statemonWTA,ii,variable='Ii',unit=pA)
        subplot(nPlots+13)
        for ii in plotStateNeurons:
            plotStatemon(startTime,endTime,statemonWTA,ii,variable='Ie',unit=pA)
        #fig.savefig('fig/'+name+'_States.png', dpi=300)
