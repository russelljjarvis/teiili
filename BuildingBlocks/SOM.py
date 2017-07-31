'''
Created 03.2017
@author: Alpha

This is an attempt to implement a spiking SOM inspired by Rumbell et al. 2014
'''

#===============================================================================
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from random import randrange, random
from random import seed as rnseed
from brian2 import ms,mV,pA,nS,nA,pF,us,volt,second,Network,prefs,SpikeGeneratorGroup,NeuronGroup,\
                   Synapses,SpikeMonitor,StateMonitor,figure, plot,show,xlabel,ylabel,\
                   seed,xlim,ylim,subplot,network_operation,set_device,device,TimedArray,\
                   defaultclock,profiling_summary,codegen
from brian2 import *

from NCSBrian2Lib.Equations.neuronEquations import ExpAdaptIF as neuronEq
#from NCSBrian2Lib.Equations.synapseEquations import fusiSynV as fusiSynapseEq
from NCSBrian2Lib.Equations.synapseEquations import StdpSynV as StdpSynapseEq
from NCSBrian2Lib.Equations.synapseEquations import reversalSynV as synapseEq
from NCSBrian2Lib.Parameters.neuronParams import gerstnerExpAIFdefaultregular as neuronPar
#from NCSBrian2Lib.Parameters.synapseParams import fusiDefault
from NCSBrian2Lib.Parameters.synapseParams import revSyn_default as synapsePar
from NCSBrian2Lib.Parameters.synapseParams import StdpSyn_default as plasticSynapsePar
from NCSBrian2Lib.Tools.tools import fkernelgauss1d, printStates, fkernel2d, xy2ind
from NCSBrian2Lib.BuildingBlocks.WTA import gen2dWTA,gen1dWTA
from NCSBrian2Lib.Plotting.WTAplot import plotWTATiles


def genSOM(# network parameters
            SOMinput,
            interval,
            nWTA1dNeurons = 64,
            nWTA2dNeurons = 10,#16
            ninputs = 3, # number of Input groups
            cutoff = 6,
            # Input inhibition:
            nInhNeuronsSOM = 2,
            nInhNeuronsInp = 2,
            # input params
            sigmGaussInput = 6.0, # this is the tau of the gaussian input to the input fields
            inputWeight = 2000, # this is the weight of the gaussian input to the input fields
            # input inh params
            synInpInh1e_weight = 0.6,   
            synInpInh1i_weight = -0.2,           
            synInhInp1i_weight = -1.5,
            synInpInh1e_taugIe = 14 * ms,
            synInpInh1i_taugIi = 5 * ms, 
            #plastic
            plasticSynapse_taupre = 10 *ms,   
            plasticSynapse_taupost = 10 *ms,   
            plasticSynapse_weight = 124,       
            plasticSynapse_LRdecay=0.99,           
            # wtaSOM params
            synWTAInh1e_weight = 0.5,                
            synInhWTA1i_weight = -1.8,               
            synWTAWTA1e_weight = 0.18,                #lateral excitation
            rpWTA = 2*ms,                            
            rpInh = 1*ms,
            # sigm SOM connectivity kernel params 
            sigmSOMlateralExc = 2.6,                         
            sigmSOMlateralExc_DecayTau = 2*second,                          
            monitor=True, debug=False):

    #SOMinput = []
        
    plasticSynapsePar["taupre" ] = plasticSynapse_taupre 
    plasticSynapsePar["taupost"] = plasticSynapse_taupost
    plasticSynapsePar["weight"] = plasticSynapse_weight
    
    neuronPar['a']    = 0*nS #4*nS #disable adaptation
    neuronPar['b']    = 0*nA #0.0805*nA
    
    start = time.time()
    
    #===============================================================================
    ## create wtaSOM
    wtaSOMGroups,wtaSOMMonitors,replaceVarsWTA  = gen2dWTA('wtaSOM',
                 neuronParameters = neuronPar, synParameters = synapsePar,
                 weInpWTA = 0, weWTAInh = synWTAInh1e_weight, wiInhWTA = synInhWTA1i_weight, weWTAWTA = synWTAWTA1e_weight,
                 rpWTA = rpWTA, rpInh = rpInh,sigm = sigmSOMlateralExc, nNeurons = nWTA2dNeurons,
                 nInhNeurons = nInhNeuronsSOM, cutoff = cutoff, monitor = monitor, numWtaInputs = 1,
                 additionalStatevars = ["latSigmaTau : second (constant)"],debug=debug)
    
    #decay of kernel sigma ???
    synwtaSOMwtaSOM1e = wtaSOMGroups['synWTAWTA1e']
    synwtaSOMwtaSOM1e.latSigmaTau = sigmSOMlateralExc_DecayTau
    synwtaSOMwtaSOM1e.run_regularly('''weight = latWeight * fkernel2d(i,j,latSigma*exp(-t/latSigmaTau),nWTA2dNeurons)''',dt=100*ms)
    
    #===============================================================================
    ## create Input groups u
    
    nInpNeurons = nWTA1dNeurons
    
    neuronEqDict= neuronEq(numInputs = 2, debug = debug,additionalStatevars = ["inputWeight: 1 (constant)"])
    gInpGroup = NeuronGroup(ninputs*nInpNeurons, name = 'gInpGroup', **neuronEqDict)
    gInpGroup.inputWeight = inputWeight
    gInpSubgroups = [gInpGroup[(inp*nInpNeurons):((inp+1)*nInpNeurons)] for inp in range(ninputs)]
    setParams(gInpGroup, neuronPar, debug = debug)
    #spikemonInp = SpikeMonitor(gInpGroup)
    #statemonInp = StateMonitor(gInpGroup, ('Vm','Iin','Iconst'), record=range(ninputs*nInpNeurons))
    
    #===============================================================================
    ## create Input Inhibition Inh_u
    
    gInpInhGroup = NeuronGroup(nInhNeuronsInp, name = 'gInpInhGroup', **neuronEqDict)
    setParams(gInpInhGroup, neuronPar, debug = debug)
    #spikemonInpInh = SpikeMonitor(gInpInhGroup)
    #statemonInpInh = StateMonitor(gInpInhGroup, ('Vm','Iin','Iconst'), record=range(nInhNeuronsInp))
    
    #===============================================================================
    ## create input inhibition synapses
    
    synDict1 = synapseEq(inputNumber = 1, debug = debug)
    synDict2 = synapseEq(inputNumber = 2, debug = debug)
    synInpInh1e = Synapses(gInpGroup, gInpInhGroup, method = "euler", name = 'sInpInh1e', **synDict1)
    synInpInh1i = Synapses(gInpGroup, gInpInhGroup, method = "euler", name = 'sInpInh1i', **synDict2)      
    synInhInp1i = Synapses(gInpInhGroup, gInpGroup, method = "euler", name = 'sInhInp1i', **synDict1)
    
    synInpInh1e.connect(True)
    synInpInh1i.connect(True)
    synInhInp1i.connect(True)
    
    synapseParInpInh1i=synapsePar
    synapseParInpInh1i["taugIi"] = synInpInh1i_taugIi
    synapseParInpInh1e=synapsePar
    synapseParInpInh1e["taugIe"] = synInpInh1e_taugIe
    
    setParams(synInpInh1e, synapseParInpInh1e, debug = debug)
    setParams(synInpInh1i, synapseParInpInh1i, debug = debug)
    setParams(synInhInp1i, synapsePar, debug = debug)
    
    synInpInh1e.weight = synInpInh1e_weight
    synInpInh1i.weight = synInpInh1i_weight          
    synInhInp1i.weight = synInhInp1i_weight
    
    #===============================================================================
    ## create plastic synapses
    
    plasticSynDict = StdpSynapseEq(inputNumber=4,debug=debug,additionalStatevars = ["LRdecay: 1 (constant)"])
    synInpSom1e = Synapses(gInpGroup,wtaSOMGroups['gWTAGroup'], method = "euler", name = 'sInpSom1e', **plasticSynDict)    
    
    synInpSom1e.connect(True)
    setParams(synInpSom1e, plasticSynapsePar, debug = debug)
    #statemonSynInpSom1e = StateMonitor(synInpSom1e,('w'), record=range(ninputs*nInpNeurons*nWTA2dNeurons*nWTA2dNeurons))
    
    synInpSom1e.w = 'w_max*rand()'
    
    synInpSom1e.LRdecay = plasticSynapse_LRdecay
    synInpSom1e.run_regularly('''diffApre = diffApre*LRdecay''',dt=100*ms)
    # :TODO decrease of learning rate 
                   
    stimTimes = range(0,len(SOMinput)*interval,interval)
    meangaussind=[TimedArray([int(sigmGaussInput+(nInpNeurons-2*sigmGaussInput)*
                                  SOMinput[int(np.floor(stimTime/interval))][inp]) for stimTime in stimTimes], dt=interval*ms) for inp in range(ninputs)]
    
    for ii in range(len(meangaussind)):
        exec('meangaussind'+str(ii)+'=meangaussind[ii]') # sorry for that. Please tell me, if you find a better way
    
    
    for inp in range(ninputs):
        gInpSubgroups[inp].run_regularly('''Iconst = inputWeight * fkernelgauss1d(int(meangaussind{inp}(t)),i,sigmGaussInput)*pA'''.format(inp=inp), dt=interval*ms)
    
    #===============================================================================

    
    #if plotInp:
    #    somNet.add((spikemonInp,statemonInp))#,statemonInp
    #    somNet.add((spikemonInpInh,statemonInpInh))
    #    
    
    
    # make dictionnaries    
    SOMGroups = {
            'gInpGroup':gInpGroup,
            'gInpSubgroups':gInpSubgroups,
            
            'synwtaSOMwtaSOM1e':synwtaSOMwtaSOM1e,
            
            'synInpSom1e':synInpSom1e,
            'gInpInhGroup':gInpInhGroup,
            'synInpInh1e':synInpInh1e,
            'synInpInh1i':synInpInh1i,
            'synInhInp1i':synInhInp1i}
    
    SOMGroups.update(wtaSOMGroups)

    # spikemons
    if monitor:
        SOMMonitors = wtaSOMMonitors        
    
    #replacevars should be the real names of the parameters, that can be changed by the arguments of this function:
    # in this case: weInpWTA, weWTAInh, wiInhWTA, weWTAWTA,rpWTA, rpInh,sigm     
    replaceVars = replaceVarsWTA 
    replaceVars += [
                    gInpGroup.name   + '_inputWeight',
                    gInpGroup.name   + '_refP',
                    gInpInhGroup.name+ '_refP',
                    
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
        print('The keys of the output dict are:')
        for key in SOMGroups :
            if key not in wtaSOMGroups:
                print(key)
    
    end = time.time()
    print ('setting up took ' + str(end - start) + ' sec')
    
    return SOMGroups,SOMMonitors,replaceVars
