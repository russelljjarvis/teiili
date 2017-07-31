# coding: utf-8
from brian2 import *
from NCSBrian2Lib.Tools.tools import *


def ExpAdaptIF(numInputs = 1,debug=False,method='euler', additionalStatevars = None):
    '''
    Brette, Gerstner 2005 Exponential adaptive IF model
    see: http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model
    @return: a dictionary of keyword arguments for NeuronGroup()
    @note: you have to set parameters for each NeuronGroup after its creation, synapses have to increment the correct variable: Ie or Ii
    Neuron group is created and prepared as follows:
    eqDict = neuronEquatioons.ExpAdaptIF()
    neurongroup = NeuronGroup(nNeurons, **eqDict , refractory = 2*ms, method='euler',name='groupname')
    tools.setParams(neurongroup , equationParams.gerstnerExpAIFdefaultregular)
    @param: 
    numInputs  # number of input currents (>0) (they are just numbered: Ie, Ie2, Ie3, ...)
    debug      # print debug info?
    '''
    
    arguments = dict(locals())
    #del(arguments['debug'])
    
    modelEq = """
    dVm/dt = (gL*(EL - Vm) + gL*DeltaT*exp((Vm - VT)/DeltaT) + Iin - wad)/C : volt (unless refractory)
    dwad/dt = (a*(Vm - EL) - wad)/tauwad : amp
    Ii      : amp                         # inh input current
    Ie      : amp                         # exc input current
    Iconst  : amp                         # constant input current
    C       : farad     (constant)        # membrane capacitance
    gL      : siemens   (constant)        # leak conductance
    EL      : volt      (constant)        # leak reversal potential
    VT      : volt      (constant)        # threshold
    DeltaT  : volt      (constant)        # slope factor
    tauwad  : second    (constant)        # adaptation time constant
    a       : siemens   (constant)        # adaptation decay parameter
    b       : amp       (constant)        # adaptation weight
    Vr      : volt      (constant)        # reset potential
    refP    : second    (constant)        # refractory period (It is still possible to set it to False)
    x       : 1         (constant)        # x location on 2d grid (only set it if you need it)
    y       : 1         (constant)        # y location on 2d grid
    """
    # add additional input currents (if you have several input currents)
    Ies = ["+ Ie" + str(i) + " " for i in range(1,numInputs+1) if i > 1]
    Iis = ["+ Ii" + str(i) + " " for i in range(1,numInputs+1) if i > 1]
    modelEq = modelEq + "Iin = Iconst + Ii + Ie " + "".join(Ies) + "".join(Iis) + " : amp # input currents\n"
    Iesline = ["    Ie" + str(i) + " : amp" for i in range(1,numInputs+1) if i > 1]
    Iisline = ["    Ii" + str(i) + " : amp" for i in range(1,numInputs+1)if i > 1]
    modelEq = modelEq + "\n".join(Iesline) +"\n" + "\n".join(Iisline)  
    
    if additionalStatevars is not None:
        if debug:
            print("added to Equation: \n" + "\n".join(additionalStatevars))
        modelEq += "\n            ".join(additionalStatevars)
        
    
    thresholdEq = "Vm > (VT + 5 * DeltaT)"   
    resetEq     = "Vm = Vr; wad += b"

    eqDict = dict(model=modelEq, threshold=thresholdEq, reset=resetEq, refractory = 'refP' , method=method)
    
    if debug:
        print('arguments of ExpAdaptIF: \n' + str(arguments))
        printEqDict(eqDict)

    return eqDict

def Silicon(numInputs = 1, debug=False, Excitatory=True, method='euler', additionalStatevars = None):
    '''Silicon Neuron as in Chicca et al. 2014
    @return: a dictionary of keyword arguments for NeuronGroup()
    @note: you have to set parameters for each NeuronGroup after its creation, synapses have to increment the correct variable: Ie or Ii
    Neuron group is created and prepared as follows:
    eqDict = neuronEquatioons.ExpAdaptIF()
    neurongroup = NeuronGroup(nNeurons, **eqDict , refractory = 2*ms, method='euler',name='groupname')
    tools.setParams(neurongroup , equationParams.gerstnerExpAIFdefaultregular)
    @param: see equation below
    '''
    
    arguments = dict(locals())
    del(arguments['debug'])
    del(arguments['Excitatory'])
 
    modelEq = """
    dImem/dt = (Ipos * (Ipos > 1*pamp)  - Imem * (1 + Iahp / Itau)) / ( taum * (1 + Ith / (Imem + noise + Io)) ) : amp (unless refractory)
    Ipos =  ( (Ith / Itau) * (Iin - Iahp - Itau)  + Ifb ) : amp
    Ifb  =  ( (Ia / Itau) * (Imem + Ith) ) : amp
    Ia   =  ( Iagain * 1 / (1 + exp(-(Imem - Iath)/ Ianorm) ) ) : amp
      
    dIahp/dt=(Iposa - Iahp ) / tauahp : amp
    dIca/dt = (Iposa - Ica ) / tauca : amp

    kappa = (kn + kp) / 2 : 1
    taum = Cmem * Ut / (kappa * Itau) : second
    tauahp = Cahp * Ut / (kappa * Itaua) : second

    Ie : amp
    Ii : amp
    Iin_teach : amp
    Iconst  : amp      # constant input current

    mu = 0.25 * pA : amp
    sigma = 0.1 * pA : amp
    b = sign(2 * rand() -1) : 1 (constant over dt)
    noise = b * (sigma * randn() + mu) : amp (constant over dt)

    kn     : 1 (constant)
    kp     : 1 (constant)
    Ut     : volt (constant)
    Io     : amp (constant)
    Csyn   : farad (constant)
    Cmem   : farad (constant)
    Cahp   : farad (constant)
    Iagain : amp (constant)
    Iath   : amp (constant)
    Ianorm : amp (constant)
    tauca  : second (constant)
    Iposa  : amp (constant)
    Iwa    : amp (constant)
    Itaua  : amp (constant)
    Ispkthr : amp (constant)
    Ispkthr_inh : amp (constant)
    Ireset : amp (constant)
    Ith    : amp (constant)
    Itau   : amp (constant)
    refP    : second (constant)        # refractory period (It is still possible to set it to False)
    
    x       : 1         (constant)        # x location on 2d grid  (only set it if you need it)
    y       : 1         (constant)        # y location on 2d grid
    """
    # add additional input currents (if you have several input currents)
    Ies = ["+ Ie" + str(i) + " " for i in range(1,numInputs+1) if i > 1]
    Iis = ["+ Ii" + str(i) + " " for i in range(1,numInputs+1) if i > 1]
    modelEq = modelEq + "Iin = Iconst + Iin_teach + Ii + Ie " + "".join(Ies) + "".join(Iis) + " : amp # input currents\n"
    Iesline = ["    Ie" + str(i) + " : amp" for i in range(1,numInputs+1) if i > 1]
    Iisline = ["    Ii" + str(i) + " : amp" for i in range(1,numInputs+1)if i > 1]
    modelEq = modelEq + "\n".join(Iesline) +"\n" + "\n".join(Iisline)  
    
    if additionalStatevars is not None:
        if debug:
            print("added to Equation: \n" + "\n".join(additionalStatevars))
        modelEq += "\n            ".join(additionalStatevars)
        
    
    
    if Excitatory:
        thresholdEq = "Imem > Ispkthr"
    else:
        thresholdEq = "Imem > Ispkthr_inh"
    
    resetEq = "Imem = Ireset"
    
    if debug:
        print('arguments of Silicon Neuron: \n' + str(arguments))
    
    eqDict = dict(model=modelEq, threshold=thresholdEq, reset=resetEq)
    
    if debug:
        printEqDict(eqDict)
    
    return eqDict

def printEqDict(eqDict):
    print( 'Model equation:')
    print( eqDict['model'])
    print( '-_-_-_-_-_-_-_-')
    print( 'threshold equation:')
    print( eqDict['threshold'])
    print( '-_-_-_-_-_-_-_-')
    print( 'reset equation:')
    print( eqDict['reset'])
    print( '-_-_-_-_-_-_-_-')
    print( 'refractory variable:')
    print( eqDict['refractory'])
    print( '-_-_-_-_-_-_-_-')
    print( 'method:')
    print( eqDict['method'])
    print( '-------------')