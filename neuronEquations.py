# coding: utf-8

from brian2 import *

from NCSBrian2Lib.tools import *


def printeqDict(eqDict):
    print( 'Model equation:')
    print( eqDict['model'])
    print( '-_-_-_-_-_-_-_-')
    print( 'threshold equation:')
    print( eqDict['threshold'])
    print( '-_-_-_-_-_-_-_-')
    print( 'reset equation:')
    print( eqDict['reset'])
    print( '-------------')


def ExpAdaptIF(C=None,gL=None,EL=None,VT=None,DeltaT=None, 
    tauwad=None,a=None,b=None,Vr=None,debug=False):
    '''Brette, Gerstner 2005 Exponential adaptive IF model
    see: http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model
    @return: a dictionary of keyword arguments for NeuronGroup()
    @note: you have to set parameters for each NeuronGroup after its creation, synapses have to increment the correct variable: Ie or Ii
    Neuron group is created and prepared as follows:
    eqDict = neuronEquatioons.ExpAdaptIF()
    neurongroup = NeuronGroup(nNeurons, **eqDict , refractory = 2*ms, method='euler',name='groupname')
    tools.setParams(neurongroup , equationParams.gerstnerExpAIFdefaultregular)
    @param: C : farad (constant)              # membrane capacitance
    gL : siemens (constant)           # leak conductance
    EL : volt (constant)              # leak reversal potential
    VT : volt (constant)              # threshold
    DeltaT : volt (constant)          # slope factor
    tauwad : second (constant)        # adaptation time constant
    a : siemens (constant)            # adaptation decay parameter
    b : amp (constant)                # adaptation weight
    Vr : volt (constant)              # reset potential
    '''
    
    arguments = dict(locals())
    del(arguments['debug'])
    
    modelEq = """
    dVm/dt = (gL*(EL - Vm) + gL*DeltaT*exp((Vm - VT)/DeltaT) + Ie + Ii - wad)/C : volt (unless refractory)
    dwad/dt = (a*(Vm - EL) - wad)/tauwad : amp
    Ie : amp
    Ii : amp
    C : farad (constant)              # membrane capacitance
    gL : siemens (constant)           # leak conductance
    EL : volt (constant)              # leak reversal potential
    VT : volt (constant)              # threshold
    DeltaT : volt (constant)          # slope factor
    tauwad : second (constant)        # adaptation time constant
    a : siemens (constant)            # adaptation decay parameter
    b : amp (constant)                # adaptation weight
    Vr : volt (constant)              # reset potential
    """
    
    thresholdEq = "Vm > (VT + 5 * DeltaT)"
    
    resetEq = "Vm = Vr; wad += b"
    
    if debug:
        print('arguments of ExpAdaptIF: \n' + str(arguments))
    
    eqDict = dict(model=modelEq, threshold=thresholdEq, reset=resetEq)

    if debug:
        printeqDict(eqDict)
    
    return eqDict, arguments


def Silicon(Ispkthr=None, Ispkthr_inh=None, Ireset=None, Ith=None, Itau=None, tauca=None, debug=False, Excitatory=True):
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
    dImem/dt = (Ipos * (Ipos > 1*pamp)  - Imem * (1 + Iahp / Itau)) / ( taum * (1 + Ith / (Imem + noise + Io)) ) : amp #
    Ipos =  ( (Ith / Itau) * (Iin - Iahp - Itau)  + Ifb ) : amp
    Ifb  =  ( (Ia / Itau) * (Imem + Ith) ) : amp
    Ia   =  ( Iagain * 1 / (1 + exp(-(Imem - Iath)/ Ianorm) ) ) : amp
      
    dIahp/dt=(Iposa - Iahp ) / tauahp : amp
    dIca/dt = (Iposa - Ica ) / tauca : amp

    kappa = (kn + kp) / 2 : 1
    taum = Cmem * Ut / (kappa * Itau) : second
    tauahp = Cahp * Ut / (kappa * Itaua) : second

    Iin = Iin_ex + Iin_inh + Iin_teach : amp
    Iin_ex : amp
    Iin_inh : amp
    Iin_teach : amp

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
    """
    
    if Excitatory:
        thresholdEq = "Imem > Ispkthr"
    else:
        thresholdEq = "Imem > Ispkthr_inh"
    
    resetEq = "Imem = Ireset"
    
    if debug:
        print('arguments of Silicon Neuron: \n' + str(arguments))
    
    eqDict = dict(model=modelEq, threshold=thresholdEq, reset=resetEq)
    
    if debug:
        printeqDict(eqDict)
    
    return eqDict, arguments


