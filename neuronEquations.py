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

def ExpAdaptIF(tauIe=None,tauIi=None,C=None,gL=None,EL=None,VT=None,DeltaT=None, 
    tauw=None,a=None,b=None,Vr=None,debug=False):
    '''Brette, Gerstner 2005 Exponential adaptive IF model
    see: http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model
    @return: a dictionary of keyword arguments for NeuronGroup()
    @note: you have to set parameters for each NeuronGroup after its creation, synapses have to increment the correct variable: Ie or Ii
    Neuron group is created and prepared as follows:
    eqDict = neuronEquatioons.ExpAdaptIF()
    neurongroup = NeuronGroup(nNeurons, **eqDict , refractory = 2*ms, method='euler',name='groupname')
    tools.setParams(neurongroup , equationParams.gerstnerExpAIFdefaultregular)
    @param:
    tauIe : second (constant)         # excitatory input time constant
    tauIi : second (constant)         # inhibitory input time constant
    C : farad (constant)              # membrane capacitance
    gL : siemens (constant)           # leak conductance
    EL : volt (constant)              # leak reversal potential
    VT : volt (constant)              # threshold
    DeltaT : volt (constant)          # slope factor
    tauw : second (constant)          # adaptation time constant
    a : siemens (constant)            # adaptation decay parameter
    b : amp (constant)                # adaptation weight
    Vr : volt (constant)              # reset potential
    '''
    
    arguments = dict(locals())
    del(arguments['debug'])
    
    modelEq = """dVm/dt = (gL*(EL - Vm) + gL*DeltaT*exp((Vm - VT)/DeltaT) + Ie + Ii - w)/C : volt (unless refractory)
    dw/dt = (a*(Vm - EL) - w)/tauw : amp
    dIe/dt = (-Ie/tauIe) : amp        # instantaneous rise, exponential decay
    dIi/dt = (-Ii/tauIi) : amp        # instantaneous rise, exponential decay
    tauIe : second (constant)         # excitatory input time constant
    tauIi : second (constant)         # inhibitory input time constant
    C : farad (constant)              # membrane capacitance
    gL : siemens (constant)           # leak conductance
    EL : volt (constant)              # leak reversal potential
    VT : volt (constant)              # threshold
    DeltaT : volt (constant)          # slope factor
    tauw : second (constant)          # adaptation time constant
    a : siemens (constant)            # adaptation decay parameter
    b : amp (constant)                # adaptation weight
    Vr : volt (constant)              # reset potential"""
    #maybe check if constant "shared" is more efficient, especially for large groups where every neuron has the same params?

    thresholdEq = "Vm > (VT + 5 * DeltaT)"
    
    resetEq = "Vm = Vr; w += b"
    
    if debug:
        print('arguments of ExpAdaptIF: \n' + str(arguments))
    
    modelEq = replaceConstants(modelEq,arguments,debug)
    thresholdEq = replaceConstants(thresholdEq,arguments,debug)
    resetEq = replaceConstants(resetEq,arguments,debug)
    
    eqDict = dict(model=modelEq, threshold=thresholdEq, reset=resetEq)
    
    if debug:
        printeqDict(eqDict)
    
    return (eqDict)



def ExpAdaptIFReversal(taugIe=None,taugIi=None,C=None,gL=None,EL=None,VT=None,DeltaT=None, 
    tauw=None,a=None,b=None,Vr=None,debug=False):
    '''Brette, Gerstner 2005 Exponential adaptive IF model
    see: http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model
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
    
    modelEq = """
    dVm/dt = (gL*(EL - Vm) + gL*DeltaT*exp((Vm - VT)/DeltaT) + gIe*(EIe - Vm) + gIi*(EIi - Vm) - w)/C : volt (unless refractory)
    dw/dt = (a*(Vm - EL) - w)/tauw : amp
    dgIe/dt = (-gIe/taugIe) : siemens # instantaneous rise, exponential decay
    dgIi/dt = (-gIi/taugIi) : siemens # instantaneous rise, exponential decay
    taugIe : second (constant)        # excitatory input time constant
    taugIi : second (constant)        # inhibitory input time constant
    C : farad (constant)              # membrane capacitance
    gL : siemens (constant)           # leak conductance
    EL : volt (constant)              # leak reversal potential
    VT : volt (constant)              # threshold
    DeltaT : volt (constant)          # slope factor
    tauw : second (constant)          # adaptation time constant
    a : siemens (constant)            # adaptation decay parameter
    b : amp (constant)                # adaptation weight
    Vr : volt (constant)              # reset potential
    EIe : volt (constant)             # excitatory reversal potential
    EIi : volt (constant)             # inhibitory reversal potential
    """
    
    thresholdEq = "Vm > (VT + 5 * DeltaT)"
    
    resetEq = "Vm = Vr; w += b"
    
    if debug:
        print('arguments of ExpAdaptIF: \n' + str(arguments))
    
    modelEq = replaceConstants(modelEq,arguments,debug)
    thresholdEq = replaceConstants(thresholdEq,arguments,debug)
    resetEq = replaceConstants(resetEq,arguments,debug)
    
    eqDict = dict(model=modelEq, threshold=thresholdEq, reset=resetEq)
    
    if debug:
        printeqDict(eqDict)
    
    return (eqDict)




