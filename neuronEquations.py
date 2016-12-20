# coding: utf-8
from brian2 import *

def ExpAdaptIF():
    """Brette, Gerstner 2005 Exponential adaptive IF model
    see: http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model
    
    returns a dictionary of keyword arguments for NeuronGroup()

    please note that you have to set parameters for each NeuronGroup after its creation
        
    please also note that synapses have to increment the correct variable: Ie or Ii
    """
    
    modelEq = """
    dVm/dt = (gL*(EL - Vm) + gL*DeltaT*exp((Vm - VT)/DeltaT) + Ie + Ii - w)/C : volt (unless refractory)
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
    Vr : volt (constant)              # reset potential
    """

    thresholdEq='Vm>(VT + 5 * DeltaT)'
    resetEq='Vm=Vr; w+=b'
    
    eqDict = dict(model=modelEq, threshold=thresholdEq, reset=resetEq)
    
    return (eqDict)
    # Neuron group is created as follows:
    # eqDict = neuronEquatioons.ExpAdaptIF()
    # neurongroup = NeuronGroup(nNeurons, **eqDict , refractory = 2*ms, method='euler',name='groupname')
    # tools.setParams(neurongroup , equationParams.gerstnerExpAIFdefaultregular)

