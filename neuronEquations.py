# coding: utf-8
from brian2 import *

def ExpAdaptIF(taugIe=5*ms,taugIi =7*ms,C =281*pF,gL=35*nS,EL =-70.6*mV,VT=-50.4*mV,DeltaT=2*mV,
               EIe=60*mV,EIi=-88*mV,
               tauw=None,a=None,b=None,Vr=None,behaviour = 'regular'):
    """Brette, Gerstner 2005 Exponential adaptive IF model
    see: http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model
    
    returns a dictionary of keyword arguments for NeuronGroup()
        
    input arguments are:
    taugIe : ms (excitatory input time constant)
    taugIi : ms (inhibitory input time constant)
    C : pF (membrane capacitance)
    gL : nS (leak conductance)
    EL : mV (leak reversal potential)
    VT : mV (threshold)
    DeltaT : mV (slope factor)
    tauw : ms (adaptation time constant)
    a : nS (adaptation decay parameter)
    b : nA (adaptation weight)
    Vr : mV (reset potential)
    behaviour selects one of the known standard behaviours of the neuron 
    if you specify one of tauw,a,b, or Vr the specified behaviour will be overwritten (partially)
    
    if you want to specify a variable parameter, you can pass a string (in the form 'par : unit') instead of a value:
    e.g. VT="VT : volts"
    but please note that you have to set those parameters manually after NeuronGroup creation if they should be different from 0
    
    please also note that synapses have to increment the correct variable: gIe or gIi
    """
    
    arguments = dict(locals())
        
    # different electrophysiological behaviour
    # if one of the arguments tauw,a,b,Vr is set, assume that behaviour should be overwritten
    if behaviour == "regular":
        # Regular spiking (as in the paper)
        if tauw==None:
            tauw = 144*ms
        if a==None:
            a = 4*nS
        if b==None:
            b = 0.0805*nA
        if Vr==None:
            Vr = -70.6*mV       
    elif behaviour == "bursting":
        # Bursting
        if tauw==None:
            tauw = 20 *ms
        if a==None:
            a = 4*nS
        if b==None:
            b = 0.5*nA
        if Vr==None:
            Vr = VT+5*mV 
    elif behaviour == "fast":
        # Fast spiking
        if tauw==None:
            tauw = 144*ms
        if a==None:
            a = 2*C/tauw
        if b==None:
            b = 0*nA
        if Vr==None:
            Vr = -70.6*mV

    #taum = C / gL
    Vcut = VT + 5 * DeltaT
    
    eqstr = """
    dvm/dt = (gL*(EL - vm) + gL*DeltaT*exp((vm - VT)/DeltaT) + gIe*(EIe - vm) + gIi*(EIi - vm) - w)/C : volt (unless refractory)
    dw/dt = (a*(vm - EL) - w)/tauw : amp
    dgIe/dt = (-gIe/taugIe) : amp #instantaneous rise, exponential decay
    dgIi/dt = (-gIi/taugIi) : amp #instantaneous rise, exponential decay
    """
    
    del(arguments['behaviour'])
    for key in arguments:
        if isinstance(arguments[key],str):
            eqstr += arguments[key] + ''' (constant)
            '''
  
    # string replacement, if we do not do it here, Brian2 will do it when a NeuronGroup is created using global variables (we do not want that)
    modelEq = Equations(eqstr,tauIe=taugIe,tauIi=taugIi,C=C,gL=gL,EL=EL,VT=VT,DeltaT=DeltaT,tauw=tauw,a=a,b=b,Vr=Vr)
    thresholdEq='vm>{Vcut}'.format(Vcut=Vcut)
    resetEq='vm={Vr}; w+={b}'.format(Vr=Vr,b=b)
    
    eqDict = dict(model=modelEq, threshold=thresholdEq, reset=resetEq)
    
    return (eqDict)
    # Neuron group is created as follows:
    # NeuronGroup(nNeurons, **eqDict , refractory = 2*ms, method='euler',name='groupname')
