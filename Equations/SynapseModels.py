# -*- coding: utf-8 -*-
from brian2 import pF, nS, mV, ms, pA, nA
from NCSBrian2Lib.Equations.SynapseEquation import SynapseEquation


def reversalSynV(numimputs, refP=False, additionalVars=None):
    SynEq = SynapseEquation(model=None, baseUnit='conductance', kernel='exponential', plasticity='nonplastic', inputNumber=numimputs, additionalStatevars=additionalVars)
    return SynEq.keywords, SynEq.parameters

def BraderFusiSynapses(numimputs, refP=False, additionalVars=None):
    SynEq = SynapseEquation(model=None, baseUnit='current', kernel='exponential', plasticity='fusi', inputNumber=numimputs, additionalStatevars=additionalVars)
    return SynEq.keywords, SynEq.parameters

def SiliconSynapses(numimputs, refP=False, additionalVars=None):
    SynEq = SynapseEquation(model=None, baseUnit='DPI', plasticity='nonplastic', inputNumber=numimputs, additionalStatevars=additionalVars)
    return SynEq.keywords, SynEq.parameters

def StdpSynV(numimputs, refP=False, additionalVars=None):
    SynEq = SynapseEquation(model=None, baseUnit='conductance', kernel='exponential', plasticity='stdp', inputNumber=numimputs, additionalStatevars=additionalVars)
    return SynEq.keywords, SynEq.parameters