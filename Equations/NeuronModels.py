# -*- coding: utf-8 -*-
from brian2 import pF, nS, mV, ms, pA, nA
from NCSBrian2Lib.Equations.NeuronEquation import NeuronEquation


def ExpAdaptIf(numimputs,refP=False, additionalVars=None):
    NeEq = NeuronEquation(baseUnit='voltage', adaptation='calciumFeedback', integrationMode='exponential', leak='non-leaky', position='none', noise='none',refractory=refP, numInputs=numimputs, additionalStatevars=additionalVars)
    return NeEq.keywords, NeEq.parameters

def Silicon(numimputs,refP=False, additionalVars=None):
    NeEq = NeuronEquation(baseUnit='current', adaptation='calciumFeedback', integrationMode='exponential', leak='leaky', position='none', noise='gaussianNoise',refractory=refP, numInputs=numimputs, additionalStatevars=additionalVars)
    return NeEq.keywords, NeEq.parameters

