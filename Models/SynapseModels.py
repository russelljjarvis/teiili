# -*- coding: utf-8 -*-
from brian2 import pF, nS, mV, ms, pA, nA
from NCSBrian2Lib.Equations.SynapseEquation import SynapseEquation


def reversalSynV(additionalVars=None):
    SynEq = SynapseEquation(model=None, baseUnit='conductance', kernel='exponential', plasticity='nonplastic', additionalStatevars=additionalVars)
    return SynEq.keywords, SynEq.parameters


def BraderFusiSynapses(additionalVars=None):
    SynEq = SynapseEquation(model=None, baseUnit='current', kernel='exponential', plasticity='fusi', additionalStatevars=additionalVars)
    return SynEq.keywords, SynEq.parameters


def DPISyn(additionalVars=None):
    SynEq = SynapseEquation(model=None, baseUnit='DPI', plasticity='nonplastic', additionalStatevars=additionalVars)
    return SynEq.keywords, SynEq.parameters


def DPI_stdp(additionalVars=None):
    SynEq = SynapseEquation(model=None, baseUnit='DPI', plasticity='stdp', additionalStatevars=additionalVars)
    return SynEq.keywords, SynEq.parameters


def StdpSynV(additionalVars=None):
    SynEq = SynapseEquation(model=None, baseUnit='conductance', kernel='exponential', plasticity='stdp', additionalStatevars=additionalVars)
    return SynEq.keywords, SynEq.parameters
