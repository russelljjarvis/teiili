# -*- coding: utf-8 -*-
from NCSBrian2Lib.models.builder.synapse_equation_builder import SynapseEquationBuilder


def reversalSynV():
    SynEq = SynapseEquationBuilder(model=None, baseUnit='conductance',
                                   kernel='exponential', plasticity='nonplastic')
    return SynEq.keywords, SynEq.parameters


def BraderFusiSynapses():
    SynEq = SynapseEquationBuilder(model=None, baseUnit='current',
                                   kernel='exponential', plasticity='fusi')
    return SynEq.keywords, SynEq.parameters


def DPISyn():
    SynEq = SynapseEquationBuilder(model=None, baseUnit='DPI',
                                   plasticity='nonplastic')
    return SynEq.keywords, SynEq.parameters


def DPI_stdp():
    SynEq = SynapseEquationBuilder(model=None, baseUnit='DPI',
                                   plasticity='stdp')
    return SynEq.keywords, SynEq.parameters


def StdpSynV():
    SynEq = SynapseEquationBuilder(model=None, baseUnit='conductance',
                                   kernel='exponential', plasticity='stdp')
    return SynEq.keywords, SynEq.parameters
