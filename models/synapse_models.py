# -*- coding: utf-8 -*-
""" This contains subclasses of SynapseEquationBuilder with predefined common parameters"""
from NCSBrian2Lib.models.builder.synapse_equation_builder import SynapseEquationBuilder


class ReversalSynV(SynapseEquationBuilder):
    """ """
    def __init__(self):
        SynapseEquationBuilder.__init__(self,model=None, baseUnit='conductance',
                                        kernel='exponential', plasticity='nonplastic')


class BraderFusiSynapses(SynapseEquationBuilder):
    """ """
    def __init__(self):
        SynapseEquationBuilder.__init__(self,model=None, baseUnit='current',
                                        kernel='exponential', plasticity='fusi')


class DPISyn(SynapseEquationBuilder):
    """ """
    def __init__(self):
        SynapseEquationBuilder.__init__(self,model=None, baseUnit='DPI',
                                        plasticity='nonplastic')


class DPIstdp(SynapseEquationBuilder):
    """ """
    def __init__(self):
        SynapseEquationBuilder.__init__(self,model=None, baseUnit='DPI',
                                        plasticity='stdp')


class StdpSynV(SynapseEquationBuilder):
    """ """
    def __init__(self):
        SynapseEquationBuilder.__init__(self,model=None, baseUnit='conductance',
                                        kernel='exponential', plasticity='stdp')
