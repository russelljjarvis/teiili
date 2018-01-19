# -*- coding: utf-8 -*-
# @Author: mrax, mmilde
# @Date:   2017-12-27 10:46:44
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-18 10:43:17

"""This contains subclasses of SynapseEquationBuilder with predefined common parameters
"""
from NCSBrian2Lib.models.builder.synapse_equation_builder import SynapseEquationBuilder


class ReversalSynV(SynapseEquationBuilder):
    """"""

    def __init__(self):
        """This class provides you with all equations to simulate synapses with reversal
        potential.
        """
        SynapseEquationBuilder.__init__(self, model=None, baseUnit='conductance',
                                        kernel='exponential', plasticity='nonplastic')


class BraderFusiSynapses(SynapseEquationBuilder):
    """"""

    def __init__(self):
        """This class provides you with all equations to simulate a bistable Brader-Fusi synapse
        as published in Brader and Fusi 2007
        """
        SynapseEquationBuilder.__init__(self, model=None, baseUnit='current',
                                        kernel='exponential', plasticity='fusi')


class DPISyn(SynapseEquationBuilder):
    """"""

    def __init__(self):
        """This class provides you with all equations to simulate a Differential Pair
        Integrator (DPI) synapse as published in Chicca et al. 2014
        """
        SynapseEquationBuilder.__init__(self, model=None, baseUnit='DPI',
                                        plasticity='nonplastic')


class DPIstdp(SynapseEquationBuilder):
    """"""

    def __init__(self):
        """This class provides you with all equations to simulate a Differential Pair
        Integrator (DPI) synapse as published in Chicca et al. 2014. However, additional
        equations are provided to make this synapse subject to learning based on
        Spike-Time Depenendent Plasticity (STDP) as published in Song, Miller and Abbott (2000)
        and Song and Abbott (2001). Also see another example at:
        https://brian2.readthedocs.io/en/stable/examples/synapses.STDP.html
        """
        SynapseEquationBuilder.__init__(self, model=None, baseUnit='DPI',
                                        plasticity='stdp')


class StdpSynV(SynapseEquationBuilder):
    """"""

    def __init__(self):
        """This class provides you with all equations to simulate a exponential decaying
        voltage-based synapse with learning based on Spike-Time Depenendent Plasticity (STDP)
        as published in Song, Miller and Abbott (2000) and Song and Abbott (2001).
        Also see another example at:
        https://brian2.readthedocs.io/en/stable/examples/synapses.STDP.html
        """
        SynapseEquationBuilder.__init__(self, model=None, baseUnit='conductance',
                                        kernel='exponential', plasticity='stdp')
