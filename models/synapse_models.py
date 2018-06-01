# -*- coding: utf-8 -*-
# @Author: mrax, mmilde
# @Date:   2017-12-27 10:46:44
# @Last Modified by:   Moritz Milde
# @Last Modified time: 2018-06-01 16:57:48

"""This contains subclasses of SynapseEquationBuilder with predefined common parameters
"""
import os
from NCSBrian2Lib.models.builder.synapse_equation_builder import SynapseEquationBuilder
import NCSBrian2Lib.models


class ReversalSynV(SynapseEquationBuilder):
    """"""

    def __init__(self):
        """This class provides you with all equations to simulate synapses with reversal
        potential.
        """
        SynapseEquationBuilder.__init__(self, base_unit='conductance',
                                        kernel='exponential', plasticity='non_plastic')


class BraderFusiSynapses(SynapseEquationBuilder):
    """"""

    def __init__(self):
        """This class provides you with all equations to simulate a bistable Brader-Fusi synapse
        as published in Brader and Fusi 2007
        """
        SynapseEquationBuilder.__init__(self, base_unit='current',
                                        kernel='exponential', plasticity='fusi')


class DPISyn(SynapseEquationBuilder):
    """"""

    def __init__(self):
        """This class provides you with all equations to simulate a Differential Pair
        Integrator (DPI) synapse as published in Chicca et al. 2014
        """
        SynapseEquationBuilder.__init__(self, base_unit='DPI',
                                        plasticity='non_plastic')


class DPIShunt(SynapseEquationBuilder):
    """"""

    def __init__(self):
        """This class provides you with all equations to simulate a Differential Pair
        Integrator (DPI) synapse as published in Chicca et al. 2014
        """
        SynapseEquationBuilder.__init__(self, base_unit='DPIShunting',
                                        plasticity='non_plastic')


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
        SynapseEquationBuilder.__init__(self, base_unit='DPI',
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
        SynapseEquationBuilder.__init__(self, base_unit='conductance',
                                        kernel='exponential', plasticity='stdp')


if __name__ == '__main__':

    path = os.path.dirname(os.path.realpath(NCSBrian2Lib.models.__file__))

    path = os.path.join(path, "equations")
    if not os.path.isdir(path):
        os.mkdir(path)

    ReversalSynV = ReversalSynV()
    ReversalSynV.export_eq(os.path.join(path, "ReversalSynV"))

    BraderFusiSynapses = BraderFusiSynapses()
    BraderFusiSynapses.export_eq(os.path.join(path, "BraderFusiSynapses"))

    DPISyn = DPISyn()
    DPISyn.export_eq(os.path.join(path, "DPISyn"))

    DPIShunt = DPIShunt()
    DPIShunt.export_eq(os.path.join(path, "DPIShunt"))

    DPIstdp = DPIstdp()
    DPIstdp.export_eq(os.path.join(path, "DPIstdp"))

    StdpSynV = StdpSynV()
    StdpSynV.export_eq(os.path.join(path, "StdpSynV"))
