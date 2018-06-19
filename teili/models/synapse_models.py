# -*- coding: utf-8 -*-
"""This contains subclasses of SynapseEquationBuilder with predefined common parameters
"""
# @Author: mrax, mmilde
# @Date:   2017-12-27 10:46:44

import os
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
import teili.models


class DoubleExponential(SynapseEquationBuilder):
    """This class provides you with all equations to simulate synapses with double
    exponential dynamics.
    """

    def __init__(self):
        """This class provides you with all equations to simulate synapses with double
        exponential dynamics.
        """
        SynapseEquationBuilder.__init__(self, base_unit='current',
                                        kernel='alpha', plasticity='non_plastic')


class ReversalSynV(SynapseEquationBuilder):
    """This class provides you with all the equations to simulate synapses with reversal
    potential.
    """

    def __init__(self):
        """This class provides you with all the equations to simulate synapses with reversal
        potential.
        """
        SynapseEquationBuilder.__init__(self, base_unit='conductance',
                                        kernel='exponential', plasticity='non_plastic')


class BraderFusiSynapses(SynapseEquationBuilder):
    """This class provides you with all the equations to simulate a bistable Brader-Fusi synapse
    as published in Brader and Fusi 2007.
    """

    def __init__(self):
        """This class provides you with all the equations to simulate a bistable Brader-Fusi synapse
        as published in Brader and Fusi 2007.
        """
        SynapseEquationBuilder.__init__(self, base_unit='current',
                                        kernel='exponential', plasticity='fusi')


class DPISyn(SynapseEquationBuilder):
    """This class provides you with all the equations to simulate a Differential Pair
    Integrator (DPI) synapse as published in Chicca et al. 2014.
    """

    def __init__(self):
        """This class provides you with all the equations to simulate a Differential Pair
        Integrator (DPI) synapse as published in Chicca et al. 2014.
        """
        SynapseEquationBuilder.__init__(self, base_unit='DPI',
                                        plasticity='non_plastic')


class DPIShunt(SynapseEquationBuilder):
    """This class provides you with all the equations to simulate a Differential Pair
    Integrator (DPI) synapse as published in Chicca et al. 2014.
    """

    def __init__(self):
        """This class provides you with all the equations to simulate a Differential Pair
        Integrator (DPI) synapse as published in Chicca et al. 2014
        """
        SynapseEquationBuilder.__init__(self, base_unit='DPIShunting',
                                        plasticity='non_plastic')


class DPIstdp(SynapseEquationBuilder):
    """This class provides the well-known DPI synapse with Spike-Time Dependent Plasticity
    mechanism.
    """

    def __init__(self):
        """This class provides you with all the equations to simulate a Differential Pair
        Integrator (DPI) synapse as published in Chicca et al. 2014. However, additional
        equations are provided to make this synapse subject to learning based on
        Spike-Time Dependent Plasticity (STDP) as published in Song, Miller and Abbott (2000)
        and Song and Abbott (2001). Also see another example at:
        https://brian2.readthedocs.io/en/stable/examples/synapses.STDP.html
        """
        SynapseEquationBuilder.__init__(self, base_unit='DPI',
                                        plasticity='stdp')


class StdpSynV(SynapseEquationBuilder):
    """This class provides you with all the equations to simulate an exponential decaying
    voltage-based synapse with learning based on Spike-Time Dependent Plasticity (STDP).
    """

    def __init__(self):
        """This class provides you with all the equations to simulate an exponential decaying
        voltage-based synapse with learning based on Spike-Time Dependent Plasticity (STDP)
        as published in Song, Miller and Abbott (2000) and Song and Abbott (2001).

        Also see another example at:
            https://brian2.readthedocs.io/en/stable/examples/synapses.STDP.html
        """
        SynapseEquationBuilder.__init__(self, base_unit='conductance',
                                        kernel='exponential', plasticity='stdp')


def main():
    path = os.path.dirname(os.path.realpath(teili.models.__file__))

    path = os.path.join(path, "equations")
    if not os.path.isdir(path):
        os.mkdir(path)

    doubleExponential = DoubleExponential()
    doubleExponential.export_eq(os.path.join(path, "DoubleExponential"))

    reversalSynV = ReversalSynV()
    reversalSynV.export_eq(os.path.join(path, "ReversalSynV"))

    braderFusiSynapses = BraderFusiSynapses()
    braderFusiSynapses.export_eq(os.path.join(path, "BraderFusiSynapses"))

    dpiSyn = DPISyn()
    dpiSyn.export_eq(os.path.join(path, "DPISyn"))

    dpiShunt = DPIShunt()
    dpiShunt.export_eq(os.path.join(path, "DPIShunt"))

    dpistdp = DPIstdp()
    dpistdp.export_eq(os.path.join(path, "DPIstdp"))

    stdpSynV = StdpSynV()
    stdpSynV.export_eq(os.path.join(path, "StdpSynV"))

    reversalSynVfusi = SynapseEquationBuilder(base_unit='conductance',
                                        kernel='exponential', plasticity='fusi')
    reversalSynVfusi.export_eq(os.path.join(path, "ReversalSynVfusi"))


if __name__ == '__main__':
    main()
