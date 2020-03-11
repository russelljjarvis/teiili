# -*- coding: utf-8 -*-
"""This contains subclasses of SynapseEquationBuilder with predefined common parameters
"""
# @Author: mrax, mmilde
# @Date:   2017-12-27 10:46:44

import os
import sys
from pathlib import Path
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder


class Exponential(SynapseEquationBuilder):
    """This class provides you with all the equations to simulate an exponential decaying
    voltage-based synapse without learning.
    """

    def __init__(self):
        """This class provides you with all the equations to simulate an exponential decaying
        voltage-based synapse without.

        Also see another example at:
            https://brian2.readthedocs.io/en/stable/examples/synapses.STDP.html
        """
        SynapseEquationBuilder.__init__(self, base_unit='current',
                                        kernel='exponential', plasticity='non_plastic')

class ExponentialStdp(SynapseEquationBuilder):
    """This class provides you with all the equations to simulate an exponential decaying
    voltage-based synapse without learning.
    """

    def __init__(self):
        """This class provides you with all the equations to simulate an exponential decaying
        voltage-based synapse without.

        Also see another example at:
            https://brian2.readthedocs.io/en/stable/examples/synapses.STDP.html
        """
        SynapseEquationBuilder.__init__(self, base_unit='current',
                                        kernel='exponential', plasticity='stdp')


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

class Alpha(SynapseEquationBuilder):
    """This class provides you with all equations to simulate synapses with double
    exponential dynamics.
    """

    def __init__(self):
        """This class provides you with all equations to simulate synapses with alpha function
         dynamics.
        """
        SynapseEquationBuilder.__init__(self, base_unit='current',
                                        kernel='alpha', plasticity='non_plastic')

class AlphaStdp(SynapseEquationBuilder):
    """This class provides you with all equations to simulate synapses with double
    exponential dynamics.
    """

    def __init__(self):
        """This class provides you with all equations to simulate synapses with alpha function
         dynamics.
        """
        SynapseEquationBuilder.__init__(self, base_unit='current',
                                        kernel='alpha', plasticity='stdp')

class Resonant(SynapseEquationBuilder):
    """This class provides you with all equations to simulate synapses with resonant
    function dynamics.
    """

    def __init__(self):
        """This class provides you with all equations to simulate synapses with double
        exponential dynamics.
        """
        SynapseEquationBuilder.__init__(self, base_unit='current',
                                        kernel='resonant', plasticity='non_plastic')

class ResonantStdp(SynapseEquationBuilder):
    """This class provides you with all equations to simulate synapses with resonant
    function dynamics with STDP learning.
    """

    def __init__(self):
        """This class provides you with all equations to simulate synapses with double
        exponential dynamics.
        """
        SynapseEquationBuilder.__init__(self, base_unit='current',
                                        kernel='resonant', plasticity='stdp')



class Alpha(SynapseEquationBuilder):
    """This class provides you with all equations to simulate synapses with double
    exponential dynamics.
    """

    def __init__(self):
        """This class provides you with all equations to simulate synapses with double
        exponential dynamics.
        """
        SynapseEquationBuilder.__init__(self, base_unit='current',
                                        kernel='alpha', plasticity='non_plastic')

class Resonant(SynapseEquationBuilder):
    """This class provides you with all equations to simulate synapses with double
    exponential dynamics.
    """

    def __init__(self):
        """This class provides you with all equations to simulate synapses with double
        exponential dynamics.
        """
        SynapseEquationBuilder.__init__(self, base_unit='current',
                                        kernel='resonant', plasticity='non_plastic')



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

class DPISyn_alpha(SynapseEquationBuilder):
    """This class provides you with all the equations to simulate a Differential Pair
    Integrator (DPI) synapse as published in Chicca et al. 2014.
    """

    def __init__(self):
        """This class provides you with all the equations to simulate a Differential Pair
        Integrator (DPI) synapse as published in Chicca et al. 2014.
        """
        SynapseEquationBuilder.__init__(self, base_unit='DPI', kernel='alpha',
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
    """This class provides the well-known DPI synapse with Spike-Timing Dependent Plasticity
    mechanism.
    """

    def __init__(self):
        """This class provides you with all the equations to simulate a Differential Pair
        Integrator (DPI) synapse as published in Chicca et al. 2014. However, additional
        equations are provided to make this synapse subject to learning based on
        Spike-Timing Dependent Plasticity (STDP) as published in Song, Miller and Abbott (2000)
        and Song and Abbott (2001). Also see another example at:
        https://brian2.readthedocs.io/en/stable/examples/synapses.STDP.html
        """
        SynapseEquationBuilder.__init__(self, base_unit='DPI',
                                        plasticity='stdp')


class StdpSynV(SynapseEquationBuilder):
    """This class provides you with all the equations to simulate an exponential decaying
    voltage-based synapse with learning based on Spike-Timing Dependent Plasticity (STDP).
    """

    def __init__(self):
        """This class provides you with all the equations to simulate an exponential decaying
        voltage-based synapse with learning based on Spike-Timing Dependent Plasticity (STDP)
        as published in Song, Miller and Abbott (2000) and Song and Abbott (2001).

        Also see another example at:
            https://brian2.readthedocs.io/en/stable/examples/synapses.STDP.html
        """
        SynapseEquationBuilder.__init__(self, base_unit='conductance',
                                        kernel='exponential', plasticity='stdp')


class DPIadp(SynapseEquationBuilder):
    def __init__(self):
        SynapseEquationBuilder.__init__(self, base_unit='DPI',
                                        modulation='activity')

class DPIstdgm(SynapseEquationBuilder):
    def __init__(self):
        SynapseEquationBuilder.__init__(self, base_unit='unit_less', 
                                        SynSTDGM='stdgm')


def main(path=None):
    if path is None:
        path = str(Path.home())
        path = os.path.join(path, "teiliApps", "equations")

    if not os.path.isdir(path):
        Path(path).mkdir(parents=True)

    #Kernel synapses
    exponential = Exponential()
    exponential.export_eq(os.path.join(path, "Exponential"))

    doubleExponential = DoubleExponential()
    doubleExponential.export_eq(os.path.join(path, "DoubleExponential"))

    alpha = Alpha()
    alpha.export_eq(os.path.join(path, "Alpha"))

    alphastdp = AlphaStdp()
    alphastdp.export_eq(os.path.join(path, "AlphaStdp"))

    resonant = Resonant()
    resonant.export_eq(os.path.join(path, "Resonant"))

    reversalSynV = ReversalSynV()
    reversalSynV.export_eq(os.path.join(path, "ReversalSynV"))

    braderFusiSynapses = BraderFusiSynapses()
    braderFusiSynapses.export_eq(os.path.join(path, "BraderFusiSynapses"))

    dpiSyn = DPISyn()
    dpiSyn.export_eq(os.path.join(path, "DPISyn"))

    dpiShunt = DPIShunt()
    dpiShunt.export_eq(os.path.join(path, "DPIShunt"))

    exponentialstdp = ExponentialStdp()
    exponentialstdp.export_eq(os.path.join(path, "ExponentialStdp"))

    dpistdp = DPIstdp()
    dpistdp.export_eq(os.path.join(path, "DPIstdp"))

    stdpSynV = StdpSynV()
    stdpSynV.export_eq(os.path.join(path, "StdpSynV"))

    reversalSynVfusi = SynapseEquationBuilder(base_unit='conductance',
                                        kernel='exponential', plasticity='fusi')
    reversalSynVfusi.export_eq(os.path.join(path, "ReversalSynVfusi"))

    dpiadp = DPIadp()
    dpiadp.export_eq(os.path.join(path, "DPIadp"))

    dpistdgm = DPIstdgm()
    dpistdgm.export_eq(os.path.join(path, "DPIstdgm"))

if __name__ == '__main__':
    main()
