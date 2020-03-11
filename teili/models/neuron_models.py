# -*- coding: utf-8 -*-
"""This contains subclasses of NeuronEquationBuilder with predefined common parameters
"""
# @Author: mmilde
# @Date:   2018-01-08 14:53:11

from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from pathlib import Path
import os
import sys


class Izhikevich(NeuronEquationBuilder):
    """This class provides you with all equations to simulate a voltage-based
    quadratic, adaptive integrate and fire neuron.
    """

    def __init__(self, num_inputs=1):
        """This initializes the NeuronEquationBuilder with Izhikevich neuron model.

        Args:
            num_inputs (int, optional): Description
        """
        NeuronEquationBuilder.__init__(self, base_unit='voltage', adaptation='calcium_feedback',
                                       integration_mode='quadratic', leak='non_leaky',
                                       position='spatial', noise='none')
        self.add_input_currents(num_inputs)

class ExpLIF(NeuronEquationBuilder):
    """This class provides you with all equations to simulate a voltage-based
    exponential leaky integrate and fire neuron.
    """

    def __init__(self, num_inputs=1):
        NeuronEquationBuilder.__init__(self, base_unit='voltage', adaptation='none',
                                       integration_mode='exponential', leak='leaky',
                                       position='spatial', noise='none')
        self.add_input_currents(num_inputs)


class ExpAdaptIF(NeuronEquationBuilder):
    """This class provides you with all equations to simulate a voltage-based
    exponential, adaptive integrate and fire neuron.
    """

    def __init__(self, num_inputs=1):
        """This initializes the NeuronEquationBuilder with ExpAdaptIF neuron model.

        Args:
            num_inputs (int, optional): Description
        """
        NeuronEquationBuilder.__init__(self, base_unit='voltage', adaptation='calcium_feedback',
                                       integration_mode='exponential', leak='non_leaky',
                                       position='spatial', noise='none')
        self.add_input_currents(num_inputs)


class ExpAdaptLIF(NeuronEquationBuilder):
    """This class provides you with all equations to simulate a voltage-based
    exponential, adaptive integrate and fire neuron.
    """

    def __init__(self, num_inputs=1):
        """This initializes the NeuronEquationBuilder with ExpAdaptIF neuron model.

        Args:
            num_inputs (int, optional): Description
        """
        NeuronEquationBuilder.__init__(self, base_unit='voltage', adaptation='calcium_feedback',
                                       integration_mode='exponential', leak='leaky',
                                       position='spatial', noise='none')
        self.add_input_currents(num_inputs)

class LinearLIF(NeuronEquationBuilder):
    """This class provides you with all equations to simulate a voltage-based
    exponential, adaptive integrate and fire neuron.
    """

    def __init__(self, num_inputs=1):
        """This initializes the NeuronEquationBuilder with ExpAdaptIF neuron model.

        Args:
            num_inputs (int, optional): Description
        """
        NeuronEquationBuilder.__init__(self, base_unit='voltage', adaptation='none',
                                       integration_mode='linear', leak='leaky',
                                       position='spatial', noise='none')
        self.add_input_currents(num_inputs)

class DPI(NeuronEquationBuilder):
    """This class provides you with all equations to simulate a current-based
    exponential, adaptive leaky integrate and fire neuron as implemented on
    the neuromorphic chips by the NCS group. The neuronmodel follows the DPI neuron
    which was published in 2014 (Chicca et al. 2014).
    """

    def __init__(self, num_inputs=1):
        """This initializes the NeuronEquationBuilder with DPI neuron model.

        Args:
            num_inputs (int, optional): Description
        """
        NeuronEquationBuilder.__init__(self, base_unit='current',
                                       adaptation='calcium_feedback',
                                       integration_mode='exponential', leak='leaky',
                                       position='spatial', noise='none')
        self.add_input_currents(num_inputs)


class OCTA_Neuron(NeuronEquationBuilder):
    """Custom equations for the OCTA network.

    octa_neuron : neuron_equation that comprises of all the components needed for octa.
        In some synaptic connections not all features are used.
    """

    def __init__(self, num_inputs=2):
        NeuronEquationBuilder.__init__(self, base_unit='current',
                                       feedback='calcium_feedback',
                                       integration='exponential', location='spatial',
                                       noise='none', gain_modulationm='gain_modulation',
                                       modulation='activity')
        self.add_input_currents(num_inputs)

def main(path=None):
    if path is None:
        path = str(Path.home())
        path = os.path.join(path, "teiliApps", "equations")

    if not os.path.isdir(path):
        Path(path).mkdir(parents=True)

    expLIF = ExpLIF()
    expLIF.export_eq(os.path.join(path, "ExpLIF"))

    expAdaptIF = ExpAdaptIF()
    expAdaptIF.export_eq(os.path.join(path, "ExpAdaptIF"))

    expAdaptLIF = ExpAdaptLIF()
    expAdaptLIF.export_eq(os.path.join(path, "ExpAdaptLIF"))

    linearLIF = LinearLIF()
    linearLIF.export_eq(os.path.join(path, "LinearLIF"))

    dpi = DPI()
    dpi.export_eq(os.path.join(path, "DPI"))

    izhikevich = Izhikevich()
    izhikevich.export_eq(os.path.join(path, "Izhikevich"))

    octa_neuron = OCTA_Neuron()
    octa_neuron.export_eq(os.path.join(path, "OCTA_neuron"))

if __name__ == '__main__':
    main()
