# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-01-08 14:53:11
# @Last Modified by:   mmilde
# @Last Modified time: 2018-05-30 10:26:09

""" This contains subclasses of NeuronEquationBuilder with predefined common parameters"""

from NCSBrian2Lib.models.builder.neuron_equation_builder import NeuronEquationBuilder
import NCSBrian2Lib.models
import os


class ExpAdaptIF(NeuronEquationBuilder):
    """ This class provides you with all equations to simulate a voltage-based
    exponential, adaptive integrate and fire neuron.
    """

    def __init__(self, num_inputs=1):
        NeuronEquationBuilder.__init__(self, baseUnit='voltage', adaptation='calciumFeedback',
                                       integrationMode='exponential', leak='non-leaky',
                                       position='spatial', noise='none')
        self.addInputCurrents(num_inputs)


class DPI(NeuronEquationBuilder):
    """ This class provides you with all equations to simulate a current-based
    exponential, adaptive leaky integrate and fire neuron as implemented on
    the neuromorphic chips by the NCS group. The neuronmodel follows the DPI neuron
    which was published in 2014 (Chicca et al. 2014).
    """

    def __init__(self, num_inputs=1):
        NeuronEquationBuilder.__init__(self, baseUnit='current', adaptation='calciumFeedback',
                                       integrationMode='exponential', leak='leaky',
                                       position='spatial', noise='none')
        self.addInputCurrents(num_inputs)


if __name__ == '__main__':

    path = os.path.dirname(os.path.realpath(NCSBrian2Lib.models.__file__))

    path = os.path.join(path, "equations")
    if not os.path.isdir(path):
        os.mkdir(path)

    expAdaptIF = ExpAdaptIF()
    expAdaptIF.exporteq(os.path.join(path, "ExpAdaptIF"))

    dpi = DPI()
    dpi.exporteq(os.path.join(path, "DPI"))
