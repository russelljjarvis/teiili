# -*- coding: utf-8 -*-
""" This contains subclasses of NeuronEquationBuilder with predefined common parameters"""

from NCSBrian2Lib.models.builder.neuron_equation_builder import NeuronEquationBuilder


class ExpAdaptIF(NeuronEquationBuilder):
    """ """
    def __init__(self):
        	NeuronEquationBuilder.__init__(self, baseUnit='voltage', adaptation='calciumFeedback',
                                             integrationMode='exponential', leak='non-leaky',
                                             position='none', noise='none')


class DPI(NeuronEquationBuilder):
    """ """
    def __init__(self):
        	NeuronEquationBuilder.__init__(self, baseUnit='current', adaptation='calciumFeedback',
                                         integrationMode='exponential', leak='leaky', position='none',
                                         noise='none')

