# -*- coding: utf-8 -*-
from NCSBrian2Lib.models.builder.neuron_equation_builder import NeuronEquationBuilder


def ExpAdaptIf(numimputs=3, refractory="refP"):
    NeEq = NeuronEquationBuilder(baseUnit='voltage', adaptation='calciumFeedback',
                                 integrationMode='exponential', leak='non-leaky',
                                 position='none', noise='none', refractory=refractory,
                                 numInputs=numimputs)
    return NeEq.keywords, NeEq.parameters


def DPI(numimputs=3, refractory="refP"):
    NeEq = NeuronEquationBuilder(baseUnit='current', adaptation='calciumFeedback',
                                 integrationMode='exponential', leak='leaky', position='none',
                                 noise='none', refractory=refractory, numInputs=numimputs)
    return NeEq.keywords, NeEq.parameters
