#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:33:37 2019

@author: matteo
"""
from teili.core.groups import Neurons, Connections

from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder

from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
num_inputs = 2
my_neu_model = NeuronEquationBuilder(base_unit='current', adaptation='calcium_feedback',
                               integration_mode='exponential', leak='leaky'   )

my_syn_model = SynapseEquationBuilder(base_unit='DPI',  plasticity='non_plastic')


test_neuron1 = Neurons(N=2, equation_builder=my_neu_model(num_inputs=2),
                                             name="testNeuron")

test_neuron2 = Neurons(N=2, equation_builder=my_neu_model(num_inputs=2),
                                             name="testNeuron1")



input_synapse = Connections(test_neuron1, test_neuron2,
                            equation_builder=my_syn_model(),
                            name="input_synapse")

test_synapse = Connections(test_neuron1, test_neuron2, equation_builder=my_syn_model)

input_synapse.connect(True)
