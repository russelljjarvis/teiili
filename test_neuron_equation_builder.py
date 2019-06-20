#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:33:37 2019
e 
@author: matteo
"""
from teili.core.groups import Neurons, Connections
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.models import synapse_models

octa_neuron = NeuronEquationBuilder(base_unit='current',feedback = 'calcium_feedback', 
                                     integration = 'exponential', location = 'spatial',
                                    gm = 'gm', var = 'var' )

SynSTDGM = SynapseEquationBuilder(base_unit='None', 
                                      SynSTDGM = 'SynSTDGM')

DPI_var = SynapseEquationBuilder(base_unit='DPI', 
                                      SynSTDGM = 'SynSTDGM')

test_neuron1 = Neurons(N=2, equation_builder=octa_neuron(num_inputs=2),
                                             name="testNeuron1")

test_neuron2 = Neurons(N=2, equation_builder=octa_neuron(num_inputs=2),
                                             name="testNeuron2")


input_synapse = Connections(test_neuron1, test_neuron2,
                            equation_builder=SynSTDGM(),
                            name="input_synapse")

input_synapse.connect(True)
