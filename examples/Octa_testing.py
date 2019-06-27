#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:45:02 2019

@author: Matteo1
"""

from teili.building_blocks.building_block import BuildingBlock
from teili.building_blocks.octa import Octa
from teili.core.groups import Neurons, Connections
from teili.stimuli.testbench import WTA_Testbench, OCTA_Testbench


from teili.tools.octa_tools.octa_param import wtaParameters, octaParameters,\
 octa_neuron

test_OCTA= Octa(name='test_OCTA', 
                wtaParams = wtaParameters,
                 octaParams = octaParameters,     
                 neuron_eq_builder=octa_neuron,
                 stacked_inp = True,
                noise= True,
                 monitor=True,
                 debug=False)

