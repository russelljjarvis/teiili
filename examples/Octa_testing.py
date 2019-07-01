#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:45:02 2019

@author: Matteo1
"""
from brian2 import ms
from teili import TeiliNetwork
from teili.building_blocks.octa import Octa
from teili.tools.octa_tools.octa_param import wtaParameters, octaParameters,\
 octa_neuron


OCTA_net= Octa(name='OCTA_net', 
                wtaParams = wtaParameters,
                 octaParams = octaParameters,     
                 neuron_eq_builder=octa_neuron,
                 stacked_inp = True,
                noise= True,
                 monitor=True,
                 debug=False)

Net = TeiliNetwork()
Net.add(      
            OCTA_net,
            OCTA_net.sub_blocks['predictionWTA'],
            OCTA_net.sub_blocks['compressionWTA']
          )
    
Net.run(octaParameters['revolutions']*ms, report='text')