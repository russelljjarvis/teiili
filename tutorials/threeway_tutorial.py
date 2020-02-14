#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 17:18:17 2018

@author: dzenn
"""

import time
from brian2 import ms, prefs, defaultclock

from teili.building_blocks.threeway import Threeway
from teili.tools.three_way_kernels import A_plus_B_equals_C
from teili import TeiliNetwork


prefs.codegen.target = "numpy"
defaultclock.dt = 0.1 * ms

#==========Threeway building block test=========================================

duration = 500 * ms

#===============================================================================
# create the network

exampleNet = TeiliNetwork()

TW = Threeway('TestTW',
              hidden_layer_gen_func = A_plus_B_equals_C,
              cutoff = 2,
              monitor=True)

exampleNet.add(TW)

#===============================================================================
# reset the spike generators
TW.reset_inputs()

#===============================================================================
# simulation    
# set the example input values
TW.set_A(0.4)
TW.set_B(0.2)

print('Starting the simulation!')
start = time.clock()
exampleNet.run(duration, report = 'text')
end = time.clock()
print('simulation took ' + str(end - start) + ' sec')
print('simulation done!')

# get the resulting population codes
a, b, c = TW.get_values()

print("A = %g, B = %g, C = %g" % (a,b,c))

#===============================================================================
#Visualization

TW_plot = TW.plot()



    
    
    
            