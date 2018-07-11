#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 17:18:17 2018

@author: dzenn
"""

import time
from brian2 import ms, prefs, set_device, defaultclock

from teili.building_blocks.threeway import Threeway
from teili.tools.three_way_kernels import A_plus_B_equals_C
from teili import teiliNetwork

prefs.codegen.target = "numpy"
defaultclock.dt = 0.1 * ms
#set_device('cpp_standalone')

#==========Threeway building block test=========================================

duration = 10000 * ms

#===============================================================================
# create the network

exampleNet = teiliNetwork()

TW = Threeway('TestTW', hidden_layer_gen_func = A_plus_B_equals_C, cutoff = 2, monitor=True)

exampleNet.add(TW)

#===============================================================================
# Select test mode: Standard or Live 
#
# Standard mode: > simulation is performed in the main thread
#                > input is provided from the start
#                > output is computed once the simulation has finished
# 
# Live mode:     > simulation is performed in a background thread
#                > network starts with no input, input is set on-line through gui
#                > output is shown live in a plotGUI (TW.plot() for raster
#                plotting is still available though)
#
#                Important: in live mode, make sure the duration of the simulation
#                is large enough (at least 10000 ms)

#test_mode = 'standard'
test_mode = 'live'

#===============================================================================
# reset the spike generators
TW.reset_inputs()

#===============================================================================
# simulation

if test_mode == 'standard':
    
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
    plot_raster = TW.plot()
    
elif test_mode == 'live':
    
    # run network as a thread
    exampleNet.run_as_thread(duration=duration)
    
    # create and show parameter and live plottin gui
    param_gui = TW.show_parameter_gui()
    plot_gui = TW.plot_live_inputs()
    
    plot_raster = TW.plot()
