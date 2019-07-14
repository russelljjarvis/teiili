#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 17:18:17 2018

@author: dzenn
"""

import time
from brian2 import ms, prefs, defaultclock

import pyqtgraph as pg
from pyqtgraph import QtGui
from teili.building_blocks.threeway import Threeway
from teili.tools.three_way_kernels import A_plus_B_equals_C
from teili import TeiliNetwork
from teili.tools.visualizer.DataControllers import Rasterplot

prefs.codegen.target = "numpy"
defaultclock.dt = 0.1 * ms
#set_device('cpp_standalone')

#==========Threeway building block test=========================================

duration = 100 * ms

#===============================================================================
# create the network

exampleNet = TeiliNetwork()

TW = Threeway('TestTW', hidden_layer_gen_func = A_plus_B_equals_C, cutoff = 2, monitor=True)

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

QtApp = QtGui.QApplication.instance()

mainfig = pg.GraphicsWindow()
subfig1 = mainfig.addPlot(row=0, col=0)
subfig2 = mainfig.addPlot(row=1, col=0)
subfig3 = mainfig.addPlot(row=2, col=0)

plot_A = Rasterplot([TW.monitors['spikemon_A']], neuron_id_range=(0,TW.A.num_neurons),
                  mainfig=mainfig, subfig_rasterplot=subfig1, backend='pyqtgraph', QtApp=QtApp,
                  show_immediately=False)

plot_B = Rasterplot([TW.monitors['spikemon_B']], neuron_id_range=(0,TW.B.num_neurons),
                  mainfig=mainfig, subfig_rasterplot=subfig2, backend='pyqtgraph', QtApp=QtApp,
                  show_immediately=False)

plot_C = Rasterplot([TW.monitors['spikemon_C']], neuron_id_range=(0,TW.C.num_neurons),
                  mainfig=mainfig, subfig_rasterplot=subfig3, backend='pyqtgraph', QtApp=QtApp,
                  show_immediately=True)


    
    
    
            