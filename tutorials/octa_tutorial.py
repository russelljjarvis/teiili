#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Thu Jun 27 14:45:02 2019

# @author: Matteo
""" Minimal working example for an OCTA (Online Clustering of Temporal Activity) network.
Detailed documentation can be found on our documentation or directly
in the docstrings of the `building_block` located in `teili/building_blocks/octa.py`.

The network's parameters can be found in `teili/models/parameters/octa_params.py`.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

from brian2 import us, ms, prefs, defaultclock, core, float64
import copy as cp

from teili import TeiliNetwork
from teili.building_blocks.octa import Octa
from teili.models.parameters.octa_params import wta_params, octa_params,\
    mismatch_neuron_param, mismatch_synap_param
from teili.models.neuron_models import OCTA_Neuron as octa_neuron
from teili.stimuli.testbench import OCTA_Testbench
from teili.tools.sorting import SortMatrix
from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataControllers.Rasterplot import Rasterplot
from teili.tools.io import monitor_init


def plot_sorted_compression(OCTA_net):
    """ Plot the spiking activity, i.e. spike rasterplot of the compression
    layer, sorted by similarity. Similarity is calculated based on euclidean
    distance. The sorting yields permuted indices which are used to re-order the
    neurons.
    The out-of-the-box network with the default testbench stimulus will show an
    alignment with time, such that the sorted spike rasterplot can be fitted by
    a line with increasing or decreasing neuron index. As the starting neuron is
    picked at random the alignment with time can vary between runs.

    Arguments:
        OCTA (TeiliNetwork): The `TeiliNetwork` which contains the OCTA `BuildingBlock`. 
    """

    weights = cp.deepcopy(np.asarray(OCTA_net.sub_blocks['compression'].groups['s_exc_exc'].w_plast))
    indices = cp.deepcopy(np.asarray(OCTA_net.sub_blocks['compression'].monitors['spikemon_exc'].i))
    time = cp.deepcopy(np.asarray(OCTA_net.sub_blocks['compression'].monitors['spikemon_exc'].t))

    s = SortMatrix(nrows=49, ncols=49, matrix=weights, axis=1)
    # We use the permuted indices to sort the neuron ids
    sorted_ind = np.asarray([np.where(np.asarray(s.permutation) == int(i))[0][0] for i in indices])

    plt.figure(1)
    plt.plot(time, sorted_ind, '.r')
    plt.xlabel('Time')
    plt.ylabel('Sorted spikes')
    plt.xlim(500,700)
    plt.title('Rasterplot compression block')
    plt.show()

def update():
    region.setZValue(10)
    minX, maxX = region.getRegion()
    p2.setXRange(minX, maxX, padding=0)
    p3.setXRange(minX, maxX, padding=0)


def updateRegion(window, viewRange):
    rgn = viewRange[0]
    region.setRegion(rgn)

if __name__ == '__main__':
    prefs.codegen.target = "numpy"
    defaultclock.dt = 500 * us
    core.default_float_dtype = float64
    visualization_backend = 'pyqtgraph'  # Or set it to 'matplotlib' to use matplotlib.pyplot to plot

    Net = TeiliNetwork()
    OCTA_net = Octa(name='OCTA_net')

    testbench_stim = OCTA_Testbench()
    testbench_stim.rotating_bar(length=10, nrows=10,
                                direction='cw',
                                ts_offset=3, angle_step=10,
                                noise_probability=0.2,
                                repetitions=300,
                                debug=False)

    OCTA_net.groups['spike_gen'].set_spikes(indices=testbench_stim.indices,
                                            times=testbench_stim.times * ms)

    Net.add(OCTA_net,
            OCTA_net.monitors['spikemon_proj'],
            OCTA_net.sub_blocks['compression'],
            OCTA_net.sub_blocks['prediction'])

    Net.run(np.max(testbench_stim.times) * ms,
            report='text')

    if visualization_backend == 'matplotlib':
        plot_sorted_compression(OCTA_net)
    else:
        app = QtGui.QApplication.instance()
        if app is None:
            app = QtGui.QApplication(sys.argv)
        else:
            print('QApplication instance already exists: %s' % str(app))

        pg.setConfigOptions(antialias=True)
        labelStyle = {'color': '#FFF', 'font-size': 18}
        MyPlotSettings = PlotSettings(fontsize_title=18,
                                      fontsize_legend=12,
                                      fontsize_axis_labels=14,
                                      marker_size=10)
        sort_rasterplot = True
        win = pg.GraphicsWindow(title="Network activity")
        win.resize(1024, 768)
        p1 = win.addPlot(title="Spike raster plot: L4")
        p2 = win.addPlot(title="Zoomed in spike raster plot: L2/3")
        win.nextRow()
        p3 = win.addPlot(title="Zoomed in spike raster plot: L5/6",
                         colspan=2)

        p1.showGrid(x=True, y=True)
        p2.showGrid(x=True, y=True)
        p3.showGrid(x=True, y=True)

        region = pg.LinearRegionItem()
        region.setZValue(10)

        p1.addItem(region, ignoreBounds=True)
        

        monitor_p1 = OCTA_net.monitors['spikemon_proj']
        monitor_p2 = monitor_init()
        monitor_p2.i = cp.deepcopy(np.asarray(
            OCTA_net.sub_blocks['compression'].monitors['spikemon_exc'].i))
        monitor_p2.t = cp.deepcopy(np.asarray(
            OCTA_net.sub_blocks['compression'].monitors['spikemon_exc'].t))
        monitor_p3 = monitor_init()
        monitor_p3.i = cp.deepcopy(np.asarray(
            OCTA_net.sub_blocks['prediction'].monitors['spikemon_exc'].i))
        monitor_p3.t = cp.deepcopy(np.asarray(
            OCTA_net.sub_blocks['prediction'].monitors['spikemon_exc'].t))
        
        if sort_rasterplot:
            weights_23 = cp.deepcopy(np.asarray(
                OCTA_net.sub_blocks['compression'].groups['s_exc_exc'].w_plast))
            s_23 = SortMatrix(nrows=OCTA_net.sub_blocks['compression'].groups['s_exc_exc'].source.N,
                              ncols=OCTA_net.sub_blocks['compression'].groups['s_exc_exc'].target.N,
                              matrix=weights_23,
                              axis=1)

            weights_23_56 = cp.deepcopy(np.asarray(
                OCTA_net.sub_blocks['prediction'].groups['s_inp_exc'].w_plast))
            s_23_56 = SortMatrix(nrows=OCTA_net.sub_blocks['prediction'].groups['s_inp_exc'].source.N,
                                 ncols=OCTA_net.sub_blocks['prediction'].groups['s_inp_exc'].target.N,
                                 matrix=weights_23_56,
                                 axis=1)

            monitor_p2.i = np.asarray([np.where(
                np.asarray(s_23.permutation) == int(i))[0][0] for i in monitor_p2.i])
            monitor_p3.i = np.asarray([np.where(
                np.asarray(s_23_56.permutation) == int(i))[0][0] for i in monitor_p3.i])


        duration = np.max(testbench_stim.times)
        Rasterplot(MyEventsModels=[monitor_p1],
                   MyPlotSettings=MyPlotSettings,
                   time_range=[0, duration],
                   neuron_id_range=None,
                   title="Input rotating bar",
                   xlabel='Time (s)',
                   ylabel="Neuron ID",
                   backend='pyqtgraph',
                   mainfig=win,
                   subfig_rasterplot=p1,
                   QtApp=app,
                   show_immediately=False)

        Rasterplot(MyEventsModels=[monitor_p2],
                   MyPlotSettings=MyPlotSettings,
                   time_range=[0, duration],
                   neuron_id_range=None,
                   title="Spike raster plot of L2/3",
                   xlabel='Time (s)',
                   ylabel="Neuron ID",
                   backend='pyqtgraph',
                   mainfig=win,
                   subfig_rasterplot=p2,
                   QtApp=app,
                   show_immediately=False)

        Rasterplot(MyEventsModels=[monitor_p3],
                   MyPlotSettings=MyPlotSettings,
                   time_range=[0, duration],
                   neuron_id_range=None,
                   title="Spike raster plot of L5/6",
                   xlabel='Time (s)',
                   ylabel="Neuron ID",
                   backend='pyqtgraph',
                   mainfig=win,
                   subfig_rasterplot=p3,
                   QtApp=app,
                   show_immediately=False)

        region.sigRegionChanged.connect(update)
        p2.sigRangeChanged.connect(updateRegion)
        p3.sigRangeChanged.connect(updateRegion)
        region.setRegion([29.6, 30])
        p1.setXRange(25, 30, padding=0)

        app.exec_()
