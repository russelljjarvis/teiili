# SPDX-License-Identifier: MIT
# Copyright (c) 2018 University of Zurich

import numpy as np
import warnings
from itertools import chain

try:
    import pyqtgraph as pg
    from PyQt5 import QtGui
except BaseException:
    warnings.warn("No method using pyqtgraph can be used as pyqtgraph or PyQt5"
                  "can't be imported.")

from teili.tools.visualizer.DataViewers import RasterplotViewer
from teili.tools.visualizer.DataViewers.HistogramViewerPyqtgraph import HistogramViewerPyqtgraph
from teili.tools.visualizer.DataViewers.DataViewerUtilsPyqtgraph import DataViewerUtilsPyqtgraph

class RasterplotViewerPyqtgraph(RasterplotViewer):
    """ Class to plot raster plot with pyqtgraph backend """

    def __init__(
            self,
            MyPlotSettings,
            mainfig=None,
            subfig_rasterplot=None,
            subfig_histogram=None,
            QtApp=None,
            add_histogram=False):
        """ Setup RasterplotViewer by initializing main figure and subfigures.
            If any of them is set to None, it will be created internally.
        If add_histogram is False, subfig_histogram is set to None.
        Args:
            MyPlotSettings (PlotSettings object): instance of class
                PlotSettings holding basic plot settings (e.g. fontsize, ...)
            mainfig (pyqtgraph window object): pyqtgraph main window
                (pg.GraphicsWindow())
            subfig_rasterplot (pyqtgraph subplot): pyqtgraph subplot of mainfig
                which will hold the rasterplot
            subfig_histogram (pyqtgraph subplot): pyqtgraph subplot of mainfig
                which will hold the histogram
            QtApp (pyqtgraph application): pyqtgraph application to run plots
                (QtGui.QApplication([])), if None: it will check for an existing
                QtApps to use or creates a new one otherwise
            add_histogram (bool): if True: add histogram of spike count per
                neuron on right side of plot
        """

        self.MyPlotSettings = MyPlotSettings
        self.set_DataViewerUtils()
        self.add_histogram = add_histogram

        # set up qt application, main-figure
        self.QtApp = self.DVUtils.set_up_QtApp(QtApp=QtApp)
        self.mainfig = self.DVUtils.set_up_mainfig(mainfig=mainfig, subfig=subfig_rasterplot)

        # subplots
        self.subfig_rasterplot = subfig_rasterplot
        if not self.subfig_rasterplot:
            self.subfig_rasterplot = self.mainfig.addPlot(row=0, column=0)

        if self.add_histogram:
            self.subfig_histogram = subfig_histogram
            if not self.subfig_histogram:
                self.mainfig.nextCol()
                self.subfig_histogram = self.mainfig.addPlot(row=0, column=5)
                self.subfig_histogram.setYLink(self.subfig_rasterplot)
        else:
            self.subfig_histogram = None

        pg.setConfigOptions(antialias=True)

    def set_DataViewerUtils(self):
        """ Set which DataViewerUtils class should be considered"""
        self.DVUtils = DataViewerUtilsPyqtgraph(viewer=self)

    def create_plot(
            self,
            all_spike_times,
            all_neuron_ids,
            subgroup_labels=None,
            time_range_axis=None,
            neuron_id_range_axis=None,
            title='raster plot',
            xlabel='time (s)',
            ylabel='neuron ids'):
        """ Function to generate raster plot (incl histogram of events per
            neuron id) from groups of event sets (spike times, neuron ids)
        Args:
            all_spike_times (list of lists): list of lists of spike times of
                events, (e.g. [[t0, t1, t2],[tt1, tt2, tt3]])
            all_neuron_ids (list of lists): list of lists of neuron ids of
                events (e.g. [[3,4,5,2],[9,7,7 8]]
            subgroup_labels (list of str): list of labels for the different
                subgroups (e.g. ['exc', 'inh'])
            time_range_axis (tuple): (t_start(float), t_end(float)) of time
                interval within which events should be shown
            neuron_id_range_axis (tuple): (min_id, max_id) of neuron ids which
                should be shown
            title (str): title of plot
            xlabel (str): label of x-axis
            ylabel (str): label for y-axis

        Remarks:
            time_range_axis and neuron_id_range_axis do not filter the spike
            times and neuron ids. It only defines the region which is shown in
            the plot. Hence, for the histogram ALL events provided are
            considered to calculate the histogram.
        """

        # check if colours ok
        self.check_num_colors(n_provided_colors=len(
            self.MyPlotSettings.colors), n_required_colors=len(all_neuron_ids))

        # set parameters on plot dimensions along time and neuron_id axis
        # +[1e-9] to to deal with cases where no spikes or only nan spiek times
        # were detected
        if time_range_axis is None:
            time_range_axis = (0,
                               np.nanmax(list(map(lambda x: x,
                                                  chain.from_iterable(
                                                      all_spike_times + [
                                                          [1e-9]])))))
        if neuron_id_range_axis is None:
            # +[[0]] to deal with cases where no spikes were detected
            neuron_id_range_axis = (0, max(map(lambda x: x,
                                               chain.from_iterable(
                                                   all_neuron_ids + [
                                                       [0]]))) + 1)

        if subgroup_labels is not None:
            self.subfig_rasterplot.addLegend()

        # TODO: change symbol in raster plot from o to |
        label = None
        for subgroup_nr, (spike_times, neuron_ids, color) in enumerate(
                zip(all_spike_times, all_neuron_ids, self.MyPlotSettings.colors)):
            if not isinstance(color, str):
                color = tuple(np.asarray(color))
            if subgroup_labels is not None:
                label = subgroup_labels[subgroup_nr]
            self.subfig_rasterplot.plot(
                x=spike_times,
                y=neuron_ids,
                name=label,
                pen=None,

                symbol='o',
                symbolPen=None,
                symbolSize=self.MyPlotSettings.marker_size,
                symbolBrush=color)

        self.DVUtils._set_title_and_labels(subfig=self.subfig_rasterplot, title=title,
                                           xlabel=xlabel, ylabel=ylabel)
        self.subfig_rasterplot.setRange(
            xRange=(
                time_range_axis[0], time_range_axis[1]), yRange=(
                neuron_id_range_axis[0], neuron_id_range_axis[1]))

        self.DVUtils.add_legend(subgroup_labels=subgroup_labels,
                                subfig=self.subfig_rasterplot)

        if self.add_histogram:
            self._add_histogram_to_rasterplot(
                all_neuron_ids=all_neuron_ids,
                num_neurons=neuron_id_range_axis[1])

    def _add_histogram_to_rasterplot(self, all_neuron_ids, num_neurons):
        """ Function to add histogram of spikes per neuron to raster plot
        Args:
            all_neuron_ids (list of lists): list of lists of neuron ids of events
                (e.g. [[3,4,5,2],[9,7,7 8]]
            num_neurons (int): total number of neurons to show on y-axis
        """

        HV = HistogramViewerPyqtgraph(
            self.MyPlotSettings,
            mainfig=self.mainfig,
            subfig=self.subfig_histogram,
            QtApp=self.QtApp)
        HV.create_plot(data=all_neuron_ids,
                            subgroup_labels=None,
                            bins=np.arange(-0.5,
                                           num_neurons + 0.5,
                                           1),
                            orientation='horizontal',
                            title=None,
                            xlabel='count [spikes]',
                            ylabel=None)

        titleStyle = {'color': '#FFF', 'size': str(
            self.MyPlotSettings.fontsize_title) + 'pt'}
        # fake, required to align with raster plot
        self.subfig_histogram.setTitle(' ', **titleStyle)

        self.subfig_histogram.setYRange(0, num_neurons)
        self.subfig_histogram.getAxis('left').setStyle(showValues=False)