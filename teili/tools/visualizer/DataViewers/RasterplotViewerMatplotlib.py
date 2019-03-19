# SPDX-License-Identifier: MIT
# Copyright (c) 2018 University of Zurich

import matplotlib.pylab as plt
import numpy as np
from itertools import chain
from matplotlib.ticker import MaxNLocator
from teili.tools.visualizer.DataViewers.HistogramViewerMatplotlib import HistogramViewerMatplotlib
from teili.tools.visualizer.DataViewers.RasterplotViewer import RasterplotViewer
from teili.tools.visualizer.DataViewers.DataViewerUtilsMatplotlib import DataViewerUtilsMatplotlib

HIST_PLOT_BORDER_LEFT=0.125  # the left side of the subplots of the figure
HIST_PLOT_BORDER_RIGHT=0.9  # the right side of the subplots of the figure
HIST_PLOT_BORDER_BOTTOM=0.1  # the bottom of the subplots of the figure
HIST_PLOT_BORDER_TOP=0.9  # the top of the subplots of the figure
HIST_PLOT_WSPACE=0.05  # the amount of width reserved for space between subplots,
                       # expressed as a fraction of the average axis width
HIST_PLOT_HSPACE=0.2  # the amount of height reserved for space between subplots,
                      # expressed as a fraction of the average axis height

class RasterPlotViewerMatplotlib(RasterplotViewer):
    """ Class to plot raster plot with matplotlib backend """

    def __init__(
            self,
            MyPlotSettings,
            mainfig=None,
            subfig_rasterplot=None,
            subfig_histogram=None,
            add_histogram=False):
        """ Setup RasterplotViewer by initializing main figure and subfigures.
            If any of them is set to None, it will be created internally.
            If add_histogram is False, subfig_histogram is set to None.
        Args:
            MyPlotSettings (PlotSettings object): instance of class
                PlotSettings holding basic plot settings (e.g. fontsize, ...)
            mainfig (matplotlib figure object): matplotlib figure which holds
                the subfig (subplots)
            subfig_rasterplot (matplotlib subplot): matplotlib subplot of
                mainfig which will hold the rasterplot
            subfig_histogram (matplotlib subplot): matplotlib subplot of
                mainfig which will hold the histogram
            add_histogram (bool): if True: add histogram of spike count per
                neuron on right side of plot
        """

        self.MyPlotSettings = MyPlotSettings
        self.set_DataViewerUtils()
        self.add_histogram = add_histogram

        # set up main-figure
        self.mainfig = self.DVUtils.set_up_mainfig(mainfig=mainfig, subfig=subfig_rasterplot)

        # subplot
        self.subfig_rasterplot = subfig_rasterplot
        if not self.subfig_rasterplot:
            if self.add_histogram:
                self.subfig_rasterplot = plt.subplot2grid(
                    (1, 3), (0, 0), rowspan=1, colspan=2, fig=self.mainfig)
            else:
                self.subfig_rasterplot = self.mainfig.add_subplot(111)

        if self.add_histogram:
            self.subfig_histogram = subfig_histogram
            if not self.subfig_histogram:
                self.subfig_histogram = plt.subplot2grid((1, 3), (0, 2),
                                                         rowspan=1,
                                                         sharey=self.subfig_rasterplot,
                                                         frameon=0,
                                                         fig=self.mainfig)
                plt.subplots_adjust(
                    left=HIST_PLOT_BORDER_LEFT,
                    right=HIST_PLOT_BORDER_RIGHT,
                    bottom=HIST_PLOT_BORDER_BOTTOM,
                    top=HIST_PLOT_BORDER_TOP,
                    wspace=HIST_PLOT_WSPACE,
                    hspace=HIST_PLOT_HSPACE)
        else:
            self.subfig_histogram = None


    def set_DataViewerUtils(self):
        """ Set which DataViewerUtils class should be considered"""
        self.DVUtils = DataViewerUtilsMatplotlib(viewer=self)

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
            :except (e.g. [[3,4,5,2],[9,7,7 8]]
            subgroup_labels (list of str): list of labels for the different
                subgroups (e.g. ['exc', 'inh'])
            time_range_axis (tuple): (t_start(float), t_end(float)) of time
                interval within which events should be show
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
        # set parameters on plot dimensions along time and neuron_id axes
        # +[1e-9] to to deal with cases where no spikes or only nan spiek times
        # were detected
        if time_range_axis is None:
            time_range_axis = (0, np.nanmax(list(map(lambda x: x,
                                                     chain.from_iterable(
                                                         all_spike_times + [
                                                             [1e-9]])))))
        if neuron_id_range_axis is None:
            # +[[0]] to deal with cases where no spikes were detected
            neuron_id_range_axis = (0, max(list(map(lambda x: x,
                                                    chain.from_iterable(
                                                        all_neuron_ids + [
                                                            [0]])))) + 1)

        label = None
        for subgroup_nr, (spike_times, neuron_ids, color) in enumerate(
                zip(all_spike_times, all_neuron_ids, self.MyPlotSettings.colors)):
            if subgroup_labels is not None:
                label = subgroup_labels[subgroup_nr]
            self.subfig_rasterplot.scatter(
                spike_times,
                neuron_ids,
                c=color,
                marker="|",
                label=label,
                s=self.MyPlotSettings.marker_size)
            self.subfig_rasterplot.set_autoscale_on(False)
            self.subfig_rasterplot.set_xlim(
                time_range_axis[0], time_range_axis[1])
            self.subfig_rasterplot.set_ylim(
                neuron_id_range_axis[0], neuron_id_range_axis[1])
            self.subfig_rasterplot.yaxis.set_major_locator(
                MaxNLocator(integer=True))

        self.DVUtils._set_title_and_labels(subfig=self.subfig_rasterplot,
                                   title=title, xlabel=xlabel, ylabel=ylabel)
        if subgroup_labels is not None:
            self.subfig_rasterplot.legend(
                fontsize=self.MyPlotSettings.fontsize_legend)

        if self.add_histogram:
            self._add_histogram_to_rasterplot(
                all_neuron_ids=all_neuron_ids,
                num_neurons=neuron_id_range_axis[1])

    def _add_histogram_to_rasterplot(self, all_neuron_ids, num_neurons):
        """ Function to add histogram of spikes per neuron to raster plot
        Args:
            all_neuron_ids (list of lists): list of lists of neuron ids of
                events (e.g. [[3,4,5,2],[9,7,7 8]]
            num_neurons (int): total number of neurons to show on y-axis
        """

        HV = HistogramViewerMatplotlib(
            self.MyPlotSettings,
            mainfig=self.mainfig,
            subfig=self.subfig_histogram)
        HV.create_plot(data=all_neuron_ids,
                            subgroup_labels=None,
                            bins=np.arange(-0.5,
                                           num_neurons + 0.5,
                                           1),
                            orientation='horizontal',
                            title=None,
                            xlabel='count [spikes]',
                            ylabel=None)

        self.subfig_histogram.get_yaxis().set_visible(False)
        self.subfig_histogram.patch.set_visible(False)
        self.subfig_histogram.xaxis.set_major_locator(
            MaxNLocator(integer=True))