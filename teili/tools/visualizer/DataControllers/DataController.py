# SPDX-License-Identifier: MIT
# Copyright (c) 2018 University of Zurich

import numpy as np
import warnings
try:
    import pyqtgraph as pg
    from PyQt5 import QtGui
except BaseException:
    warnings.warn("No method using pyqtgraph can be used as pyqtgraph or PyQt5"
                  " can't be imported.")


class DataController(object):
    """ Parent class of all DataControllers"""

    def _filter_events_for_interval(
            self,
            all_spike_times,
            all_neuron_ids,
            t_start,
            t_end):
        """Function to filter for events which are within a given time interval
            (t_start, t_end)

        Args:
            all_spike_times (list): list of spike times of events
            all_neuron_ids (list): list of neuron ids of events
            t_start (float): start time of time interval within which events
                should be considered
            t_end (float): end time of time interval within which events should
                be considered

        Returns:
            all_spike_times (array): array of all spike times of events within
                specified time interval
            all_neuron_ids (array): array of all neuron ids of events within
                specified time interval
        """

        all_spike_times = np.asarray(all_spike_times)
        all_neuron_ids = np.asarray(all_neuron_ids)
        shown_indices = np.where(
            np.logical_and(
                all_spike_times >= t_start,
                all_spike_times <= t_end
            )
        )[0]
        return (all_spike_times[shown_indices], all_neuron_ids[shown_indices])

    def _filter_events_for_neuron_ids(
            self,
            all_spike_times,
            all_neuron_ids,
            active_neuron_ids):
        """Function to filter for events of a subset of neuron ids
        Args:
            all_spike_times (list): list of spike times of events
            all_neuron_ids (list): list of neuron ids of events
            active_neuron_ids (list): list of neuron ids to filter for (neurons
                with these ids are returned)

        Returns:
            all_spike_times (array): array of all spike times of events of
                specified neuron ids
            all_neuron_ids (array): array of all neuron ids of events within
                specified neuron ids
        """

        shown_indices = [*map(lambda x: x in active_neuron_ids, all_neuron_ids)]
        return (
            np.asarray(all_spike_times)[shown_indices],
            np.asarray(all_neuron_ids)[shown_indices])

    def filter_events(
            self,
            all_spike_times,
            all_neuron_ids,
            interval=None,
            neuron_ids=None):
        """Function to filter for events which are within a given time interval
           (t_start, t_end) and of a subset of neuron ids.

        Args:
            all_spike_times (list): list of spike times of events
            all_neuron_ids (list): list of neuron ids of events
            interval (tuple(float, float)): (t_start, t_end) of time interval
                within which events should be considered. If interval is set to
                None, the interval is set to the duration of the entire spike train.
            neuron_ids (list): list of neuron ids to filter for (neurons
                with these ids are returned). If neuron_ids is set to None,
                all neuron ids are considered.

        Returns:
            all_spike_times (array): array of all spike times of events of
                specified neuron ids
            all_neuron_ids (array): array of all neuron ids of events within
                specified neuron ids
        """

        all_spike_times, all_neuron_ids = np.asarray(
            all_spike_times), np.asarray(all_neuron_ids)

        if interval is None and neuron_ids is None:
            return (all_spike_times, all_neuron_ids)

        if interval is not None:
            all_spike_times, all_neuron_ids = self._filter_events_for_interval(
                all_spike_times=all_spike_times, all_neuron_ids=all_neuron_ids,
                t_start=interval[0], t_end=interval[1]
                )
        if neuron_ids is not None:
            all_spike_times, all_neuron_ids = self._filter_events_for_neuron_ids(
                all_spike_times=all_spike_times, all_neuron_ids=all_neuron_ids,
                active_neuron_ids=neuron_ids
                )
        return (all_spike_times, all_neuron_ids)

    def _update_detailed_subplot_Xrange(self, region, detailed_plot):
        """ Function to update x range of detailed_plot depending on current
            position of the 'region' plotitem

        Args:
            region (pyqtgraph LinearRegionItem): LinearRegionItem whose
                position defines the x range of the detailed_plot
            detailed_plot (pyqtgraph PlotItem): PlotItem whose x_range will
                be updated depending on the 'region' position
        """
        minX, maxX = region.getRegion()
        detailed_plot.setXRange(minX, maxX, padding=0)

    def _update_region_position(self, region, detailed_plot):
        """ Function to update position of 'region' plotitem depending on
            current range of the detailed_plot
        plotitem.

        Args:
            region (pyqtgraph LinearRegionItem): LinearRegionItem whose
                position will be updated depending on the 'detailed_plot's range
            detailed_plot (pyqtgraph PlotItem): PlotItem whose x_range
                defines the position of the 'region' plotitem
        """
        rgn = detailed_plot.viewRange()[0]
        region.setRegion(rgn)

    def connect_detailed_subplot(self, filled_subplot_original_view, filled_subplot_detailed_view, show_plot=True):
        """ Function to connect the filled_subplot_original_view and the filled
            subplot_detailed_view via a region item.
            The region item is added to the filled_subplot_original_view.
            By changing the position of the region item in there, it
            automatically updates what is shown in the filled_subplot_detailed
            view and vice versa.
        
        Args:
             filled_subplot_original_view (pyqtgraph PlotItem): PlotItem filled
                with all the data to show to which the region is added.
             filled_subplot_detailed_view (pyqtgraph PlotItem): PlotItem filled
                with all the data as well. It will later show the detailed
                representation of the data whose x_range is connected with
                the position a RegionItem in the filled_subplot_original_view.
        Remarks:
            It is very important that the two PlotItems are two independent
            subplot objects. Do not pass the same PlotItem instance twice.
            Both PlotItems have to be already filled subplots. Meaning, both of
            them already have to contain/show the data.
        """
        region = pg.LinearRegionItem()
        filled_subplot_original_view.addItem(region, ignoreBounds=False)

        region.sigRegionChanged.connect(
            lambda: self._update_detailed_subplot_Xrange(
                region=region, detailed_plot=filled_subplot_detailed_view
                )
            )
        filled_subplot_detailed_view.sigRangeChanged.connect(
            lambda: self._update_region_position(
                region=region, detailed_plot=filled_subplot_detailed_view
                )
            )

        if show_plot:
            self.viewer.QtApp.exec_()

    def show(self):
        """ show plot """

        self.viewer.show()

    def save(self, path_to_save, figure_size=None):
        """ Save figure to path_to_save with size figure_size
        Args:
            path_to_save (str): path to location where to save figure incl filename
            figure_size (2-tuple): tuple of width and height of figure to save
        """

        self.viewer.save(path_to_save, figure_size)