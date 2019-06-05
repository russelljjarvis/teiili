# SPDX-License-Identifier: MIT
# Copyright (c) 2018 University of Zurich

import warnings

from teili.tools.visualizer.DataModels import EventsModel
from teili.tools.visualizer.DataControllers import DataController
from teili.tools.visualizer.DataViewers import RasterPlotViewerMatplotlib, RasterplotViewerPyqtgraph, PlotSettings


class Rasterplot(DataController):
    """ Class to plot rasterplots with different backends and from different DataModels
    - RasterplotController"""

    def __init__(
            self,
            MyEventsModels,
            MyPlotSettings=PlotSettings(),
            subgroup_labels=None,
            time_range=None,
            neuron_id_range=None,
            title='raster plot',
            xlabel='time',
            ylabel='count',
            backend='matplotlib',
            mainfig=None,
            subfig_rasterplot=None,
            subfig_histogram=None,
            QtApp=None,
            add_histogram=False,
            show_immediately=False):
        """ Setup Rasterplot controller and create rasterplot (incl histogram
                if add_histogram is True)
        Args:
            MyEventsModels (EventsModel or brian spike monitor object):
                EventsModel or brian spike monitor which holds data to plot
            MyPlotSettings (PlotSettings object): instance of class
                PlotSettings holding basic plot settings (e.g. fontsize, ...)
            subgroup_labels (list of str): list of labels for the different
                subgroups (e.g. ['exc', 'inh'])
            time_range (tuple): (t_start(float), t_end(float)) of time interval
                within which events should be considered
            neuron_id_range (tuple): (min_id, max_id) of neuron ids which
                should be considered
            title (str): title of plot
            xlabel (str): label of x-axis
            ylabel (str): label for y-axis
            backend (str): 'matplotlib' or 'pyqtgraph', defines which backend
                should be used for plotting
            mainfig (figure object): figure which holds the subfig (subplots)
                (plt.figure or  pg.GraphicsWindow())
            subfig_rasterplot (subplot): subplot of mainfig which will hold the
                rasterplot
            subfig_histogram (subplot): subplot of mainfig which will hold the
                histogram (if add_histogram is True)
            QtApp (pyqtgraph application): pyqtgraph application to run plots
                (QtGui.QApplication([])), only required if backend is pyqtgraph
            add_histogram (bool): if True: add histogram of spike count per
                neuron on right side of plot
            show_immediately (bool): if True: plot is shown immediately after
                it has been created
        """

        self.subgroup_labels = subgroup_labels
        self.time_range = time_range
        self.neuron_id_range = neuron_id_range
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.add_histogram = add_histogram

        if backend == 'matplotlib':
            self.viewer = RasterPlotViewerMatplotlib(
                MyPlotSettings,
                mainfig=mainfig,
                subfig_rasterplot=subfig_rasterplot,
                subfig_histogram=subfig_histogram,
                add_histogram=self.add_histogram)
        elif backend == 'pyqtgraph':
            self.viewer = RasterplotViewerPyqtgraph(
                MyPlotSettings,
                mainfig=mainfig,
                subfig_rasterplot=subfig_rasterplot,
                subfig_histogram=subfig_histogram,
                QtApp=QtApp,
                add_histogram=self.add_histogram)
        else:
            raise Exception(
                'You asked for the backend "{}" which is not supported'.format(backend))

        self.MyEventsModels = []
        for one_eventmodel in MyEventsModels:
            if not isinstance(one_eventmodel, EventsModel):
                self.MyEventsModels.append(
                    EventsModel.from_brian_spike_monitor(one_eventmodel))
            else:
                self.MyEventsModels.append(one_eventmodel)

        # prepare data for rasterplot
        self._get_data_from_eventsmodels()
        self._filter_data()

        self.create_plot()
        # to allow easier access to main- and subfigure
        self.mainfig = self.viewer.mainfig
        self.subfig_rasterplot = self.viewer.subfig_rasterplot
        self.subfig_histogram = self.viewer.subfig_histogram
        if show_immediately:
            self.show()

    def _get_data_from_eventsmodels(self):
        """ Get data from MyEventsModels and reformat it to list of neuron_ids
            and spike_times per subgroup"""
        self.all_neuron_ids, self.all_spike_times = [], []
        for one_event_model in self.MyEventsModels:
            self.all_neuron_ids.append(one_event_model.neuron_ids)
            self.all_spike_times.append(one_event_model.spike_times)

    def _filter_data(self):
        """ Filter self.neuron_ids and self.spike_times to be within time_range
            and neuron_id_range. The MyEventsModels data is copied and not
            changed in place """

        if self.time_range is None and self.neuron_id_range is None:
            return

        else:
            # turn tuple(start,end) into range
            if self.neuron_id_range is None:
                considered_neuron_ids = None
            else:
                considered_neuron_ids = range(
                    self.neuron_id_range[0], self.neuron_id_range[1] + 1)

            all_filtered_neuron_ids, all_filtered_spike_times = [], []
            for event_model_nr, one_event_model in enumerate(
                    self.MyEventsModels):
                active_spike_times, active_neuron_ids = self.filter_events(
                    all_spike_times=one_event_model.spike_times,
                    all_neuron_ids=one_event_model.neuron_ids,
                    interval=self.time_range,
                    neuron_ids=considered_neuron_ids
                    )

                if len(active_neuron_ids) == 0:
                    warnings.warn(
                        "For subgroup_nr {} there are no events left after filtering for time range {} and neuron_id range {}." .format(
                            event_model_nr, self.time_range, self.neuron_id_range))

                all_filtered_spike_times.append(active_spike_times)
                all_filtered_neuron_ids.append(active_neuron_ids)

        self.all_spike_times = all_filtered_spike_times
        self.all_neuron_ids = all_filtered_neuron_ids

    def create_plot(self):
        """ Function to create rasterplot (incl histogram if add_histogram is True)
            in subfigures defined above and with data from MyEventsModels with
            subgroups defined above"""

        self.viewer.create_plot(
            all_spike_times=self.all_spike_times,
            all_neuron_ids=self.all_neuron_ids,
            subgroup_labels=self.subgroup_labels,
            time_range_axis=self.time_range,
            neuron_id_range_axis=self.neuron_id_range,
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel)
