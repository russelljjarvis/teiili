import matplotlib.pylab as plt
try:
    from teili.tools.visualizer.DataModels import EventsModel, StateVariablesModel
    from teili.tools.visualizer.DataControllers import DataController
    from teili.tools.visualizer.DataViewers import RasterPlotViewerMatplotlib
    from teili.tools.visualizer.DataViewers import RasterplotViewerPyqtgraph
except BaseException:
    from teili.teili.tools.visualizer.DataModels import EventsModel, StateVariablesModel
    from teili.teili.tools.visualizer.DataControllers import DataController
    from teili.teili.tools.visualizer.DataViewers import RasterPlotViewerMatplotlib
    from teili.teili.tools.visualizer.DataViewers import RasterplotViewerPyqtgraph


class RasterplotController(DataController):
    """ Class to plot rasterplots with different backends and from different DataModels"""

    def __init__(
            self,
            MyPlotSettings,
            MyEventsModels,
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
            show_immediately=True):
        """ Setup Rasterplot controller and create rasterplot (incl histogram if add_histogram is True)
        Args:
            MyPlotSettings (PlotSettings object): instance of class PlotSettings holding basic plot settings (e.g. fontsize, ...)
            MyEventsModels (EventsModel or brian spike monitor object): EventsModel or brian spike monitor which holds data to be plotted
            subgroup_labels (list of str): list of labels for the different subgroups (e.g. ['exc', 'inh'])
            time_range (tuple): (t_start(float), t_end(float)) of time interval within which events should be considered
            neuron_id_range (tuple): (min_id, max_id) of neuron ids which should be considered
            title (str): title of plot
            xlabel (str): label of x-axis
            ylabel (str): label for y-axis
            backend (str): 'matplotlib' or 'pyqtgraph', defines which backend should be used for plotting
            mainfig (figure object): figure which holds the subfig (subplots) (plt.figure or  pg.GraphicsWindow())
            subfig_rasterplot (subplot): subplot of mainfig which will hold the rasterplot
            subfig_histogram (subplot): subplot of mainfig which will hold the histogram (if add_histogram is True)
            QtApp (pyqtgraph application): pyqtgraph application to run plots ( QtGui.QApplication([]) ),
                                            only required if backend is pyqtgraph
            add_histogram (bool): if True: add histogram of spike count per neuron on right side of plot
            show_immediately (bool): if True: plot is shown immediately after it has been created
        """

        self.subgroup_labels = subgroup_labels
        self.time_range = time_range
        self.neuron_id_range = neuron_id_range
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.add_histogram = add_histogram

        if backend == 'matplotlib':
            self.my_rasterplot = RasterPlotViewerMatplotlib(
                MyPlotSettings,
                mainfig=mainfig,
                subfig_rasterplot=subfig_rasterplot,
                subfig_histogram=subfig_histogram,
                add_histogram=self.add_histogram)
        elif backend == 'pyqtgraph':
            self.my_rasterplot = RasterplotViewerPyqtgraph(
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
        self._filter_data_to_be_within_ranges()
        self.create_rasterplot()
        if show_immediately:
            self.show_rasterplot()

    def _get_all_data_from_event_models(self):
        """ Get data from MyEventsModels and reformat it to list of neuron_ids and spike_times per subgroup"""

        self.all_neuron_ids, self.all_spike_times = [], []
        for one_event_model in self.MyEventsModels:
            self.all_neuron_ids.append(one_event_model.neuron_ids)
            self.all_spike_times.append(one_event_model.spike_times)

    def _filter_data_to_be_within_ranges(self):
        """ Filter data from MyEventsModels to be within time_range and neuron_id_range.
        The MyEventsModels data is copied and not changed in place """

        if self.time_range is None and self.neuron_id_range is None:
            self._get_all_data_from_event_models()
        else:
            self.all_neuron_ids, self.all_spike_times = [], []
            for one_event_model in self.MyEventsModels:
                active_spike_times, active_neuron_ids = self.filter_events(all_spike_times=one_event_model.spike_times,
                                                                           all_neuron_ids=one_event_model.neuron_ids,
                                                                           interval=self.time_range,
                                                                           neuron_ids=range(self.neuron_id_range[0],
                                                                                            self.neuron_id_range[1] + 1))
                self.all_spike_times.append(active_spike_times)
                self.all_neuron_ids.append(active_neuron_ids)

    def create_rasterplot(self):
        """ Function to create rasterplot (incl histogram if add_histogram is True) in subfigures defined above and
            with data from MyEventsModels with subgroups defined above"""

        self.my_rasterplot.create_rasterplot(
            all_spike_times=self.all_spike_times,
            all_neuron_ids=self.all_neuron_ids,
            subgroup_labels=self.subgroup_labels,
            time_range_axis=self.time_range,
            neuron_id_range_axis=self.neuron_id_range,
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel)

    def show_rasterplot(self):
        """ show plot """

        self.my_rasterplot.show_rasterplot()

    def save_rasterplot(self, path_to_save, figure_size):
        """ Save figure to path_to_save with size figure_size
        Args:
            path_to_save (str): path to location where to save figure incl filename
            figure_size (2-tuple): tuple of width and height of figure to save
        """

        self.my_rasterplot.save_rasterplot(path_to_save, figure_size)
