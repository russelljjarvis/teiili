import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

try:
    from teili.tools.visualizer.DataViewers.RasterplotViewer import RasterplotViewer
    from teili.tools.visualizer.DataViewers.HistogramViewerMatplotlib import HistogramViewerMatplotlib
except BaseException:
    from teili.teili.tools.visualizer.DataViewers.RasterplotViewer import RasterplotViewer
    from teili.teili.tools.visualizer.DataViewers.HistogramViewerMatplotlib import HistogramViewerMatplotlib


class RasterPlotViewerMatplotlib(RasterplotViewer):
    """ Class to plot raster plot with matplotlib backend """

    def __init__(
            self,
            MyPlotSettings,
            mainfig=None,
            subfig_rasterplot=None,
            subfig_histogram=None,
            add_histogram=False):
        """ Setup RasterplotViewer by initializing main figure and subfigures. If any of them is set to None, it will be
        created internally.
        If add_histogram is False, subfig_histogram is set to None.
        Args:
            MyPlotSettings (PlotSettings object): instance of class PlotSettings holding basic plot settings (e.g. fontsize, ...)
            mainfig (matplotlib figure object): matplotlib figure which holds the subfig (subplots)
            subfig_rasterplot (matplotlib subplot): matplotlib subplot of mainfig which will hold the rasterplot
            subfig_histogram (matplotlib subplot): matplotlib subplot of mainfig which will hold the histogram
            add_histogram (bool): if True: add histogram of spike count per neuron on right side of plot
        """

        self.MyPlotSettings = MyPlotSettings
        self.add_histogram = add_histogram

        # figure
        self.mainfig = mainfig
        if not self.mainfig:
            if subfig_rasterplot:
                self.mainfig = subfig_rasterplot.figure
            else:
                self.mainfig = plt.figure()

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
                self.subfig_histogram = plt.subplot2grid(
                    (1, 3), (0, 2), rowspan=1, sharey=self.subfig_rasterplot, frameon=0, fig=self.mainfig)
                plt.subplots_adjust(
                    left=0.125,
                    right=0.9,
                    bottom=0.1,
                    top=0.9,
                    wspace=0.05,
                    hspace=0.2)
        else:
            self.subfig_histogram = None

    def create_rasterplot(
            self,
            all_spike_times,
            all_neuron_ids,
            subgroup_labels=None,
            time_range_axis=None,
            neuron_id_range_axis=None,
            title='raster plot',
            xlabel='time (s)',
            ylabel='neuron ids'):
        """ Function to generate raster plot (incl histogram of events per neuron id) from groups of event sets
            (spike times, neuron ids)
        Args:
            all_spike_times (list of lists): list of lists of spike times of events, (e.g. [[t0, t1, t2],[tt1, tt2, tt3]])
            all_neuron_ids (list of lists): list of lists of neuron ids of events (e.g. [[3,4,5,2],[9,7,7 8]]
            subgroup_labels (list of str): list of labels for the different subgroups (e.g. ['exc', 'inh'])
            time_range_axis (tuple): (t_start(float), t_end(float)) of time interval within which events should be show
            neuron_id_range_axis (tuple): (min_id, max_id) of neuron ids which should be shown
            title (str): title of plot
            xlabel (str): label of x-axis
            ylabel (str): label for y-axis

        Remarks:
            time_range_axis and neuron_id_range_axis do not filter the spike times and neuron ids. It only defines the
            region which is shown in the plot. Hence, for the histogram ALL events provided are considered to calculate
            the histogram.
        """

        # check if colours ok
        assert len(
            self.MyPlotSettings.colors) >= len(all_neuron_ids), 'You have {} subgroups but only {} colors in your MyPlotSettings.colors'.format(
            len(all_neuron_ids), len(
                self.MyPlotSettings.colors))

        # set parameters on plot dimensions along time and neuron_id axis
        if time_range_axis is None:
            time_range_axis = (0, max(map(lambda x: max(x), all_spike_times)))
        if neuron_id_range_axis is None:
            neuron_id_range_axis = (
                0, max(map(lambda x: max(x), all_neuron_ids)) + 1)

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

        self._set_title_and_labels(title=title, xlabel=xlabel, ylabel=ylabel)

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
            all_neuron_ids (list of lists): list of lists of neuron ids of events (e.g. [[3,4,5,2],[9,7,7 8]]
            num_neurons (int): total number of neurons to show on y-axis
        """

        HV = HistogramViewerMatplotlib(
            self.MyPlotSettings,
            mainfig=self.mainfig,
            subfig=self.subfig_histogram)
        HV.create_histogram(data=all_neuron_ids,
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

    def _set_title_and_labels(self, title, xlabel, ylabel):
        """ Set title and label of x- and y-axis in plot
        Args:
            title (str): title of plot
            xlabel (str): label for x-axis
            ylabel (str): label for y-axis
        """
        if title is not None:
            self.subfig_rasterplot.set_title(
                title, fontsize=self.MyPlotSettings.fontsize_title)
        if xlabel is not None:
            self.subfig_rasterplot.set_xlabel(
                xlabel, fontsize=self.MyPlotSettings.fontsize_axis_labels)
        if ylabel is not None:
            self.subfig_rasterplot.set_ylabel(
                ylabel, fontsize=self.MyPlotSettings.fontsize_axis_labels)

    def show_rasterplot(self):
        """ show plot """
        plt.show()

    def save_rasterplot(
        self,
        path_to_save='rasterplot.png',
        figure_size=None):
        """ Save figure to path_to_save with size figure_size as png, pdf, ps, eps and svg.
        Args:
            path_to_save (str): path to location where to save figure incl filename
            figure_size (tuple): tuple of width and height in inch of figure to save
        """
        if figure_size is not None:
            self.mainfig.set_size_inches(figure_size[0], figure_size[1])
        self.mainfig.savefig(path_to_save)
        print('Figure saved to: ' + path_to_save)
