import matplotlib.pylab as plt
import numpy as np
import warnings

from teili.tools.visualizer.DataViewers.HistogramViewer import HistogramViewer
from teili.tools.visualizer.DataViewers.DataViewerUtilsMatplotlib import DataViewerUtilsMatplotlib

class HistogramViewerMatplotlib(HistogramViewer):
    """ Class to plot histogram with matplotlib backend """

    def __init__(self, MyPlotSettings, mainfig=None, subfig=None):
        """ Setup HistogramViewer by initializing main figure and subfigure.
            If any of them is set to None, it will be created internally.
        Args:
            MyPlotSettings (PlotSettings object): instance of class
                PlotSettings holding basic plot settings (e.g. fontsize, ...)
            mainfig (matplotlib figure object): matplotlib figure which holds
                the subfig (subplots)
            subfig (matplotlib subplot): matplotlib subplot of mainfig which
                will hold the histogram
        """
        self.MyPlotSettings = MyPlotSettings

        # figure
        self.mainfig = mainfig
        if not self.mainfig:
            if subfig:
                self.mainfig = subfig.figure
            else:
                self.mainfig = plt.figure()

        # subplot
        self.subfig = subfig
        if not self.subfig:
            self.subfig = self.mainfig.add_subplot(111)

        self.set_DataViewerUtils()

    def set_DataViewerUtils(self):
        """ Set which DataViewerUtils class should be considered"""
        self.DVUtils = DataViewerUtilsMatplotlib(mainfig=self.mainfig)

    def create_plot(
            self,
            data,
            subgroup_labels=None,
            bins=None,
            orientation='vertical',
            title='histogram',
            xlabel='bins',
            ylabel='count'):
        """ Function to create histogram for groups of event sets in
            self.subfig (spike times, neuron ids) with matplotlib
        Args:
            data (list of lists): list of lists of neuron ids of events
                (e.g. [[3,4,5,2],[9,7,7 8]]
            subgroup_labels (list of str): list of labels for the different
                subgroups (e.g. ['exc', 'inh'])
            bins (array, list): array with edges of bins in histogram
            orientation (str): orientation of histogram (vertical/horizontal)
            title (str): title of plot
            xlabel (str): label of x-axis
            ylabel (str): label for y-axis
        """

        if bins is None:
            bins = self.set_bins(data=data)

        # get max value of y axis based on max count in histogram + 5% to set
        # axis_lim of count slightly above y_max otherwise the points at the
        # border are hard to see in the plot
        axis_lim = (max([self.get_highest_count(lst) for lst in data])) * 1.05
        if orientation == 'vertical':
            self.subfig.set_ylim(bottom=0, top=axis_lim)
        elif orientation == 'horizontal':
            self.subfig.set_xlim(left=0, right=axis_lim)

        # check if num colors ok
        self.check_num_colors(n_provided_colors=len(
            self.MyPlotSettings.colors), n_required_colors=len(data))

        label = None
        for subgroup_nr, (subgroup, color) in enumerate(
                zip(data, self.MyPlotSettings.colors[:len(data)])):
            if subgroup_labels is not None:
                label = subgroup_labels[subgroup_nr]
            if (np.isnan(subgroup)).any():
                subgroup = subgroup[~np.isnan(subgroup)]
                warnings.warn("One of your subgroup contains NAN entries. They are removed and not shown in the histogram")

            self.subfig.hist(
                subgroup,
                bins=bins,
                orientation=orientation,
                label=label,
                color=color)
            self.subfig.set_autoscale_on(False)

        if subgroup_labels is not None:
            self.subfig.legend(loc='best',
                               fontsize=self.MyPlotSettings.fontsize_legend)

        self.DVUtils._set_title_and_labels(subfig=self.subfig, title=title, xlabel=xlabel, ylabel=ylabel,
                                   fontsize_title=self.MyPlotSettings.fontsize_title,
                                   fontsize_axis_labels=self.MyPlotSettings.fontsize_axis_labels)