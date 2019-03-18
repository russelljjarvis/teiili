import matplotlib.pylab as plt
import numpy as np
import warnings

from teili.tools.visualizer.DataViewers.HistogramViewer import HistogramViewer


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

    def create_histogram(
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
            max_per_dataset = []
            for x in data:
                if np.size(x) > 0:  # to avoid error by finding max of emtpy dataset
                    max_per_dataset.append(np.nanmax(x))
                else:
                    max_per_dataset.append(0)
            bins = range(int(max(max_per_dataset))+2)  # +2 to always have at least 1 bin        # get max value of y axis based on max count in histogram + 5% to set

        # axis_lim of count slightly above y_max otherwise the points at the
        # border are hard to see in the plot
        axis_lim = (max([self.get_highest_count(lst) for lst in data])) * 1.05
        if orientation == 'vertical':
            self.subfig.set_ylim(bottom=0, top=axis_lim)
        elif orientation == 'horizontal':
            self.subfig.set_xlim(left=0, right=axis_lim)

        # check if num colors ok
        assert len(
            self.MyPlotSettings.colors) >= len(data), \
                'You have {} subgroups but only {} colors in your MyPlotSettings.colors'.format(
                    len(data), len(self.MyPlotSettings.colors)
                    )

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
                color=color)  # rwidth=0.9)
            self.subfig.set_autoscale_on(False)

        if subgroup_labels is not None:
            self.subfig.legend(loc='best',
                               fontsize=self.MyPlotSettings.fontsize_legend)

        self._set_title_and_labels(title=title, xlabel=xlabel, ylabel=ylabel)

    def _set_title_and_labels(self, title, xlabel, ylabel):
        """ Set title and label of x- and y-axis in plot
        Args:
            title (str): title of plot
            xlabel (str): label for x-axis
            ylabel (str): label for y-axis
        """

        if title is not None:
            self.subfig.set_title(
                title, fontsize=self.MyPlotSettings.fontsize_title)
        if xlabel is not None:
            self.subfig.set_xlabel(
                xlabel, fontsize=self.MyPlotSettings.fontsize_axis_labels)
        if ylabel is not None:
            self.subfig.set_ylabel(
                ylabel, fontsize=self.MyPlotSettings.fontsize_axis_labels)

    def show_histogram(self):
        """ show plot """
        plt.show()

    def save_histogram(
        self,
        path_to_save='histogram.png',
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
