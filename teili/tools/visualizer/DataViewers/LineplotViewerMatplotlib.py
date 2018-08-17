import matplotlib.pylab as plt
import numpy as np


try:
    from teili.tools.visualizer.DataViewers import LineplotViewer
except BaseException:
    from teili.teili.tools.visualizer.DataViewers import LineplotViewer


class LineplotViewerMatplotlib(LineplotViewer):
    """ Class to plot lineplots with matplotlib backend """

    def __init__(self, MyPlotSettings, mainfig=None, subfig=None):
        """ Setup LineplotViewer by initializing main figure and subfigure. If any of them is set to None, it will be
        created internally.
        Args:
            MyPlotSettings (PlotSettings object): instance of class PlotSettings holding basic plot settings (e.g. fontsize, ...)
            mainfig (matplotlib figure object): matplotlib figure which holds the subfig (subplots)
            subfig (matplotlib subplot): matplotlib subplot of mainfig which will hold the lineplot
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

    def create_lineplot(
            self,
            data,
            subgroup_labels=None,
            x_range_axis=None,
            y_range_axis=None,
            title='Lineplot',
            xlabel=None,
            ylabel=None):
        """ Function to create lineplot for groups of event sets in self.subfig with matplotlib
        Args:
            data (list of tuples): list of tuples, whereby each tuple is one subgroup which will be plotted,
                                        e.g.: [(x_values_A, y_values_A), (x_values_B, y_values_B), ... ]
            subgroup_labels (list of str): list of labels for the different subgroups (e.g. ['exc', 'inh'])
            x_range_axis (tuple): (min, max) of interval within which elements are shown along x-axis
            y_range_axis (tuple): (min, max) of interval within which elements are shown along y-axis
            title (str): title of plot
            xlabel (str): label of x-axis
            ylabel (str): label for y-axis
        """
        # check if num colors ok
        assert len(
            self.MyPlotSettings.colors) >= len(data), 'You have {} subgroups but only {} colors in your MyPlotSettings.colors'.format(
            len(data), len(
                self.MyPlotSettings.colors))

        # set parameters on plot dimensions along x and y axis
        if x_range_axis is None:
            x_range_axis = (0, max(map(lambda x: np.max(x[0]), data)))
        if y_range_axis is None:
            y_range_axis = (0, max(map(lambda x: np.max(x[1]), data)))

        label = None
        for subgroup_nr, (subgroup, color) in enumerate(
                zip(data, self.MyPlotSettings.colors[:len(data)])):

            if subgroup_labels is not None:
                label = subgroup_labels[subgroup_nr]

            self.subfig.plot(
                subgroup[0],
                subgroup[1],
                label=label,
                color=color)

            self.subfig.set_xlim(xmin=x_range_axis[0], xmax=x_range_axis[1])
            self.subfig.set_ylim(ymin=y_range_axis[0], ymax=y_range_axis[1])

        if subgroup_labels is not None:
            my_handles, my_labels = [], []
            subfig_legend_handles_labels = self.subfig.get_legend_handles_labels()
            for subgroup_label in subgroup_labels:
                index_in_figure_labels = subfig_legend_handles_labels[1].index(
                    subgroup_label)
                my_handles.append(
                    subfig_legend_handles_labels[0][index_in_figure_labels])
                my_labels.append(subgroup_label)
            self.subfig.legend(
                loc='best',
                handles=my_handles,
                labels=my_labels,
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

    def show_lineplot(self):
        """ show plot """
        plt.show()

    def save_lineplot(
            self,
            path_to_save='lineplot.png',
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
