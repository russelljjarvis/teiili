# SPDX-License-Identifier: MIT
# Copyright (c) 2018 University of Zurich

import matplotlib.pylab as plt
import numpy as np

from teili.tools.visualizer.DataViewers import LineplotViewer
from teili.tools.visualizer.DataViewers.DataViewerUtilsMatplotlib import DataViewerUtilsMatplotlib


class LineplotViewerMatplotlib(LineplotViewer):
    """ Class to plot lineplots with matplotlib backend """

    def __init__(self, MyPlotSettings, mainfig=None, subfig=None):
        """ Setup LineplotViewer by initializing main figure and subfigure.
            If any of them is set to None, it will be created internally.
        Args:
            MyPlotSettings (PlotSettings object): instance of class
                PlotSettings holding basic plot settings (e.g. fontsize, ...)
            mainfig (matplotlib figure object): matplotlib figure which holds
                the subfig (subplots)
            subfig (matplotlib subplot): matplotlib subplot of mainfig which
                will hold the lineplot
        """
        self.MyPlotSettings = MyPlotSettings
        self.set_DataViewerUtils()

        # set up main- and subfigure
        self.mainfig = self.DVUtils.set_up_mainfig(mainfig=mainfig, subfig=subfig)
        self.subfig = self.DVUtils.set_up_subfig(mainfig=self.mainfig, subfig=subfig)


    def set_DataViewerUtils(self):
        """ Set which DataViewerUtils class should be considered"""
        self.DVUtils = DataViewerUtilsMatplotlib(viewer=self)

    def create_plot(
            self,
            data,
            subgroup_labels=None,
            x_range_axis=None,
            y_range_axis=None,
            title='Lineplot',
            xlabel=None,
            ylabel=None):
        """ Function to create lineplot for groups of event sets in self.subfig
            with matplotlib
        Args:
            data (list of tuples): list of tuples, whereby each tuple is one
                subgroup which will be plotted, e.g.:[(x_values_A, y_values_A),
                (x_values_B, y_values_B), ... ]
            subgroup_labels (list of str): list of labels for the different
                subgroups (e.g. ['exc', 'inh'])
            x_range_axis (tuple): (min, max) of interval within which elements
                are shown along x-axis
            y_range_axis (tuple): (min, max) of interval within which elements
                are shown along y-axis
            title (str): title of plot
            xlabel (str): label of x-axis
            ylabel (str): label for y-axis
        """
        # check if num colors ok
        self.check_num_colors(n_provided_colors=len(
            self.MyPlotSettings.colors), n_required_colors=len(data))

        # set parameters on plot dimensions along x and y axis
        if x_range_axis is None:
            all_x_data = np.concatenate(list(map(lambda x: np.asarray(x[0]).flatten(), data)))
            x_range_axis = (min(all_x_data, default=0), max(all_x_data, default=1))
        if y_range_axis is None:
            all_y_data = np.concatenate(list(map(lambda x: np.asarray(x[1]).flatten(), data)))
            y_range_axis = (min(all_y_data, default=0), max(all_y_data, default=1))

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

            self.subfig.set_xlim(left=x_range_axis[0], right=x_range_axis[1])
            self.subfig.set_ylim(bottom=y_range_axis[0], top=y_range_axis[1])

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

        self.DVUtils._set_title_and_labels(subfig=self.subfig, title=title, xlabel=xlabel, ylabel=ylabel)
