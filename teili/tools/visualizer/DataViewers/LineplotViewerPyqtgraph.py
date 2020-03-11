# SPDX-License-Identifier: MIT
# Copyright (c) 2018 University of Zurich

import numpy as np
import warnings
try:
    import pyqtgraph as pg
    from PyQt5 import QtGui
except BaseException:
    warnings.warn("No method using pyqtgraph can be used as pyqtgraph or PyQt5"
                  "can't be imported.")

from teili.tools.visualizer.DataViewers import LineplotViewer
from teili.tools.visualizer.DataViewers.DataViewerUtilsPyqtgraph import DataViewerUtilsPyqtgraph

class LineplotViewerPyqtgraph(LineplotViewer):
    """ Class to plot lineplot with pyqtgraph backend """

    def __init__(
            self,
            MyPlotSettings,
            mainfig=None,
            subfig=None,
            QtApp=None):
        """ Setup LineplotViewer by initializing main figure and subfigure.
            If any of them is set to None, it will be created internally.
        Args:
            MyPlotSettings (PlotSettings object): instance of class
                PlotSettings holding basic plot settings (e.g. fontsize, ...)
            mainfig (pyqtgraph window object): pyqtgraph main window
                (pg.GraphicsWindow())
            subfig (pyqtgraph subplot): pyqtgraph subplot of mainfig which will
                hold the lineplot
            QtApp (pyqtgraph application): pyqtgraph application to run plots
                (QtGui.QApplication([])), if None: it will check for an existing
                QtApps to use or creates a new one otherwise
        """

        self.MyPlotSettings = MyPlotSettings
        self.set_DataViewerUtils()

        # set up qt application, main- and sub-figure
        self.QtApp = self.DVUtils.set_up_QtApp(QtApp=QtApp)
        self.mainfig = self.DVUtils.set_up_mainfig(mainfig=mainfig, subfig=subfig)
        self.subfig = self.DVUtils.set_up_subfig(mainfig=self.mainfig, subfig=subfig)

        pg.setConfigOptions(antialias=True)

    def set_DataViewerUtils(self):
        """ Set which DataViewerUtils class should be considered"""
        self.DVUtils = DataViewerUtilsPyqtgraph(viewer=self)

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

        if subgroup_labels is not None:
            self.subfig.addLegend()

        # set parameters on plot dimensions along x and y axis
        if x_range_axis is None:
            all_x_data = np.concatenate(list(map(lambda x: np.asarray(x[0]).flatten(), data)))
            x_range_axis = (min(all_x_data, default=0), max(all_x_data, default=1))
        if y_range_axis is None:
            all_y_data = np.concatenate(list(map(lambda x: np.asarray(x[1]).flatten(), data)))
            y_range_axis = (min(all_y_data, default=0), max(all_y_data, default=1))

        # lineplot
        for subgroup_nr, (subgroup, color) in enumerate(
                zip(data, self.MyPlotSettings.colors)):

            # color = np.asarray(pg.colorTuple(pg.mkColor(color)))
            if not isinstance(color, str):
                color = tuple(np.asarray(color))

            n_traces_x = int(np.size(subgroup[0]) / np.max((1, np.shape(subgroup[0])[0])))
            n_traces_y = int(np.size(subgroup[1]) / np.max((1, np.shape(subgroup[1])[0])))

            subgroup_x = np.reshape(
                subgroup[0], (np.shape(
                    subgroup[0])[0], n_traces_x))
            subgroup_y = np.reshape(
                subgroup[1], (np.shape(
                    subgroup[1])[0], n_traces_y))
            for nr_trace_x in range(n_traces_x):
                for nr_trace_y in range(n_traces_y):
                    self.subfig.plot(
                        x=subgroup_x[:, nr_trace_x],
                        y=subgroup_y[:, nr_trace_y],
                        pen=color)

            if subgroup_labels is not None:
                style = pg.PlotDataItem(pen=color)
                self.subfig.legend.addItem(style, subgroup_labels[subgroup_nr])

        self.DVUtils.add_legend(subgroup_labels=subgroup_labels,
                                subfig=self.subfig)

        self.DVUtils._set_title_and_labels(subfig=self.subfig, title=title,
                                           xlabel=xlabel, ylabel=ylabel)
        self.subfig.setRange(
            xRange=(
                x_range_axis[0], x_range_axis[1]), yRange=(
                y_range_axis[0], y_range_axis[1]))