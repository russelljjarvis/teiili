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

from teili.tools.visualizer.DataViewers.HistogramViewer import HistogramViewer
from teili.tools.visualizer.DataViewers.DataViewerUtilsPyqtgraph import DataViewerUtilsPyqtgraph

class HistogramViewerPyqtgraph(HistogramViewer):
    """ Class to plot histogram with pyqtgraph backend """

    def __init__(self, MyPlotSettings, mainfig=None, subfig=None, QtApp=None):
        """ Setup HistogramViewer by initializing main figure and subfigure.
            If any of them is set to None, it will be created internally.

        Args:
            MyPlotSettings (PlotSettings object): instance of class
                PlotSettings holding basic plot settings (e.g. fontsize, ...)
            mainfig (pyqtgraph window object): pyqtgraph main window
                (pg.GraphicsWindow())
            subfig (pyqtgraph subplot): pyqtgraph subplot of mainfig which will
                hold the histogram
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
            bins=None,
            orientation='vertical',
            title='histogram',
            xlabel='bins',
            ylabel='count'):
        """ Function to generate histogram for groups of event sets (spike times, neuron ids) with pyqtgraph
        Args:
            data (list of lists): list of lists of neuron ids of events (e.g. [[3,4,5,2],[9,7,7 8]]
            subgroup_labels (list of str): list of labels for the different subgroups (e.g. ['exc', 'inh'])
            bins (array, list): array with edges of bins in histogram
            orientation (str): orientation of histogram (vertical or horizontal)
            title (str): title of plot
            xlabel (str): label of x-axis
            ylabel (str): label for y-axis
            """

        if bins is None:
            bins = self.set_bins(data=data)

        # check if num colors ok
        self.check_num_colors(n_provided_colors=len(
            self.MyPlotSettings.colors), n_required_colors=len(data))

        if subgroup_labels is not None:
            self.subfig.addLegend()

        # histogram
        for subgroup_nr, (subgroup, color) in enumerate(
                zip(data, self.MyPlotSettings.colors)):
            subgroup = self.remove_nans(subgroup)

            y, x = np.histogram(subgroup, bins=bins)
            color = np.asarray(pg.colorTuple(pg.mkColor(color)))

            if orientation == 'horizontal':
                barchart = pg.BarGraphItem(
                    x0=y * 0, y0=x[:-1], height=1.0, width=y, pen=None, brush=color)
            else:
                barchart = pg.BarGraphItem(
                    x0=x[:-1], y0=0, height=y, width=1.0, pen=None, brush=color)

            if subgroup_labels is not None:
                style = pg.PlotDataItem(pen=color)
                self.subfig.legend.addItem(style, subgroup_labels[subgroup_nr])
            self.subfig.addItem(barchart)

        self.DVUtils.add_legend(subgroup_labels=subgroup_labels,
                                subfig=self.subfig)

        self.DVUtils._set_title_and_labels(subfig=self.subfig, title=title,
                                           xlabel=xlabel, ylabel=ylabel)