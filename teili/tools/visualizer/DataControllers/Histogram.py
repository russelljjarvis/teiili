# SPDX-License-Identifier: MIT
# Copyright (c) 2018 University of Zurich

import numpy as np

from teili.tools.visualizer.DataControllers import DataController
from teili.tools.visualizer.DataViewers import HistogramViewerMatplotlib, HistogramViewerPyqtgraph, PlotSettings


class Histogram(DataController):
    """ Class to plot histograms with different backends and from different DataModels
    - HistogramController """

    def __init__(
            self,
            DataModel_to_attr,
            MyPlotSettings=PlotSettings(),
            subgroup_labels=None,
            bins=None,
            orientation='vertical',
            title='histogram',
            xlabel='bins',
            ylabel='count',
            backend='matplotlib',
            mainfig=None,
            subfig=None,
            QtApp=None,
            show_immediately=False):
        """ Setup Histogram Controller and create histogram plot. 
        
        Args:
            DataModel_to_attr (list of tuples): [tuple(::class DataModel::
                attr_of_DataModel_to_consider), ( ..., ...), ... ] for every
                subgroups one tuple (data model can also be a brian state
                monitor or spike monitor)
            MyPlotSettings (PlotSettings object): instance of class
                PlotSettings holding basic plot settings (e.g. fontsize, ...)
            subgroup_labels (list of str): list of labels for the different
                subgroups (e.g. ['exc', 'inh'])
            bins (array, list): array with edges of bins in histogram
            orientation (str): orientation of histogram (vertical/horizontal)
            title (str): title of plot
            xlabel (str): label of x-axis
            ylabel (str): label for y-axis
            backend (str): 'matplotlib' or 'pyqtgraph', defines which backend
                should be used for plotting
            mainfig (figure object): figure which holds the subfig (subplots)
                (plt.figure or  pg.GraphicsWindow())
            subfig (subplot): subplot of mainfig which will hold the histogram
            QtApp (pyqtgraph application): pyqtgraph application to run plots
                (QtGui.QApplication([]) ), only required if backend is
                pyqtgraph
            show_immediately (bool): if True: plot is shown
                immediately after it has been created.
        """

        self.subgroup_labels = subgroup_labels
        self.bins = bins
        self.orientation = orientation
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        if backend == 'matplotlib':
            self.viewer = HistogramViewerMatplotlib(
                MyPlotSettings, mainfig=mainfig, subfig=subfig)
        elif backend == 'pyqtgraph':
            self.viewer = HistogramViewerPyqtgraph(
                MyPlotSettings, mainfig=mainfig, subfig=subfig, QtApp=QtApp)
        else:
            raise Exception(
                'You asked for the backend "{}" which is not supported'.format(backend))

        self._get_data(DataModel_to_attr)
        self.create_plot()
        # to allow easier access to main- and subfigure
        self.mainfig = self.viewer.mainfig
        self.subfig = self.viewer.subfig
        if show_immediately:
            self.show()

    def _get_data(self, DataModel_to_attr):
        """ get data from DataModel_to_attr. Reformat it as list (self.data) of attributes considered"""

        self.data = []
        for data_model, attr_to_consider in DataModel_to_attr:
            self.data.append(
                np.asarray(
                    getattr(
                        data_model,
                        attr_to_consider)).flatten())

    def create_plot(self):
        """ Function to create histogram in subfigure with data from DataModel_to_attr with subgroups defined above"""

        self.viewer.create_plot(
            data=self.data,
            subgroup_labels=self.subgroup_labels,
            bins=self.bins,
            orientation=self.orientation,
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel)
