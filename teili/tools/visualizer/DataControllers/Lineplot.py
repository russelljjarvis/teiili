# SPDX-License-Identifier: MIT
# Copyright (c) 2018 University of Zurich

import numpy as np
import warnings
from teili.tools.visualizer.DataControllers import DataController
from teili.tools.visualizer.DataModels import StateVariablesModel
from teili.tools.visualizer.DataViewers import LineplotViewerMatplotlib, LineplotViewerPyqtgraph, PlotSettings


class Lineplot(DataController):
    """ Class to plot lineplots with different backends and from different
    DataModels - LineplotController """

    def __init__(
            self,
            DataModel_to_x_and_y_attr,
            MyPlotSettings=PlotSettings(),
            subgroup_labels=None,
            x_range=None,
            y_range=None,
            title='Lineplot',
            xlabel=None,
            ylabel=None,
            backend='matplotlib',
            mainfig=None,
            subfig=None,
            QtApp=None,
            show_immediately=False):
        """ Setup Lineplot Controller and create lineplot
        Args:
            DataModel_to_x_and_y_attr (list of tuples): list of tuples like
                [
                 (::class DataModel::,
                  (attr_of_DataModel_for_x_axis, ..._for_y_axis),
                 (::class DataModel::,
                  (attr_of_DataModel_to_consider_for_x_axis, ... _for_y_axis),
                  ...
                ]
                for all subgroups to be shown (data model can also be a brian
                state monitor or spike monitor)
            MyPlotSettings (PlotSettings object): instance of class
                PlotSettings holding basic plot settings (e.g. fontsize, ...)
            subgroup_labels (list of str): list of labels for the different
                subgroups (e.g. ['exc', 'inh'])
            x_range (tuple): (min, max) x-values of interval within which
                elements of data should be considered
            y_range (tuple): (min, max) y-values of interval within which
                elements of data should be considered
            title (str): title of plot
            xlabel (str): label of x-axis
            ylabel (str): label for y-axis
            backend (str): 'matplotlib' or 'pyqtgraph', defines which backend
                should be used for plotting
            mainfig (figure object): figure which holds the subfig (subplots)
                (plt.figure or  pg.GraphicsWindow())
            subfig (subplot): subplot of mainfig which will hold the histogram
            QtApp (pyqtgraph application): pyqtgraph application to run plots
                (QtGui.QApplication([])), only required if backend is pyqtgraph
            show_immediately (bool): if True: plot is shown immediately after
                it has been created
        """

        self.subgroup_labels = subgroup_labels
        self.x_range = x_range
        self.y_range = y_range
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        if backend == 'matplotlib':
            self.viewer = LineplotViewerMatplotlib(
                MyPlotSettings,
                mainfig=mainfig,
                subfig=subfig)
        elif backend == 'pyqtgraph':
            self.viewer = LineplotViewerPyqtgraph(
                MyPlotSettings,
                mainfig=mainfig,
                subfig=subfig,
                QtApp=QtApp)
        else:
            raise Exception(
                'You asked for the backend "{}" which is not supported'.format(backend))

        # prepare data for lineplot
        self._get_data_from_datamodels(DataModel_to_x_and_y_attr)
        self._filter_data()
        self.create_plot()
        # to allow easier access to main- and subfigure
        self.mainfig = self.viewer.mainfig
        self.subfig = self.viewer.subfig
        if show_immediately:
            self.show()

    def _get_data_from_datamodels(self, DataModel_to_x_and_y_attr):
        """ Get data from data model which will be shown along x and y axis of plot.

        Args:
            DataModel_to_x_and_y_attr (list of tuples): list of tuples like
                [(::class DataModel::,
                 (attr_of_DataModel_for_x_axis, ..._for_y_axis),
                (::class DataModel::,
                 (attr_of_DataModel_for_x_axis, ..._for_y_axis), ...]
                for all subgroups to be shown (data model can also be a brian
                state monitor or spike monitor)
        """

        self.data = []
        for data_model, x_y_attributes in DataModel_to_x_and_y_attr:
            x_data = np.asarray(getattr(data_model, x_y_attributes[0]))
            y_data = np.asarray(getattr(data_model, x_y_attributes[1]))

            # if data model is a brian state monitor, the array containing the variable values
            # has to be transposed to be consistent for the plotting
            # functionalities
            if not isinstance(data_model, StateVariablesModel):
                from brian2 import StateMonitor
                if isinstance(data_model, StateMonitor):
                    if x_y_attributes[0] != 't':
                        x_data = x_data.T
                    if x_y_attributes[1] != 't':
                        y_data = y_data.T

            self.data.append((x_data, y_data))

    def _filter_data(self):
        """ Filter data from self.data to be within self.x-range AND self.y_range. """

        if self.x_range is None and self.y_range is None:
            return

        if self.x_range is not None:
            for subgroup_nr, subgroup in enumerate(self.data):
                x_dim = len(subgroup[0].shape) - (subgroup[0].shape).count(1)
                y_dim = len(subgroup[1].shape) - (subgroup[1].shape).count(1)

                if x_dim != y_dim:
                    assert (x_dim == 1 or y_dim == 1), "Your data dimensions don't match, please adjust them." \
                                                   "(x: {}, y: {})".format(x_dim, y_dim)

                    assert (subgroup[0].shape[0] == subgroup[1].shape[0]), "Your data dimensions don't match, please adjust them." \
                                                   "(x: {}, y: {})".format(subgroup[0].shape, subgroup[1].shape)


                indices_within_x_range = np.where(
                    np.logical_and(
                        subgroup[0] >= self.x_range[0],
                        subgroup[0] <= self.x_range[1]))
                if len(indices_within_x_range[0]) == 0:
                    warnings.warn(
                        "For subgroup nr {} there are no datapoints left after filtering x_values to be within "
                        "range ({}, {})".format(
                            subgroup_nr, self.x_range[0], self.x_range[1]))

                self.data[subgroup_nr] = (
                    subgroup[0][indices_within_x_range[:x_dim]], subgroup[1][indices_within_x_range[:y_dim]])

        if self.y_range is not None:
            for subgroup_nr, subgroup in enumerate(self.data):
                x_dim = len(subgroup[0].shape) - (subgroup[0].shape).count(1)
                y_dim = len(subgroup[1].shape) - (subgroup[1].shape).count(1)
                if x_dim != y_dim:
                    assert (x_dim == 1 or y_dim == 1), "Your data dimensions don't match, please adjust them." \
                                                    " (x: {}, y: {})".format(x_dim, y_dim)

                indices_within_y_range = np.where(
                    np.logical_and(
                        subgroup[1] >= self.y_range[0],
                        subgroup[1] <= self.y_range[1]))
                if len(indices_within_y_range[0]) == 0:
                    warnings.warn(
                        "For subgroup_nr {} there are no datapoints left after filtering y_values to be within "
                        "range ({}, {})".format(
                            subgroup_nr, self.y_range[0], self.y_range[1]))
                self.data[subgroup_nr] = (
                    subgroup[0][indices_within_y_range[:x_dim]], subgroup[1][indices_within_y_range[:y_dim]])

    def create_plot(self):
        """ Function to create lineplot in subfigure with data from
            DataModel_to_attr with subgroups defined above"""
        self.viewer.create_plot(
            data=self.data,
            subgroup_labels=self.subgroup_labels,
            x_range_axis=self.x_range,
            y_range_axis=self.y_range,
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel)
