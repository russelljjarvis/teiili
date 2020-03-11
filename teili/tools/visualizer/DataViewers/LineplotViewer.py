# SPDX-License-Identifier: MIT
# Copyright (c) 2018 University of Zurich

from teili.tools.visualizer.DataViewers import DataViewer


class LineplotViewer(DataViewer):
    """ Parent class of Lineplot viewers with different backends (matplotlib, pyqtgraph)"""
    def set_DataViewerUtils(self):
        """ Set which DataViewerUtils class should be considered"""
        super().set_DataViewerUtils()

    def create_plot(self):
        """ Method to create plot """
        super().create_plot()