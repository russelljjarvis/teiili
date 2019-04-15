# SPDX-License-Identifier: MIT
# Copyright (c) 2018 University of Zurich

import numpy as np
import warnings

from teili.tools.visualizer.DataViewers.DataViewer import DataViewer


class HistogramViewer(DataViewer):
    """ Parent class of Histogram viewers with different backends (matplotlib, pyqtgraph)"""

    def __init__(self):
        pass

    def set_DataViewerUtils(self):
        """ Set which DataViewerUtils class should be considered"""
        super().set_DataViewerUtils()

    def create_plot(self):
        """ Method to create plot """
        super().create_plot()

    def _get_most_common_element(self, lst):
        """ Get element which occurs most often in lst
        Args:
            lst (list): list to be checked
        """

        if len(lst) == 0:
            return None
        else:
            lst = list(lst)
            return max(lst, key=lst.count)

    def get_highest_count(self, lst):
        """ Get highest number of occurrence of any element in lst
        Args:
            lst (list): list to be checked
        """

        if len(lst) == 0:
            return 0
        else:
            lst = list(lst)
            most_common_element = self._get_most_common_element(lst)
            return lst.count(most_common_element)

    def set_bins(self, data):
        ''' define bins used in histogram if not defined by user
         Args:
             data (list): list of data to define bins (for histogram) for
         '''

        max_per_dataset = []
        for x in data:
            if np.size(x) > 0:  # to avoid error by finding max of emtpy dataset
                max_per_dataset.append(np.nanmax(x))
            else:
                max_per_dataset.append(0)
        bins = range(int(max(max_per_dataset))+2)  # +2 to always have at least 1 bin
        return bins

    def remove_nans(self, subgroup):
        """ Method to remove nans from data
        Args:
            subgroup (array-like): data to filtered out nans
        """
        if (np.isnan(subgroup)).any():
            subgroup = subgroup[~np.isnan(subgroup)]
            warnings.warn("One of your subgroup contains NAN entries. They are removed and not shown in the histogram")
        return subgroup

