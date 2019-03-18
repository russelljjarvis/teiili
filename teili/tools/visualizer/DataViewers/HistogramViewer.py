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
