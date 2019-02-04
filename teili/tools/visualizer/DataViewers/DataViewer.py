from teili.tools.visualizer.Freezer import Freezer


class PlotSettings(Freezer):
    """ Data structure to hold general plot settings. Derived from Freezer to
        ensure that the correct attribute names are used (e.g. avoid creating
        the same attribute twice but spelled differently) """

    def __init__(
        self,
        fontsize_title=16,
        fontsize_legend=14,
        fontsize_axis_labels=14,
        marker_size=5,
        colors=[
            'r',
            'b',
            'g',
            'c',
            'k',
            'm',
            'y']):
        """
        Args:
            fontsize_title (int): font size of plot title
            fontsize_legend (int): font size of legend in plot
            fontsize_axis_labels (int): font size of axis labels
            marker_size (int): size of markers shown in plot
            colors (list): list of str indicating colors (e.g. 'r', 'b', ...)
                OR list of RGBA tuples indicating colours for matplotlib
                backend ([0:1], for pyqtgraph backend [0:255])
        """
        self.fontsize_title = fontsize_title
        self.fontsize_legend = fontsize_legend
        self.fontsize_axis_labels = fontsize_axis_labels
        self.marker_size = marker_size
        self.colors = colors

        self._freeze()


class DataViewer(object):
    """ Parent class of all DataViewer classes """
    pass
