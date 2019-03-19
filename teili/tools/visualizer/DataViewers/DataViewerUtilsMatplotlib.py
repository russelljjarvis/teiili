import matplotlib.pylab as plt

from teili.tools.visualizer.DataViewers.DataViewerUtils import DataViewerUtils

class DataViewerUtilsMatplotlib(DataViewerUtils):
    """ Class holding matplotlib specific methods which
        are shared between different Viewers"""
    def __init__(self, mainfig):
        """ Set up DataViewerUtils for matplotlib backend
        Args:
            mainfig (matplotlib figure object): matplotlib figure which holds
                the subfig (subplots)
        """
        self.mainfig = mainfig

    def show(self):
        """ show plot """
        plt.show()

    def save(self,
             path_to_save='plot.png',
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

    def _set_title_and_labels(self, subfig, title, xlabel, ylabel,
                                fontsize_title, fontsize_axis_labels):
        """ Set title and label of x- and y-axis in plot subfig
        Args:
            subfig (matplotlib subfigure): subfigure to which title,
                                            x- & y-axis-label are added
            title (str): title of plot
            xlabel (str): label for x-axis
            ylabel (str): label for y-axis
            fontsize_title (int): fontsize for title
            fontsize_axis_labels(int)): fontsize for x-&y-axis-label
        """
        if title is not None:
            subfig.set_title(title, fontsize=fontsize_title)
        if xlabel is not None:
            subfig.set_xlabel(xlabel, fontsize=fontsize_axis_labels)
        if ylabel is not None:
            subfig.set_ylabel(ylabel, fontsize=fontsize_axis_labels)