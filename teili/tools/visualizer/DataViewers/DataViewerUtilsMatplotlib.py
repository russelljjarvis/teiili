import matplotlib.pylab as plt

from teili.tools.visualizer.DataViewers.DataViewerUtils import DataViewerUtils

class DataViewerUtilsMatplotlib(DataViewerUtils):

    def __init__(self, mainfig):
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