import warnings
try:
    import pyqtgraph as pg
    from PyQt5 import QtGui
except BaseException:
    warnings.warn("No method using pyqtgraph can be used as pyqtgraph or PyQt5"
                  "can't be imported.")

from teili.tools.visualizer.DataViewers.DataViewerUtils import DataViewerUtils


class DataViewerUtilsPyqtgraph(DataViewerUtils):

    def __init__(self, QtApp, mainfig):
        self.QtApp = QtApp
        self.mainfig = mainfig

    def show(self):
        """ show plot """
        self.QtApp.exec_()

    def save(self, path_to_save='plot.png', figure_size=None):
        """ Save figure to path_to_save with size figure_size as svg, png, jpg and tiff
        Args:
            path_to_save (str): path to location where to save figure incl filename
            figure_size (tuple): tuple of width and height of figure to save
        """
        self.QtApp.processEvents()
        if figure_size is not None:
            self.mainfig.resize(figure_size[0], figure_size[1])

        if path_to_save.split('.')[-1] == 'svg':
            ex = pg.exporters.SVGExporter(self.mainfig.scene())
        else:
            ex = pg.exporters.ImageExporter(self.mainfig.scene())

        ex.export(fileName=path_to_save)
        print('Figure saved to: ' + path_to_save)