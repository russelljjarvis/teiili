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

    def _set_title_and_labels(self, subfig, title, xlabel, ylabel,
                              fontsize_title, fontsize_axis_labels):
        """ Set title and label of x- and y-axis in plot
        Args:
            subfig (pyqtgraph subplot): subfigure to which title,
                                            x- & y-axis-label are added
            title (str): title of plot
            xlabel (str): label for x-axis
            ylabel (str): label for y-axis
            fontsize_title (int): fontsize for title
            fontsize_axis_labels(int)): fontsize for x-&y-axis-label
        """
        if title is not None:
            titleStyle = {'color': '#FFF', 'size': str(fontsize_title) + 'pt'}
            subfig.setTitle(title, **titleStyle)

        labelStyle = {'color': '#FFF', 'font-size': str(fontsize_axis_labels) + 'pt'}
        if xlabel is not None:
            subfig.setLabel('bottom', xlabel, **labelStyle)
        if ylabel is not None:
            subfig.setLabel('left', ylabel, **labelStyle)
        subfig.getAxis('bottom').tickFont = QtGui.QFont('arial', fontsize_axis_labels)
        subfig.getAxis('left').tickFont = QtGui.QFont('arial', fontsize_axis_labels)