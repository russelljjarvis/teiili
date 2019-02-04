import numpy as np
import warnings
try:
    import pyqtgraph as pg
    from PyQt5 import QtGui
except BaseException:
    warnings.warn("No method using pyqtgraph can be used as pyqtgraph or PyQt5 can't be imported.")

from teili.tools.visualizer.DataViewers import LineplotViewer


class LineplotViewerPyqtgraph(LineplotViewer):
    """ Class to plot lineplot with pyqtgraph backend """

    def __init__(
            self,
            MyPlotSettings,
            mainfig=None,
            subfig=None,
            QtApp=None):
        """ Setup LineplotViewer by initializing main figure and subfigure. If any of them is set to None, it will be
        created internally.
        Args:
            MyPlotSettings (PlotSettings object): instance of class PlotSettings holding basic plot settings (e.g. fontsize, ...)
            mainfig (pyqtgraph window object): pyqtgraph main window ( pg.GraphicsWindow() )
            subfig (pyqtgraph subplot): pyqtgraph subplot of mainfig which will hold the lineplot
            QtApp (pyqtgraph application): pyqtgraph application to run plots ( QtGui.QApplication([]) )
        """

        self.MyPlotSettings = MyPlotSettings

        # QtApp
        self.QtApp = QtApp
        if not self.QtApp:
            self.QtApp = QtGui.QApplication([])

        # figure
        self.mainfig = mainfig
        if not self.mainfig:
            if subfig:
                pass  # TODO: !1
                # self.mainfig = subfig.figure get pyqt win
            else:
                self.mainfig = pg.GraphicsWindow()

        # subplot
        self.subfig = subfig
        if not self.subfig:
            self.subfig = self.mainfig.addPlot(row=1, column=1)

        pg.setConfigOptions(antialias=True)

    def create_lineplot(
            self,
            data,
            subgroup_labels=None,
            x_range_axis=None,
            y_range_axis=None,
            title='Lineplot',
            xlabel=None,
            ylabel=None):
        """ Function to create lineplot for groups of event sets in self.subfig with matplotlib
        Args:
            data (list of tuples): list of tuples, whereby each tuple is one subgroup which will be plotted,
                                        e.g.: [(x_values_A, y_values_A), (x_values_B, y_values_B), ... ]
            subgroup_labels (list of str): list of labels for the different subgroups (e.g. ['exc', 'inh'])
            x_range_axis (tuple): (min, max) of interval within which elements are shown along x-axis
            y_range_axis (tuple): (min, max) of interval within which elements are shown along y-axis
            title (str): title of plot
            xlabel (str): label of x-axis
            ylabel (str): label for y-axis
        """

        # check if num colors ok
        assert len(
            self.MyPlotSettings.colors) >= len(data), 'You have {} subgroups but only {} colors in your MyPlotSettings.colors'.format(
            len(data), len(
                self.MyPlotSettings.colors))

        if subgroup_labels is not None:
            self.subfig.addLegend()

        # set parameters on plot dimensions along x and y axis
        if x_range_axis is None:
            x_range_axis = (0, max(map(lambda x: np.nanmax(x[0]), data)))
        if y_range_axis is None:
            y_range_axis = (0, max(map(lambda x: np.nanmax(x[1]), data)))

        # lineplot
        for subgroup_nr, (subgroup, color) in enumerate(
                zip(data, self.MyPlotSettings.colors)):

            # color = np.asarray(pg.colorTuple(pg.mkColor(color)))
            if not isinstance(color, str):
                color = tuple(np.asarray(color))

            n_traces_x = int(np.size(subgroup[0]) / np.max((1, np.shape(subgroup[0])[0])))
            n_traces_y = int(np.size(subgroup[1]) / np.max((1, np.shape(subgroup[1])[0])))

            subgroup_x = np.reshape(
                subgroup[0], (np.shape(
                    subgroup[0])[0], n_traces_x))
            subgroup_y = np.reshape(
                subgroup[1], (np.shape(
                    subgroup[1])[0], n_traces_y))
            for nr_trace_x in range(n_traces_x):
                for nr_trace_y in range(n_traces_y):
                    self.subfig.plot(
                        x=subgroup_x[:, nr_trace_x],
                        y=subgroup_y[:, nr_trace_y],
                        pen=color)

            if subgroup_labels is not None:
                style = pg.PlotDataItem(pen=color)
                self.subfig.legend.addItem(style, subgroup_labels[subgroup_nr])

        if subgroup_labels is not None:
            legendStyle = {
                'color': '#FFF', 'size': str(
                    self.MyPlotSettings.fontsize_legend) + 'pt'}
            for item in self.subfig.legend.items:
                for single_item in item:
                    if isinstance(single_item,
                                  pg.graphicsItems.LabelItem.LabelItem):
                        single_item.setText(single_item.text, **legendStyle)

        self._set_title_and_labels(title=title, xlabel=xlabel, ylabel=ylabel)
        self.subfig.setRange(
            xRange=(
                x_range_axis[0], x_range_axis[1]), yRange=(
                y_range_axis[0], y_range_axis[1]))

    def _set_title_and_labels(self, title, xlabel, ylabel):
        """ Set title and label of x- and y-axis in plot
        Args:
            title (str): title of plot
            xlabel (str): label for x-axis
            ylabel (str): label for y-axis
        """
        if title is not None:
            titleStyle = {
                'color': '#FFF', 'size': str(
                    self.MyPlotSettings.fontsize_title) + 'pt'}
            self.subfig.setTitle(title, **titleStyle)

        labelStyle = {'color': '#FFF',
                      'font-size': str(self.MyPlotSettings.fontsize_axis_labels) + 'pt'}
        if xlabel is not None:
            self.subfig.setLabel('bottom', xlabel, **labelStyle)
        if ylabel is not None:
            self.subfig.setLabel('left', ylabel, **labelStyle)
        self.subfig.getAxis('bottom').tickFont = QtGui.QFont(
            'arial', self.MyPlotSettings.fontsize_axis_labels)
        self.subfig.getAxis('left').tickFont = QtGui.QFont(
            'arial', self.MyPlotSettings.fontsize_axis_labels)

    def show_lineplot(self):
        """ show plot """
        self.QtApp.exec_()

    def save_lineplot(
            self,
            path_to_save='lineplot.svg',
            figure_size=None):
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
