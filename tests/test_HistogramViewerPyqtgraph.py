import unittest

import pyqtgraph as pg
from PyQt5 import QtGui

from teili.tools.visualizer.DataViewers import PlotSettings, HistogramViewerPyqtgraph


SHOW_PLOTS_IN_TESTS = False
QtApp = QtGui.QApplication([])


class TestHistogramViewerPyqtgraph(unittest.TestCase):

    def test___init__(self):
        MyPlotSettings = PlotSettings()

        # without mainfig/subfig
        HV = HistogramViewerPyqtgraph(
            MyPlotSettings=MyPlotSettings,
            mainfig=None,
            subfig=None,
            QtApp=QtApp)
        self.assertNotEqual(HV.mainfig, None)
        self.assertNotEqual(HV.subfig, None)

        # with mainfig/subfig
        mainfig = pg.GraphicsWindow()
        subfig = mainfig.addPlot(row=1, column=1)
        HV = HistogramViewerPyqtgraph(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig=subfig,
            QtApp=QtApp)
        self.assertNotEqual(HV.mainfig, None)

        if SHOW_PLOTS_IN_TESTS:
            HV.show()

    def test_create_histogram(self):
        MyPlotSettings = PlotSettings()

        lst1 = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5]
        lst2 = [1, 1, 1, 3, 3, 3, 5, 5, 5, 7, 7, 7, 9]
        data = [lst1, lst2]
        subgroup_labels = ['lst1', 'lst2']

        bins = range(11)
        orientation = 'vertical'

        # with mainfig/subfig
        mainfig = pg.GraphicsWindow()
        subfig = mainfig.addPlot(row=1, column=1)
        HV = HistogramViewerPyqtgraph(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig=subfig,
            QtApp=QtApp)
        HV.create_plot(
            data=data,
            subgroup_labels=subgroup_labels,
            bins=bins,
            orientation=orientation,
            title='histogram two subgroups',
            xlabel='bins',
            ylabel='count')

        if SHOW_PLOTS_IN_TESTS:
            HV.show()


if __name__ == '__main__':
    unittest.main()
