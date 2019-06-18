import unittest

import numpy as np

from teili.tools.visualizer.DataViewers import PlotSettings, LineplotViewerPyqtgraph

SHOW_PLOTS_IN_TESTS = False

try:
    from PyQt5 import QtGui
    import pyqtgraph as pg
    QtApp = QtGui.QApplication([])
    SKIP_PYQTGRAPH_RELATED_UNITTESTS = False
except BaseException:
    SKIP_PYQTGRAPH_RELATED_UNITTESTS = True

class TestLineplotViewerPyqtgraph(unittest.TestCase):
    @unittest.skipIf(SKIP_PYQTGRAPH_RELATED_UNITTESTS,
                     "Skip unittest TestLineplotViewerPyqtgraph.test___init__ using pyqtgraph"
                     "as pyqtgraph could not be imported")
    def test___init__(self):
        MyPlotSettings = PlotSettings()

        # without mainfig/subfig
        LV = LineplotViewerPyqtgraph(
            MyPlotSettings=MyPlotSettings,
            mainfig=None,
            subfig=None,
            QtApp=QtApp)
        self.assertNotEqual(LV.mainfig, None)
        self.assertNotEqual(LV.subfig, None)

        if SHOW_PLOTS_IN_TESTS:
            LV.show()

        # with mainfig/subfig
        mainfig = pg.GraphicsWindow()
        subfig = mainfig.addPlot(row=1, column=1)
        LV = LineplotViewerPyqtgraph(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig=subfig,
            QtApp=QtApp)
        self.assertNotEqual(LV.mainfig, None)
        self.assertNotEqual(LV.subfig, None)

        if SHOW_PLOTS_IN_TESTS:
            LV.show()

    @unittest.skipIf(SKIP_PYQTGRAPH_RELATED_UNITTESTS,
                     "Skip unittest TestLineplotViewerPyqtgraph.test_create_lineplot using pyqtgraph"
                     "as pyqtgraph could not be imported")
    def test_createlineplot(self):
        MyPlotSettings = PlotSettings()

        # basics
        data_x_axis = [np.arange(0, 1.4, 0.1)]
        data_y_axis = [[1, 2, 1, 1, 1, 1, 1, 4, 3, 2, 5, 3, 4, 7]]
        data = [(data_x_axis[0], data_y_axis[0])]
        subgroup_labels = ['lst1']
        x_range_axis, y_range_axis = None, None

        mainfig = pg.GraphicsWindow()
        subfig = mainfig.addPlot(row=1, column=1)

        LV = LineplotViewerPyqtgraph(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig=subfig,
            QtApp=QtApp)
        LV.create_plot(
            data=data,
            subgroup_labels=subgroup_labels,
            x_range_axis=x_range_axis,
            y_range_axis=y_range_axis,
            title='Lineplot',
            xlabel='x-axis-label',
            ylabel='y-axis-label')

        if SHOW_PLOTS_IN_TESTS:
            LV.show()

        # create two subgroups
        data_x_axis = [np.arange(0, 1, 0.1), np.arange(0, 1.4, 0.1)]
        data_y_axis = [[1, 1, 2, 1, 3, 3, 4, 1, 2, 4],
                       [1, 2, 1, 1, 1, 1, 1, 4, 3, 2, 5, 3, 4, 7]]
        data = [(data_x_axis[0], data_y_axis[0]),
                (data_x_axis[1], data_y_axis[1])]
        subgroup_labels = ['lst1', 'lst2']
        x_range_axis, y_range_axis = None, None

        mainfig = pg.GraphicsWindow()
        subfig = mainfig.addPlot(row=1, column=1)
        LV = LineplotViewerPyqtgraph(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig=subfig,
            QtApp=QtApp)
        LV.create_plot(
            data=data,
            subgroup_labels=subgroup_labels,
            x_range_axis=x_range_axis,
            y_range_axis=y_range_axis,
            title='Lineplot',
            xlabel='x-axis-label',
            ylabel='y-axis-label')

        if SHOW_PLOTS_IN_TESTS:
            LV.show()

        # introduce x_range_axis and y_range_axis
        data_x_axis = [np.arange(0, 1, 0.1), np.arange(0, 1.4, 0.1)]
        data_y_axis = [[1, 1, 2, 1, 3, 3, 4, 1, 2, 4],
                       [1, 2, 1, 1, 1, 1, 1, 4, 3, 2, 5, 3, 4, 7]]
        data = [(data_x_axis[0], data_y_axis[0]),
                (data_x_axis[1], data_y_axis[1])]
        subgroup_labels = ['lst1', 'lst2']

        x_range_axis = (0, 1.4)
        y_range_axis = (0, 10)

        mainfig = pg.GraphicsWindow()
        subfig = mainfig.addPlot(row=1, column=1)
        LV = LineplotViewerPyqtgraph(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig=subfig,
            QtApp=QtApp)
        LV.create_plot(
            data=data,
            subgroup_labels=subgroup_labels,
            x_range_axis=x_range_axis,
            y_range_axis=y_range_axis,
            title='Lineplot',
            xlabel='x-axis-label',
            ylabel='y-axis-label')

        if SHOW_PLOTS_IN_TESTS:
            LV.show()


if __name__ == '__main__':
    unittest.main()
