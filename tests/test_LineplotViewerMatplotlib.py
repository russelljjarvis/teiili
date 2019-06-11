import unittest

import matplotlib.pylab as plt
import numpy as np

from teili.tools.visualizer.DataViewers import PlotSettings, LineplotViewerMatplotlib


SHOW_PLOTS_IN_TESTS = False


class TestLineplotViewerMatplotlib(unittest.TestCase):

    def test___init__(self):
        MyPlotSettings = PlotSettings()

        # without mainfig/subfig
        LV = LineplotViewerMatplotlib(
            MyPlotSettings=MyPlotSettings,
            mainfig=None,
            subfig=None)
        self.assertNotEqual(LV.mainfig, None)
        self.assertNotEqual(LV.subfig, None)

        # with mainfig/subfig
        mainfig = plt.figure()
        subfig = plt.subplot2grid(
            (1, 1), (0, 0), rowspan=1, colspan=1, fig=mainfig)
        LV = LineplotViewerMatplotlib(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig=subfig)
        self.assertNotEqual(LV.mainfig, None)
        self.assertNotEqual(LV.subfig, None)

        if SHOW_PLOTS_IN_TESTS:
            LV.show()
        plt.close('all')

    def test_createlineplot(self):
        MyPlotSettings = PlotSettings()

        # basics
        data_x_axis = [np.arange(0, 1.4, 0.1)]
        data_y_axis = [[1, 2, 1, 1, 1, 1, 1, 4, 3, 2, 5, 3, 4, 7]]
        data = [(data_x_axis[0], data_y_axis[0])]
        subgroup_labels = ['lst1']
        x_range_axis, y_range_axis = None, None

        mainfig = plt.figure()
        subfig = None
        LV = LineplotViewerMatplotlib(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig=subfig)
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
        plt.close('all')

        # create two subgroups
        data_x_axis = [np.arange(0, 1, 0.1), np.arange(0, 1.4, 0.1)]
        data_y_axis = [[1, 1, 2, 1, 3, 3, 4, 1, 2, 4],
                       [1, 2, 1, 1, 1, 1, 1, 4, 3, 2, 5, 3, 4, 7]]
        data = [(data_x_axis[0], data_y_axis[0]),
                (data_x_axis[1], data_y_axis[1])]
        subgroup_labels = ['lst1', 'lst2']
        x_range_axis, y_range_axis = None, None

        mainfig = plt.figure()
        subfig = None
        LV = LineplotViewerMatplotlib(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig=subfig)
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
        plt.close('all')

        # introduce x_range_axis and y_range_axis
        data_x_axis = [np.arange(0, 1, 0.1), np.arange(0, 1.4, 0.1)]
        data_y_axis = [[1, 1, 2, 1, 3, 3, 4, 1, 2, 4],
                       [1, 2, 1, 1, 1, 1, 1, 4, 3, 2, 5, 3, 4, 7]]
        data = [(data_x_axis[0], data_y_axis[0]),
                (data_x_axis[1], data_y_axis[1])]
        subgroup_labels = ['lst1', 'lst2']

        x_range_axis = (0, 1.4)
        y_range_axis = (0, 10)

        mainfig = plt.figure()
        subfig = None
        LV = LineplotViewerMatplotlib(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig=subfig)
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
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
