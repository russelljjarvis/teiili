import unittest

import matplotlib.pylab as plt

from teili.tools.visualizer.DataViewers import PlotSettings, HistogramViewerMatplotlib


SHOW_PLOTS_IN_TESTS = False


class TestHistogramViewerMatplotlib(unittest.TestCase):

    def test___init__(self):
        MyPlotSettings = PlotSettings()

        # without mainfig/subfig
        HV = HistogramViewerMatplotlib(
            MyPlotSettings=MyPlotSettings,
            mainfig=None,
            subfig=None)
        self.assertNotEqual(HV.mainfig, None)
        self.assertNotEqual(HV.subfig, None)

        # with mainfig/subfig
        mainfig = plt.figure()
        subfig = mainfig.add_subplot(111)
        HV = HistogramViewerMatplotlib(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig=subfig)
        self.assertNotEqual(HV.mainfig, None)
        self.assertNotEqual(HV.subfig, None)

        if SHOW_PLOTS_IN_TESTS:
            HV.show()
        plt.close('all')

    def test_createhistogram(self):
        MyPlotSettings = PlotSettings()

        lst1 = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5]
        lst2 = [1, 1, 1, 3, 3, 3, 5, 5, 5, 7, 7, 7, 9]
        data = [lst1, lst2]
        subgroup_labels = ['lst1', 'lst2']

        bins = range(11)
        orientation = 'vertical'

        # with mainfig/subfig
        mainfig = plt.figure()
        subfig = mainfig.add_subplot(111)
        HV = HistogramViewerMatplotlib(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig=subfig)
        HV.create_plot(
            data=data,
            subgroup_labels=subgroup_labels,
            bins=bins,
            orientation=orientation,
            title='histogram two groups',
            xlabel='bins',
            ylabel='count')

        if SHOW_PLOTS_IN_TESTS:
            HV.show()
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
