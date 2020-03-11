import unittest

import numpy as np

from teili.tools.visualizer.DataViewers import PlotSettings, RasterplotViewerPyqtgraph

SHOW_PLOTS_IN_TESTS = False

try:
    import pyqtgraph as pg
    from PyQt5 import QtGui
    QtApp = QtGui.QApplication([])
    SKIP_PYQTGRAPH_RELATED_UNITTESTS = False
except BaseException:
    SKIP_PYQTGRAPH_RELATED_UNITTESTS = True

class TestRasterplotViewerPyqtgraph(unittest.TestCase):
    @unittest.skipIf(SKIP_PYQTGRAPH_RELATED_UNITTESTS,
                     "Skip unittest TestRasterplotViewerPyqtgraph.test___init__ using pyqtgraph"
                     "as pyqtgraph could not be imported")
    def test___init__(self):
        MyPlotSettings = PlotSettings()

        # without mainfig/subfig
        RV = RasterplotViewerPyqtgraph(
            MyPlotSettings=MyPlotSettings,
            mainfig=None,
            subfig_rasterplot=None,
            subfig_histogram=None,
            QtApp=QtApp,
            add_histogram=False)
        self.assertNotEqual(RV.mainfig, None)
        self.assertNotEqual(RV.subfig_rasterplot, None)
        self.assertEqual(RV.subfig_histogram, None)

        if SHOW_PLOTS_IN_TESTS:
            RV.show()

        # with mainfig/subfig but without histogram
        mainfig = pg.GraphicsWindow()
        subfig_rasterplot = mainfig.addPlot(row=1, column=1)
        subfig_histogram = None
        RV = RasterplotViewerPyqtgraph(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig_rasterplot=subfig_rasterplot,
            subfig_histogram=subfig_histogram,
            QtApp=QtApp,
            add_histogram=False)
        self.assertNotEqual(RV.mainfig, None)
        self.assertNotEqual(RV.subfig_rasterplot, None)
        self.assertEqual(RV.subfig_histogram, None)

        if SHOW_PLOTS_IN_TESTS:
            RV.show()

        # with mainfig/subfig and with histogram
        mainfig = pg.GraphicsWindow()
        subfig_rasterplot = mainfig.addPlot(row=1, column=1)
        subfig_histogram = mainfig.addPlot(row=1, column=1)
        RV = RasterplotViewerPyqtgraph(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig_rasterplot=subfig_rasterplot,
            subfig_histogram=subfig_histogram,
            QtApp=QtApp,
            add_histogram=True)
        self.assertNotEqual(RV.mainfig, None)
        self.assertNotEqual(RV.subfig_rasterplot, None)
        self.assertNotEqual(RV.subfig_histogram, None)

        if SHOW_PLOTS_IN_TESTS:
            RV.show()

    @unittest.skipIf(SKIP_PYQTGRAPH_RELATED_UNITTESTS,
                     "Skip unittest TestRasterplotViewerPyqtgraph.test_create_rasterplot using pyqtgraph"
                     "as pyqtgraph could not be imported")
    def test_createrasterplot(self):
        MyPlotSettings = PlotSettings()

        # basics
        all_spike_times = [np.arange(0, 1.4, 0.1)]
        all_neuron_ids = [[1, 2, 1, 1, 1, 1, 1, 4, 3, 2, 5, 3, 4, 7]]
        subgroup_labels = ['lst1']
        time_range_axis = None
        neuron_id_range_axis = None
        add_histogram = False

        mainfig = pg.GraphicsWindow()
        subfig_rasterplot = mainfig.addPlot(row=1, column=1)
        subfig_histogram = None
        RV = RasterplotViewerPyqtgraph(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig_rasterplot=subfig_rasterplot,
            subfig_histogram=subfig_histogram,
            QtApp=QtApp,
            add_histogram=add_histogram)
        RV.create_plot(
            all_spike_times=all_spike_times,
            all_neuron_ids=all_neuron_ids,
            subgroup_labels=subgroup_labels,
            time_range_axis=time_range_axis,
            neuron_id_range_axis=neuron_id_range_axis,
            title='raster plot basics',
            xlabel='time (s)',
            ylabel='neuron ids')
        if SHOW_PLOTS_IN_TESTS:
            RV.show()

        # create two subgroups
        all_spike_times = [np.arange(0, 1, 0.1), np.arange(0, 1.4, 0.1)]
        all_neuron_ids = [[1, 1, 2, 1, 3, 3, 4, 1, 2, 4],
                          [1, 2, 1, 1, 1, 1, 1, 4, 3, 2, 5, 3, 4, 7]]
        subgroup_labels = ['lst1', 'lst2']
        time_range_axis = None
        neuron_id_range_axis = None
        add_histogram = False

        mainfig = pg.GraphicsWindow()
        subfig_rasterplot = mainfig.addPlot(row=1, column=1)
        subfig_histogram = None
        RV = RasterplotViewerPyqtgraph(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig_rasterplot=subfig_rasterplot,
            subfig_histogram=subfig_histogram,
            QtApp=QtApp,
            add_histogram=add_histogram)
        RV.create_plot(
            all_spike_times=all_spike_times,
            all_neuron_ids=all_neuron_ids,
            subgroup_labels=subgroup_labels,
            time_range_axis=time_range_axis,
            neuron_id_range_axis=neuron_id_range_axis,
            title='raster plot two subgroups',
            xlabel='time (s)',
            ylabel='neuron ids')
        if SHOW_PLOTS_IN_TESTS:
            RV.show()

        # introduce time_range_axis and neuron_id_range_axis
        all_spike_times = [np.arange(0, 1, 0.1), np.arange(0, 1.4, 0.1)]
        all_neuron_ids = [[1, 1, 2, 1, 3, 3, 4, 1, 2, 4],
                          [1, 2, 1, 1, 1, 1, 1, 4, 3, 2, 5, 3, 4, 7]]
        subgroup_labels = ['lst1', 'lst2']
        time_range_axis = (0, 0.9)
        neuron_id_range_axis = (0, 5)
        add_histogram = False

        mainfig = pg.GraphicsWindow()
        subfig_rasterplot = mainfig.addPlot(row=1, column=1)
        subfig_histogram = None
        RV = RasterplotViewerPyqtgraph(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig_rasterplot=subfig_rasterplot,
            subfig_histogram=subfig_histogram,
            QtApp=QtApp,
            add_histogram=add_histogram)
        RV.create_plot(
            all_spike_times=all_spike_times,
            all_neuron_ids=all_neuron_ids,
            subgroup_labels=subgroup_labels,
            time_range_axis=time_range_axis,
            neuron_id_range_axis=neuron_id_range_axis,
            title='raster plot t[0, 0.9] i[0,5]',
            xlabel='time (s)',
            ylabel='neuron ids')
        if SHOW_PLOTS_IN_TESTS:
            RV.show()

        # add histogram
        all_spike_times = [np.arange(0, 1, 0.1), np.arange(0, 1.4, 0.1)]
        all_neuron_ids = [[1, 1, 2, 1, 3, 3, 4, 1, 2, 4],
                          [1, 2, 1, 1, 1, 1, 1, 4, 3, 2, 5, 3, 4, 7]]
        subgroup_labels = ['lst1', 'lst2']
        time_range_axis = (0, 0.9)
        neuron_id_range_axis = (0, 5)
        add_histogram = True

        mainfig = pg.GraphicsWindow()
        subfig_rasterplot = mainfig.addPlot(row=1, column=1)
        subfig_histogram = mainfig.addPlot(row=1, column=1)
        RV = RasterplotViewerPyqtgraph(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig_rasterplot=subfig_rasterplot,
            subfig_histogram=subfig_histogram,
            QtApp=QtApp,
            add_histogram=add_histogram)
        RV.create_plot(
            all_spike_times=all_spike_times,
            all_neuron_ids=all_neuron_ids,
            subgroup_labels=subgroup_labels,
            time_range_axis=time_range_axis,
            neuron_id_range_axis=neuron_id_range_axis,
            title='raster plot, add histogram',
            xlabel='time (s)',
            ylabel='neuron ids')
        if SHOW_PLOTS_IN_TESTS:
            RV.show()


if __name__ == '__main__':
    unittest.main()
