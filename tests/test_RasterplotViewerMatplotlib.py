import unittest

import matplotlib.pylab as plt
import numpy as np

from teili.tools.visualizer.DataViewers import PlotSettings, RasterPlotViewerMatplotlib


SHOW_PLOTS_IN_TESTS = False


class TestRasterplotViewerMatplotlib(unittest.TestCase):

    def test___init__(self):
        MyPlotSettings = PlotSettings()

        # without mainfig/subfig
        RV = RasterPlotViewerMatplotlib(
            MyPlotSettings=MyPlotSettings,
            mainfig=None,
            subfig_rasterplot=None,
            subfig_histogram=None,
            add_histogram=False)
        self.assertNotEqual(RV.mainfig, None)
        self.assertNotEqual(RV.subfig_rasterplot, None)
        self.assertEqual(RV.subfig_histogram, None)

        # with mainfig/subfig but without histogram
        mainfig = plt.figure()
        subfig_rasterplot = mainfig.add_subplot(111)
        subfig_histogram = None
        RV = RasterPlotViewerMatplotlib(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig_rasterplot=subfig_rasterplot,
            subfig_histogram=subfig_histogram,
            add_histogram=False)
        self.assertNotEqual(RV.mainfig, None)
        self.assertNotEqual(RV.subfig_rasterplot, None)
        self.assertEqual(RV.subfig_histogram, None)

        # with mainfig/subfig and with histogram
        mainfig = plt.figure()
        subfig_rasterplot = plt.subplot2grid(
            (1, 3), (0, 0), rowspan=1, colspan=2, fig=mainfig)
        subfig_histogram = plt.subplot2grid(
            (1, 3), (0, 2), rowspan=1, sharey=subfig_rasterplot, frameon=0, fig=mainfig)
        plt.subplots_adjust(
            left=0.125,
            right=0.9,
            bottom=0.1,
            top=0.9,
            wspace=0.05,
            hspace=0.2)
        RV = RasterPlotViewerMatplotlib(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig_rasterplot=subfig_rasterplot,
            subfig_histogram=subfig_histogram,
            add_histogram=True)
        self.assertNotEqual(RV.mainfig, None)
        self.assertNotEqual(RV.subfig_rasterplot, None)
        self.assertNotEqual(RV.subfig_histogram, None)

        if SHOW_PLOTS_IN_TESTS:
            plt.show()
        plt.close('all')

    def test_createrasterplot(self):
        MyPlotSettings = PlotSettings()

        # basics
        all_spike_times = [np.arange(0, 1.4, 0.1)]
        all_neuron_ids = [[1, 2, 1, 1, 1, 1, 1, 4, 3, 2, 5, 3, 4, 7]]
        subgroup_labels = ['lst1']
        time_range_axis = None
        neuron_id_range_axis = None
        add_histogram = False

        mainfig = plt.figure()
        subfig_rasterplot = None
        subfig_histogram = None
        RV = RasterPlotViewerMatplotlib(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig_rasterplot=subfig_rasterplot,
            subfig_histogram=subfig_histogram,
            add_histogram=add_histogram)
        RV.create_plot(
            all_spike_times=all_spike_times,
            all_neuron_ids=all_neuron_ids,
            subgroup_labels=subgroup_labels,
            time_range_axis=time_range_axis,
            neuron_id_range_axis=neuron_id_range_axis,
            title='raster plot basic',
            xlabel='time (s)',
            ylabel='neuron ids')
        if SHOW_PLOTS_IN_TESTS:
            RV.show()
        plt.close('all')

        # craete two subgroups
        all_spike_times = [np.arange(0, 1, 0.1), np.arange(0, 1.4, 0.1)]
        all_neuron_ids = [[1, 1, 2, 1, 3, 3, 4, 1, 2, 4],
                          [1, 2, 1, 1, 1, 1, 1, 4, 3, 2, 5, 3, 4, 7]]
        subgroup_labels = ['lst1', 'lst2']

        time_range_axis = None
        neuron_id_range_axis = None
        add_histogram = False

        mainfig = plt.figure()
        subfig_rasterplot = None
        subfig_histogram = None
        RV = RasterPlotViewerMatplotlib(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig_rasterplot=subfig_rasterplot,
            subfig_histogram=subfig_histogram,
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
        plt.close('all')

        # introduce time_range and neuron_id_range
        all_spike_times = [np.arange(0, 1, 0.1), np.arange(0, 1.4, 0.1)]
        all_neuron_ids = [[1, 1, 2, 1, 3, 3, 4, 1, 2, 4],
                          [1, 2, 1, 1, 1, 1, 1, 4, 3, 2, 5, 3, 4, 7]]
        subgroup_labels = ['lst1', 'lst2']
        time_range_axis = (0, 0.9)
        neuron_id_range_axis = (0, 5)
        add_histogram = False

        mainfig = plt.figure()
        subfig_rasterplot = None
        subfig_histogram = None
        RV = RasterPlotViewerMatplotlib(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig_rasterplot=subfig_rasterplot,
            subfig_histogram=subfig_histogram,
            add_histogram=add_histogram)
        RV.create_plot(
            all_spike_times=all_spike_times,
            all_neuron_ids=all_neuron_ids,
            subgroup_labels=subgroup_labels,
            time_range_axis=time_range_axis,
            neuron_id_range_axis=neuron_id_range_axis,
            title='raster plot t[0, 9.0], id[0,5]',
            xlabel='time (s)',
            ylabel='neuron ids')
        if SHOW_PLOTS_IN_TESTS:
            RV.show()
        plt.close('all')

        # add histogram
        all_spike_times = [np.arange(0, 1, 0.1), np.arange(0, 1.4, 0.1)]
        all_neuron_ids = [[1, 1, 2, 1, 3, 3, 4, 1, 2, 4],
                          [1, 2, 1, 1, 1, 1, 1, 4, 3, 2, 5, 3, 4, 7]]
        subgroup_labels = ['lst1', 'lst2']
        time_range_axis = (0, 0.9)
        neuron_id_range_axis = (0, 5)
        add_histogram = True

        mainfig = plt.figure()
        subfig_rasterplot = None
        subfig_histogram = None
        RV = RasterPlotViewerMatplotlib(
            MyPlotSettings=MyPlotSettings,
            mainfig=mainfig,
            subfig_rasterplot=subfig_rasterplot,
            subfig_histogram=subfig_histogram,
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
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
