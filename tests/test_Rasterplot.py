import unittest

import numpy as np
import warnings

from utils_unittests import run_brian_network, get_plotsettings

from teili.tools.visualizer.DataControllers import Rasterplot
from teili.tools.visualizer.DataModels import EventsModel

try:
    import pyqtgraph as pg
    from PyQt5 import QtGui
    SKIP_PYQTGRAPH_RELATED_UNITTESTS = False
except BaseException:
    SKIP_PYQTGRAPH_RELATED_UNITTESTS = True

SHOW_PLOTS_IN_TESTS = False

class TestRasterplot(unittest.TestCase):

    def test_getalldatafromeventmodels(self):

        (spikemonN1, spikemonN2) = run_brian_network(statemonitors=False)

        # from DataModels & EventModels
        EM1 = EventsModel.from_brian_spike_monitor(spikemonN1)
        EM2 = EventsModel.from_brian_spike_monitor(spikemonN2)

        RC = Rasterplot(
            MyPlotSettings=get_plotsettings(),
            MyEventsModels=[
                EM1,
                EM2],
            show_immediately=SHOW_PLOTS_IN_TESTS)
        RC._get_data_from_eventsmodels()
        self.assertEqual(len(RC.all_neuron_ids), 2)
        self.assertEqual(len(RC.all_neuron_ids[0]), len(spikemonN1.i))
        self.assertEqual(len(RC.all_spike_times[0]), len(
            np.asarray(spikemonN1.t)))
        self.assertEqual(len(RC.all_neuron_ids[1]), len(spikemonN2.i))
        self.assertEqual(len(RC.all_spike_times[1]), len(
            np.asarray(spikemonN2.t)))

        # from brian state monitor and spike monitors
        RC = Rasterplot(
            MyPlotSettings=get_plotsettings(),
            MyEventsModels=[
                spikemonN1,
                spikemonN2],
            show_immediately=SHOW_PLOTS_IN_TESTS)
        RC._get_data_from_eventsmodels()
        self.assertEqual(len(RC.all_neuron_ids), 2)
        self.assertEqual(len(RC.all_neuron_ids[0]), len(spikemonN1.i))
        self.assertEqual(len(RC.all_spike_times[0]), len(
            np.asarray(spikemonN1.t)))
        self.assertEqual(len(RC.all_neuron_ids[1]), len(spikemonN2.i))
        self.assertEqual(len(RC.all_spike_times[1]), len(
            np.asarray(spikemonN2.t)))

    def test_filterdatatobewithinranges(self):

        (spikemonN1, spikemonN2) = run_brian_network(statemonitors=False)

        subgroup_labels = ['N1', 'N2']
        time_range = (0, 0.05)
        neuron_id_range = (0, 5)

        # from DataModels & EventModels
        EM1 = EventsModel.from_brian_spike_monitor(spikemonN1)
        EM2 = EventsModel.from_brian_spike_monitor(spikemonN2)

        RC = Rasterplot(
            MyPlotSettings=get_plotsettings(),
            MyEventsModels=[
                EM1,
                EM2],
            subgroup_labels=subgroup_labels,
            time_range=time_range,
            neuron_id_range=neuron_id_range,
            show_immediately=SHOW_PLOTS_IN_TESTS)
        RC._filter_data()
        self.assertLessEqual(max(RC.all_neuron_ids[0], default=0), neuron_id_range[1])
        self.assertLessEqual(max(RC.all_spike_times[0], default=0), time_range[1])

        # from brian state monitor and spike monitors
        RC = Rasterplot(
            MyPlotSettings=get_plotsettings(),
            MyEventsModels=[
                EM1,
                EM2],
            subgroup_labels=subgroup_labels,
            time_range=time_range,
            neuron_id_range=neuron_id_range,
            show_immediately=SHOW_PLOTS_IN_TESTS)
        RC._filter_data()
        self.assertLessEqual(max(RC.all_neuron_ids[0], default=0), neuron_id_range[1])
        self.assertLessEqual(max(RC.all_spike_times[0], default=0), time_range[1])

    def test_createrasterplot(self):

        (spikemonN1, spikemonN2) = run_brian_network(statemonitors=False)

        MyEventsModels = [spikemonN1, spikemonN2]
        subgroup_labels = ['N1', 'N2']
        time_range = (0, 0.05)
        neuron_id_range = (0, 5)

        # matplotlib backend, no histogram
        backend = 'matplotlib'
        add_histogram = False
        RC = Rasterplot(
            MyPlotSettings=get_plotsettings(),
            MyEventsModels=MyEventsModels,
            subgroup_labels=subgroup_labels,
            time_range=time_range,
            neuron_id_range=neuron_id_range,
            title='raster plot, no hist, matplotlib',
            xlabel='time',
            ylabel='count',
            backend=backend,
            add_histogram=add_histogram,
            show_immediately=SHOW_PLOTS_IN_TESTS)

        # matplotlib backend, with histogram
        backend = 'matplotlib'
        add_histogram = True
        RC = Rasterplot(
            MyPlotSettings=get_plotsettings(),
            MyEventsModels=MyEventsModels,
            subgroup_labels=subgroup_labels,
            time_range=time_range,
            neuron_id_range=neuron_id_range,
            title='raster plot, with hist, matplotlib',
            xlabel='time',
            ylabel='count',
            backend=backend,
            add_histogram=add_histogram,
            show_immediately=SHOW_PLOTS_IN_TESTS)

        if not SKIP_PYQTGRAPH_RELATED_UNITTESTS:
            # pyqtgraph backend, no histogram
            backend = 'pyqtgraph'
            add_histogram = False
            RC = Rasterplot(
                MyPlotSettings=get_plotsettings(),
                MyEventsModels=MyEventsModels,
                subgroup_labels=subgroup_labels,
                time_range=time_range,
                neuron_id_range=neuron_id_range,
                title='raster plot, no hist, pyqtgraph',
                xlabel='time',
                ylabel='count',
                backend=backend,
                add_histogram=add_histogram,
                show_immediately=SHOW_PLOTS_IN_TESTS)

        if not SKIP_PYQTGRAPH_RELATED_UNITTESTS:
            # pyqtgraph backend, with histogram
            backend = 'pyqtgraph'
            add_histogram = True
            RC = Rasterplot(
                MyPlotSettings=get_plotsettings(),
                MyEventsModels=MyEventsModels,
                subgroup_labels=subgroup_labels,
                time_range=time_range,
                neuron_id_range=neuron_id_range,
                title='raster plot, with hist, pyqtgraph',
                xlabel='time',
                ylabel='count',
                backend=backend,
                add_histogram=add_histogram,
                show_immediately=SHOW_PLOTS_IN_TESTS)

        time_range = (0.02, 0.05)
        neuron_id_range = (0, 5)
        # matplotlib backend, when no/empty data provided
        backend = 'matplotlib'
        add_histogram = True
        RC = Rasterplot(
            MyPlotSettings=get_plotsettings(),
            MyEventsModels=MyEventsModels,
            subgroup_labels=subgroup_labels,
            time_range=time_range,
            neuron_id_range=neuron_id_range,
            title='empty raster plot, with hist, matplotlib',
            xlabel='time',
            ylabel='count',
            backend=backend,
            add_histogram=add_histogram,
            show_immediately=SHOW_PLOTS_IN_TESTS)

        if not SKIP_PYQTGRAPH_RELATED_UNITTESTS:
            # pyqtgraph backend, when no/empty data provided
            backend = 'pyqtgraph'
            add_histogram = True
            RC = Rasterplot(
                MyPlotSettings=get_plotsettings(),
                MyEventsModels=MyEventsModels,
                subgroup_labels=subgroup_labels,
                time_range=time_range,
                neuron_id_range=neuron_id_range,
                title='empty raster plot, with hist, pyqtgraph',
                xlabel='time',
                ylabel='count',
                backend=backend,
                add_histogram=add_histogram,
                show_immediately=SHOW_PLOTS_IN_TESTS)
        else:
            warnings.warn("Skip part of unittest TestRasterplot.test_createrasterplot using pyqtgraph"
                          "as pyqtgraph could not be imported")

if __name__ == '__main__':
    unittest.main()
