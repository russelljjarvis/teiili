import unittest

import numpy as np
import warnings


from utils_unittests import get_plotsettings, run_brian_network

from teili.tools.visualizer.DataControllers import Histogram
from teili.tools.visualizer.DataModels import EventsModel

try:
    import pyqtgraph as pg
    from PyQt5 import QtGui
    SKIP_PYQTGRAPH_RELATED_UNITTESTS = False
except BaseException:
    SKIP_PYQTGRAPH_RELATED_UNITTESTS = True

SHOW_PLOTS_IN_TESTS = False

class TestHistogram(unittest.TestCase):

    def test_getdata(self):

        (spikemonN1, spikemonN2) = run_brian_network(statemonitors=False)

        # from DataModels & EventModels
        EM1 = EventsModel.from_brian_spike_monitor(spikemonN1)
        EM2 = EventsModel.from_brian_spike_monitor(spikemonN2)
        DataModel_to_attr = [(EM1, 'neuron_ids'), (EM2, 'neuron_ids')]

        HC = Histogram(
            MyPlotSettings=get_plotsettings(alpha=0.4),
            DataModel_to_attr=DataModel_to_attr,
            show_immediately=SHOW_PLOTS_IN_TESTS)
        HC._get_data(DataModel_to_attr)
        self.assertEqual(len(HC.data), len(DataModel_to_attr))
        self.assertEqual(np.size(HC.data[0]), np.size(getattr(spikemonN1, 'i')))

        # from brian state monitor and spike monitors
        DataModel_to_attr = [
            (spikemonN1, 'i'),
            (spikemonN2, 'i')]

        HC = Histogram(
            MyPlotSettings=get_plotsettings(alpha=0.4),
            DataModel_to_attr=DataModel_to_attr,
            show_immediately=SHOW_PLOTS_IN_TESTS)
        HC._get_data(DataModel_to_attr)
        self.assertEqual(len(HC.data), len(DataModel_to_attr))
        self.assertEqual(
            np.size(
                HC.data[0]), np.size(
                getattr(
                    spikemonN1, 'i')))

    def test_createhistogram(self):
        # check backends
        spikemonN1, spikemonN2 = run_brian_network(statemonitors=False)

        EM1 = EventsModel.from_brian_spike_monitor(spikemonN1)
        EM2 = EventsModel.from_brian_spike_monitor(spikemonN2)
        DataModel_to_attr = [(EM1, 'neuron_ids'), (EM2, 'neuron_ids')]

        subgroup_labels = ['EM1', 'EM2']

        backend = 'matplotlib'
        HC = Histogram(
            MyPlotSettings=get_plotsettings(alpha=0.4),
            DataModel_to_attr=DataModel_to_attr,
            subgroup_labels=subgroup_labels,
            bins=None,
            orientation='vertical',
            title='histogram',
            xlabel='bins',
            ylabel='count',
            backend=backend,
            show_immediately=SHOW_PLOTS_IN_TESTS)
        HC.create_plot()

        if not SKIP_PYQTGRAPH_RELATED_UNITTESTS:
            backend = 'pyqtgraph'
            HC = Histogram(
                MyPlotSettings=get_plotsettings(max255=True, alpha=0.4),
                DataModel_to_attr=DataModel_to_attr,
                subgroup_labels=subgroup_labels,
                bins=None,
                orientation='vertical',
                title='histogram',
                xlabel='bins',
                ylabel='count',
                backend=backend,
                show_immediately=SHOW_PLOTS_IN_TESTS)
            HC.create_plot()

        else:
            warnings.warn("Skip part of unittest TestHistogram.test_createhistogram using pyqtgraph"
                                                       "as pyqtgraph could not be imported")


if __name__ == '__main__':
    unittest.main()