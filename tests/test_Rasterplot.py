import unittest

from brian2 import us, ms, prefs, defaultclock, start_scope, SpikeGeneratorGroup, SpikeMonitor, StateMonitor
import numpy as np
import warnings

from teili.tools.visualizer.DataControllers import Rasterplot
from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataModels import EventsModel
from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import DPI
from teili.models.synapse_models import DPISyn
from teili.models.parameters.dpi_neuron_param import parameters as neuron_model_param

try:
    import pyqtgraph as pg
    from PyQt5 import QtGui
    QtApp = QtGui.QApplication([])
    SKIP_PYQTGRAPH_RELATED_UNITTESTS = False
except BaseException:
    QtApp = None
    SKIP_PYQTGRAPH_RELATED_UNITTESTS = True

def run_brian_network():
    prefs.codegen.target = "numpy"
    defaultclock.dt = 10 * us

    start_scope()
    N_input, N_N1, N_N2 = 1, 5, 3
    duration_sim = 200

    Net = TeiliNetwork()
    # setup spike generator
    spikegen_spike_times = np.sort(
        np.random.choice(
            size=30,
            a=range(
                0,
                duration_sim,
                5),
            replace=False)) * ms
    spikegen_neuron_ids = np.zeros_like(spikegen_spike_times) / ms
    gInpGroup = SpikeGeneratorGroup(
        N_input,
        indices=spikegen_neuron_ids,
        times=spikegen_spike_times,
        name='gtestInp')
    # setup neurons
    testNeurons1 = Neurons(
        N_N1, equation_builder=DPI(
            num_inputs=2), name="testNeuron")
    testNeurons1.set_params(neuron_model_param)
    testNeurons2 = Neurons(
        N_N2, equation_builder=DPI(
            num_inputs=2), name="testNeuron2")
    testNeurons2.set_params(neuron_model_param)
    # setup connections
    InpSyn = Connections(
        gInpGroup,
        testNeurons1,
        equation_builder=DPISyn(),
        name="testSyn",
        verbose=False)
    InpSyn.connect(True)
    InpSyn.weight = '100 + rand() * 50'
    Syn = Connections(
        testNeurons1,
        testNeurons2,
        equation_builder=DPISyn(),
        name="testSyn2")
    Syn.connect(True)
    Syn.weight = '100+ rand() * 50'
    # spike monitors input and network
    spikemonN1 = SpikeMonitor(testNeurons1, name='spikemon')
    spikemonN2 = SpikeMonitor(testNeurons2, name='spikemonOut')
    # # state monitor neurons
    statemonN1 = StateMonitor(
        testNeurons1, variables=[
            "Iin", "Imem"], record=[
            0, 3], name='statemonNeu')
    statemonN2 = StateMonitor(
        testNeurons2,
        variables=['Imem'],
        record=0,
        name='statemonNeuOut')

    Net.add(
        gInpGroup,
        testNeurons1,
        testNeurons2,
        InpSyn,
        Syn,
        spikemonN1,
        spikemonN2,
        statemonN1,
        statemonN2)
    Net.run(duration_sim * ms)
    print('Simulation run for {} ms'.format(duration_sim))
    return spikemonN1, spikemonN2, statemonN1, statemonN2


def get_plotsettings():
    MyPlotSettings = PlotSettings(
        fontsize_title=20,
        fontsize_legend=14,
        fontsize_axis_labels=14,
        marker_size=30,
        colors=[
            'r',
            'b',
            'g'])
    return MyPlotSettings


SHOW_PLOTS_IN_TESTS = False



class TestRasterplot(unittest.TestCase):

    def test_getalldatafromeventmodels(self):

        spikemonN1, spikemonN2, _, _ = run_brian_network()

        # from DataModels & EventModels
        EM1 = EventsModel.from_brian_spike_monitor(spikemonN1)
        EM2 = EventsModel.from_brian_spike_monitor(spikemonN2)

        RC = Rasterplot(
            MyPlotSettings=get_plotsettings(),
            MyEventsModels=[
                EM1,
                EM2],
            QtApp=QtApp,
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
            QtApp=QtApp,
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

        spikemonN1, spikemonN2, _, _ = run_brian_network()

        subgroup_labels = ['N1', 'N2']
        time_range = (0, 0.6)
        neuron_id_range = (0, 7)

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
            QtApp=QtApp,
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
            QtApp=QtApp,
            show_immediately=SHOW_PLOTS_IN_TESTS)
        RC._filter_data()
        self.assertLessEqual(max(RC.all_neuron_ids[0], default=0), neuron_id_range[1])
        self.assertLessEqual(max(RC.all_spike_times[0], default=0), time_range[1])

    def test_createrasterplot(self):

        spikemonN1, spikemonN2, _, _ = run_brian_network()

        MyEventsModels = [spikemonN1, spikemonN2]
        subgroup_labels = ['N1', 'N2']
        time_range = (0, 0.6)
        neuron_id_range = (0, 8)

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
            mainfig=None,
            subfig_rasterplot=None,
            subfig_histogram=None,
            QtApp=QtApp,
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
            mainfig=None,
            subfig_rasterplot=None,
            subfig_histogram=None,
            QtApp=QtApp,
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
                mainfig=None,
                subfig_rasterplot=None,
                subfig_histogram=None,
                QtApp=QtApp,
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
                mainfig=None,
                subfig_rasterplot=None,
                subfig_histogram=None,
                QtApp=QtApp,
                add_histogram=add_histogram,
                show_immediately=SHOW_PLOTS_IN_TESTS)

        time_range = (0.4, 0.8)
        neuron_id_range = (0, 8)
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
            mainfig=None,
            subfig_rasterplot=None,
            subfig_histogram=None,
            QtApp=QtApp,
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
                mainfig=None,
                subfig_rasterplot=None,
                subfig_histogram=None,
                QtApp=QtApp,
                add_histogram=add_histogram,
                show_immediately=SHOW_PLOTS_IN_TESTS)
        else:
            warnings.warn("Skip part of unittest TestRasterplot.test_createrasterplot using pyqtgraph"
                          "as pyqtgraph could not be imported")

if __name__ == '__main__':
    unittest.main()
