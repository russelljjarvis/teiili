import unittest

from brian2 import us, ms, prefs, defaultclock, start_scope, SpikeGeneratorGroup, StateMonitor
import numpy as np
import warnings

from teili.tools.visualizer.DataControllers import Lineplot
from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataModels import StateVariablesModel
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
    SKIP_PYQTGRAPH_RELATED_UNITTESTS = True

def run_brian_network():
    prefs.codegen.target = "numpy"
    defaultclock.dt = 10 * us

    start_scope()
    N_input, N_N1, N_N2 = 1, 5, 3
    duration_sim = 150

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
    # state monitor neurons
    statemonN1 = StateMonitor(
        testNeurons1, variables=[
            "Iin"], record=[
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
        statemonN1,
        statemonN2)
    Net.run(duration_sim * ms)
    print('Simulation run for {} ms'.format(duration_sim))
    return statemonN1, statemonN2


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


class TestLineplot(unittest.TestCase):

    def test_getdata(self):

        statemonN1, statemonN2 = run_brian_network()

        # from DataModels
        SVM = StateVariablesModel.from_brian_state_monitors(
            [statemonN1, statemonN2], skip_not_rec_neuron_ids=True)

        DataModel_to_x_and_y_attr = [(SVM, ('t_Imem', 'Imem')),
                                     (SVM, ('t_Iin', 'Iin'))]

        LC = Lineplot(
            MyPlotSettings=get_plotsettings(),
            DataModel_to_x_and_y_attr=DataModel_to_x_and_y_attr,
            title='data from DataModels',
            show_immediately=SHOW_PLOTS_IN_TESTS)
        LC._get_data_from_datamodels(DataModel_to_x_and_y_attr)
        self.assertEqual(len(LC.data), len(DataModel_to_x_and_y_attr))
        self.assertEqual(
            np.size(
                LC.data[0][0]), np.size(
                getattr(
                    statemonN2, 't')))
        self.assertEqual(
            np.size(
                LC.data[0][1]), np.size(
                getattr(
                    statemonN2, 'Imem')))
        self.assertEqual(
            np.size(
                LC.data[1][0]), np.size(
                getattr(
                    statemonN1, 't')))
        self.assertEqual(
            np.size(
                LC.data[1][1]), np.size(
                getattr(
                    statemonN1, 'Iin')))

        # from brian state monitor and spike monitors
        DataModel_to_x_and_y_attr = [(statemonN1, ('t', 'Iin')),
                                     (statemonN2, ('t', 'Imem'))]

        LC = Lineplot(
            MyPlotSettings=get_plotsettings(),
            DataModel_to_x_and_y_attr=DataModel_to_x_and_y_attr,
            title='data from StateMonitors',
            show_immediately=SHOW_PLOTS_IN_TESTS)
        LC._get_data_from_datamodels(DataModel_to_x_and_y_attr)
        self.assertEqual(len(LC.data), len(DataModel_to_x_and_y_attr))
        self.assertEqual(
            np.size(
                LC.data[0][0]), np.size(
                getattr(
                    statemonN1, 't')))
        self.assertEqual(
            np.size(
                LC.data[0][1]), np.size(
                getattr(
                    statemonN1, 'Iin')))
        self.assertEqual(
            np.size(
                LC.data[1][0]), np.size(
                getattr(
                    statemonN2, 't')))
        self.assertEqual(
            np.size(
                LC.data[1][1]), np.size(
                getattr(
                    statemonN2, 'Imem')))

    def test__filterdata(self):

        statemonN1, statemonN2 = run_brian_network()

        # from DataModels
        SVM = StateVariablesModel.from_brian_state_monitors(
            [statemonN1, statemonN2], skip_not_rec_neuron_ids=True)
        DataModel_to_x_and_y_attr = [
            (SVM, ('t_Imem', 'Imem')), (SVM, ('t_Iin', 'Iin'))]

        # no fitlering should happen
        x_range = None
        y_range = None
        LC = Lineplot(
            MyPlotSettings=get_plotsettings(),
            DataModel_to_x_and_y_attr=DataModel_to_x_and_y_attr,
            x_range=x_range,
            y_range=y_range,
            title='no filtering',
            show_immediately=SHOW_PLOTS_IN_TESTS)
        LC._filter_data()
        self.assertEqual(
            np.size(
                LC.data[0][0]), np.size(
                getattr(
                    statemonN2, 't')))
        self.assertEqual(
            np.size(
                LC.data[0][1]), np.size(
                getattr(
                    statemonN2, 'Imem')))
        self.assertEqual(
            np.size(
                LC.data[1][0]), np.size(
                getattr(
                    statemonN1, 't')))
        self.assertEqual(
            np.size(
                LC.data[1][1]), np.size(
                getattr(
                    statemonN1, 'Iin')))

        # filter only x
        x_range = (0.05, 0.1)
        y_range = None
        LC = Lineplot(
            MyPlotSettings=get_plotsettings(),
            DataModel_to_x_and_y_attr=DataModel_to_x_and_y_attr,
            x_range=x_range,
            y_range=y_range,
            title='filter x {}'.format(x_range),
            show_immediately=SHOW_PLOTS_IN_TESTS)
        LC._filter_data()
        self.assertEqual(
            np.shape(
                LC.data[0][0])[0], np.shape(
                LC.data[0][1])[0])
        self.assertEqual(
            np.shape(
                LC.data[1][0])[0], np.shape(
                LC.data[1][1])[0])

        self.assertTrue((LC.data[0][0] >= x_range[0]).all())
        self.assertTrue((LC.data[0][0] <= x_range[1]).all())
        self.assertTrue((LC.data[1][0] >= x_range[0]).all())
        self.assertTrue((LC.data[1][0] <= x_range[1]).all())

        # filter only y
        x_range = None
        y_range = (0, 3e-9)
        LC = Lineplot(
            MyPlotSettings=get_plotsettings(),
            DataModel_to_x_and_y_attr=DataModel_to_x_and_y_attr,
            x_range=x_range,
            y_range=y_range,
            title='filter y {}'.format(y_range),
            show_immediately=SHOW_PLOTS_IN_TESTS)
        LC._filter_data()
        self.assertEqual(
            np.shape(
                LC.data[0][0])[0], np.shape(
                LC.data[0][1])[0])
        self.assertEqual(
            np.shape(
                LC.data[1][0])[0], np.shape(
                LC.data[1][1])[0])

        self.assertTrue((LC.data[0][1] >= y_range[0]).all())
        self.assertTrue((LC.data[0][1] <= y_range[1]).all())
        self.assertTrue((LC.data[1][1] >= y_range[0]).all())
        self.assertTrue((LC.data[1][1] <= y_range[1]).all())

        # filter x and y
        x_range = (0.02, 0.13)
        y_range = (0, 5e-9)
        LC = Lineplot(
            MyPlotSettings=get_plotsettings(),
            DataModel_to_x_and_y_attr=DataModel_to_x_and_y_attr,
            x_range=x_range,
            y_range=y_range,
            title='filter x {} and y {}'.format(x_range, y_range),
            show_immediately=SHOW_PLOTS_IN_TESTS)
        LC._filter_data()
        self.assertEqual(
            np.shape(
                LC.data[0][0])[0], np.shape(
                LC.data[0][1])[0])
        self.assertEqual(
            np.shape(
                LC.data[1][0])[0], np.shape(
                LC.data[1][1])[0])

        self.assertTrue((LC.data[0][0] >= x_range[0]).all())
        self.assertTrue((LC.data[0][0] <= x_range[1]).all())
        self.assertTrue((LC.data[1][0] >= x_range[0]).all())
        self.assertTrue((LC.data[1][0] <= x_range[1]).all())

        self.assertTrue((LC.data[0][1] >= y_range[0]).all())
        self.assertTrue((LC.data[0][1] <= y_range[1]).all())
        self.assertTrue((LC.data[1][1] >= y_range[0]).all())
        self.assertTrue((LC.data[1][1] <= y_range[1]).all())

        # no elements left after filtering
        x_range = (0.2, 0.4)  # simulation only goes from (0.0, 0.15)
        y_range = (0, 3e-9)
        Lineplot(
            MyPlotSettings=get_plotsettings(),
            DataModel_to_x_and_y_attr=DataModel_to_x_and_y_attr,
            title='empty data',
            x_range=x_range,
            y_range=y_range,
            show_immediately=SHOW_PLOTS_IN_TESTS)

    def test_createlineplot(self):
        # check backends
        statemonN1, statemonN2 = run_brian_network()

        # from DataModels
        SVM = StateVariablesModel.from_brian_state_monitors(
            [statemonN1, statemonN2], skip_not_rec_neuron_ids=True)
        DataModel_to_x_and_y_attr = [
            (SVM, ('t_Imem', 'Imem')), (SVM, ('t_Iin', 'Iin'))]
        subgroup_labels = ['Imem', 'Iin']
        x_range = (0.02, 0.13)
        y_range = (0, 5e-9)

        backend = 'matplotlib'
        LC = Lineplot(
            MyPlotSettings=get_plotsettings(),
            DataModel_to_x_and_y_attr=DataModel_to_x_and_y_attr,
            subgroup_labels=subgroup_labels,
            x_range=x_range,
            y_range=y_range,
            title='Lineplot matplotlib with Imem (in red) and Iin (in blue)',
            xlabel='my x label',
            ylabel='my y label',
            backend=backend,
            mainfig=None,
            subfig=None,
            QtApp=None,
            show_immediately=SHOW_PLOTS_IN_TESTS)
        LC.create_plot()

        if not SKIP_PYQTGRAPH_RELATED_UNITTESTS:
            backend = 'pyqtgraph'
            LC = Lineplot(
                MyPlotSettings=get_plotsettings(),
                DataModel_to_x_and_y_attr=DataModel_to_x_and_y_attr,
                subgroup_labels=subgroup_labels,
                x_range=x_range,
                y_range=y_range,
                title='Lineplot pyqt with Imem (r) and Iin (b)',
                xlabel='my x label',
                ylabel='my y label',
                backend=backend,
                mainfig=None,
                subfig=None,
                QtApp=QtApp,
                show_immediately=SHOW_PLOTS_IN_TESTS)
            LC.create_plot()
        else:
            warnings.warn("Skip part of unittest TestLineplot.test_createlineplot using pyqtgraph"
                          "as pyqtgraph could not be imported")


if __name__ == '__main__':
    unittest.main()
