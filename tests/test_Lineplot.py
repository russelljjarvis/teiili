import unittest

import numpy as np
import warnings

from utils_unittests import get_plotsettings, run_brian_network

from teili.tools.visualizer.DataControllers import Lineplot
from teili.tools.visualizer.DataModels import StateVariablesModel

try:
    import pyqtgraph as pg
    from PyQt5 import QtGui
    SKIP_PYQTGRAPH_RELATED_UNITTESTS = False
except BaseException:
    SKIP_PYQTGRAPH_RELATED_UNITTESTS = True

SHOW_PLOTS_IN_TESTS = False

class TestLineplot(unittest.TestCase):

    def test_getdata(self):
        (statemonN1, statemonN2) = run_brian_network(statemonitors=True, spikemonitors=False)

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            (statemonN1, statemonN2) = run_brian_network(statemonitors=True, spikemonitors=False)

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
            x_range = (0.01, 0.05)
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
            y_range = (0, 1.5e-7)
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
            x_range = (0.01, 0.05)
            y_range = (0, 1.5e-7)
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            # check backends
            (statemonN1, statemonN2) = run_brian_network(statemonitors=True, spikemonitors=False)

            # from DataModels
            SVM = StateVariablesModel.from_brian_state_monitors(
                [statemonN1, statemonN2], skip_not_rec_neuron_ids=True)
            DataModel_to_x_and_y_attr = [
                (SVM, ('t_Imem', 'Imem')), (SVM, ('t_Iin', 'Iin'))]
            subgroup_labels = ['Imem', 'Iin']
            x_range = (0.01, 0.05)
            y_range = (0, 1.5e-7)

            backend = 'matplotlib'
            LC = Lineplot(
                MyPlotSettings=get_plotsettings(),
                DataModel_to_x_and_y_attr=DataModel_to_x_and_y_attr,
                subgroup_labels=subgroup_labels,
                x_range=x_range,
                y_range=y_range,
                title='Lineplot matplotlib with Imem and Iin',
                xlabel='my x label',
                ylabel='my y label',
                backend=backend,
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
                    title='Lineplot pyqt with Imem and Iin',
                    xlabel='my x label',
                    ylabel='my y label',
                    backend=backend,
                    show_immediately=SHOW_PLOTS_IN_TESTS)
                LC.create_plot()
            else:
                warnings.warn("Skip part of unittest TestLineplot.test_createlineplot using pyqtgraph"
                              "as pyqtgraph could not be imported")


if __name__ == '__main__':
    unittest.main()
