import unittest

import os
import numpy as np

from teili.tools.visualizer.DataModels import EventsModel, StateVariablesModel


class TestDataController(unittest.TestCase):

    def test_saveloaddatamodel(self):
        # EventsModel
        neuron_ids = [1, 1, 1, 2, 3, 1, 4, 5]
        spike_times = [11, 14, 14, 16, 17, 25, 36, 40]
        EM_org = EventsModel(neuron_ids=neuron_ids, spike_times=spike_times)

        outputfilename_em = './test_saveloaddatamodel_eventsmodel.npz'
        EM_org.save_datamodel(outputfilename_em)

        EM_restored = EventsModel(neuron_ids=None, spike_times=None)
        EM_restored.load_datamodel(outputfilename_em)

        self.assertTrue((np.array_equal(EM_org.neuron_ids, EM_restored.neuron_ids)))
        self.assertTrue((np.array_equal(EM_org.spike_times, EM_restored.spike_times)))

        # StateVariablesModel
        state_variable_names = ['var_name']
        num_neurons = 6
        num_timesteps = 50
        state_variables = [np.random.random((num_neurons, num_timesteps))]
        state_variables_times = [np.linspace(0, 100, num_timesteps)]
        SVM_org = StateVariablesModel(
            state_variable_names,
            state_variables,
            state_variables_times)

        outputfilename_svm = './test_saveloaddatamodel_statevariablesmodel.npz'
        SVM_org.save_datamodel(outputfilename_svm)

        SVM_restored = StateVariablesModel(
            state_variable_names=None,
            state_variables=None,
            state_variables_times=None)
        SVM_restored.load_datamodel(outputfilename_svm)

        self.assertTrue((SVM_org.var_name == SVM_restored.var_name).all())
        self.assertTrue((SVM_org.t_var_name == SVM_restored.t_var_name).all())

        os.system('rm {}'.format(outputfilename_em))
        os.system('rm {}'.format(outputfilename_svm))

    def test_fromfile(self):
        # EventsModel
        neuron_ids = [1, 1, 1, 2, 3, 1, 4, 5]
        spike_times = [11, 14, 14, 16, 17, 25, 36, 40]
        EM_org = EventsModel(neuron_ids=neuron_ids, spike_times=spike_times)

        outputfilename_em = './test_saveloaddatamodel_eventsmodel.npz'
        EM_org.save_datamodel(outputfilename_em)

        EM_restored_directly = EventsModel.from_file(outputfilename_em)
        self.assertTrue((np.array_equal(EM_org.neuron_ids, EM_restored_directly.neuron_ids)))
        self.assertTrue((np.array_equal(EM_org.spike_times, EM_restored_directly.spike_times)))

        # StateVariablesModel
        state_variable_names = ['var_name']
        num_neurons = 6
        num_timesteps = 50
        state_variables = [np.random.random((num_neurons, num_timesteps))]
        state_variables_times = [np.linspace(0, 100, num_timesteps)]
        SVM_org = StateVariablesModel(
            state_variable_names,
            state_variables,
            state_variables_times)

        outputfilename_svm = './test_saveloaddatamodel_statevariablesmodel.npz'
        SVM_org.save_datamodel(outputfilename_svm)

        SVM_restored_directly = StateVariablesModel.from_file(
            outputfilename_svm)
        self.assertTrue((SVM_org.var_name == SVM_restored_directly.var_name).all())
        self.assertTrue((SVM_org.t_var_name == SVM_restored_directly.t_var_name).all())

        os.system('rm {}'.format(outputfilename_em))
        os.system('rm {}'.format(outputfilename_svm))


if __name__ == '__main__':
    unittest.main()
