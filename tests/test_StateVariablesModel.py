import os
import sys
import unittest
from contextlib import contextmanager

import numpy as np
from brian2 import us, ms, prefs, defaultclock, start_scope, SpikeGeneratorGroup, SpikeMonitor, StateMonitor
from teili import TeiliNetwork
from teili.core.groups import Neurons, Connections
from teili.models.neuron_models import DPI
from teili.models.parameters.dpi_neuron_param import parameters as neuron_model_param
from teili.models.synapse_models import DPISyn
from teili.tools.visualizer.DataModels import StateVariablesModel


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def run_teili_network():

    prefs.codegen.target = "numpy"
    defaultclock.dt = 10 * us

    start_scope()

    # parameters
    N_input = 1
    N_N1 = 5
    N_N2 = 3
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

    # spike monitors input and network
    spikemonN1 = SpikeMonitor(testNeurons1, name='spikemon')

    # # state monitor neurons
    statemonN1 = StateMonitor(
        testNeurons1, variables=[
            "Iin", "Imem"], record=[
            0, 3], name='statemonNeu')
    statemonN2 = StateMonitor(
        testNeurons2,
        variables=['Iahp'],
        record=0,
        name='statemonNeuOut')
    statemonN2_2 = StateMonitor(
        testNeurons2,
        variables=['Imem'],
        record=0,
        name='statemonNeuOut_2')

    Net.add(gInpGroup, testNeurons1, testNeurons2,
            InpSyn, Syn,
            spikemonN1,
            statemonN1, statemonN2,
            )
    Net.run(duration_sim * ms)
    return Net, spikemonN1, statemonN1, statemonN2, statemonN2_2


class TestDataModel(unittest.TestCase):

    def test_StateVariablesModel(self):

        state_variable_names = ['var_name']
        num_neurons = 6
        num_timesteps = 50
        state_variables = [np.random.random((num_neurons, num_timesteps))]
        state_variables_times = [np.linspace(0, 100, num_timesteps)]

        SVM = StateVariablesModel(
            state_variable_names,
            state_variables,
            state_variables_times)

        self.assertTrue(SVM.var_name.shape == (num_neurons, num_timesteps))
        self.assertTrue(len(SVM.t_var_name) == num_timesteps)


        # test that raise Exception if variable names are not unique
        state_variable_names = ['var_name', 'var_name']
        num_neurons = 6
        num_timesteps = 50
        state_variables = [np.random.random((num_neurons, num_timesteps)), np.random.random((num_neurons, num_timesteps))]
        state_variables_times = [np.linspace(0, 100, num_timesteps), np.linspace(0, 100, num_timesteps)]

        with suppress_stdout():
            with self.assertRaises(Exception) as context:
                SVM = StateVariablesModel(
                    state_variable_names,
                    state_variables,
                    state_variables_times)


    def test_StateVariablesModelfrombrianstatemonitors(self):

        Net, spikemonN1, statemonN1, statemonN2, statemonN2_2 = run_teili_network()

        SVM = StateVariablesModel.from_brian_state_monitors(
            [statemonN1, statemonN2], skip_not_rec_neuron_ids=False)
        self.assertTrue(SVM.Imem.shape[0] == len(statemonN1.t))
        self.assertTrue(len(SVM.t_Imem) == len(statemonN1.t))
        self.assertTrue(SVM.Iin.shape[0] == len(statemonN1.t))
        self.assertTrue(len(SVM.t_Iin) == len(statemonN1.t))
        self.assertTrue(SVM.Iahp.shape[0] == len(statemonN2.t))
        self.assertTrue(len(SVM.t_Iahp) == len(statemonN2.t))

        SVM = StateVariablesModel.from_brian_state_monitors(
            [statemonN1, statemonN2], skip_not_rec_neuron_ids=True)
        self.assertTrue(SVM.Imem.shape[1] == len(statemonN1.record))
        self.assertTrue(SVM.Iin.shape[1] == len(statemonN1.record))
        self.assertTrue(SVM.Iahp.shape[1] == len(statemonN2.record))

        # statemonN1 & statemonN2_2 store the a variable called 'Imem'
        with suppress_stdout():
            with self.assertRaises(Exception) as context:
                SVM = StateVariablesModel.from_brian_state_monitors(
                    [statemonN1, statemonN2_2], skip_not_rec_neuron_ids=False)


if __name__ == '__main__':
    unittest.main()
