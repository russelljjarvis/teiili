import unittest

import numpy as np

from brian2 import us, ms, prefs, defaultclock, start_scope, SpikeGeneratorGroup, SpikeMonitor, StateMonitor

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import DPI
from teili.models.synapse_models import DPISyn
from teili.models.parameters.dpi_neuron_param import parameters as neuron_model_param

from teili.tools.visualizer.DataModels import EventsModel


def run_teili_network():

    prefs.codegen.target = "numpy"
    defaultclock.dt = 10 * us

    start_scope()

    # parameters
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
    # spike monitors input and network
    spikemonN1 = SpikeMonitor(testNeurons1, name='spikemon')
    # # state monitor neurons
    statemonN1 = StateMonitor(
        testNeurons1, variables=[
            "Iin", "Imem"], record=[
            0, 3], name='statemonNeu')

    Net.add(gInpGroup, testNeurons1, testNeurons2,
            InpSyn, Syn,
            spikemonN1,
            statemonN1,
            )
    Net.run(duration_sim * ms)
    print('Simulation run for {} ms'.format(duration_sim))

    return Net, spikemonN1, statemonN1


class TestEventsModel(unittest.TestCase):

    def test_EventsModel(self):

        neuron_ids = [1, 1, 1, 2, 3, 1, 4, 5]
        spike_times = [11, 14, 14, 16, 17, 25, 36, 40]
        EM = EventsModel(neuron_ids=neuron_ids, spike_times=spike_times)

        self.assertTrue(len(EM.neuron_ids) == len(neuron_ids))
        self.assertTrue(len(EM.spike_times) == len(spike_times))

    def test_EventsModel_from_brian_spike_monitor(self):
        Net, spikemonN1, statemonN1 = run_teili_network()
        EM = EventsModel.from_brian_spike_monitor(spikemonN1)

        self.assertTrue(len(EM.neuron_ids) == len(spikemonN1.i))
        self.assertTrue(len(EM.spike_times) == len(spikemonN1.t))


if __name__ == '__main__':
    unittest.main()
