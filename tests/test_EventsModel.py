import unittest

from teili.tools.visualizer.DataModels import EventsModel
from utils_unittests import run_brian_network

class TestEventsModel(unittest.TestCase):

    def test_EventsModel(self):

        neuron_ids = [1, 1, 1, 2, 3, 1, 4, 5]
        spike_times = [11, 14, 14, 16, 17, 25, 36, 40]
        EM = EventsModel(neuron_ids=neuron_ids, spike_times=spike_times)

        self.assertTrue(len(EM.neuron_ids) == len(neuron_ids))
        self.assertTrue(len(EM.spike_times) == len(spike_times))

    def test_EventsModelfrombrianspikemonitor(self):
        spikemonN1, _, statemonN1, _ = run_brian_network()
        EM = EventsModel.from_brian_spike_monitor(spikemonN1)

        self.assertTrue(len(EM.neuron_ids) == len(spikemonN1.i))
        self.assertTrue(len(EM.spike_times) == len(spikemonN1.t))


if __name__ == '__main__':
    unittest.main()
