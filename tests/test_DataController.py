import unittest

import numpy as np

from teili.tools.visualizer.DataControllers import DataController


DatCtr = DataController()


class TestDataController(unittest.TestCase):

    def test_filtereventsforinterval(self):
        all_spiketimes = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
        all_neuron_ids = [5, 4, 3, 2, 2, 6, 1, 0]

        t_start_1, t_end_1 = (0.1, 0.45)
        expected_spike_times_1, expected_neuron_ids_1 = (
            all_spiketimes, all_neuron_ids)
        returned_spike_times_1, returned_neuron_ids_1 = DatCtr._filter_events_for_interval(
            all_spiketimes, all_neuron_ids, t_start_1, t_end_1)
        self.assertTrue(
            np.array_equal(
                expected_neuron_ids_1,
                returned_neuron_ids_1))
        self.assertTrue(
            np.array_equal(
                expected_spike_times_1,
                returned_spike_times_1))

        t_start_2, t_end_2 = (0.0, 0.34)
        expected_spike_times_2, expected_neuron_ids_2 = (
            all_spiketimes[:5], all_neuron_ids[:5])
        returned_spike_times_2, returned_neuron_ids_2 = DatCtr._filter_events_for_interval(
            all_spiketimes, all_neuron_ids, t_start_2, t_end_2)
        self.assertTrue(
            np.array_equal(
                expected_neuron_ids_2,
                returned_neuron_ids_2))
        self.assertTrue(
            np.array_equal(
                expected_spike_times_2,
                returned_spike_times_2))

        t_start_3, t_end_3 = (0.1, 0.50)
        expected_spike_times_3, expected_neuron_ids_3 = (
            all_spiketimes, all_neuron_ids)
        returned_spike_times_3, returned_neuron_ids_3 = DatCtr._filter_events_for_interval(
            all_spiketimes, all_neuron_ids, t_start_3, t_end_3)
        self.assertTrue(
            np.array_equal(
                expected_neuron_ids_3,
                returned_neuron_ids_3))
        self.assertTrue(
            np.array_equal(
                expected_spike_times_3,
                returned_spike_times_3))

        t_start_4, t_end_4 = (0.6, 0.80)
        expected_spike_times_4, expected_neuron_ids_4 = (
            np.array(
                []), np.array(
                []))
        returned_spike_times_4, returned_neuron_ids_4 = DatCtr._filter_events_for_interval(
            all_spiketimes, all_neuron_ids, t_start_4, t_end_4)
        self.assertTrue(
            np.array_equal(
                expected_neuron_ids_4,
                returned_neuron_ids_4))
        self.assertTrue(
            np.array_equal(
                expected_spike_times_4,
                returned_spike_times_4))

    def test_filtereventsforneuronids(self):
        all_spiketimes = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
        all_neuron_ids = [5, 4, 3, 2, 2, 6, 1, 0]

        active_neuron_ids_1 = range(7)
        expected_spike_times_1, expected_neuron_ids_1 = (
            all_spiketimes, all_neuron_ids)
        returned_spike_times_1, returned_neuron_ids_1 = DatCtr._filter_events_for_neuron_ids(
            all_spiketimes, all_neuron_ids, active_neuron_ids_1)
        self.assertTrue(len(np.unique(returned_neuron_ids_1))
                        <= len(np.unique(active_neuron_ids_1)))
        self.assertTrue(
            np.array_equal(
                expected_neuron_ids_1,
                returned_neuron_ids_1))
        self.assertTrue(
            np.array_equal(
                expected_spike_times_1,
                returned_spike_times_1))

        active_neuron_ids_2 = range(5)
        expected_spike_times_2, expected_neuron_ids_2 = (
            [0.15, 0.20, 0.25, 0.30, 0.40, 0.45], [4, 3, 2, 2, 1, 0])
        returned_spike_times_2, returned_neuron_ids_2 = DatCtr._filter_events_for_neuron_ids(
            all_spiketimes, all_neuron_ids, active_neuron_ids_2)
        self.assertTrue(len(np.unique(returned_neuron_ids_2)) == 5)
        self.assertTrue(
            np.array_equal(
                expected_neuron_ids_2,
                returned_neuron_ids_2))
        self.assertTrue(
            np.array_equal(
                expected_spike_times_2,
                returned_spike_times_2))

        active_neuron_ids_3 = [1, 2, 8]
        expected_spike_times_3, expected_neuron_ids_3 = ([0.25, 0.30, 0.40],
                                                         [2, 2, 1])
        returned_spike_times_3, returned_neuron_ids_3 = DatCtr._filter_events_for_neuron_ids(
            all_spiketimes, all_neuron_ids, active_neuron_ids_3)
        self.assertTrue(len(np.unique(returned_neuron_ids_3))
                        <= len(np.unique(active_neuron_ids_3)))
        self.assertTrue(
            np.array_equal(
                expected_neuron_ids_3,
                returned_neuron_ids_3))
        self.assertTrue(
            np.array_equal(
                expected_spike_times_3,
                returned_spike_times_3))

    def test_filterevents(self):
        all_spiketimes = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
        all_neuron_ids = [5, 4, 3, 2, 2, 6, 1, 0]

        # all spike times, some neuron ids
        t_start_1, t_end_1 = (0.1, 0.45)
        active_neuron_ids_1 = range(7)
        expected_spike_times_1, expected_neuron_ids_1 = (
            all_spiketimes, all_neuron_ids)
        returned_spike_times_1, returned_neuron_ids_1 = DatCtr.filter_events(
            all_spiketimes, all_neuron_ids, (t_start_1, t_end_1), active_neuron_ids_1)
        self.assertTrue(len(np.unique(returned_neuron_ids_1))
                        <= len(np.unique(active_neuron_ids_1)))
        self.assertTrue(
            np.array_equal(
                expected_neuron_ids_1,
                returned_neuron_ids_1))
        self.assertTrue(
            np.array_equal(
                expected_spike_times_1,
                returned_spike_times_1))

        t_start_2, t_end_2 = (0.1, 0.45)
        active_neuron_ids_2 = [1, 2, 8]
        expected_spike_times_2, expected_neuron_ids_2 = (
            [0.25, 0.30, 0.40], [2, 2, 1])
        returned_spike_times_2, returned_neuron_ids_2 = DatCtr.filter_events(
            all_spiketimes, all_neuron_ids, (t_start_2, t_end_2), active_neuron_ids_2)
        self.assertTrue(len(np.unique(returned_neuron_ids_2))
                        <= len(np.unique(active_neuron_ids_2)))
        self.assertTrue(
            np.array_equal(
                expected_neuron_ids_2,
                returned_neuron_ids_2))
        self.assertTrue(
            np.array_equal(
                expected_spike_times_2,
                returned_spike_times_2))

        t_start_3, t_end_3 = (0.0, 0.34)
        active_neuron_ids_3 = range(7)
        expected_spike_times_3, expected_neuron_ids_3 = (
            all_spiketimes[:5], all_neuron_ids[:5])
        returned_spike_times_3, returned_neuron_ids_3 = DatCtr.filter_events(
            all_spiketimes, all_neuron_ids, (t_start_3, t_end_3), active_neuron_ids_3)
        self.assertTrue(len(np.unique(returned_neuron_ids_1))
                        <= len(np.unique(active_neuron_ids_3)))
        self.assertTrue(
            np.array_equal(
                expected_neuron_ids_3,
                returned_neuron_ids_3))
        self.assertTrue(
            np.array_equal(
                expected_spike_times_3,
                returned_spike_times_3))

        t_start_4, t_end_4 = (0.0, 0.34)
        active_neuron_ids_4 = [1, 2, 8]
        expected_spike_times_4, expected_neuron_ids_4 = ([0.25, 0.30],
                                                         [2, 2])
        returned_spike_times_4, returned_neuron_ids_4 = DatCtr.filter_events(
            all_spiketimes, all_neuron_ids, (t_start_4, t_end_4), active_neuron_ids_4)
        self.assertTrue(len(np.unique(returned_neuron_ids_4))
                        <= len(np.unique(active_neuron_ids_4)))
        self.assertTrue(
            np.array_equal(
                expected_neuron_ids_4,
                returned_neuron_ids_4))
        self.assertTrue(
            np.array_equal(
                expected_spike_times_4,
                returned_spike_times_4))

        t_start_5, t_end_5 = (0.6, 0.8)
        active_neuron_ids_5 = [1, 2, 8]
        expected_spike_times_5, expected_neuron_ids_5 = (
            np.array(
                []), np.array(
                []))
        returned_spike_times_5, returned_neuron_ids_5 = DatCtr.filter_events(
            all_spiketimes, all_neuron_ids, (t_start_5, t_end_5), active_neuron_ids_5)
        self.assertTrue(len(np.unique(returned_neuron_ids_5))
                        <= len(np.unique(active_neuron_ids_5)))
        self.assertTrue(
            np.array_equal(
                expected_neuron_ids_5,
                returned_neuron_ids_5))
        self.assertTrue(
            np.array_equal(
                expected_spike_times_5,
                returned_spike_times_5))


if __name__ == '__main__':
    unittest.main()
