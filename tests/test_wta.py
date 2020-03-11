# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-01-08 16:30:52
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-11 12:43:19
import unittest
import numpy as np
from teili.building_blocks.wta import WTA
from brian2 import prefs, ms

prefs.codegen.target = "numpy"
#TODO: test plotting;


class TestWTA(unittest.TestCase):

    def test_attributes_1D(self):
        test1DWTA = WTA(name='test1DWTA', dimensions=1, num_neurons=16, verbose=False)
        self.assertEqual(test1DWTA.spike_gen.N, 16)
        self.assertEqual(test1DWTA.num_neurons, 16)
        self.assertEqual(test1DWTA._groups['n_exc'].N, 16)

    def test_attributes_2D(self):
        test2DWTA = WTA(name='test2DWTA', dimensions=2, num_neurons=16, verbose=False)
        self.assertEqual(test2DWTA.spike_gen.N, 16**2)
        self.assertEqual(test2DWTA.num_neurons, 16)
        self.assertEqual(test2DWTA._groups['n_exc'].N, 16**2)

    def test_set_spikes(self):
        test2DWTA = WTA(name='test2DWTA', dimensions=2, num_neurons=16, verbose=False)
        neuron_ts = np.arange(0, 51, 5)
        neuron_id = np.ones((len(neuron_ts))) * (16**2 / 2)
        test2DWTA.spike_gen.set_spikes(indices=neuron_id, times=neuron_ts * ms)
        self.assertEqual((np.asarray(test2DWTA.spike_gen._neuron_index)).tolist(),
                         neuron_id.tolist())
        self.assertEqual((np.asarray(test2DWTA.spike_gen._spike_time / ms)).tolist(),
                         neuron_ts.tolist())


if __name__ == '__main__':
    unittest.main()
