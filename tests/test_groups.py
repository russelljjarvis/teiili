#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains unittest for groups.py
"""

import unittest
import numpy as np
from brian2 import seed
from teili.core.groups import Neurons
from teili.models.neuron_models import DPI

"""
NOTE: 
    if using python=3.6.3, line 38
    
    self.assertWarns(UserWarning, testNeurons.add_mismatch, std_dict={'Itau': 0.1})
    
    will raise the following error:
    RuntimeError: dictionary changed size during iteration
    
    Issue solved in python 3.6.6
"""
class TestGroups(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        seed_ = np.random.randint(10)
        np.random.seed(seed_)
        seed(seed_)

    def test_mismatch_added(self):
        """
        This tests if the function add_mismatch() changes the selected parameter.
        This tests also NameError (raised when the parameter is not included in the model)
        and UserWarning (raised if the function is called twice, which means
        adding the mismatch twice without resetting parameters)
        """
        testNeurons = Neurons(100, equation_builder=DPI(num_inputs=2))
        old_param_value = np.copy(getattr(testNeurons, 'Itau'))

        testNeurons.add_mismatch({'Itau': 0.1}, seed=10)
        self.assertFalse(
            any(old_param_value == np.asarray(getattr(testNeurons, 'Itau'))))

        # Adding mismatch twice:
        #self.assertWarns(UserWarning, testNeurons.add_mismatch, std_dict={'Itau': 0.1})

        # Trying to add mismatch to one parameter not included in the neuron
        # model
        with self.assertRaises(NameError):
            testNeurons.add_mismatch({'Vm': 0.1})

    def test_mismatch_lower_bound(self):
        """
        This tests the lower bound of the gaussian distribution using directly
        the internal function _add_mismatch_param() to add mismatch to one single
        parameter. If not set explicitely, the lower bound should be zero.
        Otherwise, the function raises a warning if the lower bound goes below zero.
        """
        testNeurons = Neurons(100, equation_builder=DPI(num_inputs=2))

        # Test default value of lower bound:
        testNeurons._add_mismatch_param(param='Itau', std=0.1)
        expected_lower = 0
        self.assertTrue(all(new_param_value > expected_lower for new_param_value
                            in getattr(testNeurons, 'Itau')))

        # Test input lower bound:
        param = 'Iath'
        lower = -0.2
        std = 0.1
        old_param = getattr(testNeurons, param)[0]
        testNeurons._add_mismatch_param(param, std, lower)
        expected_lower = lower * std * old_param + old_param
        self.assertTrue(all(
            new_param_value > expected_lower for new_param_value in getattr(testNeurons, param)))

    def test_mismatch_seed(self):
        """
        This method tests that the random number generation inside the mismatch
        function does not change the current internal state of the random generator.
        This allows to generate the same sequence of random number (if the internal
        seed is set) regardless of the mismatch generation.
        """

        np_current_state = np.random.get_state()
        testNeurons = Neurons(2, equation_builder=DPI(num_inputs=2))
        param = 'Itau'
        testNeurons.add_mismatch({param: 0.1})

        self.assertTrue(all(np_current_state[1] == np.random.get_state()[1]))


if __name__ == '__main__':
    unittest.main(verbosity=1)
