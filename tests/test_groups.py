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


class TestGroups(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        seed_ = np.random.randint(10)
        np.random.seed(seed_)
        seed(seed_)
        
    def test_add_mismatch(self):
        """
        This method tests the method add_mismatch()
        """
        self._test_mismatch_added()
        self._test_mismatch_lower_bound()
        self._test_mismatch_seed()
        
    def _test_mismatch_added(self):
        """
        This tests if the function add_mismatch() changes the selected parameter.
        This tests also NameError (raised when the parameter is not included in the model) 
        and UserWarning (raised if the function is called twice, which means 
        adding the mismatch twice without resetting parameters).
        """
        tetsNeurons1 = Neurons(2, equation_builder=DPI(num_inputs=2), name="testNeuron1")          
        param = 'Itau'
        old_param_value = np.copy(getattr(tetsNeurons1, param))
        tetsNeurons1.add_mismatch({param: 0.1})
        
        self.assertFalse(any(old_param_value == np.asarray(getattr(tetsNeurons1, param))))
        
        with self.assertRaises(NameError):
            tetsNeurons1.add_mismatch({'Cm': 0.1})
        
        tetsNeurons1.add_mismatch({param: 0.1})        
        
        self.assertWarns(UserWarning, tetsNeurons1.add_mismatch({param: 0.1}))
    
    def _test_mismatch_lower_bound(self):
        """
        This checks that mismatch added does not lower the parameter below zero.
        """
        tetsNeurons1 = Neurons(2, equation_builder=DPI(num_inputs=2), name="testNeuron1")          
        param = 'Itau'
        old_param_value = np.copy(getattr(tetsNeurons1, param))
        tetsNeurons1.add_mismatch({param: 0.1})
        
        self.assertTrue(all(new_param_value > 0 for new_param_value in old_param_value))
        
    def _test_mismatch_seed(self):
        """
        This method tests that the random number generation inside the mismatch 
        function does not change the current internal state of the random generator.
        This allows to generate the same sequence of random number (if the internal
        seed is set) regardless of the mismatch generation. 
        """
        
        np_current_state = np.random.get_state()
        tetsNeurons1 = Neurons(2, equation_builder=DPI(num_inputs=2), name="testNeuron1")          
        param = 'Itau'
        tetsNeurons1.add_mismatch({param: 0.1})
        
        self.assertTrue(all(np_current_state[1] == np.random.get_state()[1]))
        
    def test__set_mismatch(self):
        """
        This method tests the method _set_mismatch() when a lower bound for the
        standard normal distribution of the mismatch is set as a multiple of the
        standard deviation (std), specified as a fraction of the parameter current value.
        """
        tetsNeurons1 = Neurons(2, equation_builder=DPI(num_inputs=2), name="testNeuron1")          
        param = 'Itau'
        std = 0.1
        lower = 2
        old_param = np.unique(getattr(tetsNeurons1, param))
        expected_lower_value = -(lower * std * old_param + old_param)
        tetsNeurons1._set_mismatch(param, std=0.1, lower=lower)
        
        self.assertTrue(all(new_param_value > expected_lower_value for new_param_value in getattr(tetsNeurons1, param) ))
    
if __name__=='__main__':
    unittest.main(verbosity=1)    