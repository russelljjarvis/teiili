'''This script tests the functionality of the B'''
import unittest
import numpy as np
from teili.building_blocks.building_block import BuildingBlock
from teili.building_blocks.wta import WTA
from teili.core import tags as tags_parameters
from brian2 import prefs, ms

prefs.codegen.target = "numpy"


class TestBB(unittest.TestCase):

    def test_dictionaries(self):
        ''' TODO: test subblocks for hierachical blocks.
        Tests the basic dictionary structure.'''
        test1DWTA = WTA(name='test1DWTA', dimensions=1, num_neurons=16, verbose=False)
        tmp_input_groups = {'n_exc': test1DWTA._groups['n_exc']}
        tmp_output_groups = {'n_exc': test1DWTA._groups['n_exc']}
        tmp_hidden_groups = {'n_inh': test1DWTA._groups['n_inh']}

        self.assertEqual(test1DWTA.input_groups, tmp_input_groups)
        self.assertEqual(test1DWTA.output_groups, tmp_output_groups)
        self.assertEqual(test1DWTA.hidden_groups, tmp_hidden_groups)

    def test_groups(self):
        ''' TODO: test subblocks for hierachical blocks.
        Tests the `groups` property of the building block which recursively
        gathers all `Neuron` and `Connection` groups.'''
        test1DWTA = WTA(name='test1DWTA', dimensions=1, num_neurons=16, verbose=False)
        bb_groups = test1DWTA.groups
        self.assertEqual(bb_groups, test1DWTA._groups)

    def test_get_groups(self):
        '''Tests the __getter__ function to retrieve all groups with
        a certain set of tags.'''
        test1DWTA = WTA(name='test1DWTA', dimensions=1, num_neurons=16, verbose=False)
        target_group = test1DWTA._groups['n_exc']
        tags = tags_parameters.basic_wta_n_exc

        # test1DWTA._set_tags(tags, target_group)
        self.assertEqual(test1DWTA.get_groups(tags)['n_exc'], target_group)

    def test_set_tags(self):
        ''' Tests to set tags to a given group.'''
        test1DWTA = WTA(name='test1DWTA', dimensions=1, num_neurons=16, verbose=False)
        tags = tags_parameters.basic_wta_n_exc

        target_group = test1DWTA._groups['n_exc']
        test1DWTA._set_tags(tags, target_group)
        self.assertEqual(tags, test1DWTA.get_tags(target_group))


if __name__ == '__main__':
    unittest.main()
