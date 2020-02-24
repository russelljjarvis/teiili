#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module provides the parent class for (neuronal) building blocks.

Todo:
    * Hierarchy of building_blocks.
        There is a problem when 2 BBs of the same kind are added to another BB,
        as the names collide.
    * Monitor naming.
        Important all Monitors of Building blocks have to be named and named
        uniquely! Otherwise they will not be found, when a Network is rerun
        in standalone mode after rebuild without recompile
"""

# @Author: alpren, mmilde
# @Date:   2017-07-27 10:46:44


import copy
from collections import OrderedDict

import numpy as np
from brian2.core.names import Nameable


class BuildingBlock(Nameable):
    """This class is the parent class to all building blocks, e.g. WTA, SOM.

    Attributes:
        name (str, required): Name of the building_block population
        neuron_eq_builder (class, optional): neuron class as imported from
            models/neuron_models
        synapse_eq_builder (class, optional): synapse class as imported from
            models/synapse_models
        params (dictionary, optional): Dictionary containing all relevant
            parameters for each building block
        verbose (bool, optional): Flag to gain additional information
        groups (dictionary): Keys to all neuron and synapse groups
        monitors (dictionary): Keys to all spike and state monitors
        monitor (bool, optional): Flag to auto-generate spike and state monitors
        standalone_params (dictionary): Dictionary for all parameters to create
            a standalone network
        sub_blocks (dictionary): Dictionary for all parent building blocks
        input_groups (dictionary): Dictionary containing all possible groups which are
            potential inputs
        output_groups (dictionary): Dictionary containing all possible groups which are
            potential outputs
        hidden_groups (dictionary): Dictionary containing all remaining groups which are
            neither inputs nor outputs
    """

    def __init__(self, name, neuron_eq_builder, synapse_eq_builder,
                 block_params, verbose, monitor=False):
        """This function initializes the BuildBlock parent class. All attributes
        are shared among building_blocks, such as WTA, OCTA etc.

        Args:
            name (str, required): Name of the building_block population
            neuron_eq_builder (class, optional): Class as imported from
                models/neuron_models
            synapse_eq_builder (class, optional): Class as imported from
                models/synapse_models
            block_params (dict): Dictionary which holds building_block specific
                parameters
            verbose (bool, optional): Flag to gain additional information
            monitor (bool, optional): Flag to auto-generate spike and state
                monitors
        """
        self.neuron_eq_builder = neuron_eq_builder
        self.synapse_eq_builder = synapse_eq_builder
        self.params = block_params
        self.verbose = verbose
        self._groups = {}
        self.monitors = {}
        self.monitor = monitor
        self.standalone_params = OrderedDict()
        self.sub_blocks = {}
        self.input_groups = {}
        self.output_groups = {}
        self.hidden_groups = {}

        Nameable.__init__(self, name)

    def __iter__(self):
        """this allows us to iterate over the BrianObjects and directly add the
        Block to a Network

        Returns:
            TYPE: Returns a dictionary wich contains all brian objects
        """
        allBrianObjects = self.groups
        if self.monitor:
            allBrianObjects.update(self.monitors)
        # return iter([{**self.Groups,**self.Monitors}[key] for key in
        # {**self.Groups,**self.Monitors}]) #not Python 2 compatible
        return iter([allBrianObjects[key] for key in allBrianObjects])

    def get_run_args(self):
        """this collects the arguments to cpp main() for standalone run

        Returns:
            TYPE: Arguments which can be changed during/between runs
        """
        # asarray is to remove units. It is the way proposed in the tutorial
        run_args = [str(np.asarray(self.standalone_params[key]))
                    for key in self.standalone_params]
        return run_args

    @property
    def groups(self):
        """ This property will collect all available groups from the respective
        building block. The property follows a recursive strategy to collect all
        available groups. The intention is to easily update all available groups
        for stacked building blocks.
        NOTE Avoid any kind of loops between Building Blocks. Loops are
        forbidden as they lead to infinite collection of groups.

        Returns:
            tmp_groups (dict): Dictionary containing all groups of all sub_blocks
        """
        # Collects all groups including all sub_blocks
        tmp_groups = {}
        tmp_groups.update(self._groups)
        for sub_block in self.sub_blocks:
            # Needs to be groups and not _groups for recursive collection
            for group in self.sub_blocks[sub_block].groups:
                tmp_groups.update(
                    {self.sub_blocks[sub_block].groups[group].name: self.sub_blocks[sub_block].groups[group]})
        return tmp_groups

    def __getitem__(self, key):
        return self.groups[key]

    def _set_tags(self, tags, target_group):
        """ This method allows the user to set a list of tags to a specific
        target group. Normally the tags are already assigned by each building
        block. So this method only adds convinience and a way to replace the
        default tags if this is needed by any user. Typically this should not
        be the user's concern, that's why it is private method.

        Args:
            tags (dict): A dictionary of tags
                {'mismatch' : False,
                'noise' : False,
                'level': 0 ,
                'sign': 'None',
                'target sign': 'None',
                'num_inputs' : 0,
                'bb_type' : 'None',
                'group_type' : 'None',
                'connection_type' : 'None',
                }
            target_group (str): Name of group to set tags
        """
        tags = copy.deepcopy(tags)
        if type(target_group) == str:
            if hasattr(self._groups[target_group], '_tags'):
                self._groups[target_group]._tags.update(tags)
            else:
                self._groups[target_group]._tags = {}
                self._groups[target_group]._tags.update(tags)

        else:
            if hasattr(target_group, '_tags'):
                target_group._tags.update(tags)
            else:
                target_group._tags = {}
                target_group._tags.update(tags)

    def print_tags(self, target_group):
        """ Get the currently set tags for a given group.

        Args:
            target_group (str): Name of group to get tags from
        """
        print(self.groups[target_group]._tags)

    def get_tags(self, target_group):
        """ Get the currently set tags for a given group.

        Args:
            target_group (str): Name of group to get tags from or
               direct `TeiliGroup`

        Returns:
           (dict): Dictionary containing all assigned _tags of provided
               group
        """
        if type(target_group) == str:
            return self.groups[target_group]._tags
        else:
            return target_group._tags

    def get_groups(self, tags):
        """ Get all groups which have a certain set of tags

        Args:
            tags (dict): A dictionary of tags

        Returns:
            target_dict (dict): List of all group objects which
                share the same tags as specified.
        """

        target_dict = {}
        for group in self.groups:

            try:
                if tags.items() <= self.groups[group]._tags.items():
                    target_dict[group] = self.groups[group]
                else:
                    continue
            except AttributeError as e:
                pass
        return target_dict
