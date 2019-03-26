#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module provides the parent class for (neuronal) building blocks.

Todo:
    * Hierarchy of building_blocks.
        There is a problem when 2 BBs of the same kind are added to another BB, as the
        names collide.
    * Monitor naming.
        Important all Monitors of Building blocks have to be named and named uniquely!
        Otherwise they will not be found, when a Network is rerun in standalone mode
        after rebuild without recompile
"""

# @Author: alpren, mmilde
# @Date:   2017-07-27 10:46:44


from brian2 import asarray
from collections import OrderedDict


class BuildingBlock:
    """This class is the parent class to all building blocks such as WTA, SOM etc.

    Attributes:
        name (str, required): Name of the building_block population
        neuron_eq_builder (class, optional): neuron class as imported from models/neuron_models
        synapse_eq_builder (class, optional): synapse class as imported from models/synapse_models
        params (dictionary, optional): Dictionary containing all relevant parameters for
            each building block
        debug (bool, optional): Flag to gain additional information
        groups (dictionary): Keys to all neuron and synapse groups
        monitors (dictionary): Keys to all spike and state monitors
        monitor (bool, optional): Flag to auto-generate spike and state monitors
        standalone_params (dictionary): Dictionary for all parameters to create a standalone network
        sub_blocks (dictionary): Dictionary for all parent building blocks
        input (dictionary): Dictionary containing all possible groups which are potential inputs
        output (dictionary): Dictionary containing all possible groups which are potential outputs
        hidden (dictionary): Dictionary containing all remaining groups which are neither
            inputs nor outputs 
    """

    def __init__(self, name, neuron_eq_builder, synapse_eq_builder, block_params, debug, monitor=False):
        """This function initializes the BuildBlock parent class. All attributes are shared among
        building_blocks, such as WTA, OCTA etc.

        Args:
            name (str, required): Name of the building_block population
            neuron_eq_builder (class, optional): Class as imported from models/neuron_models
            synapse_eq_builder (class, optional): Class as imported from models/synapse_models
            block_params (dict): Dictionary which holds building_block specific parameters
            debug (bool, optional): Flag to gain additional information
            monitor (bool, optional): Flag to auto-generate spike and state monitors
        """
        self.name = name
        self.neuron_eq_builder = neuron_eq_builder
        self.synapse_eq_builder = synapse_eq_builder
        self.params = block_params
        self.debug = debug
        self._groups = {}
        self.monitors = {}
        self.monitor = monitor
        self.standalone_params = OrderedDict()
        self.sub_blocks{}
        self.input = {}
        self.output = {}
        self.hidden = {}

    def __iter__(self):
        """this allows us to iterate over the BrianObjects and directly add the Block to a Network

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
        run_args = [str(asarray(self.standalone_params[key]))
                    for key in self.standalone_params]
        return run_args

    @property
    def groups(self):
        """ This property will collect all available groups from the respective building block.
        To use it as a property it can easily be updated, if for example one building block is
        incorporated in another one.
        Recursive strategy
        Add note that loops between bb are forbidden and lead to infinite loops
        """
        # Collects all groups including all sub_blocks
        tmp_groups = {}
        tmp_groups.update(_groups)
        for g in self.sub_blocks
             tmp_groups.update(g.groups)
        return tmp_groups


    def get_eq_builder_dict(self):
        # Check name in dict if nkey erro than use default if not use specified eq builder obj.
