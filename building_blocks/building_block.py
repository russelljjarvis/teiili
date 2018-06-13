#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module provides the parent class for (neuronal) building blocks.

Todo:
    * Hierarchy of building_blocks.
        There is a problem, when 2 BBs of the same kind are added to another BB, as the
        names collide. There is a problem, when 2 BBs of the same kind are added to
        another BB, as the names collide.
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
        debug (bool, optional): Flag to gain additional information
        Groups (dictionary): Keys to all neuron and synapse groups
        monitor (bool, optional): Flag to auto-generate spike and statemonitors
        Monitors (dictionary): Keys to all spike- and statemonitors
        name (str, required): Name of the building_block population
        neuron_eq_builder (class, optional): neuron class as imported from models/neuron_models
        params (TYPE): Description
        standalone_params (dictionary): Dictionary which holds all parameters to create a standalone network
        synapse_eq_builder (class, optional): synapse class as imported from models/synapse_models
    """

    def __init__(self, name, neuron_eq_builder, synapse_eq_builder, block_params, debug, monitor=False):
        """Summary

        Args:
            name (str, required): Name of the building_block population
            neuron_eq_builder (class, optional): neuron class as imported from models/neuron_models
            synapse_eq_builder (class, optional): synapse class as imported from models/synapse_models
            blockParams (dict): Dictionary which holds building_block specific parameters
            debug (bool, optional): Flag to gain additional information
            monitor (bool, optional): Flag to auto-generate spike and statemonitors
        """
        self.name = name
        self.neuron_eq_builder = neuron_eq_builder
        self.synapse_eq_builder = synapse_eq_builder
        self.params = block_params
        self.debug = debug
        self.Groups = {}
        self.Monitors = {}
        self.monitor = monitor
        self.standalone_params = OrderedDict()

    def __iter__(self):
        """this allows us to iterate over the BrianObjects and directly add the Block to a Network

        Returns:
            TYPE: Description
        """
        allBrianObjects = self.Groups
        if self.monitor:
            allBrianObjects.update(self.Monitors)
        # return iter([{**self.Groups,**self.Monitors}[key] for key in
        # {**self.Groups,**self.Monitors}]) #not Python 2 compatible
        return iter([allBrianObjects[key] for key in allBrianObjects])

    def get_run_args(self):
        """this collects the arguments to cpp main() for stadalone run

        Returns:
            TYPE: Description
        """
        # asarray is to remove units. It is the way proposed in the tutorial
        run_args = [str(asarray(self.standalone_params[key]))
                    for key in self.standalone_params]
        return run_args
