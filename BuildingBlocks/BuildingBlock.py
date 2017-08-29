#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:41:10 2017

@author: alpha
"""
from brian2 import asarray
from collections import OrderedDict

# TODO: There is a problem, when 2 BBs of the same kind are added to another BB, as the names collide.
# We could use a hierarchy of blocks


# TODO: Important all Monitors of Building blocks have to be named and named uniquely!
# Otherwise they will not be found, when a Network is rerun in standalone mode after rebuild without recompile

class BuildingBlock:

    def __init__(self, name, neuronEq, synapseEq, neuronParams, synapseParams, blockParams, debug, monitor=False):

        self.name = name
        self.neuronEq = neuronEq
        self.synapseEq = synapseEq
        self.neuronParams = neuronParams
        self.synapseParams = synapseParams
        # self.plasticSynEq = plasticSynEq
        # self.plasticSynParams = plasticSynParams
        self.params = blockParams
        self.debug = debug
        # self.plastic = plastic
        self.Groups = {}
        self.Monitors = {}
        self.monitor = monitor
        self.standaloneParams = OrderedDict()

    def __iter__(self):
        "this allows us to iterate over the BrianObjects and directly add the Block to a Network"
        allBrianObjects = self.Groups
        if self.monitor:
            allBrianObjects.update(self.Monitors)
        # return iter([{**self.Groups,**self.Monitors}[key] for key in {**self.Groups,**self.Monitors}]) #not Python 2 compatible
        return iter([allBrianObjects[key] for key in allBrianObjects])

    def get_run_args(self):
        "this collects the arguments to cpp main() for stadalone run"
        # asarray is to remove units. It is the way proposed in the tutorial
        run_args = [str(asarray(self.standaloneParams[key]))
                    for key in self.standaloneParams]
        return run_args
