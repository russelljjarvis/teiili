#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:41:10 2017

@author: alpha
"""
from brian2 import device,asarray
from collections import OrderedDict

class BuildingBlock:
   
    def __init__(self,name,neuronEq,synapseEq,neuronParams,synapseParams,blockParams,debug):
        self.name           = name
        self.neuronEq       = neuronEq
        self.synapseEq      = synapseEq
        self.neuronParams   = neuronParams
        self.synapseParams  = synapseParams
        self.params         = blockParams
        self.debug          = debug
        self.Groups={}
        self.Monitors={}
        self.standaloneParams=OrderedDict()
                
        
        # this allows us to iterate over the BrianObjects and directly add the Block to a Network
    def __iter__(self):
        allBrianObjects = self.Groups
        allBrianObjects.update(self.Monitors)
        # return iter([{**self.Groups,**self.Monitors}[key] for key in {**self.Groups,**self.Monitors}]) #not Python 2 compatible
        return iter([allBrianObjects[key] for key in allBrianObjects])
    
    
    def get_run_args (self):
        run_args=[str(asarray(self.standaloneParams[key])) for key in self.standaloneParams] # asarray is to remove units. It is the way proposed in the tutorial
        return run_args
        


