#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:41:10 2017

@author: alpha
"""

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
        self.replaceVars=[]
                
        
        # this allows us to iterate over the BrianObjects and directly add the Block to a Network
    def __iter__(self):
        return iter([{**self.Groups,**self.Monitors}[key] for key in {**self.Groups,**self.Monitors}])
        
