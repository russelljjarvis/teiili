#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:41:10 2017

@author: alpha
"""

class BuildingBlock:
   
    def __init__(self,blockName,neuronEq,synapseEq,neuronParams,synapseParams,blockParams,debug):
        self.blockName      = blockName
        self.neuronEq       = neuronEq
        self.synapseEq      = synapseEq
        self.neuronParams   = neuronParams
        self.synapseParams  = synapseParams
        self.params         = blockParams
        self.debug          = debug
        self.Groups={}
        self.Monitors={}
        self.replaceVars=[]
        
