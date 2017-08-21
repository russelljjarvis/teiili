#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:16:28 2017

@author: alpha
"""

#todo function that plots the whole network

from brian2 import Network, second, device, set_device, prefs, get_device,ms
from NCSBrian2Lib.Tools.cppTools import buildCppAndReplace, collectStandaloneParams, run_standalone, printDict, params2run_args
from NCSBrian2Lib.BuildingBlocks.BuildingBlock import BuildingBlock
import time
from collections import OrderedDict
import pprint


class StandaloneNetwork(Network):

    hasRun = False

    def __init__(self, isStandalone = True, standaloneDir = 'output', *objs, **kwds):


        self.isStandalone = isStandalone
        self.standaloneDir = standaloneDir

        self.blocks = []
        self.standaloneParams = OrderedDict()
        self.standaloneParams['duration'] = 0 *ms

        Network.__init__(self, *objs, **kwds)

    def add_standaloneParams(self, **params):

        for key in params:
            self.standaloneParams[key] = params[key]

    def add(self, *objs):

        Network.add(self, *objs)

        for obj in objs:
            if isinstance(obj, BuildingBlock):
                self.blocks.append(obj)
                print('added to network building blocks: ' , obj)

            try:
                # add all standaloneParams from BBs, neurons and synapses to Network.standaloneParams
                self.standaloneParams.update(obj.standaloneParams)
            except AttributeError:
                pass


    def build(self, report=None, report_period=10*second,
            namespace=None, profile=True, level=0, recompile=False, standaloneParams=None, clean=True):

        if recompile or not StandaloneNetwork.hasRun:

            print('building network...')
            Network.run(self, duration = 0*ms, report=report, report_period=report_period,
                        namespace=namespace, profile=profile, level=level+1)
            StandaloneNetwork.hasRun = True

            if standaloneParams is None:
                standaloneParams = self.standaloneParams

            buildCppAndReplace(standaloneParams, self.standaloneDir, clean=clean)
        else:
            print('Network was not recompiled, standaloneParams are changed, but Network structure is not!')
            print('This might lead to unexpected behaviour.')


    def run(self, duration = None, standaloneParams=None):

        if standaloneParams is None:
            standaloneParams = self.standaloneParams

        if duration is not None:
            standaloneParams['duration'] = duration

        startSim = time.time()
        # run simulation
        printDict(standaloneParams)
        run_args = params2run_args(standaloneParams)
        device.run(directory=self.standaloneDir, with_output=True, run_args=run_args)
        end = time.time()
        print ('simulation in c++ took ' + str(end - startSim) + ' sec')
        print('simulation done!')

    def printParams(self):
        pp = pprint.PrettyPrinter()
        pp.pprint(self.standaloneParams)
