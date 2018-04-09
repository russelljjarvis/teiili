#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: alpren
# @Date:   2017-08-2 18:16:28
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-18 17:14:02

"""
Wrapper class for Netwotrk class of brian2 to provide a more flexible
interface, especially to change parameters on the fly after compilation
"""

# todo function that plots the whole network
import os
import time
from collections import OrderedDict
import pprint

from brian2 import Network, second, device, get_device, ms, all_devices
from brian2 import SpikeMonitor, StateMonitor, NeuronGroup, Synapses
from NCSBrian2Lib.tools.cpptools import build_cpp_and_replace,\
    print_dict, params2run_args
from NCSBrian2Lib.building_blocks.building_block import BuildingBlock


class NCSNetwork(Network):
    """this is a subclass of brian2.Network and does the same thing plus
    some additional methods for convenience
    and functionality to allow real time plotting and gui

    Attributes:
        blocks (list): Description
        hasRun (bool): Flag to indicate if network had been simulated already
        standaloneParams (dict): Dictionary of standalone parameters
    """
    hasRun = False

    @property
    def spikemonitors(self):
        return {att.name : att for att in self.__dict__['objects'] if type(att) == SpikeMonitor}

    @property
    def statemonitors(self):
        return {att.name : att for att in self.__dict__['objects'] if type(att) == StateMonitor}

    @property
    def neurongroups(self):
        return {att.name : att for att in self.__dict__['objects'] if type(att) == NeuronGroup}

    @property
    def synapses(self):
        return {att.name : att for att in self.__dict__['objects'] if type(att) == Synapses}


    def __init__(self, *objs, **kwds):
        """Summary

        Args:
            *objs: Description
            **kwds: Description
        """
        self.blocks = []
        self.standaloneParams = OrderedDict()
        self.standaloneParams['duration'] = 0 * ms

        # Network.__init__(self, *objs, **kwds)
        Network.__init__(self)

    def add_standaloneParams(self, **params):
        """Function to a add standalone parameter to the standaloneParam dict.
        These parammeters can be changed after building w/o recompiling the network

        Args:
            **params (dict, required): Dictionary with parameter to be added to
                stanaloneParamss
        """
        for key in params:
            self.standaloneParams[key] = params[key]

    def add(self, *objs):
        """does the same thing as Network.add (adding Groups to the Network)
        It furthermore adds the groups to a list for the parameter gui

        Args:
            *objs: arguments (brian2 objects which should be added to the network)
        """
        Network.add(self, *objs)

        for obj in objs:
            if isinstance(obj, BuildingBlock):
                self.blocks.append(obj)
                print('added to network building blocks: ', obj)

            try:
                # add all standaloneParams from BBs, neurons and synapses
                # to Network.standaloneParams
                self.standaloneParams.update(obj.standaloneParams)
            except AttributeError:
                pass

    def build(self, report=None, report_period=10 * second,
              namespace=None, profile=True, level=0, recompile=False,
              standaloneParams=None, clean=True):
        """Building the network

        Args:
            report (bool, optional): Flag to provide more detailed information during run
            report_period (brian2.unit, optional): how often should be reported (unit time)
            namespace (None, optional): Description
            profile (bool, optional): Flag to enable profiling of the network in terms of
                executin time, resources etc.
            level (int, optional): Description
            recompile (bool, optional): Flag to indicate if network should rather be recompiled
                than used based on a prioir build. Set this to False if you want to only change
                parameters rather than network topology
            standaloneParams (dict, optional): Dictionary with standalone parametes which
                should be changed
            clean (bool, optional): Flag to clean-up standalone directory
        """
        if get_device() == all_devices['cpp_standalone']:
            if recompile or not NCSNetwork.hasRun:

                print('building network...')
                Network.run(self, duration=0 * ms, report=report, report_period=report_period,
                            namespace=namespace, profile=profile, level=level + 1)
                NCSNetwork.hasRun = True

                if standaloneParams is None:
                    standaloneParams = self.standaloneParams

                build_cpp_and_replace(standaloneParams, get_device(
                ).build_options['directory'], clean=clean)
            else:
                print("""Network was not recompiled, standaloneParams are changed,
                      but Network structure is not!
                      This might lead to unexpected behavior.""")
        else:
            print('Network was compiled, as you have not set the device to \
                  cpp_standalone, you can still run() it using numpy code generation')

    def run(self, duration=None, standaloneParams=dict(), **kwargs):
        """Wrapper function to simulate a network given the duration time.
        Parameters which should be changeable especially after cpp compilation need to
        be provided to standaloneParams

        Args:
            duration (brain2.unit, optional): Simulation time in ms, i.e. 100 * ms
            standaloneParams (dict, optional): Dictionary whichs keys refer to parameters
                which should be changeable in cpp standalone mode
            **kwargs (optional): addtional keyword arguments
        """
        # kwargs are if you want to use the StandaloneNetwork as a simple brian2
        # network with numpy code generation

        if get_device() == all_devices['cpp_standalone']:

            if all_devices['cpp_standalone'].build_on_run:
                # this does not really make sense, as the whole point here is to
                # avoid recompilation on every run, but some people might still want to use it
                all_devices['cpp_standalone'].build_on_run = False
                self.build(**kwargs)

            if standaloneParams == {}:
                standaloneParams = self.standaloneParams

            if duration is not None:
                standaloneParams['duration'] = duration

            startSim = time.time()
            # run simulation
            print_dict(standaloneParams)
            run_args = params2run_args(standaloneParams)
            device.run(directory=os.path.abspath(
                get_device().build_options['directory']), with_output=True, run_args=run_args)
            end = time.time()
            print('simulation in c++ took ' + str(end - startSim) + ' sec')
            print('simulation done!')

        else:
            if duration is not None:
                standaloneParams['duration'] = duration
            else:
                duration = standaloneParams['duration']

            Network.run(self, duration=duration, **kwargs)

    def printParams(self):
        """This functions prints all standalone parameters (cpp standalone network)
        """
        pprinter = pprint.PrettyPrinter()
        pprinter.pprint(self.standaloneParams)
