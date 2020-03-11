#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrapper class for Network class of brian2.

This wrapper provides a more flexible interface, especially to
change parameters on the fly after compilation

Todo:
    * function that plots the whole network
"""
# @Author: alpren
# @Date:   2017-08-2 18:16:28

import os
import time
from collections import OrderedDict
import pprint

from brian2 import Network, second, device, get_device, ms, all_devices
from brian2 import SpikeMonitor, StateMonitor, NeuronGroup, Synapses, Quantity
from teili.tools.cpptools import build_cpp_and_replace, \
    print_dict, params2run_args
from teili.building_blocks.building_block import BuildingBlock


class TeiliNetwork(Network):
    """This is a subclass of brian2.Network.

    This subclass does the same thing plus some additional methods for
    convenience and functionality to allow real time plotting and gui.

    Attributes:
        blocks (list): Description
        has_run (bool): Flag to indicate if network has been simulated already.
        standalone_params (dict): Dictionary of standalone parameters.
        thread (TYPE): Description
    """
    has_run = False

    def __init__(self, *objs, **kwds):
        """All parameters are passed to the Brian2 network.

        Args:
            *objs: Description
            **kwds: Description
        """
        self.blocks = []
        self.standalone_params = OrderedDict()
        self.standalone_params['duration'] = 0 * ms
        self.thread = None

        Network.__init__(self, *objs, **kwds)
        # Network.__init__(self)

    @property
    def spikemonitors(self):
        """property to conveniently get all spikemonitors in the network

        Returns:
            dict: A dictionary of all spike monitors (e.g. for looping over them)
        """
        return {att.name: att for att in self.__dict__['objects'] if isinstance(att, SpikeMonitor)}

    @property
    def statemonitors(self):
        """property to conveniently get all statemonitors in the network

        Returns:
            dict: A dictionary of all statemonitors (e.g. for looping over them)
        """
        return {att.name: att for att in self.__dict__['objects'] if isinstance(att, StateMonitor)}

    @property
    def neurongroups(self):
        """property to conveniently get all neurongroups in the network

        Returns:
            dict: A dictionary of all neurongroups (e.g. for looping over them)
        """
        return {att.name: att for att in self.__dict__['objects'] if isinstance(att, NeuronGroup)}

    @property
    def synapses(self):
        """property to conveniently get all synapses in the network

        Returns:
            dict: A dictionary of all synapses (e.g. for looping over them).
        """
        return {att.name: att for att in self.__dict__['objects'] if isinstance(att, Synapses)}

    def add_standalone_params(self, **params):
        """Function to a add standalone parameter to the standaloneParam dict.

        These parameters can be changed after building w/o recompiling the network.

        Args:
            **params (dict, required): Dictionary with parameter to be added to
                standalone_params.
        """
        for key in params:
            self.standalone_params[key] = params[key]

    def add(self, *objs):
        """Does the same thing as Network.add (adding Groups to the Network)

        It also adds the groups to a list for the parameter gui.

        Args:
            *objs: arguments (brian2 objects which should be added to the network).
        """
        Network.add(self, *objs)

        for obj in objs:
            if isinstance(obj, BuildingBlock):
                self.blocks.append(obj)
                print('added to network building blocks: ', obj)

            try:
                # add all standalone_params from BBs, neurons and synapses
                # to Network.standalone_params
                self.standalone_params.update(obj.standalone_params)
            except AttributeError:
                pass

    def build(self, report="stdout", report_period=10 * second,
              namespace=None, profile=True, level=0, recompile=False,
              standalone_params=None, clean=True, verbose=True):
        """Building the network.

        Args:
            report (bool, optional): Flag to provide more detailed information during run.
            report_period (brian2.unit, optional): How often should be reported (unit time).
            namespace (None, optional): Namespace containing all names of the network to be built.
            profile (bool, optional): Flag to enable profiling of the network in terms of
                execution time, resources etc. .
            level (int, optional): Description.
            recompile (bool, optional): Flag to indicate if network should rather be recompiled
                than used based on a prior build. Set this to False if you want to only change
                parameters rather than network topology.
            standalone_params (dict, optional): Dictionary with standalone parameters which
                should be changed.
            clean (bool, optional): Flag to clean-up standalone directory.
        """
        if get_device() == all_devices['cpp_standalone']:
            if recompile or not TeiliNetwork.has_run:

                print('building network...')
                Network.run(self, duration=0 * ms, report=report, report_period=report_period,
                            namespace=namespace, profile=profile, level=level + 1)
                TeiliNetwork.has_run = True

                if standalone_params is None:
                    standalone_params = self.standalone_params

                try:
                    directory = get_device().build_options['directory']
                except KeyError:
                    directory = os.path.join(os.path.expanduser("~"), "Brian2Standalone")

                build_cpp_and_replace(standalone_params, standalone_dir=directory,
                                      clean=clean, verbose=verbose)
            else:
                print("""Network was not recompiled, standalone_params are changed,
                      but Network structure is not!
                      This might lead to unexpected behavior.""")
        else:
            print('Network was not compiled (net.build was ignored), as you have not set the device to \
                  cpp_standalone. You can still run() it using numpy code generation.')

    def run(self, duration=None, standalone_params=dict(), verbose=True, **kwargs):
        """Wrapper function to simulate a network for the given duration.

        Parameters which should be changeable, especially after cpp compilation, need to
        be provided to standalone_params.

        Args:
            duration (brain2.unit, optional): Simulation time in s, i.e. 1000 * ms.
            standalone_params (dict, optional): Dictionary whose keys refer to parameters
                which should be changeable in cpp standalone mode.
            verbose (bool, optional) : set to False if you don't want the prints, it is set to True
                by default, as there are things that can go wrong during string replacement etc. so it is
                better to have a look manually.
            **kwargs (optional): Additional keyword arguments.
        """
        # kwargs are if you want to use the StandaloneNetwork as a simple brian2
        # network with numpy code generation

        if get_device() == all_devices['cpp_standalone']:

            if all_devices['cpp_standalone'].build_on_run:
                # this does not really make sense, as the whole point here is to
                # avoid recompilation on every run, but some people might still
                # want to use it
                all_devices['cpp_standalone'].build_on_run = False
                print('building network, as you have set build_on_run = True')
                self.build(**kwargs)

            if standalone_params == {}:
                standalone_params = self.standalone_params

            if duration is not None:
                standalone_params['duration'] = duration
            if verbose:
                start_sim = time.time()
                print_dict(standalone_params)
            # run simulation
            run_args = params2run_args(standalone_params)
            directory = os.path.abspath(
                get_device().build_options['directory'])
            if not os.path.isdir(directory):
                os.mkdir(directory)
            if verbose:
                print('standalone files are written to: ', directory)

            device.run(directory=directory,
                       with_output=True, run_args=run_args)

            if verbose:
                end = time.time()
                print('simulation in c++ took ' + str(end - start_sim) + ' sec')
                print('simulation done!')

        else:
            if duration is not None:
                standalone_params['duration'] = duration
            else:
                duration = standalone_params['duration']

            Network.run(self, duration=duration, **kwargs)

    def run_as_thread(self, duration, **kwargs):
        """Running network in a thread.

        Args:
            duration (brain2.unit, optional): Simulation time in ms, i.e. 100 * ms.
            **kwargs (optional): Additional keyword arguments.
        """
        import threading
        from functools import partial
        self.thread = threading.Thread(
            target=partial(self.run, duration, **kwargs))
        self.thread.start()

    def print_params(self):
        """This functions prints all standalone parameters (cpp standalone network).
        """
        pprinter = pprint.PrettyPrinter()
        pprinter.pprint(self.standalone_params)
