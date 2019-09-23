#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is used to run large parameter sweeps in parallel
It uses cpp code generation (single compile per worker) and multiprocessing, often making the simulation up to
2 orders of magnitude faster than with numpy on a single core.

Not sure, if this works well on Windows, let me know, if you tried.

# TODO: Make this compatible with Brian2GeNN for even more performance using GPU

@author: alpha
"""

import os
import time
import multiprocessing
from multiprocessing.util import Finalize
import shutil
# import psutil
import numpy as np
import json
import traceback
import itertools
import json
from multiprocessing import Pool, cpu_count
import sys
from brian2 import get_device
import pandas as pd


class SweepParameter:
    """
    a single parameter including sweep range (can contain multiple brian2 variables)
    """

    def __init__(self, variables, sweep_range, friendly_name=None, normalize_by=None, round_decimals=9):
        """
        :param variables (list of brian2.core.variables.VariableView): the variables that are used for the sweep that
            have the same range and a changed together
        :param sweep_range (numpy.array): the range of values
        :param friendly_name (str): unique name, that is used for this variable (e.g. for plotting)
            if no friendly name is provided, the full name of the first var is used
        """
        sweep_range = np.asarray(sweep_range)
        self.variables = variables
        self.range = np.round(sweep_range, decimals=round_decimals)
        if any((self.range - sweep_range) > sweep_range*0.01):
            raise Warning('please adjust round_decimals argument of SweepParameter to a higher value, your sweep range'
                          ' seems to have values that are of higher precision')
        self.full_names = [var.group.name + '_' + var.name for var in variables]
        if friendly_name is None:
            self.name = variables[0].name
        else:
            self.name = friendly_name

    def __repr__(self):
        return 'sweep of variable(s) ' + ' and '.join(self.full_names) + ' over range ' + str(self.range) + ' '


class ParameterSweep:
    def __init__(self, net, sweep_parameters, process_results, process_results_args=None,
                 resultdir=os.path.expanduser('~/brian2_parameter_sweep/'),
                 param_condition=None):
        """
        This class makes a parameter sweep of a brian2 network using cpp codegen and multiprocessing

        :param net: the brian2 net
        :param sweep_parameters: list of SweepParameter
        :param process_results: function that is applied to the net after simulation and returns a result, can also be
            used for saving of results after each iteration. It receives the following input:
            process_results(net, resultdir, sweep_parameters, loop_param_dict, process_results_args)
            net is the net after simulation
            sweep_parameters is the list of sweep_parameters that was passed to ParameterSweep
            loop_param_dict is the dict of current parameters that are used for the iteration in which the function is called
            process_results_args are additional arguments that are passed through to the function
        :param resultdir: the directory in which results are stored
        :param param_condition: this is a function that takes the friendly names of the sweep parameters as keyword arguments
            and gives a boolean output that determines if the specific parameter combination should be considered or not.
            This is used if not all parameter combinations should be run. E.g. if you want to run only par1 > par2
            def param_condition(par1,par2):
                return par1 > par2
            By default, all conditions are True
        """
        self.sweep_parameters = sweep_parameters
        self.net = net
        self.process_results = process_results
        if process_results_args is None:
            process_results_args = dict()
        self.process_results_args = process_results_args
        self.resultdir = resultdir
        self.standalone_dir = None
        self.result = None
        if param_condition is None:
            self.param_condition = lambda *args, **kwargs: True  # all conditions True by default
        else:
            self.param_condition = param_condition

    def run(self, standalone_dir, num_workers=cpu_count() - 1, repetitions=1):
        starttime = time.time()
        starttime_pathname = time.strftime("%Y%m%d_%H%M")

        self.sweep_parameters.append(SweepParameter(variables=[],
                                                    sweep_range=range(repetitions), friendly_name='repetition'))
        paramranges = {spar.name: spar.range for spar in self.sweep_parameters}
        parameter_combinations = itertools.product(*[paramranges[k] for k in paramranges.keys()])
        parameter_combinations = ({k: combi[i] for i, k in enumerate(paramranges.keys())} for combi in
                                  parameter_combinations)
        parameter_combinations = [combi for combi in parameter_combinations if self.param_condition(combi)]
        self.parameter_combinations = parameter_combinations
        print('this needs', len(parameter_combinations), 'runs')

        result_subdir = os.path.join(self.resultdir, starttime_pathname + '__'.join(
            [pname for pname in paramranges.keys()]))
        if not os.path.exists(self.resultdir):
            os.makedirs(self.resultdir)
        if not os.path.exists(result_subdir):
            os.makedirs(result_subdir)

        np.savez_compressed(os.path.join(self.resultdir, result_subdir + "_loop_parameters.npz"), paramranges)
        np.savez_compressed(os.path.join(self.resultdir, result_subdir + "_param_combinations.npz"),
                            parameter_combinations)
        with open(os.path.join(self.resultdir, result_subdir + "_loop_parameters.txt"), 'w') as file:
            file.write(json.dumps({key: list(np.asarray(paramranges[key],dtype=float)) for key in paramranges}, indent=3))

        self.standalone_dir = standalone_dir

        pool = Pool(num_workers, initializer=initialize_worker, initargs=(self.standalone_dir, self.net, result_subdir,
                                                                          self.process_results,
                                                                          self.sweep_parameters,
                                                                          self.process_results_args))
        res = pool.map(run_worker, parameter_combinations)
        pool.close()
        pool.join()
        print(res)
        endtime = time.time()
        print("loop took " + str(endtime - starttime) + " seconds")
        print("= " + str((endtime - starttime) / 60) + " minutes")
        print("= " + str((endtime - starttime) / 60 / 60) + " hours")

        df = pd.DataFrame(parameter_combinations)
        df_res = pd.DataFrame(res)
        self.result = pd.concat([df,df_res], axis=1)

        return self.result


def cleanup_standalone():
    global glob_standalone_dir
    print('cleaning up ' + glob_standalone_dir)
    shutil.rmtree(glob_standalone_dir)


def initialize_worker(standalone_dir, net, resultdir, process_results, sweep_parameters,
                      process_results_args, verbose = False):
    '''
    this function initializes a worker (only runs once (at the beginning) per worker)
    it compiles a cpp_standalone in its own directory and stores global variables
    of the standalone function and of all other variables that a worker needs
    global means local to one worker here!
    there seems to be no other easy way to get variables from the initializer
    to the worker functions
    '''
    global init_failed
    init_failed = None

    try:
        global glob_standalone_dir
        glob_standalone_dir = standalone_dir + '_' + str(os.getpid())

        get_device().build_options['directory'] = os.path.abspath(glob_standalone_dir)
        get_device().project_dir = os.path.abspath(glob_standalone_dir)

        # define globals that can be used by the worker, there seems to be no better way to pass variables to the worker
        global glob_net
        global glob_resultdir
        global glob_process_results
        global glob_process_results_args
        global glob_num_done
        global glob_sweep_param_dict
        global glob_verbose

        glob_net = net
        glob_resultdir = resultdir
        glob_process_results = process_results
        glob_process_results_args = process_results_args
        glob_num_done = 0
        glob_sweep_param_dict = {sp.name: sp for sp in sweep_parameters}
        glob_verbose = verbose

        # compile once and just copy the folder to different locations in order to avoid mixing up result files
        shutil.copytree(standalone_dir, os.path.abspath(glob_standalone_dir))

        # this should avoid that they all start compiling at the same time,
        # which uses too much memory and cpu
        try:
            workernumber = int(multiprocessing.current_process().name.split('-')[1])
            print('initializing', workernumber)
            # time.sleep((workernumber - 1) * 10)
        except IndexError:  # if no multiprocessig
            pass

        print(get_device().build_options['directory'])
        print('project_dir:', get_device().project_dir)

        # Register a finalizer
        Finalize(None, cleanup_standalone, exitpriority=16)
    except:  # manual handling of init failure to stop workers
        init_failed = sys.exc_info()[1]
        Finalize(None, cleanup_standalone, exitpriority=16)


def run_worker(loop_param_dict):
    '''
    runs the compiled standalone with the tuple given as loop params
    the names of the parameters have to be given in the initialization of the worker
    '''
    global init_failed
    if init_failed is not None:
        raise RuntimeError(init_failed) from init_failed

    try:
        global glob_net
        global glob_resultdir
        global glob_process_results
        global glob_process_results_args
        global glob_num_done

        if glob_verbose:
            print('parameters for this run:')
            for par in loop_param_dict.keys():
                print(str(par), ' = ', loop_param_dict[par])

        for par in loop_param_dict:
            for full_par_name in glob_sweep_param_dict[par].full_names:
                glob_net.standalone_params[full_par_name] = loop_param_dict[par]

        glob_net.run(verbose = False)
        output = glob_process_results(glob_net, glob_resultdir, glob_sweep_param_dict, loop_param_dict, glob_process_results_args)

        glob_num_done += 1
        #print(' ')
        #print('####################################')
        print(multiprocessing.current_process().name, ': ')
        print('done ', glob_num_done)
        #print('####################################')
        #print(' ')
        return output

    except KeyboardInterrupt as ki:
        raise ki
    except Exception as e:
        print(str(e))
        traceback.print_tb(e.__traceback__)
        return str(e)
