#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collection of tools to add some features to cpp standalone mode
"""
# @Author: alpren
# @Date:   2017-07-28 19:02:05

import os
import time
import warnings
import numpy as np
from collections import OrderedDict
from brian2 import prefs, device, codegen, set_device


def activate_standalone(directory='Brian2Network_standalone', build_on_run=False):
    """Enables cpp standalone mode

    Args:
        directory (str, optional): Standalone directory containing all compiled files
        build_on_run (bool, optional): Flag to (re-)build network before simulating
    """
    set_device('cpp_standalone', directory=directory, build_on_run=build_on_run)
    device.reinit()
    device.activate(directory=directory, build_on_run=build_on_run)
    device.project_dir = os.path.abspath(directory)


def deactivate_standalone():
    """Disables cpp standalone mode
    """
    set_device('runtime')


def build_cpp_and_replace(standalone_params, standalone_dir='output', clean=True, do_compile=True):
    """Builds cpp standalone network and replaces variables/parameters with standalone_params
    This does string replacement in the generated c++ code.

    Args:
        standalone_params (dict, required): Dictionary containing all parameters which can be changed
            after building the network
        standalone_dir (str, optional): Directory containing output generated by network
        clean (bool, optional): Flag to clean build network
        do_compile (bool, optional): Flag to compile network
    """
    startBuild = time.time()
    prefs['codegen.cpp.extra_compile_args_gcc'].append('-std=c++14')
    prefs['codegen.cpp.extra_compile_args_msvc'].append('/std:c++14')
    prefs['codegen.cpp.headers'].append('<string>')
    # prefs['codegen.cpp.extra_compile_args_gcc'].append('-std=c++11')
    device.build(compile=False, run=False, directory=standalone_dir, clean=clean, debug=False)

    end = time.time()
    print('build took ' + str(end - startBuild) + ' sec')

    # ===============================================================================
    # Code that needs to be added to the main.cpp file

    maincppPath = os.path.join(os.getcwd(), standalone_dir, 'main.cpp')  # this should always be the correct path
    replace_vars = [key for key in standalone_params]
    replace_variables_in_cpp_code(replace_vars, replace_file_location=maincppPath)
    # ===============================================================================
    # compile
    if do_compile:
        startMake = time.time()
        # out = check_output(["make","-C","~/Code/SOM_standalone"])
        compiler, args = codegen.cpp_prefs.get_compiler_and_args()
        device.compile_source(directory=standalone_dir, compiler=compiler, clean=clean, debug=False)
        # print(out)
        end = time.time()
        print('make took ' + str(end - startMake) + ' sec')
        print('\n\nstandalone was built and compiled, ready to run!')
    else:
        print('\n\nstandalone was built, ready to compile!')


def replace_variables_in_cpp_code(replace_vars, replace_file_location):
    '''Replaces a list of variables in CPP code for standalone code generation with changeable parameters
    and it adds duration as a changeable parameter (it is always the first argument)


    Args:
        replace_vars (list, str): List of strings, variables that are replaced
        replace_file_location (str): Location of the file in which the variables are replaced
    '''

    # generate arg code
    cppArgCode = ""
    for ivar, rvar in enumerate(replace_vars):
        cppArgCode += """\n float {replvar}_p = std::stof(argv[{num}],NULL);
    std::cout << "variable {replvar} is argument {num} with value " << {replvar}_p << std::endl;\n""".format(
            num=(ivar + 1), replvar=rvar)
        print("variable {replvar} is main() argument {num}".format(num=(ivar + 1), replvar=rvar))

    print('\n*********************************\n')

    # read main.cpp
    f = open(replace_file_location, "r")
    contents = f.readlines()
    f.close()

    # insert arg code
    for i_line, line in enumerate(contents):
        if "int main(int argc, char **argv)" in line:
            insertLine = i_line + 2
    contents.insert(insertLine, cppArgCode)

    # replace var code
    replaceTODOlist = list(
        replace_vars)  # copy replace_vars in order to create a todolist where we can delete elements that are done
    f = open(replace_file_location, "w")
    for i_line, line in enumerate(contents):
        replaced = False
        for rvar in replace_vars:
            if rvar + "[i]" in line:  # for array variables
                replaced = True
                keepFirstPart = line.split('=', 1)[0]
                f.write(keepFirstPart + '= ' + rvar + '_p;\n')
                print("replaced array " + rvar + " in line " + str(i_line))
                replaceTODOlist = [elem for elem in replaceTODOlist if not elem == rvar]  # check element in todolist
            if rvar + "[0]" in line:  # for scalar (shared) variables
                replaced = True
                keepFirstPart = line.split('=', 1)[0]
                f.write(keepFirstPart + '= ' + rvar + '_p;\n')
                print("replaced scalar " + rvar + " in line " + str(i_line))
                replaceTODOlist = [elem for elem in replaceTODOlist if not elem == rvar]  # check element in todolist
        if ('.run(' in line) and ("duration" in replace_vars):
            replaced = True
            keepNetworkName = line.split('.', 1)[0]  # this is actually the same as net.name
            keepSecondPart = line.split(',', 1)[1]
            f.write(keepNetworkName + '.run(duration_p,' + keepSecondPart)
            print("replaced duration in line " + str(i_line))
            replaceTODOlist = [elem for elem in replaceTODOlist if not elem == 'duration']
        if not replaced:
            f.write(line)
    f.close()
    if len(replaceTODOlist) > 0:
        warnings.warn("could not find matching variables in cpp code for " + str(replaceTODOlist),
                      Warning)  # warning, items left in todolist
        print('\n* * * * * * * * * * * * * * * * *\n NOT all variables successfully replaced in cpp code! \n')
        print("could not find matching variables in cpp code for " + str(replaceTODOlist))

        # maybe we should raise an exception here as this is rather serious?
        raise Exception("could not find matching variables in cpp code for " + str(replaceTODOlist))
    else:
        print('\n*********************************\nall variables successfully replaced in cpp code! \n')


def print_dict(pdict):
    """Wrapper function to print dictionary

    Args:
        pdict (dictionary): Dictionary to be printed
    """
    print('The following parameters are set for this run:')
    for key in pdict:
        print(key, ' = ', pdict[key])


def params2run_args(standalone_params):
    """Add standalone parameter to run arguments

    Args:
        standalone_params (dict): Dictionary containing standalone parameters to be added to
            run arguments

    Returns:
        list: run arguments
    """
    # asarray is to remove units. It is the way proposed in the tutorial
    run_args = [str(np.asarray(standalone_params[key])) for key in standalone_params]
    return run_args


def collect_standalone_params(params=OrderedDict(), *building_blocks):
    '''This just collect the parameters of all buildingblocks and adds additional
    parameters (not from buildingblocks)

    Args:
        params (OrderedDict, optional): Dictionary with parameters. Needs to be ordered.
        *buildingBlocks: The network building block to assign standalone parameters to

    Returns:
        dict: standalone parameters
    '''
    standalone_params = OrderedDict()
    for block in building_blocks:
        standalone_params.update(block.standalone_params)

    standalone_params.update(params)
    return standalone_params


def run_standalone(standalone_params):
    """Runnung standalone networks

    Args:
        standalone_params (dict): Dictionary of standalone parameters
    """
    start_sim = time.time()
    # run simulation
    print_dict(standalone_params)
    run_args = params2run_args(standalone_params)
    device.run(directory=device.project_dir, with_output=True, run_args=run_args)
    end_sim = time.time()
    print('simulation in c++ took ' + str(end_sim - start_sim) + ' sec')
    print('simulation done!')
