#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 19:02:05 2017

@author: alpha
"""

from brian2 import implementation, check_units, ms, exp, mean, diff, declare_types,\
     prefs, figure, subplot, plot, xlim, ylim, ones, zeros, xticks, xlabel, ylabel,\
     device, codegen, asarray, set_device
from brian2 import *
import numpy as np
import os
import time
import warnings
from collections import OrderedDict


def activate_standalone(directory='Brian2Network_standalone', build_on_run=False):
    set_device('cpp_standalone', directory=directory, build_on_run=build_on_run)
#print('Network has been built already, rebuild it...')
    device.reinit()
    device.activate(directory=directory, build_on_run=build_on_run)

def deactivate_standalone():
    set_device('runtime')

def buildCppAndReplace(standaloneParams, standaloneDir='output', clean=True, do_compile=True):

    startBuild = time.time()
    prefs['codegen.cpp.extra_compile_args_gcc'].append('-std=c++14')
    #prefs['codegen.cpp.extra_compile_args_gcc'].append('-std=c++11')
    device.build(compile=False, run=False, directory=standaloneDir, clean=clean, debug=False)

    end = time.time()
    print ('build took ' + str(end - startBuild) + ' sec')

    #===============================================================================
    # Code that needs to be added to the main.cpp file

    maincppPath = os.path.join(os.getcwd(), standaloneDir, 'main.cpp')  # this should always be the correct path
    replaceVars = [key for key in standaloneParams]
    replaceVariablesInCPPcode(replaceVars, replaceFileLocation=maincppPath)
    #===============================================================================
    # compile
    if do_compile:
        startMake = time.time()
        #out = check_output(["make","-C","~/Code/SOM_standalone"])
        compiler, args = codegen.cpp_prefs.get_compiler_and_args()
        device.compile_source(directory=standaloneDir, compiler=compiler, clean=clean, debug=False)
        # print(out)
        end = time.time()
        print ('make took ' + str(end - startMake) + ' sec')
        print('\n\nstandalone was built and compiled, ready to run!')
    else:
        print('\n\nstandalone was built, ready to compile!')


def replaceVariablesInCPPcode(replaceVars, replaceFileLocation):
    ''' replaces a list of variables in CPP code for standalone code generation with changeable parameters
    and it adds duration as a changeable parameter (it is always the first argument)
    @params:
        replaceVars : List of strings, variables that are replaced
        replaceFileLocation : string, location of the file in which the variables are replaced
    '''
    # generate arg code
    cppArgCode = ""
    for ivar, rvar in enumerate(replaceVars):
        cppArgCode += """\n float {replvar}_p = std::stof(argv[{num}],NULL);
    std::cout << "variable {replvar} is argument {num} with value " << {replvar}_p << std::endl;\n""".format(num=(ivar + 1), replvar=rvar)
        print("variable {replvar} is main() argument {num}".format(num=(ivar + 1), replvar=rvar))

    print('\n*********************************\n')

    # read main.cpp
    f = open(replaceFileLocation, "r")
    contents = f.readlines()
    f.close()

    # insert arg code
    for i_line, line in enumerate(contents):
        if "int main(int argc, char **argv)" in line:
            insertLine = i_line + 2
    contents.insert(insertLine, cppArgCode)

    # replace var code
    replaceTODOlist = list(replaceVars)  # copy replaceVars in order to create a todolist where we can delete elements that are done
    f = open(replaceFileLocation, "w")
    for i_line, line in enumerate(contents):
        replaced = False
        for rvar in replaceVars:
            if rvar + "[i]" in line: # for array variables
                replaced = True
                keepFirstPart = line.split('=', 1)[0]
                f.write(keepFirstPart + '= ' + rvar + '_p;\n')
                print("replaced array " + rvar + " in line " + str(i_line))
                replaceTODOlist = [elem for elem in replaceTODOlist if not elem == rvar]  # check element in todolist
            if rvar + "[0]" in line: # for scalar (shared) variables
                replaced = True
                keepFirstPart = line.split('=', 1)[0]
                f.write(keepFirstPart + '= ' + rvar + '_p;\n')
                print("replaced scalar " + rvar + " in line " + str(i_line))
                replaceTODOlist = [elem for elem in replaceTODOlist if not elem == rvar]  # check element in todolist
        if '.run(' in line:
            replaced = True
            keepNetworkName = line.split('.', 1)[0] #this is actually the same as net.name
            keepSecondPart = line.split(',', 1)[1]
            f.write(keepNetworkName + '.run(duration_p,'+ keepSecondPart)
            print("replaced duration in line " + str(i_line))
            replaceTODOlist = [elem for elem in replaceTODOlist if not elem == 'duration']
        if not replaced:
            f.write(line)
    f.close()
    if len(replaceTODOlist) > 0:
        # maybe we should raise an exception here as this is rather serious?
        warnings.warn("could not find matching variables in cpp code for " + str(replaceTODOlist), Warning)  # warning, items left in todolist
        print('\n* * * * * * * * * * * * * * * * *\n NOT all variables successfully replaced in cpp code! \n')
        print("could not find matching variables in cpp code for " + str(replaceTODOlist))
    else:
        print('\n*********************************\nall variables successfully replaced in cpp code! \n')

def printDict(pdict):
    print('The following parameters are set for this run:')
    for key in pdict:
        print(key, ' = ', pdict[key])


def params2run_args(standaloneParams):
    run_args = [str(asarray(standaloneParams[key])) for key in standaloneParams]  # asarray is to remove units. It is the way proposed in the tutorial
    return run_args


def collectStandaloneParams(params=OrderedDict(), *buildingBlocks):
    '''This just collect the parameters of all buildingblocks and adds additional parameters (not from buildingblocks) '''
    standaloneParams = OrderedDict()
    for block in buildingBlocks:
        standaloneParams.update(block.standaloneParams)

    standaloneParams.update(params)
    return standaloneParams


def run_standalone(standaloneParams, standaloneDir):
    startSim = time.time()
    # run simulation
    printDict(standaloneParams)
    run_args = params2run_args(standaloneParams)
    device.run(directory=standaloneDir, with_output=True, run_args=run_args)
    end = time.time()
    print ('simulation in c++ took ' + str(end - startSim) + ' sec')
    print('simulation done!')
