#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 19:02:05 2017

@author: alpha
"""

from brian2 import implementation,check_units,ms,exp,mean,diff,declare_types,prefs,\
                   figure,subplot,plot,xlim,ylim,ones,zeros,xticks,xlabel,ylabel,device
from brian2 import *
import numpy as np
import os
import time

def buildCppAndReplace(replaceVars, standaloneDir='output'):
    
    startBuild = time.time()
    prefs['codegen.cpp.extra_compile_args_gcc'].append('-std=c++14')
    #prefs['codegen.cpp.extra_compile_args_gcc'].append('-std=c++11')
    device.build(compile=False, run = False, directory=standaloneDir, clean=True, debug=False)
    
    end = time.time()
    print ('build took ' + str(end - startBuild) + ' sec')
    
    #===============================================================================
    # Code that needs to be added to the main.cpp file

    maincppPath = os.path.join(os.getcwd(),standaloneDir,'main.cpp') #this should always be the correct path
    replaceVariablesInCPPcode(replaceVars,replaceFileLocation=maincppPath)
    #===============================================================================
    # compile
    startMake = time.time()  
    #out = check_output(["make","-C","~/Code/SOM_standalone"])
    compiler, args = codegen.cpp_prefs.get_compiler_and_args()
    device.compile_source(directory=standaloneDir, compiler=compiler, clean=True, debug=False)
    #print(out)
    end = time.time()
    print ('make took ' + str(end - startMake) + ' sec')
    print('\n\nstandalone SOM was built and compiled, ready to run!')

def replaceVariablesInCPPcode(replaceVars,replaceFileLocation):
    ''' replaces a list of variables in CPP code for standalone code generation with changeable parameters
    @params:
        replaceVars : List of strings, variables that are replaced
        replaceFileLocation : string, location of the file in which the variables are replaced 
    '''
    # generate arg code
    cppArgCode = ""
    for ivar, rvar in enumerate(replaceVars):
        cppArgCode += """\n	float {replvar}_p = std::stof(argv[{num}],NULL);
	std::cout << "variable {replvar} is argument {num} with value " << {replvar}_p << std::endl;\n""".format(num=(ivar+1),replvar=rvar)
        
    # read main.cpp
    f = open(replaceFileLocation, "r")
    contents = f.readlines()
    f.close()
    
    # insert arg code
    for i_line, line in enumerate(contents):
            if "int main(int argc, char **argv)" in line:
                insertLine = i_line+2
    contents.insert(insertLine, cppArgCode)
    
    # replace var code
    f = open(replaceFileLocation, "w")
    for i_line, line in enumerate(contents):
        replaced = False
        for rvar in replaceVars:
            if rvar+"[i]" in line :
                replaced = True
                keepFirstPart = line.split('=', 1)[0]
                f.write(keepFirstPart + '= ' + rvar + '_p;\n')
                print("replaced " + rvar + " in line " + str(i_line))
        if not replaced:
            f.write(line)
    f.close()

def printRunParams(replaceVars,run_args):
    print('The following parameters are set for this run:')
    for ii in range(len(run_args)):
        print(replaceVars[ii],' = ',run_args[ii])