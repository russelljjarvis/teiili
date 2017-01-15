from brian2 import *
import numpy as np


def setParams(briangroup, params, ndargs=None, debug=False):
    for par in params:
        if hasattr(briangroup, par):
            if ndargs is not None and par in ndargs:
                if ndargs[par] is None:
                    setattr(briangroup, par, params[par])
                else:
                    setattr(briangroup, par, ndargs[par])
            else:
                setattr(briangroup, par, params[par])
    if debug:
        states = briangroup.get_states()
        print '\n' 
        print '-_-_-_-_-_-_-_', '\n', 'Parameters set'
        for key in states.keys():
            if key in params:
                print key, states[key]
        print '----------'


# function that calculates 1D index from 2D index
@implementation('numpy', discard_units=True)
@check_units(x0=1,x1=1,n2dNeurons=1,result=1)
def xy2ind(x0,x1,n2dNeurons):
    return int(x0)+int(x1)*n2dNeurons

# function that calculates 2D index from 1D index
@implementation('numpy', discard_units=True)
@check_units(ind=1,n2dNeurons=1,result=1)
def ind2xy(ind,n2dNeurons):
    ret = (np.mod(np.round(ind),n2dNeurons), np.floor_divide(np.round(ind),n2dNeurons))     
    return ret

# from Brian2 Equations class
def replaceEqVar(eq , varname, replacement, debug=False):
    "replaces variables in equations like brian 2, helper for replaceConstants"
    if isinstance(replacement, str):
        eq = eq.replace(varname,replacement)
    else:
        eq = eq.replace(varname,'(' + repr(replacement) + ')')
            
    if debug:
        print('replaced ' + str(varname) + ' by ' + str(repr(replacement)))
    return (eq)


def replaceConstants(equation,replacedict, debug=False):
    "replaces constants in equations and deletes the respective definitions, given a dictionary of replacements"
    for key in replacedict:
        if replacedict[key] is not None:            
            # delete line from model eq
            neweq = ''
            firstline = True
            for line in equation.splitlines():
                if not all([kw in line for kw in [key,'(constant)']]):
                    if firstline:
                        neweq = neweq + line
                        firstline = False
                    else:
                        neweq = neweq  +'\n' + line
                else:
                    print('deleted ' + str(key) + ' from equation constants')
            equation = neweq
            # replace variable in eq with constant
            equation = replaceEqVar(equation ,key,replacedict[key],debug)
    return (equation)
        
