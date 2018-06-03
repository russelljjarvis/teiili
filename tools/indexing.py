# -*- coding: utf-8 -*-
# @Author: mmilde, alpren
# @Date:   2018-01-09 17:26:00
# @Last Modified by:   alpren
# @Last Modified time: 2018-05-30
"""
Collections of functions which convert indices to x, y coordinates and vice versa

the suffix "_cpp" avoids that variables are string replaced by brian2 if the same name
is used in the network

"""
from brian2 import implementation, check_units, declare_types
import numpy as np


@implementation('cpp', '''
    inline int xy2ind(int x, int y, int nrows_cpp, int ncols_cpp) {
    return x*ncols_cpp+y;
    }
     ''')
@declare_types(x='integer', y='integer', nrows='integer', ncols='integer', result='integer')
@check_units(x=1, y=1, nrows=1, ncols=1, result=1)
def xy2ind(x, y, nrows, ncols):
    """Given a pair of x, y (pixel) coordinates this function
    will return an index that correspond to a flattened pixel array
    It is a wrapper around np.ravel_multi_index with a cpp implementation

    Beware that the cpp version does not check if your input is OK
    (if your coordinates are actually inside of the array)


    Args:
        x (int, required): x-coordinate
        y (int, rquired): y-coordinate
        nrows (int, required): number of rows of the 2d array
        ncols (int, required): number of cols of the 2d array

    Returns:
        ind (int): Converted index (e.g. flattened array)
    """
    return np.ravel_multi_index((x, y), (nrows, ncols))
    # return x*ncols+y


@implementation('cpp', '''
    inline int ind2x(int ind, int nrows_cpp, int ncols_cpp) {
    return ind / ncols_cpp;
    }
     ''')
@declare_types(ind='integer', nrows='integer', ncols='integer', result='integer')
@check_units(ind=1, nrows=1, ncols=1, result=1)
def ind2x(ind, nrows, ncols):
    """Given an index of an array this function will provide
    you with the corresponding x coordinate

    Beware that the cpp version does not check if your input is OK
    (if your coordinates are actually inside of the array)

    Args:
        ind (int, required): index of flattened array that should be converted back to pixel cooridnates
        n2dNeurons (int, required): Longest edge of the original array

    Returns:
        x (int): The x coordinate of the respective index in the unflattened array
    """
    return np.unravel_index(ind, (nrows, ncols))[0]
    # return int(ind/ncols)


@implementation('cpp', '''
    inline int ind2y(int ind, int nrows_cpp, int ncols_cpp) {
    return ind % ncols_cpp;
    }
     ''')
@declare_types(ind='integer', nrows='integer', ncols='integer', result='integer')
@check_units(ind=1, nrows=1, ncols=1, result=1)
def ind2y(ind, nrows, ncols):
    """Given an index of an array this function will provide
    you with the corresponding y coordinate

    Beware that the cpp version does not check if your input is OK
    (if your coordinates are actually inside of the array)

    Args:
        ind (int, required): index of flattened array that should be converted back to pixel cooridnates
        n2dNeurons (int, required): Longest edge of the original array

    Returns:
        y (int): The y coordinate of the respective index in the unflattened array
    """
    return np.unravel_index(ind, (nrows, ncols))[1]
    # return int(ind%ncols)


@implementation('numpy', discard_units=True)
@declare_types(ind='integer', nrows='integer', ncols='integer',  result='integer')
@check_units(ind=1, nrows=1, ncols=1, result=1)
def ind2xy(ind, nrows, ncols):
    """Given an index of an array this function will provide
    you with the corresponding x and y coordinate of the original array
    This is basically a wrapper around numpys unravel index

    We do not provide a cpp implementation here, because it would return an array,
    which cannot easily be indexed, so please use ind2x and ind2y for that purpose!

    Args:
        ind (int, required): index of flattened array that should be converted back to pixel cooridnates
        n2dNeurons (int, required): Longest edge of the original array

    Returns:
        tuple (x, y): The corresponding x, y coordinates
    """
    return np.unravel_index(ind, (nrows, ncols))
    # return (int(ind/ncols), int(ind%ncols))


# TODO: make this consistent with the other functions in this module! (or maybe it is not at the right place here)
@implementation('numpy', discard_units=True)
@declare_types(ind='integer', ts='integer', pol='boolean', n2dNeurons='integer', result='integer')
@check_units(ind=1, ts=1, pol=1, n2dNeurons=1, result=1)
def ind2events(ind, ts, pol=True, n2dNeurons=10):
    """This function converts spikes from a brain2 group into an
    event-like structure as provided by a DVS. Events will have the structure
    of (x, y, ts, pol)

    Args:
        ind (TYPE): index of neurons that spikes (brian2group.i)
        ts (TYPE): times when neurons spikes (brian2group.t)
        pol (None, optional): Either vector with same length as ind or None
        n2dNeurons (int, required): Longest edge of the original array
    """
    x, y = np.unravel_index(ind, (n2dNeurons, n2dNeurons))
    events = np.zeros((4, len(x)))
    events[0, :] = np.asarray(x)
    events[1, :] = np.asarray(y)
    events[2, :] = np.asarray(ts)
    if pol:
        events[3, :] = np.ones((len(x)))
    elif not pol:
        events[3, :] = np.zeros((len(x)))
    elif len(pol) > 1:
        events[3, :] = pol
    return events
