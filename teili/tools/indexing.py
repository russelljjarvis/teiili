# -*- coding: utf-8 -*-
"""Collections of functions which convert indices to x, y coordinates and vice versa.

The suffix "_cpp" avoids variables being string-replaced by brian2 if the same name
is used in the network.

Todo:
    * TODO: make `ind2events` consistent with the other functions
        in this module! (or maybe it is not at the right place here).
"""
# @Author: mmilde, alpren
# @Date:   2018-01-09 17:26:00

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
    will return an index that corresponds to a flattened pixel array.

    It is a wrapper around np.ravel_multi_index with a cpp implementation

    Beware that the cpp version does not check if your input is OK
    (i.e. whether your coordinates are actually inside the array).


    Args:
        x (int, required): x-coordinate.
        y (int, rquired): y-coordinate.
        nrows (int, required): number of rows of the 2d array.
        ncols (int, required): number of cols of the 2d array.

    Returns:
        ind (int): Converted index (e.g. flattened array).
    """
    return np.ravel_multi_index((x, y), (nrows, ncols))


@implementation('cpp', '''
    inline int ind2x(int ind_cpp, int nrows_cpp, int ncols_cpp) {
    return ind_cpp / ncols_cpp;
    }
     ''')
@declare_types(ind='integer', nrows='integer', ncols='integer', result='integer')
@check_units(ind=1, nrows=1, ncols=1, result=1)
def ind2x(ind, nrows, ncols):
    """Given an index of an array this function will provide
    you with the corresponding x coordinate.

    Beware that the cpp version does not check if your input is OK
    (whether your coordinates are actually inside the array).

    Args:
        ind (int, required): index of flattened array that should be converted
            back to pixel cooridnates.
        nrows(int, required): Longest edge of the original array.
        ncols(int, required): Shortest edge of the original array.

    Returns:
        x (int): The x coordinate of the respective index in the unflattened array.
    """
    return np.unravel_index(ind, (nrows, ncols))[0]


@implementation('cpp', '''
    inline int ind2y(int ind_cpp, int nrows_cpp, int ncols_cpp) {
    return ind_cpp % ncols_cpp;
    }
     ''')
@declare_types(ind='integer', nrows='integer', ncols='integer', result='integer')
@check_units(ind=1, nrows=1, ncols=1, result=1)
def ind2y(ind, nrows, ncols):
    """Given an index of an array this function will provide
    you with the corresponding y coordinate.

    Beware that the cpp version does not check if your input is OK
    (whether your coordinates are actually inside the array).

    Args:
        ind (int, required): index of flattened array that should be converted
            back to pixel cooridnates.
        nrows (int, required): Longest edge of the original array.
        ncols (int, required): Shortest edge of the original array.

    Returns:
        y (int): The y coordinate of the respective index in the unflattened array.
    """
    return np.unravel_index(ind, (nrows, ncols))[1]


@implementation('numpy', discard_units=True)
@declare_types(ind='integer', nrows='integer', ncols='integer', result='integer')
@check_units(ind=1, nrows=1, ncols=1, result=1)
def ind2xy(ind, nrows, ncols):
    """Given an index of an array this function will provide
    you with the corresponding x and y coordinate of the original array.

    This is basically a wrapper around numpy's unravel index

    We do not provide a cpp implementation here, because it would return an array,
    which cannot easily be indexed, so please use ind2x and ind2y for that purpose!

    Args:
        ind (int, required): index of flattened array that should be converted
            back to pixel cooridnates.
        nrows (int, required): Longest edge of the original array.
        ncols (int, required): Shortest edge of the original array.

    Returns:
        tuple (x, y): The corresponding x, y coordinates.
    """
    return np.unravel_index(ind, (nrows, ncols))


@implementation('numpy', discard_units=True)
@declare_types(ind='integer', ts='integer', pol='boolean', nrows='integer', ncols='integer', result='integer')
@check_units(ind=1, ts=1, pol=1, nrows=1, ncols=1, result=1)
def ind2events(ind, ts, pol=True, nrows=10, ncols=10):
    """This function converts spikes from a brian2 group into an
    event-like structure as provided by a DVS.

    Events will have the structure of (x, y, ts, pol).

    Args:
        ind (TYPE): index of neurons that spikes (brian2group.i).
        ts (TYPE): times when neurons spikes (brian2group.t).
        pol (None, optional): Either vector with same length as ind or None.
        nrows (int, required): Longest edge of the original array.
        ncols (int, required): Shortest edge of the original array.
    """
    x, y = np.unravel_index(ind, (nrows, ncols))
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
