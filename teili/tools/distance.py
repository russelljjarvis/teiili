#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions to compute distance (e.g. in 2D).

The suffix "_cpp" avoids variables being string-replaced by brian2 if the same name
is used in the network.
"""
# @Author: mmilde, alpren
# @Date:   2018-05-30 11:54:09

from brian2 import implementation, check_units, declare_types
import numpy as np
from teili.tools.indexing import ind2xy


@implementation('cpp', '''
    float dist1d2dfloat(float i, float j, int nrows_cpp, int ncols_cpp) {
    int ix = i / ncols_cpp;
    int iy = i % ncols_cpp;
    int jx = j / ncols_cpp;
    int jy = j % ncols_cpp;
    return sqrt(pow((ix - jx),2) + pow((iy - jy),2));
    }
     ''')
@declare_types(i='float', j='float', nrows='integer', ncols='integer', result='float')
@check_units(i=1, j=1, ncols=1, nrows=1, result=1)
def dist1d2dfloat(i, j, nrows, ncols):
    """Function that calculates distance in 2D field from two 1D indices.

    Args:
        i (float, required): 1D index of source neuron.
        j (float, required): 1D index of target neuron.
        nrows (int, required): number of rows of 2D neuron population.
        ncols (int, required): number of colums of 2D neuron population.

    Returns:
        float: Distance in 2D field.
    """
    (ix, iy) = ind2xy(i, nrows, ncols)
    (jx, jy) = ind2xy(j, nrows, ncols)
    return dist2d2dfloat(ix, iy, jx, jy)


@implementation('cpp', '''
    float dist1d2dint(int i, int j, int nrows_cpp, int ncols_cpp) {
    int ix = i / ncols_cpp;
    int iy = i % ncols_cpp;
    int jx = j / ncols_cpp;
    int jy = j % ncols_cpp;
    return sqrt(pow((ix - jx),2) + pow((iy - jy),2));
    }
     ''')
@declare_types(i='integer', j='integer', nrows='integer', ncols='integer', result='float')
@check_units(i=1, j=1, ncols=1, nrows=1, result=1)
def dist1d2dint(i, j, nrows, ncols):
    """Function that calculates distance in 2D field from two 1D indices.

    Args:
        i (int, required): 1D index of source neuron.
        j (int, required): 1D index of target neuron.
        nrows (int, required): number of rows of 2D neuron population.
        ncols (int, required): number of colums of 2D neuron population.

    Returns:
        int: Distance in 2D field.
    """
    (ix, iy) = ind2xy(i, nrows, ncols)
    (jx, jy) = ind2xy(j, nrows, ncols)
    return dist2d2dint(ix, iy, jx, jy)


@implementation('cpp', '''
    float dist2d2dint(int ix_cpp, int iy_cpp,int jx_cpp, int jy_cpp) {
    return sqrt(pow((ix_cpp - jx_cpp),2) + pow((iy_cpp - jy_cpp),2));
    }
     ''')
@declare_types(ix='integer', iy='integer', jx='integer', jy='integer', result='float')
@check_units(ix=1, iy=1, jx=1, jy=1, result=1)
def dist2d2dint(ix, iy, jx, jy):
    """Function that calculates distance in 2D field from four integer 2D indices.

    Args:
        ix (int, required): x component of 2D source neuron coordinate.
        iy (int, required): y component of 2D source neuron coordinate.
        jx (int, required): x component of 2D target neuron coordinate.
        jy (int, required): y component of 2D target neuron coordinate.

    Returns:
        int: Distance in 2D field.
    """
    return np.sqrt((ix - jx)**2 + (iy - jy)**2)


@implementation('cpp', '''
    float dist2d2dfloat(float ix_cpp, float iy_cpp,float jx_cpp, float jy_cpp) {
    return sqrt(pow((ix_cpp - jx_cpp),2) + pow((iy_cpp - jy_cpp),2));
    }
     ''')
@declare_types(ix='float', iy='float', jx='float', jy='float', result='float')
@check_units(ix=1, iy=1, jx=1, jy=1, result=1)
def dist2d2dfloat(ix, iy, jx, jy):
    """Function that calculates distance in 2D field from four 2D position values.

    Args:
        ix (float, required): x component of 2D source neuron coordinate.
        iy (float, required): y component of 2D source neuron coordinate.
        jx (float, required): x component of 2D target neuron coordinate.
        jy (float, required): y component of 2D target neuron coordinate.

    Returns:
        float: Distance in 2D field.
    """
    return np.sqrt((ix - jx)**2 + (iy - jy)**2)


# this is not consistent with the other functions as this assumes normalized x and y coordinates
@implementation('cpp', '''
    float torus_dist2d2dfloat(float ix_cpp, float iy_cpp, float jx_cpp, float jy_cpp) {
    float xdiff = ix_cpp - jx_cpp;
    float ydiff = iy_cpp - jy_cpp;
    float one = 1.0;
    float dx = min( min(abs(xdiff), abs(xdiff + one)), abs(xdiff - one));
    float dy = min( min(abs(ydiff), abs(ydiff + one)), abs(ydiff - one));

    return sqrt(pow(dx,2) + pow(dy,2));
    }
     ''')
@declare_types(ix='float', iy='float', jx='float', jy='float', result='float')
@check_units(ix=1, iy=1, jx=1, jy=1, result=1)
def torus_dist2d2dfloat(ix, iy, jx, jy):
    """Function that calculates distance in torus (field with periodic boundary conditions),
    !!! assuming that width and length are 1.

    Args:
        ix (float, required): x component of 2D source neuron coordinate.
        iy (float, required): y component of 2D source neuron coordinate.
        jx (float, required): x component of 2D target neuron coordinate.
        jy (float, required): y component of 2D target neuron coordinate.

    Returns:
        float: Distance in 2D field with periodic boundary conditions.
    """
    xdiff = ix - jx
    ydiff = iy - jy
    dx = np.minimum(np.minimum(abs(xdiff), abs(xdiff + 1.0)), abs(xdiff - 1.0))
    dy = np.minimum(np.minimum(abs(ydiff), abs(ydiff + 1.0)), abs(ydiff - 1.0))
    return np.sqrt(dx**2 + dy**2)


# this is not consistent with the other functions as this assumes normalized x and y coordinates
@implementation('cpp', '''
    float torus_dist2d2dfloat(float ix_cpp, float iy_cpp, float jx_cpp, float jy_cpp) {

    float dx = min( min(abs(ix_cpp - jx_cpp), abs(ix_cpp - jx_cpp + 1.0)), abs(ix_cpp - jx_cpp - 1.0));
    float dy = min( min(abs(iy_cpp - jy_cpp), abs(iy_cpp - jy_cpp + 1.0)), abs(iy_cpp - jy_cpp - 1.0));

    return sqrt(pow(dx,2) + pow(dy,2));
    }
     ''')
@declare_types(ix='float', iy='float', jx='float', jy='float', result='float')
@check_units(ix=1, iy=1, jx=1, jy=1, result=1)
def torus_dist2d2dfloat_backup(ix, iy, jx, jy):
    """Function that calculates distance in torus (field with periodic boundary conditions),
    !!! assuming that width and length are 1.

    Args:
        ix (float, required): x component of 2D source neuron coordinate.
        iy (float, required): y component of 2D source neuron coordinate.
        jx (float, required): x component of 2D target neuron coordinate.
        jy (float, required): y component of 2D target neuron coordinate.

    Returns:
        float: Distance in 2D field with periodic boundary conditions.
    """
    xdiff = ix - jx
    ydiff = iy - jy
    dx = np.minimum(np.minimum(abs(xdiff), abs(xdiff + 1.0)), abs(xdiff - 1.0))
    dy = np.minimum(np.minimum(abs(ydiff), abs(ydiff + 1.0)), abs(ydiff - 1.0))
    return np.sqrt(dx**2 + dy**2)


def circle_dist1d(x,y,N):
    return np.min([np.abs(x-y),np.abs(x-y+N),np.abs(x-y-N)])
