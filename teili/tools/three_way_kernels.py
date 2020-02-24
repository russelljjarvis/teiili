#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 00:23:51 2018

@author: dzenn


This tool provides sets of indices for "hard-wired" connectivity of a threeway
relation unit.
"""


def coord_to_index(i, j, inputSize):
    m = (i+inputSize)%inputSize
    k = (j+inputSize)%inputSize
    return m + (k)*inputSize


def A_plus_B_equals_C(pre_key, post_key, inputSize):

    if pre_key == 'A' and post_key == 'H':
        output_i = set_i('X', inputSize)
        output_j = set_j('X', inputSize)
    elif pre_key == 'H' and post_key == 'A':
        output_i = set_j('X', inputSize)
        output_j = set_i('X', inputSize)
    elif pre_key == 'B' and post_key == 'H':
        output_i = set_i('Y', inputSize)
        output_j = set_j('Y', inputSize)
    elif pre_key == 'H' and post_key == 'B':
        output_i = set_j('Y', inputSize)
        output_j = set_i('Y', inputSize)
    elif pre_key == 'C' and post_key == 'H':
        output_i = set_diagonal_i(inputSize)
        output_j = set_diagonal_j(inputSize)
    elif pre_key == 'H' and post_key == 'C':
        output_i = set_diagonal_j(inputSize)
        output_j = set_diagonal_i(inputSize)
    else:
        raise NotImplementedError('Invalid combination of population keys')
    
    return output_i, output_j
        

def set_i(inputAxis, inputSize):
    output = []

    for k in range(inputSize):
        for m in range(inputSize):
            if inputAxis != 'Z':
                output.append(k) 
            else:
                output.append()
    return output

def set_j(inputAxis, inputSize):
    output = []
    for k in range(inputSize):
        for m in range(inputSize):
            if inputAxis == 'X':
                output.append(coord_to_index(k,m,inputSize))    
            elif inputAxis == 'Y':
                output.append(coord_to_index(m,k,inputSize))  
    return output

def set_diagonal_i(inputSize):
    output = []
    for k in range(inputSize):
        for m in range(inputSize):
            output.append(int(k))
    return output

def set_diagonal_j(inputSize):
    output = []
    for k in range(inputSize):
        for m in range(inputSize):
            output.append(coord_to_index(m, (k - m + inputSize)%inputSize, inputSize))
    return output

