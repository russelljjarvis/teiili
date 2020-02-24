#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from teili.models.parameters.octa_params import mismatch_neuron_param,\
    mismatch_synap_param


def add_bb_mismatch(bb, seed=42):
    """This allows to add mismatch to all the neuron and connection groups
    present in a building block.

    args:
        bb (type): building block object to which mismatch should be added
        seed (int, optional): random seed to sample the mismatch from
    """
    for i in bb.groups:
        if bb.groups[i]._tags['group_type'] == 'Neuron':
            bb.groups[i].add_mismatch(mismatch_neuron_param, seed=seed)
            bb.groups[i]._tags['mismatch'] = True
        elif bb.groups[i]._tags['group_type'] == 'Connection':
            bb.groups[i].add_mismatch(mismatch_synap_param, seed=seed)
            bb.groups[i]._tags['mismatch'] = True
