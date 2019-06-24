# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-10-31 16:31:27
# @Last Modified by:   mmilde
# @Last Modified time: 2018-10-31 16:52:15
mismatch_neuron_param = {
    'Inoise': 0,
    'Iconst': 0,
    'kn': 0,
    'kp': 0,
    'Ut': 0,
    'Io': 0,
    'Cmem': 0.2,
    'Iath': 0.2,
    'Iagain': 0.2,
    'Ianorm': 0.2,
    'Ica': 0.2,
    'Itauahp': 0.2,
    'Ithahp': 0.2,
    'Cahp': 0.2,
    'Ishunt': 0.2,
    'Ispkthr': 0.2,
    'Ireset': 0.2,
    'Ith': 0.2,
    'Itau': 0.2,
    'refP': 0.2,
}

mismatch_synap_param = {
    'Io_syn': 0,
    'kn_syn': 0,
    'kp_syn': 0,
    'Ut_syn': 0,
    'Csyn': 0.2,
    'Ie_tau': 0.2,
    'Ii_tau': 0.2,
    'Ie_th': 0.2,
    'Ii_th': 0.2,
    'Ie_syn': 0.2,
    'Ii_syn': 0.2,
    'w_plast': 0,
    'baseweight_e': 0.2,
    'baseweight_i': 0.2,
}


def add_device_mismatch(group, seed=1337, group_type='neuron'):
    if group_type == 'neuron':
        teili.modelgroup.add_mismatch(std_dict=mismatch_neuron_param, seed=seed)
    elif group_type == 'synapse':
        group.add_mismatch(std_dict=mismatch_synap_param, seed=seed)
