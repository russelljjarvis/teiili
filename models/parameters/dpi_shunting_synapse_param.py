# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-01-10 15:35:29
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-17 15:50:36

"""This file contains default parameter for dpi shunting synapse. For more details on model see
models/equations/dpi_shunting_synapse.py

Attributes:
    parameters (dict): Synapse parameters
"""

from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second
from NCSBrian2Lib import constants

parameters = {
    'Csyn': 1.5 * pF,
    'Io_syn': constants.I0,
    'Ii_tau': 10. * pA,
    'Ut_syn': constants.UT,
    'baseweight_i': 50. * pA,
    'kn_syn': constants.KAPPA_N,
    'kp_syn': constants.KAPPA_P,
    'wPlast': 1,
    'Ii_th': 10 * pA,
    'Ii_syn': constants.I0
}
