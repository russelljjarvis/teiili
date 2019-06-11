# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-01-10 15:35:29
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-17 15:50:31

"""This file contains default parameter for dpi dendritic synapse. For more details on model see
models/equations/dpi_synapse.py

Attributes:
    parameters (dict): Synapse parameters
"""

from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second
from teili import constants

parameters = {
    "Igain": 15 * pA,
    'Csyn': 1.5 * pF,
    'Io_syn': constants.I0,
    'I_tau': 10. * pA,
    'Ut_syn': constants.UT,
    'baseweight': 50. * pA,
    'kn_syn': constants.KAPPA_N,
    'kp_syn': constants.KAPPA_P,
    'wPlast': 1,
    'I_th': 10 * pA,
    'I_syn': constants.I0
}
