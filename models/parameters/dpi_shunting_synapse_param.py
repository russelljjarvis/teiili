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
