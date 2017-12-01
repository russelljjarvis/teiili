from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second
from NCSBrian2Lib.Parameters import constants

parameters = {
    "Igain": 15 * pA,
    'Csyn': 1.5 * pF,
    'Io_syn': 0.5 * pA,
    'Ie_tau': 10. * pA,
    'Ii_tau': 10. * pA,
    'Ut_syn': constants.Ut,
    'baseweight_e': 50. * pA,
    'baseweight_i': 50. * pA,
    'kn_syn': constants.kappa_n,
    'kp_syn': constants.kappa_p,
    'wPlast': 1,
    'Ie_th': 10 * pA,
    'Ii_th': 10 * pA,
    'Ie_syn': 0.5 * pA,
    'Ii_syn': 0.5 * pA,
}
