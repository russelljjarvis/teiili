from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second
from NCSBrian2Lib.Parameters import constants

parameters = {
    'Csyn': 1.5 * pF,
    'Io_syn': constants.Io,
    'Ii_tau': 10. * pA,
    'Ut_syn': constants.Ut,
    'baseweight_i': 50. * pA,
    'kn_syn': constants.kappa_n,
    'kp_syn': constants.kappa_p,
    'wPlast': 1,
    'Ii_th': 10 * pA,
    'Ii_syn': constants.Io
}
